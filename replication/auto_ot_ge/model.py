# Save this as auto_ot_ge/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from typing import List, Optional, Tuple

# Import the sinkhorn function
from .sinkhorn import sinkhorn_log_domain

# --- 1. Added Minimal AutoEncoder dependency ---
# (This was missing from your provided code)
class AutoEncoder(nn.Module):
    """Minimal AutoEncoder stub required by GraphEncoder."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Simple weight initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

# --- 2. Your GraphEncoder class, modified for OT ---

class GraphEncoder(torch.nn.Module):
    def __init__(
        self: 'GraphEncoder', 
        input_dim: int, 
        hidden_dims: List[int],
        k: int  # Number of clusters
    ) -> None:
        """
        Initialize the GraphEncoder.
        
        Args:
            input_dim (int): Dimension of the input features (e.g., n).
            hidden_dims (List[int]): List of hidden layer dimensions.
                                      The last one is the latent_dim.
            k (int): The number of clusters.
        """
        super().__init__()
        self.autoencoders = torch.nn.ModuleList()
        prev_dim = input_dim
        
        if not hidden_dims:
            raise ValueError("hidden_dims list cannot be empty")
            
        for hidden_dim in hidden_dims:
            self.autoencoders.append(AutoEncoder(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            
        self.latent_dim = prev_dim
        self.k = k

        # --- 3. Initialize Cluster Centroids ---
        # We register this as a Parameter so it's part of the model state,
        # but we will update it manually in the M-step, not via gradients.
        self.cluster_centroids = nn.Parameter(
            torch.empty(self.k, self.latent_dim),
            requires_grad=True # Gradients flow to Z, but M is updated manually
        )
        nn.init.xavier_uniform_(self.cluster_centroids.data)

        # Store initial clustering from pre-trained embeddings
        self.register_buffer('initial_assignments', torch.zeros(input_dim, dtype=torch.long))


    # Your existing forward/train methods for pre-training
    # (I've added type hints and fixed a small bug)
    
    def forward(
        self: 'GraphEncoder', 
        X: torch.Tensor, 
        train_mode: str, 
        **kwargs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if train_mode == 'layerwise':
            layer_number = kwargs.get('layer_number', None)
            if layer_number is None or layer_number < 0 or layer_number >= len(self.autoencoders):
                raise ValueError("Invalid layer number for layerwise training")
            
            # Use the forward method of the AutoEncoder stub
            encoded, decoded = self.autoencoders[layer_number](X)
            return encoded, decoded
            
        elif train_mode == 'endtoend':
            # Encode
            encoded = self.encode(X)
            # Decode
            X_decoded = encoded
            for autoencoder in reversed(self.autoencoders):
                X_decoded = torch.sigmoid(autoencoder.decoder(X_decoded))
            decoded = X_decoded
            return encoded, decoded
        
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")
    
    def train_pretrain(
        self: 'GraphEncoder',
        X: torch.Tensor,
        compile_model: bool, # 'compile' is a bad variable name
        train_mode: str,
        iters: int,
        optimizer: torch.optim.Optimizer,
        rho: float = 0.01,
        beta: float = 1.0,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Pre-trains the autoencoder, either layer-wise or end-to-end.
        This is your original 'train' method, renamed for clarity.
        """
        if batch_size is None:
            batch_size : int = X.shape[0]

        train_model = torch.compile(self) if compile_model else self
        
        if train_mode == 'layerwise':
            X_layer_input = X
            for layer_number in range(len(self.autoencoders)):
                # Note: You should re-create the optimizer for each layer
                # if parameters are different, but we'll assume one optimizer
                # for the whole 'self' model.
                pbar = tqdm.tqdm(
                    range(iters), 
                    desc=f"Pre-training layer {layer_number}"
                )
                for _ in pbar:
                    # Handle case where n < batch_size
                    b_size = min(X_layer_input.shape[0], batch_size)
                    batch_idx = torch.randint(0, X_layer_input.shape[0] - b_size + 1, (1,)).item()
                    X_batch = X_layer_input[batch_idx : batch_idx + b_size]
                    
                    optimizer.zero_grad()
                    encoded, decoded = train_model(
                        X_batch, 
                        train_mode='layerwise', 
                        layer_number=layer_number
                    )
                    
                    loss_1 = F.mse_loss(decoded, X_batch, reduction='sum')
                    
                    # KL sparsity [cite: 135]
                    rho_hat = torch.mean(encoded, dim=0)
                    # Add clamp for numerical stability
                    rho_hat = torch.clamp(rho_hat, min=1e-6, max=1.0 - 1e-6) 
                    loss_2 = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
                    
                    loss = loss_1 + beta * loss_2
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
                    
                # Get the output for the next layer
                with torch.no_grad():
                    X_layer_input = torch.sigmoid(
                        self.autoencoders[layer_number].encoder(X_layer_input)
                    ).detach()

        elif train_mode == 'endtoend':
            pbar = tqdm.tqdm(range(iters), desc="Pre-training end-to-end")
            for _ in pbar:
                b_size = min(X.shape[0], batch_size)
                batch_idx = torch.randint(0, X.shape[0] - b_size + 1, (1,)).item()
                X_batch = X[batch_idx : batch_idx + b_size]
                
                optimizer.zero_grad()
                encoded, decoded = train_model(X_batch, train_mode='endtoend')
                
                loss_1 = F.mse_loss(decoded, X_batch, reduction='sum')
                
                rho_hat = torch.mean(encoded, dim=0)
                # Add clamp for numerical stability
                rho_hat = torch.clamp(rho_hat, min=1e-6, max=1.0 - 1e-6) 
                loss_2 = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
            
                loss = loss_1 + beta * loss_2
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

    
    def encode(self: 'GraphEncoder', X: torch.Tensor) -> torch.Tensor:
        X_encoded = X
        for autoencoder in self.autoencoders:
            X_encoded = torch.sigmoid(autoencoder.encoder(X_encoded))
        return X_encoded
    
    
    def decode(self: 'GraphEncoder', Z: torch.Tensor) -> torch.Tensor:
        X_decoded = Z
        for autoencoder in reversed(self.autoencoders):
            X_decoded = torch.sigmoid(autoencoder.decoder(X_decoded))
        return X_decoded

    
    def initialize_centroids(
        self, 
        X: torch.Tensor,
        pretrain_optimizer: torch.optim.Optimizer,
        pretrain_iters: int = 100
    ) -> None:
        """
        Initializes centroids by pre-training the AE and running k-means.
        Also stores the initial cluster assignments.
        """
        print("Initializing centroids: Pre-training autoencoder...")
        self.train_pretrain(
            X=X,
            compile_model=False,
            train_mode='layerwise',
            iters=pretrain_iters,
            optimizer=pretrain_optimizer,
            batch_size=256
        )
        
        print("Initializing centroids: Running k-means...")
        Z = self.encode(X).detach().cpu().numpy()
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        kmeans.fit(Z)
        
        # Set centroids
        initial_centroids = torch.tensor(
            kmeans.cluster_centers_,
            dtype=self.cluster_centroids.dtype,
            device=self.cluster_centroids.device
        )
        self.cluster_centroids.data = initial_centroids
        
        # Save initial cluster assignments
        self.initial_assignments = torch.tensor(
            kmeans.labels_,
            dtype=torch.long,
            device=X.device
        )
        print("Centroids initialized and initial clustering stored.")
    
    @torch.no_grad()
    def get_initial_assignments(self) -> torch.Tensor:
        """
        Returns the initial k-means assignments from pre-training.
        """
        return self.initial_assignments

    def compute_cost_matrix(
        self: 'GraphEncoder',
        Z: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the squared Euclidean cost matrix ||z_i - mu_j||^2.
        
        
        Args:
            Z (torch.Tensor): Embeddings (n, d_emb).
            M (torch.Tensor): Centroids (k, d_emb).
            
        Returns:
            torch.Tensor: Cost matrix (n, k).
        """
        # C(z, mu) = ||z||^2 + ||mu||^2 - 2 <z, mu>
        Z_norm_sq = torch.sum(Z**2, dim=1).unsqueeze(1)  # (n, 1)
        M_norm_sq = torch.sum(M**2, dim=1).unsqueeze(0)  # (1, k)
        
        Z_M_dot = torch.mm(Z, M.t())  # (n, k)
        
        cost = Z_norm_sq + M_norm_sq - 2.0 * Z_M_dot
        
        # Clamp for numerical stability (distances can't be negative)
        return torch.clamp(cost, min=0.0)

    # --- 4. Main Auto-OT-GE Training Loop ---
    def train_auto_ot_em(
        self: 'GraphEncoder',
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        T_epochs: int,
        gamma: float,       # OT loss weight [cite: 172]
        epsilon_0: float,   # Initial epsilon [cite: 189]
        rho: float,         # Annealing rate [cite: 190]
        beta: float = 1.0,  # Sparsity weight
        rho_kl: float = 0.01 # Sparsity target
    ) -> None:
        """
        Trains the Auto-OT-GE model using the EM-style loop.
        Implements Algorithm 1 from the paper. 
        """
        
        # Define the target marginals for OT
        n = X.shape[0]
        # a_i = 1/n (uniform over samples) [cite: 255]
        a = torch.full((n,), 1.0 / n, device=X.device, dtype=X.dtype)
        # w_j = 1/k (uniform over clusters) [cite: 255]
        w = torch.full((self.k,), 1.0 / self.k, device=X.device, dtype=X.dtype)
        
        pbar = tqdm.tqdm(range(T_epochs), desc="Training Auto-OT-GE")
        for t in pbar:
            # --- 1. Encoder Forward Pass & Reconstruction ---
            optimizer.zero_grad()
            
            # Get current embeddings Z [cite: 239]
            Z = self.encode(X) 
            # Get reconstruction X_hat
            X_hat = self.decode(Z)
            
            # Compute Reconstruction Loss [cite: 245]
            loss_rec = F.mse_loss(X_hat, X, reduction='sum')
            
            # Compute KL Sparsity Loss [cite: 135]
            # Compute KL Sparsity Loss
            rho_hat = torch.mean(Z, dim=0)
            # Add clamp for numerical stability
            rho_hat = torch.clamp(rho_hat, min=1e-6, max=1.0 - 1e-6)
            kl_div = rho_kl * torch.log(rho_kl / rho_hat) + \
                     (1 - rho_kl) * torch.log((1 - rho_kl) / (1 - rho_hat))
            loss_kl = beta * torch.sum(kl_div)

            # --- 2. Anneal Regularization [cite: 248] ---
            epsilon_t = epsilon_0 * (rho ** (t / T_epochs))
            
            # --- 3. Compute Cost Matrix C(Z, M) ---
            # We use .detach() on centroids so that gradients from C
            # only flow back to Z, not to M.
            M_t = self.cluster_centroids.detach()
            cost_matrix = self.compute_cost_matrix(Z, M_t) # 

            # --- 4. E-Step: Compute OT plan pi [cite: 254] ---
            with torch.no_grad(): # E-step is non-differentiable w.r.t pi
                pi_t = sinkhorn_log_domain(
                    cost_matrix.detach(), # Detach cost matrix for E-step
                    epsilon=epsilon_t,
                    a=a,
                    b=w,
                    n_iters=50
                )

            # --- 5. M-Step: Update Centroids M [cite: 260] ---
            # M-step is also done outside the gradient graph
            with torch.no_grad():
                # mu_j = (sum_i pi_ij * z_i) / (sum_i pi_ij)
                sum_pi_j = pi_t.sum(dim=0) # Shape (k,)
                # Avoid division by zero if a cluster is empty
                sum_pi_j = torch.clamp(sum_pi_j, min=1e-8)
                
                # (k, n) @ (n, d_emb) -> (k, d_emb)
                new_centroids = (pi_t.t() @ Z) / sum_pi_j.unsqueeze(1)
                
                # Update centroids in-place [cite: 264]
                self.cluster_centroids.data = new_centroids

            # --- 6. Encoder Update: Compute Differentiable OT Loss ---
            # Now we compute the loss for the encoder.
            # We use the *updated* centroids M_t+1 (self.cluster_centroids)
            # and the *current* embeddings Z.
            # This is L_total(Theta) = L_rec + gamma * L_OT(Z_Theta, M_t+1)
            # as per [cite: 183]
            
            # Re-compute cost matrix with *updated* centroids
            cost_matrix_for_loss = self.compute_cost_matrix(
                Z, 
                self.cluster_centroids.detach() # Detach M for the backward pass
            )
            
            # Compute the OT loss: L_OT = <C, pi> [cite: 155]
            # We use the pi_t from the E-step, treated as a constant.
            loss_ot = torch.sum(cost_matrix_for_loss * pi_t)
            
            # --- 7. Total Loss & Backward Pass [cite: 271] ---
            loss_total = loss_rec + loss_kl + gamma * loss_ot
            
            loss_total.backward()
            optimizer.step()
            
            pbar.set_postfix(
                loss=loss_total.item(),
                rec=loss_rec.item(),
                ot=loss_ot.item(),
                eps=epsilon_t
            )

    # --- 5. (Option 2) Fully Joint Training (My addition) ---
    def train_auto_ot_joint(
        self: 'GraphEncoder',
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        T_epochs: int,
        gamma: float,       # OT loss weight [cite: 172]
        epsilon_0: float,   # Initial epsilon [cite: 189]
        rho: float,         # Annealing rate [cite: 190]
        beta: float = 1.0,  # Sparsity weight
        rho_kl: float = 0.01 # Sparsity target
    ) -> None:
        """
        Trains the Auto-OT-GE model using a fully joint optimization.
        Both the Autoencoder (Theta) and Centroids (M) are updated
        via backpropagation from a single composite loss. 
        
        **IMPORTANT**: The 'optimizer' MUST be initialized with
        list(self.parameters()) so it includes 'self.cluster_centroids'.
        """
        
        n = X.shape[0]
        a = torch.full((n,), 1.0 / n, device=X.device, dtype=X.dtype) # [cite: 255]
        w = torch.full((self.k,), 1.0 / self.k, device=X.device, dtype=X.dtype) # [cite: 255]
        
        pbar = tqdm.tqdm(range(T_epochs), desc="Training Auto-OT (Joint)")
        for t in pbar:
            # --- 1. Zero Gradients ---
            optimizer.zero_grad()
            
            # --- 2. Anneal Regularization [cite: 248] ---
            epsilon_t = epsilon_0 * (rho ** (t / T_epochs))

            # --- 3. Forward Pass (Encoder -> Z, Decoder -> X_hat) ---
            Z = self.encode(X) # [cite: 239]
            X_hat = self.decode(Z) # [cite: 243]
            
            # --- 4. Compute Reconstruction & Sparsity Losses ---
            loss_rec = F.mse_loss(X_hat, X, reduction='sum') # [cite: 245]
            
            rho_hat = torch.mean(Z, dim=0).clamp(min=1e-6, max=1.0 - 1e-6)
            kl_div = rho_kl * torch.log(rho_kl / rho_hat) + \
                     (1 - rho_kl) * torch.log((1 - rho_kl) / (1 - rho_hat))
            loss_kl = beta * torch.sum(kl_div) # [cite: 135]
            
            # --- 5. Compute Differentiable OT Loss ---
            
            # Get current centroids (which are nn.Parameters)
            M = self.cluster_centroids # [cite: 149]
            
            # Compute cost matrix. Gradients will flow to Z and M.
            cost_matrix = self.compute_cost_matrix(Z, M) # [cite: 252]
            
            # Compute OT plan pi.
            # This is done *inside* the gradient graph.
            pi_t = sinkhorn_log_domain(
                cost_matrix,
                epsilon=epsilon_t,
                a=a,
                b=w,
                n_iters=50
            ) # [cite: 167-168]
            
            # Compute the entropic OT loss value [cite: 165, 269]
            # L_OT_e = <C, pi> - e * H(pi)
            # H(pi) = -sum(pi * log(pi))
            transport_cost_term = torch.sum(pi_t * cost_matrix)
            # Add entropy term (note: paper loss has -epsilon * H(pi))
            entropy_term = epsilon_t * torch.sum(pi_t * torch.log(pi_t + 1e-10))
            loss_ot = transport_cost_term + entropy_term
            
            # --- 6. Total Loss & Backward Pass [cite: 271] ---
            loss_total = loss_rec + loss_kl + gamma * loss_ot
            
            loss_total.backward()
            optimizer.step()
            
            pbar.set_postfix(
                loss=loss_total.item(),
                rec=loss_rec.item(),
                ot=loss_ot.item(),
                eps=epsilon_t
            )
    
    @torch.no_grad()
    def get_cluster_assignments(self: 'GraphEncoder', X: torch.Tensor) -> torch.Tensor:
        """
        Computes the final hard cluster assignments by finding the
        closest centroid for each node's embedding.
        
        Args:
            X (torch.Tensor): The input feature matrix (n, d_in).
            
        Returns:
            torch.Tensor: The cluster assignment for each node (n,).
        """
        self.eval() # Set model to evaluation mode
        
        # 1. Get embeddings
        Z = self.encode(X) # (n, d_emb)
        
        # 2. Get final centroids
        M = self.cluster_centroids # (k, d_emb)
        
        # 3. Compute cost matrix
        cost_matrix = self.compute_cost_matrix(Z, M) # (n, k)
        
        # 4. Find the closest centroid for each node
        # argmin_j ||z_i - mu_j||^2
        assignments = torch.argmin(cost_matrix, dim=1)
        
        return assignments