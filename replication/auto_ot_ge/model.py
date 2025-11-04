# Save this as auto_ot_ge/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
        k: int,  # Number of clusters,
        X: torch.Tensor
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
        self.number_clusters = k

        # --- 3. Initialize Cluster Centroids ---
        # We register this as a Parameter so it's part of the model state,
        # but we will update it manually in the M-step, not via gradients.
        self.cluster_centroids = nn.Parameter(
            torch.empty(self.number_clusters, self.latent_dim),
            requires_grad=True # Gradients flow to Z, but M is updated manually
        )
        nn.init.xavier_uniform_(self.cluster_centroids.data)

        # Initialize cluster logits for soft assignments
        self.centroid_logits = torch.nn.Parameter(
            torch.randn(X.shape[0], self.number_clusters)  # n × k
        )

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

    
    # --- 1. MODIFIED initialize_centroids function ---
    def initialize_centroids(
        self, 
        X: torch.Tensor,
        pretrain_optimizer: torch.optim.Optimizer,
        pretrain_iters: int = 100,
        plot_initialization: bool = True # --- ADDED ARGUMENT ---
    ) -> None:
        """
        Initializes centroids by pre-training the AE and running k-means.
        Also stores the initial cluster assignments.
        """
        print("Initializing centroids: Pre-training autoencoder...")
        self.train_pretrain(
            X=X,
            compile_model=False,
            train_mode='endtoend',
            iters=pretrain_iters,
            optimizer=pretrain_optimizer,
            batch_size=256
        )
        
        print("Initializing centroids: Running k-means...")
        # Get embeddings on CPU as numpy for sklearn
        Z_cpu_np = self.encode(X).detach().cpu().numpy()
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.number_clusters, n_init=10, random_state=42)
        kmeans.fit(Z_cpu_np)
        
        # Set centroids (on the model's device)
        initial_centroids_np = kmeans.cluster_centers_
        initial_centroids = torch.tensor(
            initial_centroids_np,
            dtype=self.cluster_centroids.dtype,
            device=self.cluster_centroids.device
        )
        self.cluster_centroids.data = initial_centroids
        
        # Save initial cluster assignments (on the model's device)
        initial_assignments_np = kmeans.labels_
        self.initial_assignments = torch.tensor(
            initial_assignments_np,
            dtype=torch.long,
            device=X.device
        )
        print("Centroids initialized and initial clustering stored.")

        # --- ADDED PLOTTING BLOCK ---
        if plot_initialization:
            print("Plotting k-means initialization snapshot...")
            self.plot_initialization_snapshot(
                Z_cpu_np, 
                initial_centroids_np, 
                initial_assignments_np
            )

    # --- 2. NEW plot_initialization_snapshot function ---
    @torch.no_grad()
    def plot_initialization_snapshot(
        self: 'GraphEncoder',
        embeddings: np.ndarray,
        initial_centroids: np.ndarray,
        initial_assignments: np.ndarray
    ) -> None:
        """
        Plots the initial k-means centroids against the pre-trained embeddings.
        Uses PCA if latent_dim > 2.
        
        Args:
            embeddings (np.ndarray): The pre-trained embeddings (n, d_emb).
            initial_centroids (np.ndarray): The initial k-means centroids (k, d_emb).
            initial_assignments (np.ndarray): The initial cluster assignments (n,) from k-means.
        """
        print("Visualizing k-means initialization...")
        latent_dim = embeddings.shape[1]
        
        if latent_dim < 2:
            print(f"Cannot plot: Latent dimension is {latent_dim} (must be >= 2).")
            return
            
        Z_2d = embeddings
        M_2d = initial_centroids

        if latent_dim > 2:
            print(f"Latent dim is {latent_dim}. Running PCA for 2D visualization...")
            # Fit PCA on both data and centroids for consistent projection
            all_data_for_pca = np.vstack([embeddings, initial_centroids])
            
            pca = PCA(n_components=2, random_state=42)
            all_data_2d = pca.fit_transform(all_data_for_pca)
            
            # Split them back up
            Z_2d = all_data_2d[:len(embeddings)]
            M_2d = all_data_2d[len(embeddings):]
        
        print("Generating plot...")
        plt.figure(figsize=(10, 8))
        
        # 1. Plot the pre-trained data points, colored by k-means assignment
        plt.scatter(
            Z_2d[:, 0], 
            Z_2d[:, 1], 
            c=initial_assignments, 
            cmap='viridis',
            alpha=0.3, 
            s=15,
            label='Data Embeddings (by k-means)'
        )
        
        # 2. Plot the initial centroids
        plt.scatter(
            M_2d[:, 0],
            M_2d[:, 1],
            marker='*',          # Star marker
            c='red',             # Bright color
            s=300,               # Large size
            edgecolors='black',  # Black outline
            linewidth=1.5,
            label='Initial Centroids (k-means)'
        )

        plt.title('Centroid Initialization (K-Means on Pre-trained Embeddings)')
        plt.xlabel('Principal Component 1' if latent_dim > 2 else 'Latent Dimension 1')
        plt.ylabel('Principal Component 2' if latent_dim > 2 else 'Latent Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        print("Displaying plot. You may need to close the plot window to continue execution.")
        plt.show()
    
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

# --- MODIFIED train_auto_ot_joint function ---
    def train_auto_ot_joint(
        self: 'GraphEncoder',
        X: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        T_epochs: int,
        gamma: float = 0.1,
        epsilon_0: float = 0.5,       # slightly larger to allow meaningful OT gradients
        rho: float = 0.5,             # annealing factor
        beta: float = 1.0,
        rho_kl: float = 0.01,
        max_sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-6,
        clamp_centroids: float = 0.5,
        use_gumbel_sinkhorn: bool = True, # sharper OT plan
        gumbel_tau: float = 0.1,
        plot_movement: bool = True,
        Y_true: Optional[torch.Tensor] = None # --- ADDED ARGUMENT for true labels ---
    ) -> None:
        """
        Improved joint training for Auto-OT-GE.
        - Normalized embeddings and centroids
        - Stable Sinkhorn / Gumbel-Sinkhorn updates
        - Pre-trained KMeans initialization
        - NCut tracking
        - Optional plotting of centroid movement
        """
        n = X.shape[0]
        device, dtype = X.device, X.dtype
        
        centroid_history = [] 

        a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        w = F.softmax(self.centroid_logits, dim=0)

        if hasattr(self, 'cluster_centroids_init'):
            self.cluster_centroids.data = self.cluster_centroids_init.clone()

        print("Starting improved joint training with robust OT...")

        for t in range(T_epochs):
            self.train() 
            optimizer.zero_grad()

            w = F.softmax(self.centroid_logits, dim=0)

            epsilon_final = 1e-5
            epsilon_t = epsilon_final + (epsilon_0 - epsilon_final) * (rho ** (t / T_epochs))

            Z = self.encode(X)
            X_hat = self.decode(Z)

            loss_rec = F.mse_loss(X_hat, X, reduction='sum')
            rho_hat = Z.mean(dim=0).clamp(1e-6, 1-1e-6)
            kl_div = rho_kl * torch.log(rho_kl / rho_hat) + (1 - rho_kl) * torch.log((1 - rho_kl) / (1 - rho_hat))
            loss_kl = beta * torch.sum(kl_div)

            Z_norm = (Z - Z.mean(dim=0)) / (Z.std(dim=0) + 1e-6)

            # Updtae the centroids while ensuring they remain in the convex hull of the data.
            with torch.no_grad():
                self.cluster_centroids.data = self.compute_convex_centroids(Z)

            M_norm = (self.cluster_centroids - self.cluster_centroids.mean(dim=0)) / (self.cluster_centroids.std(dim=0) + 1e-6)

            # Compute the cost matrix to move the embeddings to centroids
            cost_matrix = self.compute_cost_matrix(Z_norm, M_norm)
            cost_matrix = cost_matrix - cost_matrix.min()
            target_ratio = 5.0
            alpha = (target_ratio * epsilon_t) / (cost_matrix.mean().item() + 1e-8)
            cost_matrix = alpha * cost_matrix 

            # Compute OT plan via Sinkhorn or Gumbel-Sinkhorn
            if use_gumbel_sinkhorn:
                pi_t = gumbel_sinkhorn(cost_matrix, epsilon=epsilon_t, tau=gumbel_tau, n_iters=50, a=a, b=w)
            else:
                pi_t = sinkhorn_log_domain(cost_matrix, epsilon=epsilon_t, a=a, b=w, n_iters=max_sinkhorn_iters)

            transport_cost = torch.sum(pi_t * cost_matrix)
            entropy = epsilon_t * torch.sum(pi_t * torch.log(pi_t + 1e-10))
            loss_ot = transport_cost + entropy

            gamma_t = gamma * min(1.0, t / (T_epochs // 3))

            loss_total = loss_rec + loss_kl + gamma_t * loss_ot
            loss_total.backward(retain_graph=True)
            optimizer.step()

            if t % max(1, T_epochs // 10) == 0:
                print(f"[DEBUG] pi_t shape: {pi_t.shape}")
                print(f"[DEBUG] cost_matrix shape: {cost_matrix.shape}")

                ncut_val = compute_ncut(X, pi_t)
                print(f"\n[Epoch {t}/{T_epochs}] NCut: {ncut_val:.6f}")
                print(f"loss_total: {loss_total.item():.4f}, rec: {loss_rec.item():.4f}, kl: {loss_kl.item():.4f}, ot: {loss_ot.item():.4f}")
                print(f"epsilon_t: {epsilon_t:.6f}")
                print("Cluster mass:", pi_t.sum(dim=0).detach().cpu().numpy())
                
                if plot_movement:
                    centroid_history.append(self.cluster_centroids.data.clone().cpu().numpy())

        if plot_movement:
            centroid_history.append(self.cluster_centroids.data.clone().cpu().numpy())
            
            print("\nTraining finished. Generating centroid movement plot...")
            with torch.no_grad():
                self.eval()
                Z_final = self.encode(X).cpu().numpy()
                assignments_final = torch.argmin(
                    self.compute_cost_matrix(self.encode(X), self.cluster_centroids), 
                    dim=1
                ).cpu().numpy()
            
            # --- Pass Y_true to the plotting function ---
            y_true_np = Y_true.cpu().numpy() if Y_true is not None else None
            self.plot_centroid_movement(centroid_history, Z_final, assignments_final, y_true_np)
    
    # Inside your GraphEncoder class
    def compute_convex_centroids(self, Z):
        """
        Computes centroids as convex combinations of embeddings Z.
        Ensures centroids lie inside the convex hull of Z.
        """
        # self.centroid_logits is an (n, k) learnable matrix
        weights = F.softmax(self.centroid_logits, dim=0)  # normalize over nodes
        centroids = torch.matmul(weights.T, Z)            # (k, d)
        return centroids


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
        print("Centroids cluster" , self.cluster_centroids )
        print("centroid Logit" , self.centroid_logits)
        print("Cost matrix", cost_matrix)
        
        # 4. Find the closest centroid for each node
        # argmin_j ||z_i - mu_j||^2
        assignments = torch.argmin(cost_matrix, dim=1)
        
        return assignments
    
    # --- MODIFIED plot_centroid_movement function ---
    @torch.no_grad()
    def plot_centroid_movement(
        self: 'GraphEncoder',
        centroid_history: List[np.ndarray],
        final_embeddings: np.ndarray,
        final_assignments: np.ndarray,
        Y_true: Optional[np.ndarray] = None # --- ADDED ARGUMENT for true labels ---
    ) -> None:
        """
        Plots the movement of cluster centroids against the final data embeddings.
        Data points can be colored by true labels if provided, otherwise by learned assignments.
        Uses PCA if latent_dim > 2.
        
        Args:
            centroid_history (List[np.ndarray]): A list of centroid arrays (k, d_emb) 
                                                 from different epochs.
            final_embeddings (np.ndarray): The final embeddings (n, d_emb).
            final_assignments (np.ndarray): The final cluster assignments (n,) for coloring 
                                            if Y_true is not provided.
            Y_true (Optional[np.ndarray]): The true cluster assignment for each node (n,).
                                           If provided, data points will be colored by these labels.
        """
        print("Visualizing centroid movement...")
        latent_dim = final_embeddings.shape[1]
        
        if latent_dim < 2:
            print(f"Cannot plot: Latent dimension is {latent_dim} (must be >= 2).")
            return
            
        history_stack = np.stack(centroid_history) # (n_steps, k, d_emb)
        n_steps, k, _ = history_stack.shape

        Z_2d = final_embeddings
        history_2d_stack = history_stack

        if latent_dim > 2:
            print(f"Latent dim is {latent_dim}. Running PCA for 2D visualization...")
            all_data_for_pca = np.vstack([
                final_embeddings, 
                history_stack.reshape(-1, latent_dim)
            ])
            
            pca = PCA(n_components=2, random_state=42)
            all_data_2d = pca.fit_transform(all_data_for_pca)
            
            Z_2d = all_data_2d[:len(final_embeddings)]
            history_2d_stack = all_data_2d[len(final_embeddings):].reshape(n_steps, k, 2)
        
        print("Generating plot...")
        plt.figure(figsize=(12, 9))
        
        # --- Determine coloring for data points ---
        if Y_true is not None:
            data_colors = Y_true
            color_label = 'True Labels'
            # Adjust cmap if true labels don't start from 0 or have gaps
            cmap = plt.cm.get_cmap('coolwarm', len(np.unique(Y_true))) 
        else:
            data_colors = final_assignments
            color_label = 'Learned Assignments'
            cmap = plt.cm.viridis

        # 1. Plot the final data points, colored by cluster (true or learned)
        plt.scatter(
            Z_2d[:, 0], 
            Z_2d[:, 1], 
            c=data_colors, # --- UPDATED COLORING ---
            cmap=cmap,     # --- UPDATED CMAP ---
            alpha=0.3,     # Slightly more transparent for clarity
            s=15,
            label=f'Data Points ({color_label})' # --- UPDATED LABEL ---
        )
        
        # 2. Plot the centroid paths
        colors_centroids = plt.cm.viridis(np.linspace(0, 1, k)) # Use a consistent cmap for centroids
        for j in range(k):
            path = history_2d_stack[:, j, :]
            
            plt.plot(
                path[:, 0], 
                path[:, 1], 
                'o-', 
                markersize=4, 
                alpha=0.7, 
                color=colors_centroids[j],
                label=f'Centroid {j} Path' if n_steps < 20 else None
            )
            
            plt.plot(
                path[0, 0], 
                path[0, 1], 
                'x', 
                color='red', 
                markersize=10, 
                mew=2,
                label='Centroid Start' if j == 0 else None # Clarify legend
            )
            
            plt.plot(
                path[-1, 0], 
                path[-1, 1], 
                '*', 
                color='black', 
                markersize=15, 
                mew=1,
                label='Centroid End' if j == 0 else None # Clarify legend
            )

        if n_steps >= 20:
             plt.plot([], [], 'o-', color='gray', label='Centroid Paths')

        plt.title('Centroid Movement During Joint Training')
        plt.xlabel('Principal Component 1' if latent_dim > 2 else 'Latent Dimension 1')
        plt.ylabel('Principal Component 2' if latent_dim > 2 else 'Latent Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        print("Displaying plot. You may need to close the plot window to continue execution.")
        plt.show()

@torch.no_grad()
def compute_ncut(S_or_X: torch.Tensor, pi: torch.Tensor, use_cosine: bool = True) -> float:
    """
    Computes the approximate NCut of a soft cluster assignment pi.

    Args:
        S_or_X: Either
            - similarity matrix S of shape (n, n), or
            - feature/embedding matrix X of shape (n, d)
        pi: soft cluster assignment, shape (n, k)
        use_cosine: if S_or_X is embedding matrix, whether to use cosine similarity (True)
                    or raw dot-product (False) to construct S.

    Returns:
        ncut_value: Python float
    """
    # Ensure pi is 2D [n, k]
    if pi.dim() != 2:
        raise ValueError("pi must be shape (n, k)")

    n = pi.shape[0]
    device = pi.device
    dtype = pi.dtype

    # If S_or_X is square and matches n, treat it as similarity matrix
    if S_or_X.dim() == 2 and S_or_X.shape[0] == S_or_X.shape[1] == n:
        S = S_or_X.to(device=device, dtype=dtype)
    else:
        # Treat S_or_X as embedding matrix X of shape (n, d)
        X = S_or_X.to(device=device, dtype=dtype)
        if X.dim() != 2 or X.shape[0] != n:
            raise ValueError(f"Expected S (n,n) or X (n,d) with same n as pi; got {tuple(S_or_X.shape)} vs pi n={n}")
        # Build similarity matrix (n x n). Use cosine by default (numerically stable).
        if use_cosine:
            # normalize rows
            X_norm = X - X.mean(dim=1, keepdim=True)  # optional centering
            denom = X_norm.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            Xn = X_norm / denom
            S = torch.matmul(Xn, Xn.t())
        else:
            S = torch.matmul(X, X.t())

    # degrees: sum over columns for each row
    degrees = S.sum(dim=1)  # shape (n,)

    # defensive casts
    degrees = degrees.to(device=device, dtype=dtype)
    pi = pi.to(device=device, dtype=dtype)

    k = pi.shape[1]
    ncut = torch.tensor(0.0, device=device, dtype=dtype)

    # compute NCut = sum_j cut(Aj, Abar)/vol(Aj)
    # where vol(Aj) = sum_i pi_ij * degree_i
    # and cut_j = sum_{i,l} S_il * assign_i * (1 - assign_l)
    for j in range(k):
        assign = pi[:, j]                        # (n,)
        vol_j = torch.sum(assign * degrees) + 1e-8  # scalar
        # pairwise term (n x n) computed via outer product and elementwise multiply
        # assign_i * (1 - assign_l) -> assign.unsqueeze(1) * (1 - assign.unsqueeze(0))
        pairwise = assign.unsqueeze(1) * (1.0 - assign.unsqueeze(0))  # (n,n)
        cut_j = torch.sum(S * pairwise)
        ncut = ncut + (cut_j / vol_j)

    return float(ncut.item())


def sample_gumbel(shape, device='cpu', eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sinkhorn(
    log_alpha: torch.Tensor,
    epsilon: float = 0.1,
    tau: float = 0.05,
    n_iters: int = 20,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched Gumbel-Sinkhorn / entropic OT solver.
    Accepts log_alpha shape (..., n, k) or (n, k) and returns same shape.
    Normalizes rows to a (default uniform) and columns to b (default uniform).
    """
    orig_shape = log_alpha.shape
    if log_alpha.dim() < 2:
        raise ValueError("log_alpha must be at least 2-D with last two dims (n,k)")

    device = log_alpha.device
    dtype = log_alpha.dtype

    # Work on a view where `...` is flattened into batch dimension
    *batch, n, k = log_alpha.shape
    batch_size = int(np.prod(batch)) if len(batch) > 0 else 1
    if batch_size == 1 and len(batch) == 0:
        log_alpha_flat = log_alpha.view(n, k)  # (n,k)
        has_batch = False
    else:
        log_alpha_flat = log_alpha.view(batch_size, n, k)  # (B, n, k)
        has_batch = True

    # Prepare marginals: default uniform (same for all batches)
    if a is None:
        a_vec = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    else:
        a_vec = a.to(device=device, dtype=dtype)
    if b is None:
        b_vec = torch.full((k,), 1.0 / k, device=device, dtype=dtype)
    else:
        b_vec = b.to(device=device, dtype=dtype)

    # Expand marginals to batch shape
    if has_batch:
        a_exp = a_vec.unsqueeze(0).expand(batch_size, -1)  # (B, n)
        b_exp = b_vec.unsqueeze(0).expand(batch_size, -1)  # (B, k)
    else:
        a_exp = a_vec
        b_exp = b_vec

    # Add Gumbel noise with the same shape as log_alpha_flat
    gumbel_noise = sample_gumbel(log_alpha_flat.shape, device=device)
    perturbed = log_alpha_flat / (tau + 1e-12) + gumbel_noise

    # Sinkhorn iterations in log-domain
    # log_pi initialized as -perturbed/epsilon
    log_pi = -perturbed / (epsilon + 1e-12)

    # iterate
    for _ in range(n_iters):
        # row normalization: subtract logsumexp over k dim
        if has_batch:
            # logsumexp over dim=2 (k)
            log_pi = log_pi - torch.logsumexp(log_pi, dim=2, keepdim=True) + torch.log(a_exp).unsqueeze(2)
            # column normalization: over dim=1 (n)
            log_pi = log_pi - torch.logsumexp(log_pi, dim=1, keepdim=True) + torch.log(b_exp).unsqueeze(1)
        else:
            log_pi = log_pi - torch.logsumexp(log_pi, dim=1, keepdim=True) + torch.log(a_exp).unsqueeze(1)
            log_pi = log_pi - torch.logsumexp(log_pi, dim=0, keepdim=True) + torch.log(b_exp).unsqueeze(0)

    pi_flat = torch.exp(log_pi)

    # reshape back to original shape
    if has_batch:
        pi = pi_flat.view(*batch, n, k)
    else:
        pi = pi_flat.view(n, k)

    return pi
