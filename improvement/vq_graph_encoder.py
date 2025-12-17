import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module as described in the VQ-VAE paper.
    Maintains a codebook of embeddings and quantizes inputs to the nearest codebook vector.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: K x D
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch, Dim) - Latent vectors from the encoder
        Returns:
            quantized: (Batch, Dim) - Quantized vectors
            loss: Scalar - VQ Loss (Codebook + Commitment)
            perplexity: Scalar - Measure of codebook usage
        """
        # 1. Compute distances
        # inputs: (B, D)
        # embedding: (K, D)
        # distance = (inputs - embedding)^2 = inputs^2 + embedding^2 - 2*inputs*embedding
        
        # (B, 1)
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            
        # 2. Get nearest codebook indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (B, 1)
        
        # 3. Quantize
        # Create one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight) # (B, D)
        
        # 4. Compute Loss
        # Loss = ||sg[z_e(x)] - e||^2 + beta * ||z_e(x) - sg[e]||^2
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 5. Straight-Through Estimator
        # z_q(x) = e_k + (z_e(x) - z_e(x)).detach() -> Gradients flow from z_q to z_e
        quantized = inputs + (quantized - inputs).detach()
        
        # 6. Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices

class GraphVQVAE(nn.Module):
    """
    Graph VQ-VAE model.
    Encoder -> Vector Quantizer -> Decoder
    """
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256, num_embeddings=10, commitment_cost=0.25):
        super(GraphVQVAE, self).__init__()
        
        # Encoder: n -> 1024 -> D
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim), # Output matches embedding_dim
            nn.Tanh() # Bound latent space
        )
        
        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # Decoder: D -> 1024 -> n
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, _ = self.vq_layer(z_e)
        x_hat = self.decoder(z_q)
        
        return x_hat, vq_loss, perplexity

    def pretrain_forward(self, x):
        """Forward pass without quantization (for pre-training)"""
        z_e = self.encoder(x)
        x_hat = self.decoder(z_e)
        return x_hat


    def encode(self, x):
        """Returns the quantized latent vectors"""
        z_e = self.encoder(x)
        z_q, _, _, _ = self.vq_layer(z_e)
        return z_q

    def get_cluster_assignments(self, x):
        """Returns the cluster indices for inputs"""
        z_e = self.encoder(x)
        _, _, _, indices = self.vq_layer(z_e)
        return indices.squeeze()

    def initialize_codebook(self, x):
        """
        Initialize codebook using K-Means on the latent representations of x.
        Args:
            x: Input data (Batch, Input Dim)
        """
        import sklearn.cluster
        
        with torch.no_grad():
            z_e = self.encoder(x).cpu().numpy()
            
        kmeans = sklearn.cluster.KMeans(n_clusters=self.vq_layer.num_embeddings, n_init=10)
        kmeans.fit(z_e)
        
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(x.device)
        self.vq_layer.embedding.weight.data.copy_(centroids)

