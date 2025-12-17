import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import sklearn.datasets
import sklearn.preprocessing
import sklearn.metrics
from scipy.optimize import linear_sum_assignment
import argparse

import networkx as nx
import os

from vq_graph_encoder import GraphVQVAE

# -------------------------
# 1. Data Loading
# -------------------------

def load_wine_data():
    """
    Loads Wine dataset and computes Normalized Similarity Matrix (D^-1 S).
    Returns:
        X (np.ndarray): Input features (N, N)
        y (np.ndarray): Ground truth labels
    """
    print("Loading Wine dataset...")
    X, y_true = sklearn.datasets.load_wine(return_X_y=True, as_frame=False)
    X_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
    
    # Cosine similarity matrix
    S = sklearn.metrics.pairwise.cosine_similarity(X_scaled, X_scaled)
    # D^-1 * S
    NTS = S / np.sum(S, axis=1, keepdims=True)
    return S, NTS, y_true

def load_graph_dataset(dataset_name):
    """
    Loads a graph dataset from GML file.
    Returns:
        X (np.ndarray): Normalized Adjacency Matrix (D^-1 A)
        y (np.ndarray): Ground truth labels
    """
    print(f"Loading {dataset_name} dataset...")
    dataset_rel_path = f"real/{dataset_name}/{dataset_name}.gml"
    
    # Check current directory
    path = os.path.join("datasets", dataset_rel_path)
    
    # Check parent directory if not found
    if not os.path.exists(path):
        parent_path = os.path.join("..", "datasets", dataset_rel_path)
        if os.path.exists(parent_path):
            path = parent_path
            
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path} (checked current and parent directories)")
        
    G = nx.read_gml(path)
    
    # Extract Adjacency Matrix
    A = nx.to_numpy_array(G)
    
    # Add self-loops to avoid zero division and preserve node info
    np.fill_diagonal(A, 1.0)
    
    # Normalize: D^-1 * A
    D_inv = 1.0 / np.sum(A, axis=1, keepdims=True)
    NTS = D_inv * A
    
    # Extract Labels
    labels = []
    nodes = list(G.nodes(data=True))
    
    # Determine label attribute
    label_attr = 'value'
    if dataset_name == 'karate':
        label_attr = 'club'
        
    for _, data in nodes:
        if label_attr in data:
            labels.append(data[label_attr])
        else:
            # Fallback or error
            labels.append(0)
            
    y_true = np.array(labels)
    
    # Encode string labels if necessary
    if y_true.dtype.kind in {'U', 'S'}:
        le = sklearn.preprocessing.LabelEncoder()
        y_true = le.fit_transform(y_true)
        
    return A, NTS, y_true

def best_cluster_mapping(y_true, y_pred):
    """
    Permute cluster labels in y_pred to match y_true as well as possible
    using the Hungarian algorithm.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.size == y_pred.size

    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = dict(zip(row_ind, col_ind))

    y_pred_aligned = np.array([mapping.get(label, label) for label in y_pred])
    return y_pred_aligned

def compute_ncut(s, labels):
    unique_labels = np.unique(labels)
    degrees = s.sum(axis=1)
    ncut = 0.0
    
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if idx.size == 0: continue
        
        volume = degrees[idx].sum()
        if volume != 0:
            internal_connections = s[np.ix_(idx, idx)].sum()
            link = volume - internal_connections
            ncut += link / volume
    return ncut

def evaluate_clustering(y_true, y_pred, S, method_name="Method"):
    y_pred_aligned = best_cluster_mapping(y_true, y_pred)
    nmi = sklearn.metrics.normalized_mutual_info_score(y_true, y_pred_aligned)
    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred_aligned)
    ncut = compute_ncut(S, y_pred_aligned)
    print(f"{method_name} Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print(f"  Ncut: {ncut:.4f}\n")
    return nmi, ari, ncut

def train(args):
    # 1. Load Data
    if args.dataset == 'wine':
        S, X, y_true = load_wine_data()
    else:
        S, X, y_true = load_graph_dataset(args.dataset)
        
    if args.use_raw_adj:
        print("Using Raw Adjacency Matrix as Input")
        X = S
        
    n_samples, input_dim = X.shape
    
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        print(f"Using specified number of clusters: {n_clusters}")
    else:
        n_clusters = len(np.unique(y_true))
        print(f"Using ground truth number of clusters: {n_clusters}")
        
    print(f"Data shape: {X.shape}, Classes: {n_clusters}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    print(f"Using device: {device}")

    # 2. Initialize Model
    # Use low commitment cost as found in testing
    model = GraphVQVAE(
        input_dim=input_dim, 
        hidden_dim=args.hidden_dim, 
        latent_dim=args.latent_dim, 
        num_embeddings=n_clusters, 
        commitment_cost=args.commitment_cost
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    # Define Loss Function
    if args.use_bce:
        if args.pos_weight is not None:
            pos_weight = torch.tensor(args.pos_weight, device=device)
        else:
            # Calculate pos_weight
            num_pos = X_tensor.sum()
            num_neg = (X_tensor.shape[0] * X_tensor.shape[1]) - num_pos
            pos_weight = num_neg / (num_pos + 1e-6)
            
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.2f}")
    else:
        criterion = lambda pred, target: F.mse_loss(pred, target) * 1000.0
        print("Using MSE Loss")

    # 3. Pre-training (Warmup)
    print(f"Starting Pre-training (Autoencoder) for {args.pretrain_epochs} epochs...")
    for epoch in range(args.pretrain_epochs):
        model.train()
        optimizer.zero_grad()
        x_hat = model.pretrain_forward(X_tensor)
        # Scale loss to handle small input variance
        loss = criterion(x_hat, X_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Pre-train Epoch {epoch+1}/{args.pretrain_epochs} | Loss: {loss.item():.6f}")

    # 4. Initialize Codebook
    print("Initializing codebook with K-Means...")
    model.initialize_codebook(X_tensor)
    
    # 5. VQ Training
    print(f"Starting VQ training for {args.epochs} epochs...")
    # Reset LR for VQ training
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    
    # New scheduler for VQ phase
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
        
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        x_hat, vq_loss, perplexity = model(X_tensor)
        
        # Reconstruction Loss (scaled)
        recon_loss = criterion(x_hat, X_tensor)
        
        # Total Loss
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"VQ Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.6f} | Recon: {recon_loss.item():.6f} | VQ: {vq_loss.item():.6f} | Perplexity: {perplexity.item():.2f}")

    # 6. Evaluation
    print("\nTraining complete. Evaluating...")
    model.eval()
    with torch.no_grad():
        y_pred = model.get_cluster_assignments(X_tensor).cpu().numpy()
    
    nmi, ari, ncut = evaluate_clustering(y_true, y_pred, S, "Graph VQ-VAE")

    # 7. Save Model
    if args.save_path:
        print(f"Saving model to {args.save_path}...")
        torch.save(model.state_dict(), args.save_path)
        print("Model saved successfully.")

    return nmi, ncut

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph VQ-VAE")
    parser.add_argument("--epochs", type=int, default=500, help="Number of VQ training epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=1000, help="Number of pre-training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (will be reduced for VQ phase)")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension") 
    parser.add_argument("--commitment_cost", type=float, default=0.01, help="Commitment cost beta")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters (K). If None, uses ground truth count.")
    parser.add_argument("--dataset", type=str, default="wine", choices=["wine", "karate", "football", "polbooks", "email"], help="Dataset to use")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--use_bce", action="store_true", help="Use BCEWithLogitsLoss instead of MSE")
    parser.add_argument("--use_raw_adj", action="store_true", help="Use raw Adjacency Matrix as input (instead of normalized)")
    parser.add_argument("--pos_weight", type=float, default=None, help="Manual pos_weight for BCE loss")
    
    args = parser.parse_args()
    train(args)
