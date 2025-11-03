import numpy as np
import torch
import torch.optim as optim
import copy  # <-- Added for saving model state
import sklearn.datasets
import sklearn.preprocessing
import sklearn.metrics
import sklearn.cluster
from scipy.optimize import linear_sum_assignment

# --- Import your model ---
from auto_ot_ge.model import GraphEncoder

# -------------------------
# 1. Load and preprocess data
# -------------------------
print("Loading Wine dataset...")
X, y_true = sklearn.datasets.load_wine(return_X_y=True, as_frame=False)
X_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

# Cosine similarity matrix
S = sklearn.metrics.pairwise.cosine_similarity(X_scaled, X_scaled)
# D^-1 * S for autoencoder input
NTS = S / np.sum(S, axis=1, keepdims=True)

n_samples, input_dim = NTS.shape
n_clusters = len(np.unique(y_true))

print(f"Data shape: {NTS.shape}, Classes: {n_clusters}")

# Convert to PyTorch tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(NTS, dtype=torch.float32).to(device)
print(f"Using device: {device}")

# -------------------------
# 2. Hyperparameters
# -------------------------
hidden_dims = [64, 32]  # AE layers, last one is latent_dim
latent_dim = hidden_dims[-1]
T_epochs_pre_training = 5000
T_epochs_OT_training = 1000
T_epochs_JOINT_training = 100
gamma = 0.01
epsilon_0 = 0.05
rho = 0.5
lr = 1e-2


# -------------------------
# 3. Initialize Model
# -------------------------
model = GraphEncoder(input_dim=input_dim, hidden_dims=hidden_dims, k=n_clusters).to(device)

# -------------------------
# 4. Pre-train and initialize centroids
# -------------------------
print("Initializing centroids via layer-wise pre-training and k-means...")
pretrain_optim = optim.Adam(model.parameters(), lr=lr)
model.initialize_centroids(
    X_tensor,
    pretrain_optimizer=pretrain_optim,
    pretrain_iters=T_epochs_pre_training
)

# --- Save the initial state for the second experiment ---
print("Pre-training complete. Saving initial state for comparison.")
initial_state_dict = copy.deepcopy(model.state_dict())
init_labels = model.get_initial_assignments().cpu().numpy()

# -------------------------
# 5. Evaluation Helper
# -------------------------
def evaluate_clustering(y_true, y_pred, method_name="Method"):
    y_pred_aligned = best_cluster_mapping(y_true, y_pred)
    nmi = sklearn.metrics.normalized_mutual_info_score(y_true, y_pred_aligned)
    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred_aligned)
    print(f"{method_name} Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}\n")
    return nmi, ari

def best_cluster_mapping(y_true, y_pred):
    """
    Permute cluster labels in y_pred to match y_true as well as possible
    using the Hungarian algorithm (linear_sum_assignment).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.size == y_pred.size

    # Build contingency matrix (confusion matrix)
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1

    # Hungarian algorithm maximizes matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = dict(zip(row_ind, col_ind))

    # Apply permutation to predictions
    y_pred_aligned = np.array([mapping.get(label, label) for label in y_pred])
    return y_pred_aligned

# -------------------------
# 6. Auto-OT-GE Training (Experiment 1: EM-Style)
# -------------------------
print(f"Starting Auto-OT-GE training (EM-Style) for {T_epochs_OT_training} epochs...")
# We use a fresh optimizer for this training run
em_optim = optim.Adam(model.parameters(), lr=lr)
model.train()
# --- Corrected function call to match model file ---
model.train_auto_ot_em(
    X=X_tensor,
    optimizer=em_optim,
    T_epochs=T_epochs_OT_training,
    gamma=gamma,
    epsilon_0=epsilon_0,
    rho=rho,
    beta=1.0,  # sparsity weight
    rho_kl=0.01
)
print("EM training complete.\n")

# --- Evaluate EM Method ---
y_pred_auto_ot_em = model.get_cluster_assignments(X_tensor).cpu().numpy()
nmi_em, ari_em = evaluate_clustering(y_true, y_pred_auto_ot_em, "Auto-OT-GE (EM)")

print("Evaluating clustering via KMeans on (EM) latent space...")
latent_em = model.encode(X_tensor).cpu().detach().numpy()
kmeans_em = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
y_pred_kmeans_em = kmeans_em.fit_predict(latent_em)
nmi_kmeans_em, ari_kmeans_em = evaluate_clustering(y_true, y_pred_kmeans_em, "KMeans on Latent Space (EM)")


# -------------------------
# 7. Auto-OT-GE Training (Experiment 2: Joint)
# -------------------------
print("--- Resetting model to initial state for Joint training ---")
model.load_state_dict(initial_state_dict)

print(f"Starting Auto-OT-GE training (Joint) for {T_epochs_OT_training} epochs...")
# This optimizer MUST track all parameters
joint_optim = optim.Adam(model.parameters(), lr=lr)
model.train()
print("Starting Joint training...")
model.train_auto_ot_joint(
    X=X_tensor,
    optimizer=joint_optim,
    T_epochs=T_epochs_JOINT_training,
    gamma=gamma,
    epsilon_0=epsilon_0,
    rho=rho,
    beta=1.0,  # sparsity weight
    rho_kl=0.01
)
print("Joint training complete.\n")

# --- Evaluate Joint Method ---
y_pred_auto_ot_joint = model.get_cluster_assignments(X_tensor).cpu().numpy()
nmi_joint, ari_joint = evaluate_clustering(y_true, y_pred_auto_ot_joint, "Auto-OT-GE (Joint)")

print("Evaluating clustering via KMeans on (Joint) latent space...")
latent_joint = model.encode(X_tensor).cpu().detach().numpy()
kmeans_joint = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
y_pred_kmeans_joint = kmeans_joint.fit_predict(latent_joint)
nmi_kmeans_joint, ari_kmeans_joint = evaluate_clustering(y_true, y_pred_kmeans_joint, "KMeans on Latent Space (Joint)")


# -------------------------
# 8. Evaluate Initial Clustering
# -------------------------
nmi_init, ari_init = evaluate_clustering(y_true, init_labels, "Initial KMeans on Pre-trained Embeddings")

# -------------------------
# 9. Summary
# -------------------------
print("--- Summary of Clustering Performance ---")
print(f"{'Method':45} {'NMI':>6} {'ARI':>6}")
print("-" * 59)
print(f"{'Auto-OT-GE (EM)':45} {nmi_em:6.4f} {ari_em:6.4f}")
print(f"{'Auto-OT-GE (Joint)':45} {nmi_joint:6.4f} {ari_joint:6.4f}")
print("-" * 59)
print(f"{'KMeans on Latent Space (EM)':45} {nmi_kmeans_em:6.4f} {ari_kmeans_em:6.4f}")
print(f"{'KMeans on Latent Space (Joint)':45} {nmi_kmeans_joint:6.4f} {ari_kmeans_joint:6.4f}")
print(f"{'Initial KMeans (Pre-trained)':45} {nmi_init:6.4f} {ari_init:6.4f}")