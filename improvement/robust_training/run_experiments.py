import optuna
import torch
import numpy as np
import tqdm
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.cluster
import sklearn.mixture
import sklearn.preprocessing
import networkx as nx
import random
import warnings
import os
import csv
import concurrent.futures
import multiprocessing

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("error", category=sklearn.exceptions.ConvergenceWarning)

# Define constants
NUMBER_OF_EPOCHS = 25000

# Optimization for GPU
if torch.cuda.is_available():
    # TensorFloat-32 requires Compute Capability >= 8.0 (Ampere)
    if torch.cuda.get_device_capability()[0] >= 8 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('high')

# ==========================================
# Model Classes
# ==========================================

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.sigmoid(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.autoencoders = torch.nn.ModuleList()
        self.hidden_dims = hidden_dims
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            # Assuming AutoEncoder class is defined elsewhere as per your snippet
            self.autoencoders.append(AutoEncoder(prev_dim, hidden_dim))
            prev_dim = hidden_dim

    def forward(self, x):
        # Stacked forward pass
        for autoencoder in self.autoencoders:
            x = torch.sigmoid(autoencoder.encoder(x))
        encoded = x
        
        # Reconstruction
        for autoencoder in reversed(self.autoencoders):
            x = torch.sigmoid(autoencoder.decoder(x))
        decoded = x
        return encoded, decoded

    def _compute_loss(self, x, decoded, encoded, rho, beta):
        loss_mse = torch.nn.functional.mse_loss(decoded, x, reduction='sum')
        
        rho_hat = torch.mean(encoded, dim=0)
        rho_hat = torch.clamp(rho_hat, min=1e-6, max=1-1e-6) 
        
        kl_term = torch.sum(
            rho * torch.log(rho / rho_hat) + 
            (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )
        
        return loss_mse + beta * kl_term

    def fit(self, x_data, epochs_per_layer, lr, batch_size, beta, rho, device='cpu'):
        """Performs Greedy Layer-wise Pretraining."""
        current_input = x_data.clone().to(device)
        
        for i, autoencoder in enumerate(self.autoencoders):
            autoencoder = autoencoder.to(device)
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
            training_model = autoencoder

            dataset = torch.utils.data.TensorDataset(current_input)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            desc = f"Pretraining Layer {i+1}/{len(self.autoencoders)}"
            pbar = tqdm.tqdm(range(epochs_per_layer), desc=desc, leave=False)
            for _ in pbar:
                for (x_batch,) in dataloader:
                    x_batch = x_batch.to(device)
                    optimizer.zero_grad()
                    encoded, decoded = training_model(x_batch)
                    loss = self._compute_loss(x_batch, decoded, encoded, rho, beta)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
            
            with torch.no_grad():
                current_input = torch.sigmoid(autoencoder.encoder(current_input))
                
        return self

    def fine_tune(self, x_data, epochs, lr, batch_size, beta, rho, device='cpu'):
        """
        Performs Global Fine-Tuning.
        Optimizes the entire stack simultaneously to minimize total reconstruction error.
        """
        self.to(device)
        
        # Optimizer for ALL parameters in the network
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        
        dataset = torch.utils.data.TensorDataset(x_data.clone())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        pbar = tqdm.tqdm(range(epochs), desc="Global Fine-Tuning", leave=False)
        for _ in pbar:
            total_loss = 0
            for (x_batch,) in dataloader:
                x_batch = x_batch.to(device)
                optimizer.zero_grad()
                
                # Full forward pass through the entire stack
                encoded, decoded = self.forward(x_batch)
                
                # Compute loss on final reconstruction against original input
                loss = self._compute_loss(x_batch, decoded, encoded, rho, beta)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            pbar.set_postfix(avg_loss=total_loss / len(dataloader))
            
        return self


class DenoisingGraphEncoder(GraphEncoder):
    def __init__(self, input_dim, hidden_dims, noise_level=0.1):
        super().__init__(input_dim, hidden_dims)
        self.noise_level = noise_level

    # fit method overrides the parent's fit to add noise
    def fit(self, x_data, epochs_per_layer, lr, batch_size, beta, rho, device='cpu'):
        """Performs Greedy Layer-wise Pretraining with Masking Noise."""
        current_input = x_data.clone().to(device)
        
        for i, autoencoder in enumerate(self.autoencoders):
            autoencoder = autoencoder.to(device)
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
            training_model = autoencoder

            dataset = torch.utils.data.TensorDataset(current_input)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            desc = f"Pretraining Robust Layer {i+1}/{len(self.autoencoders)}"
            pbar = tqdm.tqdm(range(epochs_per_layer), desc=desc, leave=False)
            for _ in pbar:
                for (x_batch,) in dataloader:
                    x_batch = x_batch.to(device)
                    
                    # Apply Masking Noise
                    mask = (torch.rand_like(x_batch) > self.noise_level).float()
                    x_corrupted = x_batch * mask
                    
                    optimizer.zero_grad()
                    # Forward pass with CORRUPTED input
                    encoded, decoded = training_model(x_corrupted)
                    
                    # Compute loss against CLEAN input (x_batch)
                    loss = self._compute_loss(x_batch, decoded, encoded, rho, beta)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
            
            # Generate next layer's input using CLEAN previous layer output
            with torch.no_grad():
                current_input = torch.sigmoid(autoencoder.encoder(current_input))
                
        return self

# ==========================================
# Metrics and Utils
# ==========================================

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

def load_dataset(dataset_name):
    print(f"[*] Loading dataset: {dataset_name}")
    
    if dataset_name.lower() == 'wine':
        x, y = sklearn.datasets.load_wine(return_X_y=True, as_frame=False)
        x = sklearn.preprocessing.MinMaxScaler().fit_transform(x)
        s = sklearn.metrics.pairwise.cosine_similarity(x, x)
        nts = s / np.sum(s, axis=1, keepdims=True)
        return s, nts, y
    
    path = dataset_name
    if not os.path.exists(path):
        candidates = [
            f"/kaggle/input/graph-clustering-datasets/real/{dataset_name}/{dataset_name}.gml",
            f"/kaggle/input/graph-clustering-datasets/synthetic/{dataset_name}.gml",
            f"../../datasets/real/{dataset_name}/{dataset_name}.gml",
            f"../../datasets/synthetic/{dataset_name}.gml",
            f"../../datasets/synthetic/{dataset_name}",
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
                
    if not os.path.exists(path):
         raise FileNotFoundError(f"Could not find dataset file: {dataset_name}")

    print(f"[*] Reading GML from: {path}")
    nxg = nx.read_gml(path)
    
    node_sample = list(nxg.nodes(data=True))[0][1]
    label_key = next((k for k in ["value", "club", "community"] if k in node_sample), "value")
    y = [nxg.nodes[n][label_key] for n in nxg.nodes]
    
    if isinstance(y[0], str):
        y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    else:
        y = np.array(y)

    s = nx.to_numpy_array(nxg)
    s = s + np.diag(np.ones(nxg.number_of_nodes()))
    nts = s / np.sum(s, axis=1, keepdims=True)
    
    return s, nts, y

# ==========================================
# Evaluation Logic
# ==========================================

def evaluate_clustering(encoded_features, s_matrix, true_labels, n_tests=20):
    
    n_clusters = len(set(true_labels))
    
    # 1. K-Means
    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_clusters, 
        algorithm="lloyd", 
        n_init=n_tests, 
        random_state=42
    )
    y_pred_km = kmeans.fit_predict(encoded_features)
    
    nmi_km = sklearn.metrics.normalized_mutual_info_score(true_labels, y_pred_km)
    ncut_km = compute_ncut(s_matrix, y_pred_km)

    # 2. Gaussian Mixture Model (GMM)
    # GMM can be sensitive to initialization. We use n_init to restart.
    
    # Preprocessing: Scale features to improve convergence and stability
    scaler = sklearn.preprocessing.StandardScaler()
    encoded_features_scaled = scaler.fit_transform(encoded_features)
    
    try:
        # Try with default full covariance and slightly higher regularization
        gmm = sklearn.mixture.GaussianMixture(
            n_components=n_clusters,
            n_init=n_tests,
            random_state=42,
            reg_covar=1e-5  # Increased from default 1e-6 for stability
        )
        y_pred_gmm = gmm.fit_predict(encoded_features_scaled)
        nmi_gmm = sklearn.metrics.normalized_mutual_info_score(true_labels, y_pred_gmm)
        ncut_gmm = compute_ncut(s_matrix, y_pred_gmm)
        
    except Exception as e:
        print(f"[!] GMM (Full) failed: {e}. Retrying with Diagonal covariance...")
        try:
            # Fallback: Diagonal covariance is much more stable
            gmm = sklearn.mixture.GaussianMixture(
                n_components=n_clusters,
                n_init=n_tests,
                random_state=42,
                covariance_type='diag',
                reg_covar=1e-4
            )
            y_pred_gmm = gmm.fit_predict(encoded_features_scaled)
            nmi_gmm = sklearn.metrics.normalized_mutual_info_score(true_labels, y_pred_gmm)
            ncut_gmm = compute_ncut(s_matrix, y_pred_gmm)
        except Exception as e2:
            print(f"[!] GMM (Diag) failed: {e2}")
            nmi_gmm = 0.0
            ncut_gmm = float('inf')
        
    return {
        "kmeans": {"nmi": nmi_km, "ncut": ncut_km},
        "gmm":    {"nmi": nmi_gmm, "ncut": ncut_gmm}
    }

def run_experiment(dataset_name, n_trials=20, nb_kmeans_tests=100, use_cpu_only=False):
    """
    Refactored to accept a CPU-force flag for better parallel stability.
    """
    # 1. Force CPU if requested (prevents GPU OOM during parallel runs)
    if use_cpu_only:
        device = 'cpu'
    else:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load Dataset
    try:
        s, nts, y = load_dataset(dataset_name)
    except Exception as e:
        print(f"[!] Failed to load dataset {dataset_name}: {e}")
        return None

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dataset shape: {nts.shape}, Device: {device}")
    
    x_tensor = torch.tensor(nts, dtype=torch.float32)
    n_clusters = len(set(y))

    # --- 1. Baseline: Standard K-Means & GMM on Raw Data ---
    print("[-] Running Baseline: K-Means & GMM on Raw Data...")
    res_base = evaluate_clustering(nts, s, y, n_tests=nb_kmeans_tests)
    km_nmi = res_base['kmeans']['nmi']
    km_ncut = res_base['kmeans']['ncut']
    gmm_nmi = res_base['gmm']['nmi']
    gmm_ncut = res_base['gmm']['ncut'] 
    
    # --- 2. Baseline: Spectral Clustering ---
    print("[-] Running Baseline: Spectral Clustering...")
    try:
        spectral = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            n_init=20, 
            assign_labels='discretize',
            random_state=42
        )
        y_spec = spectral.fit_predict(s)
        spec_nmi = sklearn.metrics.normalized_mutual_info_score(y, y_spec)
        spec_ncut = compute_ncut(s, y_spec)
    except Exception as e:
        print(f"[!] Spectral Clustering failed: {e}")
        spec_nmi, spec_ncut = 0.0, float('inf')

    # --- 3. Optuna Study for Autoencoder ---
    print(f"[-] Starting Optuna Optimization ({n_trials} trials)...")
    
    # Variables to track "Real Life" performance
    # We track best Ncut for both K-Means and GMM independently
    best_ae_kmeans_ncut = float('inf')
    best_ae_kmeans_nmi = 0.0
    
    best_ae_gmm_ncut = float('inf')
    best_ae_gmm_nmi = 0.0
    
    def objective(trial):
        nonlocal best_ae_kmeans_ncut, best_ae_kmeans_nmi, best_ae_gmm_ncut, best_ae_gmm_nmi
        
        seed = 97
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dim_decay_rate = trial.suggest_float("dim_decay_rate", 0.5, 0.9, step=0.05)
        rho = trial.suggest_float("rho", 1e-4, 1e-1, log=True)
        beta = trial.suggest_float("beta", 1e-2, 1e3, log=True)
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        
        nb_epochs_options = [NUMBER_OF_EPOCHS]
        epochs = nb_epochs_options[trial.suggest_int("nb_epochs_index", 0, len(nb_epochs_options)-1)]

        latent_dim = int(x_tensor.shape[1] * dim_decay_rate)
        hidden_dims = []
        hidden_dims.append(latent_dim)
        
        while latent_dim * dim_decay_rate >= n_clusters and len(hidden_dims) < 5:
            latent_dim = int(latent_dim * dim_decay_rate)
            if latent_dim == hidden_dims[-1]: break
            hidden_dims.append(latent_dim)
            
        n_layers = trial.suggest_int("n_layers", 1, 5)
        final_hidden_dims = hidden_dims[:n_layers]

        model = GraphEncoder(input_dim=x_tensor.shape[1], hidden_dims=final_hidden_dims).to(device)

        try:
            model.fit(x_tensor, epochs, lr, x_tensor.shape[0], beta, rho, device=device)
            model.fine_tune(x_tensor, epochs, lr, x_tensor.shape[0], beta, rho, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[!] Autoencoder training failed: {e}")
            return float('inf')

        with torch.no_grad():
            encoded, _ = model(x_tensor.to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encoded_np = encoded.cpu().numpy()
            
            if np.isnan(encoded_np).any():
                print("[!] Encoded representation contains NaNs.")
                return float('inf')
            
            try:
                res = evaluate_clustering(encoded_np, s, y, n_tests=nb_kmeans_tests)
                
                # Update Best KMeans
                if res['kmeans']['ncut'] < best_ae_kmeans_ncut:
                    best_ae_kmeans_ncut = res['kmeans']['ncut']
                    best_ae_kmeans_nmi = res['kmeans']['nmi']
                
                # Update Best GMM
                if res['gmm']['ncut'] < best_ae_gmm_ncut:
                    best_ae_gmm_ncut = res['gmm']['ncut']
                    best_ae_gmm_nmi = res['gmm']['nmi']
                
                # Optimization Target: Minimize the BEST Ncut found by either method
                # This encourages the model to find a representation good for AT LEAST one clustering method
                return min(res['kmeans']['ncut'], res['gmm']['ncut'])
            except sklearn.exceptions.ConvergenceWarning:
                return float('inf')

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials)

    # --- 4. Optuna Study for Robust Autoencoder ---
    print(f"[-] Starting Optuna Optimization for Robust SAE ({n_trials} trials)...")
    
    study_robust_best_kmeans_nmi = 0.0
    study_robust_best_kmeans_ncut = float('inf')
    
    study_robust_best_gmm_nmi = 0.0
    study_robust_best_gmm_ncut = float('inf')

    def objective_robust(trial):
        nonlocal study_robust_best_kmeans_nmi, study_robust_best_kmeans_ncut
        nonlocal study_robust_best_gmm_nmi, study_robust_best_gmm_ncut
        
        seed = 97
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dim_decay_rate = trial.suggest_float("dim_decay_rate", 0.5, 0.9, step=0.05)
        rho = trial.suggest_float("rho", 1e-4, 1e-1, log=True)
        beta = trial.suggest_float("beta", 1e-2, 1e3, log=True)
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        noise_level = trial.suggest_float("noise_level", 0.1, 0.2)
        
        nb_epochs_options = [NUMBER_OF_EPOCHS]
        epochs = nb_epochs_options[trial.suggest_int("nb_epochs_index", 0, len(nb_epochs_options)-1)]

        latent_dim = int(x_tensor.shape[1] * dim_decay_rate)
        hidden_dims = []
        hidden_dims.append(latent_dim)
        
        while latent_dim * dim_decay_rate >= n_clusters and len(hidden_dims) < 5:
            latent_dim = int(latent_dim * dim_decay_rate)
            if latent_dim == hidden_dims[-1]: break
            hidden_dims.append(latent_dim)
            
        n_layers = trial.suggest_int("n_layers", 1, 5)
        final_hidden_dims = hidden_dims[:n_layers]

        model = DenoisingGraphEncoder(input_dim=x_tensor.shape[1], hidden_dims=final_hidden_dims, noise_level=noise_level).to(device)

        try:
            model.fit(x_tensor, epochs, lr, x_tensor.shape[0], beta, rho, device=device)
            model.fine_tune(x_tensor, epochs, lr, x_tensor.shape[0], beta, rho, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[!] Robust SAE training failed: {e}")
            return float('inf')

        with torch.no_grad():
            encoded, _ = model(x_tensor.to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encoded_np = encoded.cpu().numpy()

            if np.isnan(encoded_np).any():
                print("[!] Encoded representation contains NaNs.")
                return float('inf')
            
            try:
                res = evaluate_clustering(encoded_np, s, y, n_tests=nb_kmeans_tests)
                
                # Update Best KMeans
                if res['kmeans']['ncut'] < study_robust_best_kmeans_ncut:
                    study_robust_best_kmeans_ncut = res['kmeans']['ncut']
                    study_robust_best_kmeans_nmi = res['kmeans']['nmi']

                # Update Best GMM
                if res['gmm']['ncut'] < study_robust_best_gmm_ncut:
                    study_robust_best_gmm_ncut = res['gmm']['ncut']
                    study_robust_best_gmm_nmi = res['gmm']['nmi']
                
                return min(res['kmeans']['ncut'], res['gmm']['ncut'])
            except sklearn.exceptions.ConvergenceWarning:
                return float('inf')

    sampler_robust = optuna.samplers.TPESampler(seed=42)
    study_robust = optuna.create_study(sampler=sampler_robust, direction="minimize")
    study_robust.optimize(objective_robust, n_trials=n_trials)

    # ==========================================
    # FINAL REPORT
    # ==========================================
    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*80)
    print(f"FINAL RESULTS FOR DATASET: {dataset_name}")
    print("="*80)
    
    # Header
    print(f"{'METHOD':<30} | {'NMI (Higher is better)':<22} | {'Ncut (Lower is better)':<22}")
    print("-" * 80)
    
    # Rows
    print(f"{'K-Means (Raw Data)':<30} | {km_nmi:<22.4f} | {km_ncut:<22.4f}")
    print(f"{'GMM (Raw Data)':<30} | {gmm_nmi:<22.4f} | {gmm_ncut:<22.4f}")
    print(f"{'Spectral Clustering':<30} | {spec_nmi:<22.4f} | {spec_ncut:<22.4f}")
    print("-" * 80)
    print(f"{'AE + K-Means':<30} | {best_ae_kmeans_nmi:<22.4f} | {best_ae_kmeans_ncut:<22.4f}")
    print(f"{'AE + GMM':<30} | {best_ae_gmm_nmi:<22.4f} | {best_ae_gmm_ncut:<22.4f}")
    print(f"{'AE (Optuna Min)':<30} | {'-':<22} | {study.best_value:<22.4f}")
    print("-" * 80)
    print(f"{'Robust SAE + K-Means':<30} | {study_robust_best_kmeans_nmi:<22.4f} | {study_robust_best_kmeans_ncut:<22.4f}")
    print(f"{'Robust SAE + GMM':<30} | {study_robust_best_gmm_nmi:<22.4f} | {study_robust_best_gmm_ncut:<22.4f}")
    print(f"{'Robust SAE (Optuna Min)':<30} | {'-':<22} | {study_robust.best_value:<22.4f}")
    
    print("-" * 80)
    print(f"[*] Best AE Params: {study.best_params}")
    print(f"[*] Best Robust AE Params: {study_robust.best_params}")
    print("="*80 + "\n")

    return {
        "Dataset": dataset_name,
        
        "KMeans_Raw_NMI": km_nmi,
        "KMeans_Raw_Ncut": km_ncut,
        
        "GMM_Raw_NMI": gmm_nmi,
        "GMM_Raw_Ncut": gmm_ncut,
        
        "Spectral_NMI": spec_nmi,
        "Spectral_Ncut": spec_ncut,
        
        "AE_KMeans_NMI": best_ae_kmeans_nmi,
        "AE_KMeans_Ncut": best_ae_kmeans_ncut,
        
        "AE_GMM_NMI": best_ae_gmm_nmi,
        "AE_GMM_Ncut": best_ae_gmm_ncut,
        
        "AE_Optuna_Min": study.best_value,
        
        "RobustSAE_KMeans_NMI": study_robust_best_kmeans_nmi,
        "RobustSAE_KMeans_Ncut": study_robust_best_kmeans_ncut,
        
        "RobustSAE_GMM_NMI": study_robust_best_gmm_nmi,
        "RobustSAE_GMM_Ncut": study_robust_best_gmm_ncut,
        
        "RobustSAE_Optuna_Min": study_robust.best_value,

        "AE_Params": str(study.best_params),
        "RobustSAE_Params": str(study_robust.best_params)
    }
# ==========================================
# PARALLEL EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # 1. Configuration
    datasets = ['karate'] # Add your datasets here
    n_trials = 10
    kmeans_tests = 100
    output_file = "experiment_results_karate.csv"
    
    # Number of concurrent processes. 
    # Usually set to number of CPU cores or slightly less.
    # If using GPU, set this low (e.g., 2) to avoid OOM.
    MAX_WORKERS = 4 
    
    # FORCE_CPU: Set True if you get CUDA Out Of Memory errors.
    FORCE_CPU = False 

    # 2. Setup CSV
    fieldnames = [
        "Dataset", 
        "KMeans_Raw_NMI", "KMeans_Raw_Ncut", 
        "GMM_Raw_NMI", "GMM_Raw_Ncut",
        "Spectral_NMI", "Spectral_Ncut", 
        "AE_KMeans_NMI", "AE_KMeans_Ncut",
        "AE_GMM_NMI", "AE_GMM_Ncut",
        "AE_Optuna_Min",
        "RobustSAE_KMeans_NMI", "RobustSAE_KMeans_Ncut", 
        "RobustSAE_GMM_NMI", "RobustSAE_GMM_Ncut",
        "RobustSAE_Optuna_Min",
        "AE_Params", "RobustSAE_Params"
    ]
    
    # Initialize file
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    print(f"[*] Starting parallel execution with {MAX_WORKERS} workers...")
    
    # 3. Process Pool Executor
    # This creates a pool of separate processes (bypassing the Python GIL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # Submit all tasks to the pool
        # We store the 'future' object as a key and the dataset name as value
        future_to_dataset = {
            executor.submit(run_experiment, d, n_trials, kmeans_tests, FORCE_CPU): d 
            for d in datasets
        }
        
        # Process results as they complete (in any order)
        for future in concurrent.futures.as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                result = future.result() # Blocks until this specific task is done
                
                if result:
                    # Write to CSV immediately (Thread-safe because only Main Thread writes)
                    with open(output_file, mode='a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow(result)
                    print(f"[+] Finished processing: {dataset_name}")
                else:
                    print(f"[-] returned None for: {dataset_name}")
                    
            except Exception as exc:
                print(f"[!] {dataset_name} generated an exception: {exc}")

    print(f"\nAll experiments completed. Results saved to {output_file}")