import optuna
import torch
import numpy as np
import tqdm
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.cluster
import sklearn.preprocessing
import networkx as nx
import random
import warnings
import argparse
import os

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
            
            # Optimization: Compile the model
            training_model = autoencoder

            dataset = torch.utils.data.TensorDataset(current_input)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            desc = f"Training Layer {i+1}/{len(self.autoencoders)}"
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

class DenoisingGraphEncoder(GraphEncoder):
    def __init__(self, input_dim, hidden_dims, noise_level=0.1):
        super().__init__(input_dim, hidden_dims)
        self.noise_level = noise_level

    def fit(self, x_data, epochs_per_layer, lr, batch_size, beta, rho, device='cpu'):
        """Performs Greedy Layer-wise Pretraining with Masking Noise."""
        current_input = x_data.clone().to(device)
        
        for i, autoencoder in enumerate(self.autoencoders):
            autoencoder = autoencoder.to(device)
            optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-4)
            
            # Optimization: Compile the model
            training_model = autoencoder

            

            dataset = torch.utils.data.TensorDataset(current_input)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            desc = f"Training Robust Layer {i+1}/{len(self.autoencoders)}"
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
    
    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_clusters, 
        algorithm="lloyd", 
        n_init=n_tests, 
        random_state=42
    )
    y_pred = kmeans.fit_predict(encoded_features)
    
    nmi = sklearn.metrics.normalized_mutual_info_score(true_labels, y_pred)
    ncut = compute_ncut(s_matrix, y_pred)
        
    return nmi, ncut

def run_experiment(dataset_name, n_trials=20, nb_kmeans_tests=100):
    try:
        s, nts, y = load_dataset(dataset_name)
    except Exception as e:
        print(f"[!] Failed to load dataset {dataset_name}: {e}")
        return None

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dataset shape: {nts.shape}, Device: {device}")
    
    x_tensor = torch.tensor(nts, dtype=torch.float32)
    n_clusters = len(set(y))

    # --- 1. Baseline: Standard K-Means on Raw Data ---
    print("[-] Running Baseline: K-Means on Raw Data...")
    km_nmi, km_ncut = evaluate_clustering(nts, s, y, n_tests=nb_kmeans_tests) 
    
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
    # We track the NMI specifically associated with the lowest Ncut found so far.
    best_ncut_val = float('inf')
    nmi_of_best_ncut = 0.0
    
    def objective(trial):
        nonlocal best_ncut_val, nmi_of_best_ncut
        
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
                nmi_test, ncut_test = evaluate_clustering(encoded_np, s, y, n_tests=nb_kmeans_tests)
            
                # We only record the NMI if the Ncut is better than what we've seen.
                # This mimics "Real Life" selection where we only know Ncut.
                if ncut_test < best_ncut_val:
                    best_ncut_val = ncut_test
                    nmi_of_best_ncut = nmi_test
                
                return ncut_test
            except sklearn.exceptions.ConvergenceWarning:
                return float('inf')

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials)

    # --- 4. Optuna Study for Robust Autoencoder ---
    print(f"[-] Starting Optuna Optimization for Robust SAE ({n_trials} trials)...")
    
    study_robust_best_nmi = 0.0
    study_robust_best_ncut = float('inf')

    def objective_robust(trial):
        nonlocal study_robust_best_nmi, study_robust_best_ncut
        
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
                nmi_test, ncut_test = evaluate_clustering(encoded_np, s, y, n_tests=nb_kmeans_tests)
                
                if ncut_test < study_robust_best_ncut:
                    study_robust_best_ncut = ncut_test
                    study_robust_best_nmi = nmi_test
                
                return ncut_test
            except sklearn.exceptions.ConvergenceWarning:
                return float('inf')

    sampler_robust = optuna.samplers.TPESampler(seed=42)
    study_robust = optuna.create_study(sampler=sampler_robust, direction="minimize")
    study_robust.optimize(objective_robust, n_trials=n_trials)

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*70)
    print(f"FINAL RESULTS FOR DATASET: {dataset_name}")
    print("="*70)
    
    # Header
    print(f"{'METHOD':<25} | {'NMI (Higher is better)':<22} | {'Ncut (Lower is better)':<22}")
    print("-" * 70)
    
    # Rows
    print(f"{'K-Means (Raw Data)':<25} | {km_nmi:<22.4f} | {km_ncut:<22.4f}")
    print(f"{'Spectral Clustering':<25} | {spec_nmi:<22.4f} | {spec_ncut:<22.4f}")
    print(f"{'Autoencoder (Optuna)':<25} | {nmi_of_best_ncut:<22.4f} | {study.best_value:<22.4f} ")
    print(f"{'Robust SAE (Optuna)':<25} | {study_robust_best_nmi:<22.4f} | {study_robust.best_value:<22.4f} | {study_robust_best_ncut:<22.4f}")
    
    print("-" * 70)
    print(f"[*] Best AE Params: {study.best_params}")
    print(f"[*] Best Robust AE Params: {study_robust.best_params}")
    print("="*70 + "\n")

    return {
        "Dataset": dataset_name,
        "KMeans_NMI": km_nmi,
        "KMeans_Ncut": km_ncut,
        "Spectral_NMI": spec_nmi,
        "Spectral_Ncut": spec_ncut,
        "AE_NMI": nmi_of_best_ncut,
        "AE_Ncut": best_ncut_val,
        "AE_Ncut_Optuna": study.best_value,
        "RobustSAE_NMI": study_robust_best_nmi,
        "RobustSAE_Ncut": study_robust_best_ncut,
        "RobustSAE_Ncut_Optuna": study_robust.best_value,
        "AE_Params": str(study.best_params),
        "RobustSAE_Params": str(study_robust.best_params)
    }

if __name__ == "__main__":
    import csv
    
    # List of all datasets to run
    datasets = [ 
        'polbooks', 'football'
    ]
    
    # Configuration
    n_trials = 10
    kmeans_tests = 100
    output_file = "experiment_results_1.csv"
    
    results = []
    
    # Initialize CSV file with headers
    fieldnames = [
        "Dataset", 
        "KMeans_NMI", "KMeans_Ncut", 
        "Spectral_NMI", "Spectral_Ncut", 
        "AE_NMI", "AE_Ncut", "AE_Ncut_Optuna",
        "RobustSAE_NMI", "RobustSAE_Ncut", "RobustSAE_Ncut_Optuna",
        "AE_Params", "RobustSAE_Params"
    ]
    
    # Check if file exists to avoid overwriting if we want to append (optional, here we overwrite)
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        result = run_experiment(dataset, n_trials=n_trials, nb_kmeans_tests=kmeans_tests)
        
        if result:
            results.append(result)
            
            # Append result to CSV immediately
            with open(output_file, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(result)
                
    print(f"\nAll experiments completed. Results saved to {output_file}")