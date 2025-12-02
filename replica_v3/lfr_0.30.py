import optuna
import torch
import numpy as np
import tqdm
import sklearn
import networkx as nx
import random
import warnings
import time
from contextlib import redirect_stdout

f = open("lfr_0.30.log", "w")
def my_print(*args, **kwargs):
    print(*args, **kwargs)
    with redirect_stdout(f):
        print(*args, **kwargs)
    f.flush()
    pass

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
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.autoencoders.append(AutoEncoder(prev_dim, hidden_dim))
            prev_dim = hidden_dim

    def forward(self, x):
        for autoencoder in self.autoencoders:
            x = torch.sigmoid(autoencoder.encoder(x))
        encoded = x
        for autoencoder in reversed(self.autoencoders):
            x = torch.sigmoid(autoencoder.decoder(x))
        decoded = x
        return encoded, decoded

def compute_ncut(s, labels):
    """
    Compute  normalized cut for given similarity matrix s and cluster labels:
      Ncut = sum_k cut(C_k, V\C_k) / assoc(C_k, V)
    where
      cut(C, V\C) = sum_{i in C, j not in C} A[i,j]
      assoc(C, V) = sum_{i in C, j in V} A[i,j]  (i.e., volume of C)
    A : symmetric adjacency/similarity numpy array
    labels : length-n array of integer cluster labels
    Returns float Ncut value.
    """

    # Get the unique labels in the community assignment
    unique_labels = np.unique(labels)
    
    # Precompute degrees
    degrees = s.sum(axis=1)  # degree/volume per node
    
    # Initialize ncut
    ncut = 0.0
    
    # For each cluster compute link and volume, then sum up to get ncut
    for lab in unique_labels:
        
        # Get the indices of nodes in cluster lab
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            raise Exception("compute_ncut_from_labels: empty cluster found in labels.")
        
        # Compute volume = sum of degrees of nodes in idx
        volume = degrees[idx].sum()
        
        # If volume is not zero, compute link to get the local cut then sum to ncut, otherwise skip (i.e. cut = 0)
        if volume != 0:

            # Compute link = sum over i in C, j not in C, of A[i,j]
            # = volume - internal connections
            internal_connections = s[np.ix_(idx, idx)].sum()
            link = volume - internal_connections
            
            # Compute local cut contribution
            local_cut = link / volume

            # Sum to ncut
            ncut += local_cut
    
    return ncut

warnings.filterwarnings("error", category=sklearn.exceptions.ConvergenceWarning)

nxg = nx.read_gml("../datasets/synthetic/lfr_0.30.gml") # read the football gml file into a networkx graph
y = [nxg.nodes[n]["value"] for n in nxg.nodes] # extract the ground-truth community labels
s = nx.to_numpy_array(nxg) # generate the similarity matrix
s = s + np.diag(np.ones(nxg.number_of_nodes())) # we add self-loops (not indicated in the original paper but improves performance)
nts = s / np.sum(s, axis=1, keepdims=True) # generate the normalized training set
my_print("[*] nts.shape:", nts.shape)
my_print("[*] number of clusters:", len(set(y)))
y_pred = sklearn.cluster.KMeans(n_clusters=len(set(y)), n_init=100, random_state=97).fit_predict(nts)
nmi = sklearn.metrics.normalized_mutual_info_score(y, y_pred)
ncut = compute_ncut(nts, y_pred)
my_print("[*] nmi:", nmi)
my_print("[*] ncut:", ncut)

y_pred = sklearn.cluster.SpectralClustering(n_clusters=len(set(y)), affinity='precomputed', assign_labels='kmeans', n_init=100, random_state=97,).fit_predict(s)
nmi = sklearn.metrics.normalized_mutual_info_score(y, y_pred)
ncut = compute_ncut(nts, y_pred)
my_print("[*] nmi:", nmi)
my_print("[*] ncut:", ncut)


def objective(trial):

    # my_print trial number
    my_print(f"\ntrial {trial.number}----------------------------")
    
    # Set globals
    global best_nmi
    global best_ncut
    global best_ncut_nmi
    global loss_tolerance
    global stab_tolerance
    global max_time_per_layer
    
    # Set random seeds
    torch.manual_seed(97)
    np.random.seed(97)
    random.seed(97)

    # Suggest a decay rate for hidden dimensions
    dim_decay_rate = trial.suggest_float("dim_decay_rate", 0.6, 0.9, step=0.05)

    # Compute the hidden dimensions
    latent_dim = int(x_train.shape[1] * dim_decay_rate)
    hidden_dims = []
    hidden_dims.append(latent_dim)
    while latent_dim * dim_decay_rate >= len(set(y)):
        latent_dim = int(latent_dim * dim_decay_rate)
        hidden_dims.append(latent_dim)

    # Suggest the number of layers
    n_layers = trial.suggest_int("n_layers", 1, len(hidden_dims), step=1)
    hidden_dims = hidden_dims[:n_layers]
    
    # Create the model using the hidden dimensions
    model = GraphEncoder(input_dim=x_train.shape[1], hidden_dims=hidden_dims).to(device)

    # Suggest rho and beta for the sparsity constraint
    rho = trial.suggest_float("rho", 1e-4, 1e-1, log=True)
    beta = trial.suggest_float("beta", 1e-2, 1e3, log=True)
    
    # Suggest a learning rate for the optimizer and create the optimizer    
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Create initial dataloader
    current_x_train = x_train.clone().to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(current_x_train),
        batch_size=batch_size,
        shuffle=True
    )
    dataloader_iter = iter(dataloader)

    # Suggest nb_epochs_per_layer
    # nb_epochs_per_layer = nb_epochs_per_layer_pool[trial.suggest_int("nb_epochs_per_layer", 0, len(nb_epochs_per_layer_pool)-1)]
    # nb_train_iters = nb_epochs_per_layer * len(dataloader)

    # my_print some hyper parameters
    my_print("> hidden dims =", hidden_dims)
    my_print("> rho =", rho)
    my_print("> beta =", beta)
    
    # Launch the training loop
    # For each layer in the stacked autoencoder: train the layer
    for layer_number in range(len(model.autoencoders)):
        stop = False
        last_loss = None
        start_time = time.time()
        pb = tqdm.tqdm(desc=f"layer: {layer_number}")
        stab = 0
        while not stop:
            try:
                (x_batch,) = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                (x_batch,) = next(dataloader_iter)
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            encoded, decoded = model.autoencoders[layer_number](x_batch)
            loss_1 = torch.nn.functional.mse_loss(decoded, x_batch, reduction='sum')
            rho_hat = torch.mean(encoded, dim=0)
            loss_2 = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
            loss = loss_1 + beta * loss_2
            loss.backward()
            optimizer.step()
            
            # Stop criteria
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_per_layer:
                f.write(pb.format_meter(**pb.format_dict) + '\n')
                f.flush()
                my_print(f"[!] stopping layer {layer_number} training after {elapsed_time:.2f}s (> {max_time_per_layer}s)")
                pb.close()
                break
            if last_loss is None:
                last_loss = loss.item()
            else:
                if abs(last_loss - loss.item()) < loss_tolerance:
                    stab += 1
                    if stab == stab_tolerance:
                        stop = True
                        f.write(pb.format_meter(**pb.format_dict) + '\n')
                        f.flush()
                        pb.close()
                else:
                    stab = 0
                last_loss = loss.item()
            pb.set_postfix({"loss": loss.item(), "stab": stab})
            pb.update(1)

        # Create new dataloader on the latent representations
        with torch.no_grad():
            current_x_train, _ = model.autoencoders[layer_number](current_x_train)
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(current_x_train),
                batch_size=batch_size,
                shuffle=True
            )
            dataloader_iter = iter(dataloader)
    
    try:
        # Evaluate the model
        with torch.no_grad():
            
            # Get the encoded representations
            encoded, _ = model(x_train)
            encoded = encoded.to('cpu')

            y_pred = sklearn.cluster.KMeans(n_clusters=len(set(y)), n_init=100, random_state=97).fit_predict(encoded.numpy())
            nmi = sklearn.metrics.normalized_mutual_info_score(y, y_pred)
            ncut = compute_ncut(nts, y_pred)
            
            # my_print average nmi and ncut
            my_print("[*] nmi =", nmi)
            my_print("[*] ncut =", ncut)
            
            # If average nmi is better than the best so far, update best_nmi
            if nmi > best_nmi:
                best_nmi = nmi
            
            # If average ncut is better than the best so far, update best_ncut and its corresponding average nmi (i.e. best_ncut_nmi)
            if ncut < best_ncut:
                best_ncut = ncut
                best_ncut_nmi = nmi
    
    except sklearn.exceptions.ConvergenceWarning:
        my_print("[!] KMeans did not converge (not enough distinct points) --> Returning inf for ncut")
        ncut = float('inf')

    # Return ncut as the objective to minimize
    return ncut


# Set global parameters
nb_epochs_per_layer_pool = [10, 100, 500, 1000, 2500, 5000]
nb_kmeans_tests = 100
nb_trials = 20
device = ('cuda' if torch.cuda.is_available() else 'cpu'); my_print("[*] using device:", device)
x_train = torch.tensor(nts, dtype=torch.float32).to(device)
batch_size = x_train.shape[0]
max_time_per_layer = 3 * 60  # seconds
loss_tolerance = 1e-4
stab_tolerance = 5

# Set globals to track best results
best_nmi = 0.0
best_ncut = float('inf')
best_ncut_nmi = 0.0

# Run optuna study
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=nb_trials)

# Display the best results
my_print("========================================================")
my_print("========================================================")
my_print("[*] best nmi =", best_nmi)
my_print("[*] best ncut =", best_ncut)
my_print("[*] best ncut nmi =", best_ncut_nmi)


