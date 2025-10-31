# Save this as tests/test_model_ot.py
import torch
import torch.optim as optim
import copy
import pytest
from auto_ot_ge.model import GraphEncoder, AutoEncoder


def test_train_auto_ot_em_runs(): # <-- Renamed for clarity
    """
    Tests if the train_auto_ot_em loop runs and updates parameters.
    (This was your test_train_auto_ot_runs)
    """
    n, d_in = 100, 50
    k = 5
    hidden_dims = [30, 10]
    latent_dim = hidden_dims[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create data and model
    X = torch.rand(n, d_in, device=device)
    model = GraphEncoder(input_dim=d_in, hidden_dims=hidden_dims, k=k).to(device)
    
    # 2. Setup optimizers
    pretrain_optim = optim.Adam(model.parameters(), lr=1e-3)
    
    # Optimizer for the main EM training
    # This optimizer only *needs* to update the AE, but passing
    # model.parameters() is fine.
    em_optim = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Initialize centroids
    try:
        model.initialize_centroids(
            X, 
            pretrain_optimizer=pretrain_optim, 
            pretrain_iters=5 # Just 5 iters for a quick test
        )
    except ImportError:
        pytest.skip("scikit-learn not installed. Skipping centroid init test.")
        
    # 4. Store initial state
    init_centroids = model.cluster_centroids.data.clone()
    init_encoder_weights = model.autoencoders[0].encoder.weight.data.clone()
    
    # 5. Run one step of the Auto-OT (EM) training
    model.train_auto_ot_em(
        X,
        optimizer=em_optim,
        T_epochs=1,
        gamma=0.1,
        epsilon_0=1.0,
        rho=0.99
    )
    
    # 6. Check if parameters were updated
    
    # Check M-Step: Centroids should have changed (manual update)
    final_centroids = model.cluster_centroids.data
    assert not torch.allclose(init_centroids, final_centroids)
    assert final_centroids.shape == (k, latent_dim)
    
    # Check Encoder Update: Encoder weights should have changed (optim update)
    final_encoder_weights = model.autoencoders[0].encoder.weight.data
    assert not torch.allclose(init_encoder_weights, final_encoder_weights)

# --- NEW TEST FOR THE JOINT FUNCTION ---

def test_train_auto_ot_joint_runs():
    """
    Tests if the train_auto_ot_joint loop runs and updates ALL parameters
    (encoder and centroids) simultaneously via the optimizer.
    """
    n, d_in = 100, 50
    k = 5
    hidden_dims = [30, 10]
    latent_dim = hidden_dims[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create data and model
    X = torch.rand(n, d_in, device=device)
    model = GraphEncoder(input_dim=d_in, hidden_dims=hidden_dims, k=k).to(device)
    
    # 2. Setup optimizers
    pretrain_optim = optim.Adam(model.parameters(), lr=1e-3)
    
    # Optimizer for the main JOINT training
    # MUST include all model parameters: model.parameters()
    joint_optim = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Initialize centroids
    try:
        model.initialize_centroids(
            X, 
            pretrain_optimizer=pretrain_optim, 
            pretrain_iters=5 
        )
    except ImportError:
        pytest.skip("scikit-learn not installed. Skipping centroid init test.")
        
    # 4. Store initial state
    init_centroids = model.cluster_centroids.data.clone()
    init_encoder_weights = model.autoencoders[0].encoder.weight.data.clone()
    
    # 5. Run one step of the Auto-OT JOINT training
    model.train_auto_ot_joint(
        X,
        optimizer=joint_optim,
        T_epochs=1,
        gamma=0.1,
        epsilon_0=1.0,
        rho=0.99
    )
    
    # 6. Check if parameters were updated
    
    # Centroids should have changed (updated by joint_optim)
    final_centroids = model.cluster_centroids.data
    assert not torch.allclose(init_centroids, final_centroids)
    assert final_centroids.shape == (k, latent_dim)
    
    # Encoder weights should have changed (updated by joint_optim)
    final_encoder_weights = model.autoencoders[0].encoder.weight.data
    assert not torch.allclose(init_encoder_weights, final_encoder_weights)

# --- YOUR OTHER TESTS (UNCHANGED) ---

def test_autoencoder_shapes():
    """Test the basic AutoEncoder stub's shapes."""
    ae = AutoEncoder(input_dim=100, hidden_dim=20)
    x = torch.rand(10, 100)
    encoded, decoded = ae(x)
    assert encoded.shape == (10, 20)
    assert decoded.shape == (10, 100)

def test_graph_encoder_init_error():
    """Test that GraphEncoder raises an error for empty hidden_dims."""
    with pytest.raises(ValueError, match="hidden_dims list cannot be empty"):
        GraphEncoder(input_dim=50, hidden_dims=[], k=5)

def test_graph_encoder_shapes():
    """Test the encode and decode shapes of the full GraphEncoder."""
    n, d_in = 100, 50
    hidden_dims = [30, 10]
    latent_dim = hidden_dims[-1]
    k = 5
    
    model = GraphEncoder(input_dim=d_in, hidden_dims=hidden_dims, k=k)
    X = torch.rand(n, d_in)
    
    # Test encode shape
    Z = model.encode(X)
    assert Z.shape == (n, latent_dim)
    
    # Test decode shape
    X_hat = model.decode(Z)
    assert X_hat.shape == (n, d_in)

def test_compute_cost_matrix_properties():
    """Test the cost matrix for known values, including zero cost."""
    model = GraphEncoder(input_dim=2, hidden_dims=[2], k=2)
    
    # Z = [[0, 0], [1, 2]]
    Z = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    
    # M = [[0, 0], [1, 2], [5, 5]]
    M = torch.tensor([[0.0, 0.0], [1.0, 2.0], [5.0, 5.0]])
    
    cost = model.compute_cost_matrix(Z, M)
    
    # Shape
    assert cost.shape == (2, 3)
    
    # Z[0] to M[0] should be 0
    # ||[0,0] - [0,0]||^2 = 0
    assert torch.allclose(cost[0, 0], torch.tensor(0.0))
    
    # Z[1] to M[1] should be 0
    # ||[1,2] - [1,2]||^2 = 0
    assert torch.allclose(cost[1, 1], torch.tensor(0.0))
    
    # Z[0] to M[2]
    # ||[0,0] - [5,5]||^2 = 5^2 + 5^2 = 50
    assert torch.allclose(cost[0, 2], torch.tensor(50.0))
    
    # Z[1] to M[2]
    # ||[1,2] - [5,5]||^2 = (-4)^2 + (-3)^2 = 16 + 9 = 25
    assert torch.allclose(cost[1, 2], torch.tensor(25.0))
    
    # All costs should be non-negative
    assert torch.all(cost >= 0)

def test_train_auto_ot_m_step_fixed_point():
    """
    Tests the M-Step (centroid update) logic in train_auto_ot_em.
    If Z = M, the centroids should not move.
    """
    n, d_latent, k = 2, 2, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Create model
    model = GraphEncoder(input_dim=d_latent, hidden_dims=[d_latent], k=k).to(device)
    
    # 2. Create "perfect" data X, which is also our embeddings Z
    # Z0 = [10, 10], Z1 = [-10, -10]
    X = torch.tensor([[10.0, 10.0], [-10.0, -10.0]], device=device, requires_grad=True)
    
    # 3. Set initial centroids M to be exactly X
    model.cluster_centroids.data = X.clone()
    
    # 4. Mock encoder/decoder as identity but keep gradients
    model.encode = lambda x: x
    model.decode = lambda x: x


    
    # Optimizer (only for encoder, but won't be used as loss_rec=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Run one training step of the EM method
    model.train_auto_ot_em(
        X,
        optimizer=optimizer,
        T_epochs=1,
        gamma=0.1,      # OT loss weight
        epsilon_0=0.01, # Small epsilon for "hard" assignments
        rho=1.0,        # No annealing
        beta=0.0        # No KL loss
    )
    
    # 6. Assertions
    # E-step: Cost matrix C[0,0]=0, C[1,1]=0, C[0,1] & C[1,0] are large.
    # pi_t should be ~[[0.5, 0], [0, 0.5]]
    # M-step: new_mu_0 = (pi_00*Z0 + pi_10*Z1) / (pi_00 + pi_10)
    #                   = (0.5 * Z0 + 0 * Z1) / 0.5 = Z0
    # The centroids should not have moved.
    
    assert torch.allclose(
        model.cluster_centroids.data, 
        X, 
        atol=1e-5 # Allow for tiny numerical float error
    )


def test_get_cluster_assignments():
    """
    Tests if the hard assignment function correctly assigns
    each node to its closest centroid.
    """
    n, d_in, k = 2, 5, 2
    latent_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create model
    model = GraphEncoder(input_dim=d_in, hidden_dims=[latent_dim], k=k).to(device)
    
    # 2. Create dummy input X
    X = torch.rand(n, d_in, device=device)

    # 3. Define known embeddings (Z) and centroids (M)
    
    # Z: node 0 is at [1.], node 1 is at [10.]
    known_Z = torch.tensor([[1.0], [10.0]], device=device)
    
    # M: centroid 0 is at [0.], centroid 1 is at [11.]
    known_M = torch.tensor([[0.0], [11.0]], device=device)

    # 4. Mock the model's internals
    # Force .encode() to return our known Z
    model.encode = lambda x: known_Z
    # Set the model's centroids
    model.cluster_centroids.data = known_M

    # 5. Define expected output
    # node 0 ([1.]) is closer to centroid 0 ([0.]) -> assignment 0
    # node 1 ([10.]) is closer to centroid 1 ([11.]) -> assignment 1
    expected_assignments = torch.tensor([0, 1], device=device, dtype=torch.long)

    # 6. Run the function
    assignments = model.get_cluster_assignments(X)

    # 7. Assert
    assert assignments.shape == (n,)
    assert torch.equal(assignments, expected_assignments)
    assert assignments.dtype == torch.long