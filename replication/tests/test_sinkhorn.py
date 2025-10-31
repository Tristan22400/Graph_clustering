# Save this as tests/test_sinkhorn.py
import torch
import pytest
from auto_ot_ge.sinkhorn import sinkhorn_log_domain

def test_sinkhorn_marginals():
    """
    Test if the Sinkhorn output plan respects the marginal constraints.
    """
    n, k = 100, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a random cost matrix
    cost_matrix = torch.rand(n, k, device=device)
    
    # Create non-uniform marginals
    a = torch.rand(n, device=device)
    a = a / a.sum()
    
    b = torch.rand(k, device=device)
    b = b / b.sum()
    
    epsilon = 0.1
    
    # Compute the transport plan
    pi = sinkhorn_log_domain(
        cost_matrix, 
        epsilon, 
        a, 
        b, 
        n_iters=200, 
        tol=1e-7
    )
    
    # Check shape
    assert pi.shape == (n, k)
    
    # Check marginals (with a reasonable tolerance)
    source_marginals = pi.sum(dim=1)
    target_marginals = pi.sum(dim=0)
    
    assert torch.allclose(source_marginals, a, atol=1e-5)
    assert torch.allclose(target_marginals, b, atol=1e-5)

def test_sinkhorn_uniform_marginals():
    """
    Test if the Sinkhorn output works with default uniform marginals.
    """
    n, k = 50, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cost_matrix = torch.rand(n, k, device=device)
    epsilon = 0.05
    
    pi = sinkhorn_log_domain(cost_matrix, epsilon, n_iters=200, tol=1e-7)
    
    assert pi.shape == (n, k)
    
    # Check default marginals
    expected_a = torch.full((n,), 1.0 / n, device=device, dtype=pi.dtype)
    expected_b = torch.full((k,), 1.0 / k, device=device, dtype=pi.dtype)
    
    assert torch.allclose(pi.sum(dim=1), expected_a, atol=1e-5)
    assert torch.allclose(pi.sum(dim=0), expected_b, atol=1e-5)

def test_sinkhorn_large_epsilon_property():
    """
    Test if epsilon is large, the plan converges to the outer product ab^T.
    """
    n, k = 20, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cost_matrix = torch.rand(n, k, device=device) * 100.0 # Random costs
    
    a = torch.rand(n, device=device)
    a = a / a.sum()
    
    b = torch.rand(k, device=device)
    b = b / b.sum()
    
    # Very large epsilon
    epsilon = 1e6
    
    pi = sinkhorn_log_domain(cost_matrix, epsilon, a, b, n_iters=100)
    
    # Expected plan: pi_ij = a_i * b_j
    pi_expected = a.unsqueeze(1) @ b.unsqueeze(0)
    
    # We need a looser tolerance because epsilon isn't truly infinite
    assert torch.allclose(pi, pi_expected, atol=1e-4)