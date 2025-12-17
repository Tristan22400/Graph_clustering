import torch
from typing import Optional

def sinkhorn_log_domain(
    cost_matrix: torch.Tensor,
    epsilon: float,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Computes the optimal transport plan using the Sinkhorn-Knopp algorithm
    in the log-domain for numerical stability.
    
    This solves:
    min_P <C, P> - epsilon * H(P)
    subject to P 1 = a, P^T 1 = b
    
    Args:
        cost_matrix (torch.Tensor): The n x k cost matrix ||z_i - mu_j||^2.
        epsilon (float): The entropic regularization parameter.
        a (torch.Tensor, optional): The source marginals (n,). 
                                    Defaults to uniform (1/n).
        b (torch.Tensor, optional): The target marginals (k,). 
                                    Defaults to uniform (1/k).
        n_iters (int): Maximum number of Sinkhorn iterations.
        tol (float): Tolerance for convergence.
        
    Returns:
        torch.Tensor: The n x k optimal transport plan (pi).
    """
    n, k = cost_matrix.shape
    device = cost_matrix.device

    # Initialize marginals if not provided
    if a is None:
        a = torch.full((n,), 1.0 / n, device=device, dtype=cost_matrix.dtype)
    if b is None:
        b = torch.full((k,), 1.0 / k, device=device, dtype=cost_matrix.dtype)

    # Initialize scaling factors (dual variables) in log-space
    u = torch.zeros_like(a)  # (n,)
    v = torch.zeros_like(b)  # (k,)

    # Pre-compute log marginals
    log_a = torch.log(a)
    log_b = torch.log(b)

    # Pre-compute the Gibbs kernel in log-space
    log_K = -cost_matrix / epsilon

    # Sinkhorn iterations
    for _ in range(n_iters):
        u_prev = u.clone()

        # v-update (log-domain)
        # v = log(b) - log(sum_i exp(log_K + u_i))
        #   = log(b) - logsumexp_i (log_K_ij + u_i)
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

        # u-update (log-domain)
        # u = log(a) - log(sum_j exp(log_K + v_j))
        #   = log(a) - logsumexp_j (log_K_ij + v_j)
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)

        # Check for convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Compute the transport plan pi from the dual variables u, v
    # The "Gibbs Kernel" in log-space
    log_K = -cost_matrix / epsilon
    
    # log(pi) = log(u) + log(v) + log(K)
    # Our u, v are already log(u) and log(v)
    log_pi = u.unsqueeze(1) + v.unsqueeze(0) + log_K
    
    pi = torch.exp(log_pi)
    
    return pi
    