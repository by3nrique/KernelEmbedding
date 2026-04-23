import torch

def gaussian_kernel(X, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix using pure PyTorch, MPS-compatible.

    Args:
        X (torch.Tensor): Input data matrix [N x D]
        sigma (float): Bandwidth of the Gaussian kernel

    Returns:
        K (torch.Tensor): Kernel matrix [N x N]
    """
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)

    # Compute squared pairwise distances manually (MPS-safe)
    norm = (X**2).sum(dim=1).reshape(-1, 1)
    dists = norm - 2 * X @ X.T + norm.T
    dists = torch.clamp(dists, min=0)

    # Compute the Gaussian kernel
    K = torch.exp(-dists / (2 * sigma ** 2))
    return K


def linear_kernel(X):
    """
    Compute the linear kernel matrix K = X @ X^T

    Args:
        X (torch.Tensor): Input data [n x d]

    Returns:
        torch.Tensor: Kernel matrix [n x n]
    """
    return X @ X.T


def pairwise_distances(X):
    """
    Computes pairwise squared Euclidean distances (safe for MPS/GPU).
    Args:
        X: Tensor of shape [N, D]
    Returns:
        D: Tensor of shape [N, N] with squared distances
    """
    norms = (X ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    dists = norms - 2 * X @ X.T + norms.T      # [N, N]
    return torch.clamp(dists, min=0.0)

def knn_indices(X, k):
    """
    Computes indices of the k nearest neighbors for each point (excluding itself).
    Args:
        X: Tensor of shape [N, D]
        k: Number of neighbors
    Returns:
        indices: LongTensor of shape [N, k]
    """
    dists = pairwise_distances(X)
    _, idx = torch.topk(dists, k=k + 1, dim=1, largest=False)
    return idx[:, 1:]  # exclude self

def compute_local_inverse_covariances(X, k, regularization=1e-5):
    """
    Computes inverse local covariance matrices for each point based on its neighbors.
    Args:
        X: Tensor of shape [N, D]
        k: Number of neighbors
        regularization: float added to diagonal for stability
    Returns:
        inv_covs: list of N tensors of shape [D, D]
    """
    N, D = X.shape
    inv_covs = []
    for i in range(N):
        neighbors = X[knn_indices(X, k)[i]]  # [k, D]
        mean = neighbors.mean(dim=0, keepdim=True)
        centered = neighbors - mean  # [k, D]
        cov = centered.T @ centered / (centered.shape[0] - 1)  # empirical covariance
        cov += regularization * torch.eye(D, device=X.device, dtype=X.dtype)
        inv_cov = torch.inverse(cov)
        inv_covs.append(inv_cov)
    return inv_covs


def rbf_anisotropic_kernel(X, inv_covs):
    """
    Computes the anisotropic RBF kernel matrix.
    Args:
        X: Tensor of shape [N, D]
        inv_covs: list of N [D x D] inverse covariance tensors
    Returns:
        K: Tensor of shape [N, N] with kernel values
    """
    N = X.shape[0]
    K = torch.zeros((N, N), device=X.device, dtype=X.dtype)

    for i in range(N):
        xi = X[i]
        inv_i = inv_covs[i]
        for j in range(i, N):
            xj = X[j]
            inv_j = inv_covs[j]
            diff = xi - xj
            M = 0.5 * (inv_i + inv_j)
            quad = torch.dot(diff, M @ diff)
            val = torch.exp(-0.5 * quad)
            K[i, j] = K[j, i] = val

    return K