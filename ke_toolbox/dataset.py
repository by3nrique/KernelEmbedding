import torch
from scipy.io import loadmat
from .utils import get_default_device

def prepare_dataset(path: str, subset_ratio: float = None, normalize: bool = True, seed: int = 0):
    """
    Load, shuffle, optionally normalize and subset a dataset from .mat file.

    Args:
        path (str): Path to .mat file containing 'X' and 'y'
        subset_ratio (float or None): Fraction of data to use (None = full dataset)
        normalize (bool): Whether to apply min-max normalization
        seed (int): Random seed

    Returns:
        X (torch.Tensor): Feature matrix [N x D]
        y (torch.Tensor): Label vector [N]
        device (torch.device): Active device (CPU, CUDA, MPS)
    """
    data = loadmat(path)

    if 'X' not in data or 'y' not in data:
        raise ValueError("Expected keys 'X' and 'y' in the .mat file.")

    X = torch.tensor(data['X'], dtype=torch.float32)

    # Convert nested strings like [[['5']]] to flat int labels
    y_raw = data['y']
    if 'mnist' in path:
        y= torch.tensor([int(label[0][0]) for label in y_raw], dtype=torch.int32)
    elif 'swiss_roll' in path:
        y = torch.tensor(y_raw, dtype=torch.float32)

    if subset_ratio is not None:
        X, y = select_subset(X, y, subset_ratio=subset_ratio, seed=seed)

    if normalize:
        X = minmax_normalize(X)

    device = get_default_device()
    return X.to(device), y.to(device), device


def select_subset(X, y, subset_ratio=0.10, seed=0):
    """Shuffle and select a subset of the dataset."""
    N = X.shape[0]
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    X, y = X[perm], y[perm]
    n_subset = int(subset_ratio * N)
    return X[:n_subset], y[:n_subset]


def minmax_normalize(X):
    """Apply min-max normalization to each feature."""
    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    range_ = X_max - X_min
    range_[range_ == 0] = 1.0
    return (X - X_min) / range_