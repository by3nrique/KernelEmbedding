import torch
import numpy as np

def get_default_device(verbose=True,mps=False):
    if torch.backends.mps.is_available() and mps:
        device = torch.device("mps")
        torch.set_default_tensor_type(torch.FloatTensor)
        if verbose:
            print("Using MPS device for all tensors by default.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        if verbose:
            print("Using CUDA device for all tensors by default.")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Falling back to CPU.")
    return device


def generate_concentric_circles(
        samples_per_circle=200,
        radii=(2, 6, 10),
        noise_mean=0.0,
        noise_std=0.5,
        seed: int | None = 42,
):
    """
    Generate points on several concentric circles with optional Gaussian noise.

    Parameters
    ----------
    samples_per_circle : int | Sequence[int]
        • If an int, that many points are placed on every circle.  
        • If a sequence, it must have the same length as `radii`, and each
          element gives the number of points for the corresponding circle.
    radii : Sequence[float]
        Radii of the concentric circles (len == # circles).
    noise_mean : float, default 0.0
        Mean μ of the additive N(μ, σ²) noise.
    noise_std : float, default 0.5
        Standard deviation σ of the additive noise.
        Set to 0 for perfectly clean circles.
    seed : int | None, default 42
        Random-seed to make the result reproducible.  
        Pass `None` to get fresh randomness each call.

    Returns
    -------
    tensor : np.ndarray,  shape (∑ samples_per_circle, 2)
        Cartesian coordinates of all generated points.
    target : np.ndarray,  shape (∑ samples_per_circle,)
        Integer label 0 … (# circles – 1) identifying which circle
        each point belongs to.
    """

    # -------------- housekeeping ------------------------------------------
    rng = np.random.default_rng(seed)

    # Ensure samples_per_circle is an iterable matching radii
    if np.isscalar(samples_per_circle):
        samples_per_circle = [int(samples_per_circle)] * len(radii)
    elif len(samples_per_circle) != len(radii):
        raise ValueError("samples_per_circle and radii must have the same length")

    # Pre-allocate output arrays
    total_pts = sum(samples_per_circle)
    tensor = np.empty((total_pts, 2), dtype=float)
    target = np.empty(total_pts, dtype=int)

    # -------------- populate each circle ----------------------------------
    idx = 0
    for k, (r, n_pts) in enumerate(zip(radii, samples_per_circle)):
        # Uniform angles on [0, 2π)
        angles = rng.uniform(0, 2 * np.pi, n_pts)

        # Ideal (noise-free) coordinates
        x = r * np.cos(angles)
        y = r * np.sin(angles)

        # Insert into big tensor
        tensor[idx: idx + n_pts, 0] = x
        tensor[idx: idx + n_pts, 1] = y
        target[idx: idx + n_pts] = k
        idx += n_pts

    # -------------- add Gaussian noise globally ---------------------------
    if noise_std != 0.0:
        tensor += rng.normal(noise_mean, noise_std, tensor.shape)

    return tensor, target