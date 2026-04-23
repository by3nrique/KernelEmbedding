import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import prepare_dataset
from .kernels import gaussian_kernel, linear_kernel
from .optimization import optimize_rkhs_reconstruction_general


# ────────────────────────────────────────────────────────────────────
# Step 1 – solve for β
# ────────────────────────────────────────────────────────────────────
def optimize_step1_beta(
    X: torch.Tensor,
    device: torch.device,
    *,
    K_precomputed: torch.Tensor | None = None,
    kernel_type: str = "rbf",
    sigma: float = 1.0,
    beta_solver: str = "exact",
    num_iters: int = 5_000,
    lr: float = 1e-2,
    seed: int = 0,
    # ─ new diagnostics ───────────────────────────────────────────────
    plot_every: int = 0,
    plot_loss_curve: bool = False,
):
    """
    Returns  (β, K, last_loss).

    If `K_precomputed` is supplied it is forwarded untouched; otherwise
    the chosen `kernel_type` is used to build K.
    """
    if K_precomputed is None:
        if kernel_type == "rbf":
            kernel_fn = lambda X_: gaussian_kernel(X_, sigma=sigma).to(device)
        elif kernel_type == "linear":
            kernel_fn = lambda X_: linear_kernel(X_).to(device)
        else:
            raise NotImplementedError(f"Unsupported kernel type: {kernel_type}")
        K_precomputed = kernel_fn(X)
    else:
        kernel_fn = None                         # not needed downstream
        K_precomputed = K_precomputed.to(device)

    print("\n[Step 1] Optimising β …")
    beta, final_loss, loss_curve = optimize_rkhs_reconstruction_general(
        X=X,
        K_precomputed=K_precomputed,
        kernel_fn=None,
        optimize_mode="beta",
        beta_solver=beta_solver,
        num_iters=num_iters if beta_solver == "gd" else 0,
        lr=lr,
        seed=seed,
        verbose=True,
        return_history=True,
        plot_every=plot_every,
    )
    print(f"Step 1 complete – final β-loss: {final_loss:.6e}")

    if plot_loss_curve and loss_curve:
        plt.figure(figsize=(4, 2.4))
        plt.plot(loss_curve); plt.yscale("log")
        plt.xlabel("iteration"); plt.ylabel("loss"); plt.title("β-loss")
        plt.tight_layout(); plt.show()

    return beta, K_precomputed, final_loss


# ────────────────────────────────────────────────────────────────────
# Step 2 – solve for α  (β fixed)
# ────────────────────────────────────────────────────────────────────
def optimize_step2_alpha(
    X: torch.Tensor,
    device: torch.device,
    *,
    kernel_type: str = "rbf",
    sigma: float = 1.0,
    num_iters: int = 5_000,
    lr: float = 1e-2,
    latent_dim: int = 2,
    beta: torch.Tensor,
    K_precomputed: torch.Tensor,
    seed: int = 0,
    # ─ new diagnostics ───────────────────────────────────────────────
    plot_every: int = 0,
    plot_loss_curve: bool = False,
):
    if kernel_type == "rbf":
        kernel_fn = lambda Z: gaussian_kernel(Z, sigma=sigma).to(device)
    elif kernel_type == "linear":
        kernel_fn = lambda Z: linear_kernel(Z).to(device)
    else:
        raise NotImplementedError(f"Unsupported kernel type: {kernel_type}")

    print("\n[Step 2] Optimising α …")
    alpha, final_loss, loss_curve = optimize_rkhs_reconstruction_general(
        X=X,
        kernel_fn=kernel_fn,
        K_precomputed=K_precomputed,
        optimize_mode="alpha",
        beta=beta,
        latent_dim=latent_dim,
        num_iters=num_iters,
        lr=lr,
        seed=seed,
        verbose=True,
        return_history=True,
        plot_every=plot_every,
    )
    print(f"Step 2 complete – final α-loss: {final_loss:.6e}")
    print(f"Latent representation shape: {alpha.shape}")

    if plot_loss_curve and loss_curve:
        plt.figure(figsize=(4, 2.4))
        plt.plot(loss_curve); plt.yscale("log")
        plt.xlabel("iteration"); plt.ylabel("loss"); plt.title("α-loss")
        plt.tight_layout(); plt.show()

    return alpha, kernel_fn, final_loss


# ────────────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────────────
def optimize_pipeline(
    dataset_name: str,
    dataset_path: str,
    *,
    subset_ratio: float = 1.0,
    seed: int = 0,
    # — Step 1
    kernel_type_reconstruction: str = "rbf",
    sigma_reconstruction: float = 1.0,
    beta_solver: str = "exact",
    num_iters_beta: int = 5_000,
    lr_beta: float = 1e-2,
    # — Step 2
    kernel_type_embedding: str = "rbf",
    sigma_embedding: float = 1.0,
    num_iters_alpha: int = 5_000,
    lr_alpha: float = 1e-2,
    latent_dim: int = 2,
    # — plots
    plot: bool = True,
):
    print(f"Preparing dataset ‘{dataset_name}’ from {dataset_path} …")
    X, y, device = prepare_dataset(
        path=dataset_path,
        subset_ratio=subset_ratio,
        normalize=True,
        seed=seed,
    )
    print(f"Device: {device}  |  X shape: {X.shape}  |  y shape: {y.shape}")

    # — Step 1: β ————————————————————————————————————————————————
    beta, K_precomputed, _ = optimize_step1_beta(
        X=X,
        device=device,
        kernel_type=kernel_type_reconstruction,
        sigma=sigma_reconstruction,
        beta_solver=beta_solver,
        num_iters=num_iters_beta,
        lr=lr_beta,
        seed=seed,
        plot_every=0,
        plot_loss_curve=False,
    )

    # — Step 2: α ————————————————————————————————————————————————
    alpha, kernel_fn_alpha, _ = optimize_step2_alpha(
        X=X,
        device=device,
        kernel_type=kernel_type_embedding,
        sigma=sigma_embedding,
        num_iters=num_iters_alpha,
        lr=lr_alpha,
        latent_dim=latent_dim,
        beta=beta,
        K_precomputed=K_precomputed,   # ← needed for α-mode
        seed=seed,
        plot_every=0,
        plot_loss_curve=False,
    )

    # — Optional visualisation ————————————————————————————————
    if plot:
        print("[Step 3] Plotting heat-maps and latent scatter …")
        K_orig_np = K_precomputed.cpu().numpy()
        Z = (alpha @ K_precomputed).T
        K_latent = kernel_fn_alpha(Z).cpu().numpy()
        H = Z.cpu().numpy(); y_np = y.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(K_orig_np, ax=axes[0], cmap="viridis", cbar=False)
        axes[0].set_title("Original kernel K")

        sns.heatmap(K_latent, ax=axes[1], cmap="viridis", cbar=False)
        axes[1].set_title("Latent kernel $K_{\\text{latent}}$")

        axes[2].scatter(H[:, 0], H[:, 1], c=y_np, cmap="viridis", s=10)
        axes[2].set_title("Latent coords (Z = αK)")
        axes[2].set_xlabel("Z₁"); axes[2].set_ylabel("Z₂"); axes[2].grid(True)
        plt.tight_layout(); plt.show()

    return H, y_np, beta.cpu(), alpha.cpu()