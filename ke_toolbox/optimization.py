"""rkhs_opt.py — RKHS reconstruction optimisation
================================================
Single entry‑point: `optimize_rkhs_reconstruction_general`.

* **β‑mode** (`optimize_mode="beta"`)
  * `beta_solver="exact"` – closed‑form (requires `torch.linalg.solve`).
  * `beta_solver="gd"`    – Adam gradient descent.
* **α‑mode** (`optimize_mode="alpha"`) learns a latent projection α with
  fixed β (always Adam).

Diagnostics
-----------
* `return_history=True` → returns the full loss curve.
* `plot_every=N`        → live log‑scale loss plot every N iterations.

All variants minimise the quadratic loss

    L = βᵀ A β − 2 cᵀ β + tr(K),   A = K ∘ (K̃ @ K̃),   c = row‑sum(K ∘ K̃)

where K̃ = K − diag(K) and "∘" is the Hadamard product.
"""
from __future__ import annotations

from typing import Callable, Literal, Tuple, List
import torch
import matplotlib.pyplot as plt
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────
# Helper routines
# ──────────────────────────────────────────────────────────────────────

def _zero_diag(K: torch.Tensor) -> torch.Tensor:
    """
    Return K with its diagonal zeroed, using K̃ = K - diag(diag(K)).
    Works on any device / dtype and keeps autograd intact.
    """
    return K - torch.diag(torch.diag(K))

def _compute_A_c(K: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute A and c for the quadratic loss."""
    K_tilde = _zero_diag(K)
    A = K * (K_tilde @ K_tilde)  # Hadamard product with ẼK²
    c = torch.diag(K @ K_tilde)
    return A, c

def _quadratic_loss(beta: Tensor, A: Tensor, c: Tensor, trace_term: float) -> Tensor:
    quad = beta.T @ (A @ beta)
    linear = c @ beta
    return quad - 2 * linear + trace_term


def _solve_beta_exact(A: Tensor, c: Tensor) -> Tensor:
    return torch.linalg.solve(A, c)

# ──────────────────────────────────────────────────────────────────────
# Optimiser
# ──────────────────────────────────────────────────────────────────────

def optimize_rkhs_reconstruction_general(
    X: Tensor,
    *,
    kernel_fn: Callable[[Tensor], Tensor] | None = None,
    K_precomputed: Tensor | None = None,
    # ─ general ─
    num_iters: int = 1000,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = True,
    early_stopping_patience: int = 10000,
    scheduler_patience: int = 500,
    min_lr: float = 1e-6,
    # ─ diagnostics ─
    return_history: bool = False,
    plot_every: int = 0,
    # ─ mode ─
    optimize_mode: Literal["beta", "alpha"] = "beta",
    latent_dim: int = 2,
    beta: Tensor | None = None,
    # ─ β‑mode options ─
    beta_solver: Literal["exact", "gd"] = "gd",
) -> Tuple[Tensor, float] | Tuple[Tensor, float, List[float]]:

    assert optimize_mode in ("beta", "alpha")
    if optimize_mode == "alpha":
        assert beta is not None, "α‑mode requires a fixed β."
    if optimize_mode == "beta":
        assert beta_solver in ("exact", "gd")

    torch.manual_seed(seed)
    device = X.device
    # Verify if X in an empty tensor
    if X.numel() == 0 and K_precomputed is not None:
        n = K_precomputed.shape[0]
    else:
        n = X.shape[0]

    # ─ Kernel ─
    if K_precomputed is not None:
        K_orig = K_precomputed.to(device)
    else:
        assert kernel_fn is not None, "Need kernel_fn or K_precomputed"
        K_orig = kernel_fn(X).to(device)
    trace_term = torch.trace(K_orig)

    # =========================================================================
    # β‑MODE
    # =========================================================================
    if optimize_mode == "beta":
        A, c = _compute_A_c(K_orig)

        # ─ exact ─
        if beta_solver == "exact":
            beta_opt = _solve_beta_exact(A, c)
            last_loss = _quadratic_loss(beta_opt, A, c, trace_term).item()
            if return_history:
                return beta_opt.detach(), last_loss, [last_loss]
            return beta_opt.detach(), last_loss

        # ─ gradient descent (Adam) ─
        var = torch.zeros(n, device=device, requires_grad=True)
        opt = torch.optim.Adam([var], lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", 0.5, scheduler_patience, min_lr=min_lr, verbose=False
        )
        losses: List[float] = []
        patience = 0
        best_loss = float("inf")

        if plot_every:
            plt.ion(); fig, ax = plt.subplots(); line, = ax.plot([])
            ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("loss")
            ax.set_title("β-loss")

        for t in range(1, num_iters + 1):
            opt.zero_grad()
            loss = _quadratic_loss(var, A, c, trace_term)
            loss.backward(); opt.step(); sch.step(loss.item())

            current = loss.item(); last_loss = current
            if return_history:
                losses.append(current)
            if plot_every and t % plot_every == 0:
                line.set_data(range(len(losses)), losses)
                ax.relim(); ax.autoscale_view(); plt.pause(0.01)

            if current < best_loss - 1e-9:
                best_loss = current; patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    if verbose:
                        print("[Early Stop] no improvement; breaking")
                    break
            if verbose and t % 1000 == 0:
                print(f"[β-GD] {t:4d}  loss={current:.3e}  lr={opt.param_groups[0]['lr']:.1e}")

        if plot_every:
            plt.ioff(); plt.show()
        if return_history:
            return var.detach(), last_loss, losses
        return var.detach(), last_loss

    # =========================================================================
    # α‑MODE  (β fixed)
    # =========================================================================
    var = torch.randn(latent_dim, n, device=device, requires_grad=True)
    opt = torch.optim.Adam([var], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", 0.5, scheduler_patience, min_lr=min_lr, verbose=False)

    losses: List[float] = []
    patience = 0
    best_loss = float("inf")

    if plot_every:
        plt.ion(); fig, ax = plt.subplots(); line, = ax.plot([])
        ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("loss")
        ax.set_title("α-loss")

    for t in range(1, num_iters + 1):
        opt.zero_grad()

        # recompute K(α)
        Z = var @ K_orig
        K_alpha = kernel_fn(Z.T)
        A_alpha, c_alpha = _compute_A_c(K_alpha)

        loss = _quadratic_loss(beta, A_alpha, c_alpha, torch.trace(K_alpha))
        loss.backward(); opt.step(); sch.step(loss.item())

        current = loss.item(); last_loss = current
        if return_history:
            losses.append(current)
        if plot_every and t % plot_every == 0:
            line.set_data(range(len(losses)), losses)
            ax.relim(); ax.autoscale_view(); plt.pause(0.01)

        if current < best_loss - 1e-9:
            best_loss = current; patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                if verbose:
                    print("[Early Stop] no improvement; breaking")
                break
        if verbose and t % 1000 == 0:
            print(f"[α-Adam] {t:4d}  loss={current:.3e}  lr={opt.param_groups[0]['lr']:.1e}")

    if plot_every:
        plt.ioff(); plt.show()
    if return_history:
        return var.detach(), last_loss, losses
    return var.detach(), last_loss
