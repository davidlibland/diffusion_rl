#!/usr/bin/env python3
"""Diagnose collapse of single_seed_mc on the moons problem.

Plan:
  (1) Replay the single_seed_mc forward pass (mirrors `_single_seed_forward`)
      with the SAME settings as ssmc_vs_offpolicy_sweep:
        drift = base_drift (GMM drift), log_tau = h, mc=10, n_steps=100, a=1.
      Track:
        - 10 sample seed trajectories,
        - ESS / N at each step (effective sample size after weighting).

  (2) If the average ESS fraction is below 60%, sweep k for log_tau = k*h
      to find k* such that mean ESS/N ≈ 0.6. Re-run with k* and mc=30,
      plot the new trajectories + ESS to confirm diversity is restored.
"""

from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ── Moons setup (matches ssmc_vs_offpolicy_sweep) ──────────────────────────
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])


def gmm_drift(xt, ts, a_):
    ts = ts.reshape(-1, 1)
    xt_ = xt[..., None]
    means_ = _means.float().to(xt).T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = _sigmas.float().to(xt).T
    weights_ = _weights_col.float().to(xt).T
    denom = 2 * a_ * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a_ * (1 - ts) / denom) * D / 2
    lrw = torch.log(weights_) + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), torch.log(weights_), lw)
    nm = (2 * a_ * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
    us = (nm - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")


def base_drift(x, t):
    return gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)


def reward_fn(x):
    return -10 * (x - c.to(x)).square().sum(dim=1)


# ── Diagnostic forward pass (mirrors `_single_seed_forward`) ──────────────
@torch.no_grad()
def ssmc_diagnostic(drift, log_tau, a_, batch_size, mc_samples, dim, n_steps,
                    device, dtype=torch.float32):
    """Run single_seed_mc forward, returning seed trajectories + per-step ESS.

    Returns:
        traj:     (B, T+1, dim)        seed positions at t = 0, dt, ..., 1
        ts:       (T+1,)
        ess_frac: (T,)                 mean over B of ESS/N at each step
    """
    dt = 1.0 / n_steps
    N = mc_samples
    x = torch.zeros(batch_size, dim, dtype=dtype, device=device)
    log_tau_x = log_tau(
        x,
        torch.full((batch_size, 1), 0.0, dtype=dtype, device=device),
    ).reshape([-1, 1])

    traj = [x.clone()]
    ts = [0.0]
    ess_frac = []

    for _t in torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]:
        t_curr = float(_t)
        t_next = t_curr + dt

        x_exp_flat = x.unsqueeze(1).expand(batch_size, N, dim).reshape(-1, dim)
        t_curr_vec = torch.full(
            (batch_size * N, 1), t_curr, dtype=dtype, device=device
        )
        dx = drift(x_exp_flat, t_curr_vec) * dt
        db = sqrt(2.0 * a_ * dt) * torch.randn_like(x_exp_flat)
        x_next_flat = x_exp_flat + dx + db
        x_next = x_next_flat.reshape(batch_size, N, dim)

        t_next_vec = torch.full(
            (batch_size * N, 1), t_next, dtype=dtype, device=device
        )
        log_tau_next = log_tau(x_next_flat, t_next_vec).reshape(batch_size, N)
        log_w = log_tau_next - log_tau_x  # (B, N)

        # ESS = (sum w_i)^2 / sum w_i^2;  ESS/N in [1/N, 1].
        log_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        log_ess = -torch.logsumexp(2 * log_norm, dim=1)
        ess_frac.append((log_ess.exp() / N).mean().item())

        # Categorical resample (matches `_single_seed_forward`).
        log_w_stable = log_w - log_w.amax(dim=1, keepdim=True)
        ix = torch.multinomial(
            log_w_stable.exp(), num_samples=N, replacement=True
        )
        x_next_r = torch.gather(
            x_next, 1, ix.unsqueeze(-1).expand(batch_size, N, dim)
        )
        x = x_next_r[:, 0, :]

        log_tau_x = log_tau(
            x,
            torch.full((batch_size, 1), t_next, dtype=dtype, device=device),
        ).reshape([-1, 1])

        traj.append(x.clone())
        ts.append(t_next)

    traj = torch.stack(traj, dim=1)  # (B, T+1, dim)
    return traj, torch.tensor(ts), torch.tensor(ess_frac)


# ── Plotting helpers ───────────────────────────────────────────────────────
def plot_trajectories(traj, ess_frac, title, out_path):
    """Two-panel plot: trajectories on x1-x2, ESS/N over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Trajectories panel (with moons cloud + reward target).
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=2, c="lightgray", alpha=0.4, label="moons base p")
    ax.scatter(c[0], c[1], marker="*", s=300, c="red", edgecolor="black",
               linewidth=1, zorder=5, label="reward target c")
    traj_np = traj.cpu().numpy()
    for b in range(traj_np.shape[0]):
        ax.plot(traj_np[b, :, 0], traj_np[b, :, 1], lw=0.8, alpha=0.7)
        ax.scatter(traj_np[b, -1, 0], traj_np[b, -1, 1], s=12, alpha=0.9)
    ax.set_title(f"{traj_np.shape[0]} sample trajectories  (start at origin)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ESS panel
    ax = axes[1]
    ess = ess_frac.cpu().numpy()
    ts = np.linspace(1.0 / len(ess), 1.0, len(ess))
    ax.plot(ts, ess, lw=1.5)
    ax.axhline(0.6, color="red", ls=":", alpha=0.5, label="60% target")
    ax.axhline(1.0 / len(ess) * len(ess), color="black", ls="--", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time t")
    ax.set_ylabel("ESS / N")
    ax.set_title(f"Effective sample size (mean ESS/N = {ess.mean():.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    device = "cpu"  # analysis is small; mps avoids no benefit here
    torch.manual_seed(0)
    np.random.seed(0)

    # ----- Stage 1: baseline (k = 1, mc = 10) -----
    print("\n[1] Baseline: log_tau = h(x), mc = 10")
    traj1, ts1, ess1 = ssmc_diagnostic(
        drift=base_drift,
        log_tau=lambda x, t: reward_fn(x),
        a_=a,
        batch_size=10,
        mc_samples=10,
        dim=D,
        n_steps=100,
        device=device,
    )
    mean_ess1 = ess1.mean().item()
    min_ess1 = ess1.min().item()
    print(f"  mean ESS/N = {mean_ess1:.3f}   min ESS/N = {min_ess1:.3f}")
    plot_trajectories(
        traj1, ess1,
        f"single_seed_mc baseline  (log_tau = h, mc = 10) — mean ESS/N = {mean_ess1:.2f}",
        "experiments/misc/2026-05-28_ssmc_sweeps/ssmc_diversity_baseline.png",
    )

    # Diagnose collapse: ESS < 60% on average?
    if mean_ess1 >= 0.60:
        print(f"\n  Mean ESS/N = {mean_ess1:.3f} ≥ 0.60 — no collapse, stopping.")
        return

    print(f"\n  Mean ESS/N = {mean_ess1:.3f} < 0.60 — collapse confirmed.")

    # ----- Stage 2: sweep k for log_tau = k*h, target mean ESS/N ≈ 0.6 -----
    # Use a separate, larger batch for a more stable ESS estimate.
    diag_bs = 64
    print("\n[2] Sweeping k for log_tau = k * h (target mean ESS/N ≈ 0.6, mc=30)")

    ks = np.geomspace(0.005, 1.0, 16)
    results = []
    for k in ks:
        torch.manual_seed(0)
        np.random.seed(0)
        _, _, ess_k = ssmc_diagnostic(
            drift=base_drift,
            log_tau=lambda x, t, kk=k: kk * reward_fn(x),
            a_=a,
            batch_size=diag_bs,
            mc_samples=30,
            dim=D,
            n_steps=100,
            device=device,
        )
        m = ess_k.mean().item()
        results.append((k, m))
        print(f"  k = {k:7.4f}   mean ESS/N = {m:.3f}")

    # Pick k whose mean ESS/N is closest to 0.6 (interp on log-k axis).
    arr = np.array(results)
    log_k = np.log(arr[:, 0])
    ess_v = arr[:, 1]
    order = np.argsort(ess_v)
    log_k = log_k[order]
    ess_v = ess_v[order]
    target = 0.6
    if ess_v.min() > target:
        k_star = float(np.exp(log_k.min()))
    elif ess_v.max() < target:
        k_star = float(np.exp(log_k.max()))
    else:
        k_star = float(np.exp(np.interp(target, ess_v, log_k)))
    print(f"\n  Chosen k* = {k_star:.4f}")

    # ----- Stage 3: re-run with k*, mc = 30 -----
    print("\n[3] Re-running with k* and branch (mc) = 30")
    torch.manual_seed(0)
    np.random.seed(0)
    traj2, ts2, ess2 = ssmc_diagnostic(
        drift=base_drift,
        log_tau=lambda x, t, kk=k_star: kk * reward_fn(x),
        a_=a,
        batch_size=10,
        mc_samples=30,
        dim=D,
        n_steps=100,
        device=device,
    )
    mean_ess2 = ess2.mean().item()
    print(f"  mean ESS/N = {mean_ess2:.3f}   min = {ess2.min().item():.3f}")
    plot_trajectories(
        traj2, ess2,
        f"single_seed_mc fixed  (log_tau = {k_star:.3f}·h, mc = 30) — mean ESS/N = {mean_ess2:.2f}",
        "experiments/misc/2026-05-28_ssmc_sweeps/ssmc_diversity_fixed.png",
    )

    # ----- ESS-vs-k summary plot -----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(arr[:, 0], arr[:, 1], "o-", lw=1.5)
    ax.axhline(0.6, color="red", ls=":", alpha=0.5, label="60% target")
    ax.axvline(k_star, color="green", ls="--", alpha=0.7, label=f"k* = {k_star:.3f}")
    ax.set_xlabel("k  (log_tau = k · h)")
    ax.set_ylabel("mean ESS / N")
    ax.set_title(f"ESS vs k (mc = 30, n_steps = 100, batch = {diag_bs})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("experiments/misc/2026-05-28_ssmc_sweeps/ssmc_diversity_k_sweep.png", dpi=150, bbox_inches="tight")
    print("  saved experiments/misc/2026-05-28_ssmc_sweeps/ssmc_diversity_k_sweep.png")


if __name__ == "__main__":
    main()
