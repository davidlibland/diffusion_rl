#!/usr/bin/env python3
"""Visualize the training data each method generates.

Grid layout:
  rows = t-bins  (t∈[0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0])
  cols = method:  off-policy | ssmc k=0 | ssmc k=0.01 | ssmc k=0.1 | ssmc k=1

Each cell scatters (x₁, x₂) for samples whose t falls into that bin,
coloured by the regression target the value network is trained on.
The moons base distribution is shown faintly underneath, and the reward
target c=[1,0] is marked with a red star.
"""

import json
from math import ceil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from diffusion_rl.models.on_policy import single_seed_mc


# ── Setup (matches sweeps) ────────────────────────────────────────────────
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
means_np = clf.means_
sigmas_np = np.sqrt(clf.covariances_)
weights_np = clf.weights_


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


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


def make_log_tau(k):
    if k == 0.0:
        def _zero(x, t):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return _zero

    def _scaled(x, t):
        return k * reward_fn(x)
    return _scaled


# `value` is unused by single_seed_mc when log_tau = k*h (only its terminal
# branch reads h), but it is still called at non-terminal steps.  Stub it
# with zeros to make sampling self-contained.
def zero_value(x, t):
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


# ── Sampling ──────────────────────────────────────────────────────────────
DEVICE = "cpu"
torch.manual_seed(0)
np.random.seed(0)


@torch.no_grad()
def sample_offpolicy(n):
    x1_np = gmm_sample(n)
    x1 = torch.from_numpy(x1_np).float()
    eps = torch.randn_like(x1)
    t = torch.rand(n, 1)
    x = t * x1 + torch.sqrt(2 * a * t * (1 - t)) * eps
    target = reward_fn(x1)
    return x, t.squeeze(-1), target


@torch.no_grad()
def sample_ssmc(k, n_traj, mc=10, n_steps=100):
    """Pull (x, t, target) from single_seed_mc with log_tau = k*h."""
    all_x, all_t, all_tgt = single_seed_mc(
        drift=base_drift,
        value=zero_value,
        log_tau=make_log_tau(k),
        h=reward_fn,
        a=a,
        batch_size=n_traj,
        mc_samples=mc,
        dim=D,
        n_steps=n_steps,
        device=DEVICE,
    )
    # all_x: (n_traj * n_steps, dim);  all_t: (...);  all_tgt: (...)
    return all_x, all_t, all_tgt


print("Sampling off-policy data ...")
N_OFF = 12_000
x_off, t_off, y_off = sample_offpolicy(N_OFF)

K_VALUES = [0.0, 0.01, 0.1, 1.0]
N_TRAJ = 120  # 120 × 100 = 12,000 samples per k

ssmc_data = {}
for k in K_VALUES:
    print(f"Sampling single_seed_mc (k = {k}) ...")
    x, t, y = sample_ssmc(k, n_traj=N_TRAJ)
    ssmc_data[k] = (x, t, y)
    print(f"  shapes: x={tuple(x.shape)} t={tuple(t.shape)} y={tuple(y.shape)}")

# ── Bin and plot ──────────────────────────────────────────────────────────
T_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
T_LABELS = [f"t∈[{lo:.1f},{hi:.1f})" for lo, hi in zip(T_EDGES[:-1], T_EDGES[1:])]
T_LABELS[-1] = f"t∈[{T_EDGES[-2]:.1f},{T_EDGES[-1]:.1f}]"

methods = [("off-policy", (x_off, t_off, y_off))]
for k in K_VALUES:
    methods.append((f"ssmc k={k:g}", ssmc_data[k]))

# Robust colour scale across all targets (1st-99th percentiles).
all_targets = torch.cat([m[1][2].flatten() for m in methods]).numpy()
vmin, vmax = np.quantile(all_targets, [0.01, 0.99])
print(f"\nTarget colour scale: [{vmin:.2f}, {vmax:.2f}]")

# Plot bounds (cover both moons cloud and points found near c).
x_lim = (-2.0, 2.5)
y_lim = (-2.0, 2.5)

n_rows, n_cols = len(T_LABELS), len(methods)
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(2.6 * n_cols, 2.6 * n_rows),
    sharex=True, sharey=True,
)

cmap = plt.cm.viridis
for r, (lo, hi, t_label) in enumerate(zip(T_EDGES[:-1], T_EDGES[1:], T_LABELS)):
    for ccol, (m_name, (xm, tm, ym)) in enumerate(methods):
        ax = axes[r, ccol]

        # Moons cloud underneath
        ax.scatter(X[:, 0], X[:, 1], s=1, c="lightgray", alpha=0.25, zorder=1)
        ax.scatter(c[0], c[1], marker="*", s=120, c="red",
                   edgecolor="black", linewidth=0.6, zorder=4)

        t_np = tm.numpy().flatten()
        mask = (t_np >= lo) & (t_np < hi if hi < 1.0 else t_np <= hi)
        if mask.sum() > 0:
            sub_x = xm[mask].numpy()
            sub_y = ym[mask].numpy().flatten()
            sc = ax.scatter(sub_x[:, 0], sub_x[:, 1],
                            c=sub_y, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=4, alpha=0.7, zorder=2, edgecolor="none")
        else:
            sc = None

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.set_title(m_name, fontsize=10, fontweight="bold")
        if ccol == 0:
            ax.set_ylabel(t_label, fontsize=9)
        ax.grid(True, alpha=0.2)

# Shared colourbar
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
    ax=axes, orientation="vertical", fraction=0.025, pad=0.02,
)
cbar.set_label("training target  log V̂(x,t)", fontsize=9)

fig.suptitle(
    "Training-data distribution by method × t-bin  (colour = regression target)",
    fontsize=13, fontweight="bold", y=0.995,
)
out = "experiments/misc/2026-05-28_ssmc_sweeps/ssmc_k_data_grid.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved {out}")

# Save quick stats: count per (method, t-bin)
stats = {}
for m_name, (xm, tm, ym) in methods:
    t_np = tm.numpy().flatten()
    counts = []
    for lo, hi in zip(T_EDGES[:-1], T_EDGES[1:]):
        mask = (t_np >= lo) & (t_np < hi if hi < 1.0 else t_np <= hi)
        counts.append(int(mask.sum()))
    stats[m_name] = counts
print("\nSample counts per t-bin:")
hdr = "  " + " ".join(f"{lab:>15}" for lab in T_LABELS)
print(hdr)
for m_name, cnt in stats.items():
    row = "  " + " ".join(f"{n:>15}" for n in cnt)
    print(f"{m_name:<14}{row}")

with open("experiments/misc/2026-05-28_ssmc_sweeps/ssmc_k_data_grid_counts.json", "w") as f:
    json.dump(stats, f, indent=2)
