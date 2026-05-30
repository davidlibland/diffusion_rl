"""
Data Quality Analysis v2
========================
Comprehensive bias/variance analysis of target values produced by different
on-policy sampling methods compared to the analytical value function V_anal(x,t).

Stages:
  1. Unbiased MC baseline — λ≈1, uniform SMC (no twist)
  2. Reward-guided MC    — λ≈1, reward-based SMC resampling
  3. Oracle lower bound  — oracle V + oracle SMC, λ sweep
  4. Oracle SMC, model V — best trained V, oracle SMC twist, λ sweep
  5. Reward SMC, model V — best trained V, reward SMC twist, λ sweep
  6. Self-consistent (best) — best model for both V and SMC, λ sweep
  7a. Self-consistent (early)— early checkpoint for both V and SMC, λ sweep
  7b. Self-consistent (mid)  — mid checkpoint for both V and SMC, λ sweep

Usage:
  python experiments/data_quality/data_quality_v2.py            # full run (collect data + plot)
  python experiments/data_quality/data_quality_v2.py --replot   # reload saved JSON and replot only

Outputs: experiments/data_quality/dq2_stage{N}.png + experiments/data_quality/data_quality_v2_results.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM Setup (standard boilerplate)
# ---------------------------------------------------------------------------
print("Setting up GMM...")
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

_means   = torch.from_numpy(clf.means_).double()
_sigma2  = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas  = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])

# ---------------------------------------------------------------------------
# Analytical Value Function
# ---------------------------------------------------------------------------
class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means",   means.float())
        self.register_buffer("sigma2",  sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        c = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * c).sum(-1)
               + 200.0 * v * (c ** 2).sum()) / denom
            - 10.0 * (c ** 2).sum()
        )

    def forward(self, x, t):
        x = x.double()
        t = t.double().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        means   = self.means.double()
        sigma2  = self.sigma2.double()
        weights = self.weights.double()
        eps = 1e-40
        dk = t_ * sigma2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * means[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + eps
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(weights)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV  = 2 * self.a * (1 - t_) * sigma2[None, :] / dk
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        V = torch.logsumexp(log_pw + log_zk, dim=1)
        return V.float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)

# anal_fn wraps CPU-only AnalyticalValue; returns result on input device
def anal_fn(x, t):
    result = _anal_vm_cpu(x.cpu(), t.cpu())
    return result.to(x.device)


# ---------------------------------------------------------------------------
# GMM Drift
# ---------------------------------------------------------------------------
def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    orig_log_weights = torch.log(weights_)
    denominator = 2 * a * (1 - ts) + ts * sigmas_ ** 2
    likelihood_exp_numerator = reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum")
    likelihood_exp = -likelihood_exp_numerator / (2 * ts * denominator)
    log_std_factor = torch.log(2 * a * (1 - ts) / denominator) * d / 2
    log_rel_weights = orig_log_weights + likelihood_exp + log_std_factor
    normalization = torch.logsumexp(log_rel_weights, dim=1, keepdim=True)
    log_weights = log_rel_weights - normalization
    log_weights = torch.where((ts == 0), orig_log_weights, log_weights)
    std_factor = torch.exp(log_std_factor)
    new_means = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denominator[:, None, :]
    new_sigmas = torch.sqrt(0.5 * std_factor * sigmas_ ** 2)
    return {"log_weights": log_weights, "means": new_means, "sigmas": new_sigmas}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt, ts,
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "cpu"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)

# ---------------------------------------------------------------------------
# SMC value functions
# ---------------------------------------------------------------------------
# Uniform resampling (constant smc_value = log(1) = 0)
smc_const  = lambda x, t: torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
# Reward-based importance weights (no t-dependence)
smc_reward = lambda x, t: reward(x)
# Analytical value function as smc_value
smc_anal   = lambda x, t: anal_fn(x, t)


# ---------------------------------------------------------------------------
# GMM sample function for off-policy
# ---------------------------------------------------------------------------
means_np   = clf.means_
sigmas_np  = np.sqrt(clf.covariances_)
weights_np = clf.weights_


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


# ---------------------------------------------------------------------------
# Load analytical target E_OPT
# ---------------------------------------------------------------------------
with open("notebooks/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]
print(f"E_OPT = {E_OPT:.4f}  V(0,0) = {V_0_0:.4f}")

# Constant value function = V(0,0).  Used in stages 1/2 with lambda=1 so that
# the bootstrap target carries no oracle information (pure MC).  With lam=1,
# the one-step bootstrap is completely ignored anyway; this is just a safeguard.
def value_const(x, _t):
    return torch.full((x.shape[0],), V_0_0, device=x.device, dtype=x.dtype)

# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------
all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()


def load_value_fn(ckpt_path, model_class):
    """Load a value function from a checkpoint file."""
    vm = ValueNetwork(D, bias=bias_val)
    dummy_drift = lambda x, t: torch.zeros_like(x)
    if model_class == "on_policy":
        tmp = OnPolicyValue(
            base_score_module=dummy_drift, value_module=vm,
            dim=D, a=a, lr=1e-3, loss_type="quad"
        )
    else:
        tmp = OffPolicyValue(
            base_score_module=dummy_drift, reward_function=reward,
            value_module=vm, a=a, lr=1e-3, dim=D
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tmp.load_state_dict(ckpt["state_dict"])
    vm_loaded = tmp.value_module.eval()
    return lambda x, t: vm_loaded(x.cpu(), t.cpu().flatten())


print("Loading checkpoints...")

CKPT_EARLY = "checkpoints/convergence_run/single_seed_td_lam0.6/best.ckpt"
CKPT_MID   = "checkpoints/convergence_run/single_seed_td_lam0.6/best-v1.ckpt"
CKPT_BEST  = "checkpoints/offpolicy_convergence/best.ckpt"

early_model_fn = load_value_fn(CKPT_EARLY, "on_policy")
print(f"  Loaded early model (step~3600) from {CKPT_EARLY}")

mid_model_fn = load_value_fn(CKPT_MID, "on_policy")
print(f"  Loaded mid model (step~10400) from {CKPT_MID}")

best_model_fn = load_value_fn(CKPT_BEST, "off_policy")
print(f"  Loaded best model (off-policy, step~11668) from {CKPT_BEST}")

# For smc_value versions of the model functions (same signature)
def make_smc_model_fn(model_fn):
    return lambda x, t: model_fn(x, t)

early_smc_fn = make_smc_model_fn(early_model_fn)
mid_smc_fn   = make_smc_model_fn(mid_model_fn)
best_smc_fn  = make_smc_model_fn(best_model_fn)


# Blended value functions: t*r(x) + (1-t)*V(x,t)
# This is what one_step_bootstrap was accidentally using before the raw_value_fn fix.
def make_blended_fn(model_fn):
    def blended(x, t):
        v = model_fn(x, t)
        r = reward(x)
        t_flat = t.flatten()
        return t_flat * r + (1 - t_flat) * v
    return blended

early_blended_fn = make_blended_fn(early_model_fn)
mid_blended_fn   = make_blended_fn(mid_model_fn)
best_blended_fn  = make_blended_fn(best_model_fn)

# ---------------------------------------------------------------------------
# Lambda sweep configuration
# ---------------------------------------------------------------------------
# lambda_eff values.  Per-step lambda = lambda_eff^(1/n_steps).
# With n_steps=100:
#   λ=0       → per_step=0
#   1e-100    → per_step=0.1
#   7.89e-31  → per_step=0.5
#   0.1       → per_step≈0.977
#   0.5       → per_step≈0.993
#   0.8       → per_step≈0.998
#   1.0       → per_step=1.0
LAMBDA_VALUES = [0.0, 1e-100, 7.89e-31, 0.1, 0.5, 0.8, 1.0]
LAMBDA_LABELS = ["λ=0", "λ_s=0.1", "λ_s=0.5", "λ_eff=0.1", "λ_eff=0.5", "λ_eff=0.8", "λ=1"]

# ---------------------------------------------------------------------------
# Binning configuration
# ---------------------------------------------------------------------------
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_MIDS  = [0.1, 0.3, 0.5, 0.7, 0.9]

# ---------------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------------
METHOD_STYLES = {
    "ancestral_td_lambda":    {"ls": "-",  "marker": "o"},
    "single_seed_td_lambda":  {"ls": "--", "marker": "s"},
    "ancestral_mc_td_lambda": {"ls": ":",  "marker": "^"},
    "single_seed_mc":         {"ls": "-.", "marker": "D"},
    "one_step_bootstrap":     {"ls": "-",  "marker": "P"},
    "fbrrt":                  {"ls": "-",  "marker": "X"},
    "fbrrt_td_lambda":        {"ls": "--", "marker": "X"},
    "fbrrt_cv":               {"ls": ":",  "marker": "*"},
    "fbrrt_mc_z":             {"ls": "-.", "marker": "v"},
}
METHOD_COLORS = {
    "ancestral_td_lambda":    "#e74c3c",
    "single_seed_td_lambda":  "#3498db",
    "ancestral_mc_td_lambda": "#2ecc71",
    "single_seed_mc":         "#9b59b6",
    "one_step_bootstrap":     "#f39c12",
    "fbrrt":                  "#1abc9c",
    "fbrrt_td_lambda":        "#e67e22",
    "fbrrt_cv":               "#16a085",
    "fbrrt_mc_z":             "#c0392b",
    "off_policy":             "#2c3e50",
}
METHOD_DISPLAY = {
    "ancestral_td_lambda":    "Ancestral TD(λ)",
    "single_seed_td_lambda":  "Single-Seed TD(λ)",
    "ancestral_mc_td_lambda": "Ancestral MC-TD(λ)",
    "single_seed_mc":         "Single-Seed MC",
    "one_step_bootstrap":     "One-Step Bootstrap",
    "fbrrt":                  "FBRRT",
    "fbrrt_td_lambda":        "FBRRT-TD(λ)",
    "fbrrt_cv":               "FBRRT-CV",
    "fbrrt_mc_z":             "FBRRT-MCZ",
    "off_policy":             "Off-Policy",
}

# Viridis colormap for lambda values
_viridis = cm.get_cmap("viridis", len(LAMBDA_VALUES))
LAMBDA_COLORS = [_viridis(i) for i in range(len(LAMBDA_VALUES))]

# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def collect_onpolicy(sampling_method, value_fn, smc_value_fn, lambda_eff,
                     n_batches=20, batch_size=512):
    """Collect (x, t, target) tensors from OnPolicySMCDataset.

    OnPolicySMCDataset kwargs: batch_size=32, n_steps=100, mc_samples_per_step=10.
    Per-call output sizes (before DataLoader batching):
      - single_seed_td_lambda / single_seed_mc: 32 * 100 = 3200 samples
      - ancestral_td_lambda: ceil(32/10) * 10 * (100-1) = 3960 samples  [t=0 excluded]
      - ancestral_mc_td_lambda: ceil(32/10) * 10 * 100 = 4000 samples
    n_batches * batch_size = 10240 total, ~2048 per time bin (5 bins × 20 steps).
    """
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=value_fn, smc_value=smc_value_fn,
        reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
    )
    loader = DataLoader(ds, batch_size=batch_size)
    xs, ts, tgts = [], [], []
    for i, (y, x, t) in enumerate(loader):
        xs.append(x)
        ts.append(t.flatten())
        tgts.append(y.flatten())
        if i + 1 >= n_batches:
            break
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


def collect_fbrrt_direct(v_fn, n_calls=10, n_steps=100, n_particles=10, branch=4,
                          entropy_lambda=1.0, alpha=1.0):
    """Collect (x, t, target) by calling fbrrt_smc_grad_control directly.
    Mirrors collect_fbrrt_cv_direct so that branch / n_particles / n_steps can
    be varied without going through OnPolicySMCDataset (which fixes branch=4)."""
    from diffusion_rl.models.on_policy import fbrrt_smc_grad_control

    xs, ts, tgts = [], [], []
    for _ in range(n_calls):
        out = fbrrt_smc_grad_control(
            a=a, n_steps=n_steps, n_particles=n_particles, branch=branch,
            f=base_drift, v_theta=v_fn, reward=reward,
            d=D, alpha=alpha, entropy_lambda=entropy_lambda,
            device=torch.device(DEVICE),
        )
        xs.append(out.x)
        ts.append(out.t)
        tgts.append(out.v_hat)
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


def collect_fbrrt_mc_z_direct(v_policy_fn, v_target_fn, n_calls=10,
                                n_steps=100, n_particles=10, branch=4,
                                entropy_lambda=1.0, alpha=1.0):
    """Collect (x, t, target) by calling fbrrt_smc_grad_mc_Z directly with
    separate v_policy / v_target functions."""
    from diffusion_rl.models.on_policy import fbrrt_smc_grad_mc_Z

    xs, ts, tgts = [], [], []
    for _ in range(n_calls):
        out = fbrrt_smc_grad_mc_Z(
            a=a, n_steps=n_steps, n_particles=n_particles, branch=branch,
            f=base_drift, v_policy=v_policy_fn, v_target=v_target_fn,
            reward=reward, d=D, alpha=alpha, entropy_lambda=entropy_lambda,
            device=torch.device(DEVICE),
        )
        xs.append(out.x)
        ts.append(out.t)
        tgts.append(out.v_hat)
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


def collect_fbrrt_cv_direct(v_policy_fn, v_target_fn, n_calls=10,
                             n_steps=100, n_particles=10, branch=4,
                             entropy_lambda=1.0, alpha=1.0):
    """Collect (x, t, target) by calling fbrrt_smc_grad_control_variate directly
    with separate v_policy / v_target functions.

    The OnPolicySMCDataset wiring forces v_policy = v_target = self.value, which
    zeroes out the residual control variate term.  This helper bypasses the
    dataset so we can pass distinct functions and actually exercise the RCV.
    """
    from diffusion_rl.models.on_policy import fbrrt_smc_grad_control_variate

    xs, ts, tgts = [], [], []
    for _ in range(n_calls):
        out = fbrrt_smc_grad_control_variate(
            a=a, n_steps=n_steps, n_particles=n_particles, branch=branch,
            f=base_drift, v_policy=v_policy_fn, v_target=v_target_fn,
            reward=reward, d=D, alpha=alpha, entropy_lambda=entropy_lambda,
            device=torch.device(DEVICE),
        )
        xs.append(out.x)
        ts.append(out.t)
        tgts.append(out.v_hat)
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


def collect_offpolicy(n_batches=10, batch_size=512):
    """Collect (x, t, target) tensors from InterpolatingNumpyDataset."""
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=batch_size)
    loader = DataLoader(ds, batch_size=batch_size)
    xs, ts, tgts = [], [], []
    for i, (x1, x, t) in enumerate(loader):
        xs.append(x)
        ts.append(t.flatten())
        tgts.append(reward(x1))
        if i + 1 >= n_batches:
            break
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


N_PER_BIN = 1000  # samples per bin used for stats (truncated for equal comparison)


def binned_stats(x, t, target, n_per_bin=N_PER_BIN, seed=42):
    """Returns dict: bin_name -> {'mean', 'std', 'var', 'n'} for target - V_anal.

    Each bin is truncated to at most n_per_bin samples (randomly subsampled
    without replacement) before computing statistics.  This ensures all methods
    are compared on the same number of samples per bin, eliminating any
    artefacts from different per-call output sizes.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        v_anal = anal_fn(x, t)
    err = target - v_anal
    stats = {}
    for name, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (t >= lo) & (t < hi)
        n_avail = mask.sum().item()
        if n_avail < 2:
            stats[name] = {"n": n_avail, "mean": float("nan"),
                           "std": float("nan"), "var": float("nan")}
            continue
        e = err[mask]
        if n_avail > n_per_bin:
            idx = torch.randperm(n_avail, generator=rng)[:n_per_bin]
            e = e[idx]
        stats[name] = {
            "n":    len(e),
            "mean": e.mean().item(),
            "std":  e.std().item(),
            "var":  e.var().item(),
        }
    return stats


def avg_stats(stats):
    """Returns (avg_var, avg_abs_bias) for a stats dict."""
    vars_   = [v["var"]        for v in stats.values() if not np.isnan(v["var"])]
    biases  = [abs(v["mean"])  for v in stats.values() if not np.isnan(v["mean"])]
    avg_var  = float(np.mean(vars_))   if vars_   else float("nan")
    avg_bias = float(np.mean(biases))  if biases  else float("nan")
    return avg_var, avg_bias


def print_stats_table(label, stats):
    avg_var, avg_bias = avg_stats(stats)
    print(f"    {label:<40}  avg_var={avg_var:8.4f}  avg|bias|={avg_bias:8.4f}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style(entry):
    """Return (lw, ls, color, marker, zorder) for a result entry."""
    method  = entry["method"]
    lam_idx = entry.get("lambda_idx", None)
    is_offp = entry.get("is_offpolicy", False)
    if is_offp:
        return 2.5, "--", METHOD_COLORS["off_policy"], "x", 10
    return (1.5,
            METHOD_STYLES.get(method, {}).get("ls", "-"),
            LAMBDA_COLORS[lam_idx] if lam_idx is not None else METHOD_COLORS.get(method, "#888888"),
            METHOD_STYLES.get(method, {}).get("marker", "o"),
            5)


def _plot_lines(ax, entries, key, log_scale=True):
    """
    Plot one metric (key = 'var' or 'bias') for all entries.
    Returns list of (handle, label) for legend construction.
    """
    handles = []
    for entry in entries:
        stats  = entry["stats"]
        label  = entry["label"]
        lw, ls, color, marker, zorder = _style(entry)
        if key == "var":
            vals = [max(stats[b]["var"],           1e-9) for b in BIN_NAMES]
        else:
            vals = [max(abs(stats[b]["mean"]),     1e-9) for b in BIN_NAMES]
        h, = ax.plot(BIN_MIDS, vals, ls=ls, color=color, lw=lw,
                     marker=marker, markersize=4, zorder=zorder, label=label)
        handles.append((h, label))
    ax.set_xticks(BIN_MIDS)
    ax.set_xticklabels([f"{m:.1f}" for m in BIN_MIDS])
    ax.grid(True, alpha=0.3, which="both")
    if log_scale:
        ax.set_yscale("log")
    return handles


def _plot_scatter(ax, entries):
    """Bias-variance scatter on log-log axes."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    for entry in entries:
        avg_var, avg_bias = avg_stats(entry["stats"])
        if np.isnan(avg_var) or np.isnan(avg_bias):
            continue
        avg_var  = max(avg_var,  1e-9)
        avg_bias = max(avg_bias, 1e-9)
        lw, ls, color, marker, zorder = _style(entry)
        sz = 200 if entry.get("is_offpolicy", False) else 60
        ax.scatter(avg_bias, avg_var, marker=marker, s=sz, color=color,
                   zorder=zorder, edgecolors="black", linewidths=0.4)
    ax.grid(True, alpha=0.3, which="both")


def plot_variance_bias(stage_results, stage_name, output_path, include_scatter=False):
    """
    Layout (2 rows × 2 cols, all stages):
      [0,0] Variance — all methods, log scale
      [0,1] |Bias|   — all methods, log scale  + legend
      [1,0] Variance — on-policy only, log scale  (zoom)
      [1,1] Scatter (log-log, λ-sweep stages)  OR  |Bias| on-policy zoom
    """
    onpolicy = [e for e in stage_results if not e.get("is_offpolicy", False)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_all_v, ax_all_b, ax_zoom, ax_br = (axes[0, 0], axes[0, 1],
                                           axes[1, 0], axes[1, 1])

    # Row 0 — all methods
    ax_all_v.set_title(f"{stage_name} — Variance (all, log)")
    ax_all_v.set_xlabel("t"); ax_all_v.set_ylabel("Var(target − V_anal)")
    _plot_lines(ax_all_v, stage_results, "var")

    ax_all_b.set_title(f"{stage_name} — |Bias| (all, log)")
    ax_all_b.set_xlabel("t"); ax_all_b.set_ylabel("|E[target − V_anal]|")
    hl = _plot_lines(ax_all_b, stage_results, "bias")
    ax_all_b.legend([h for h, _ in hl], [l for _, l in hl],
                    bbox_to_anchor=(1.02, 1), loc="upper left",
                    borderaxespad=0, fontsize=7)

    # Row 1 left — on-policy variance zoom
    ax_zoom.set_title(f"{stage_name} — Variance (on-policy zoom, log)")
    ax_zoom.set_xlabel("t"); ax_zoom.set_ylabel("Var(target − V_anal)")
    _plot_lines(ax_zoom, onpolicy, "var")

    # Row 1 right — scatter (λ-sweep) or on-policy bias zoom
    if include_scatter:
        ax_br.set_title(f"{stage_name} — Bias-Variance Scatter (log-log)")
        ax_br.set_xlabel("avg |bias|"); ax_br.set_ylabel("avg variance")
        _plot_scatter(ax_br, stage_results)
    else:
        ax_br.set_title(f"{stage_name} — |Bias| (on-policy zoom, log)")
        ax_br.set_xlabel("t"); ax_br.set_ylabel("|E[target − V_anal]|")
        _plot_lines(ax_br, onpolicy, "bias")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# All results container
# ---------------------------------------------------------------------------
ALL_RESULTS = {}

# ---------------------------------------------------------------------------
# Replot-from-JSON path  (python data_quality_v2.py --replot)
# ---------------------------------------------------------------------------
RESULTS_JSON = "experiments/data_quality/data_quality_v2_results.json"
REPLOT_ONLY  = "--replot" in sys.argv or (
    os.path.exists(RESULTS_JSON) and "--replot" in sys.argv
)

# Optional stage filter: DQ2_ONLY=stage11a,stage11b skips other stages and
# merges new results into the existing JSON instead of clobbering it.
_DQ2_ONLY = os.environ.get("DQ2_ONLY", "").strip()
ONLY_STAGES = set(s.strip() for s in _DQ2_ONLY.split(",") if s.strip()) or None

def _should_run(stage_key):
    return ONLY_STAGES is None or stage_key in ONLY_STAGES

STAGE_META = [
    ("stage1",  "Stage 1: Unbiased MC baseline\n(λ=1, uniform SMC, const V)",
                "experiments/data_quality/dq2_stage1.png", False),
    ("stage2",  "Stage 2: Reward-guided MC\n(λ=1, reward SMC, const V)",
                "experiments/data_quality/dq2_stage2.png", False),
    ("stage3",  "Stage 3: Oracle lower bound\n(oracle V + oracle SMC)",
                "experiments/data_quality/dq2_stage3.png", True),
    ("stage4",  "Stage 4: Oracle SMC, model V\n(best model V + oracle SMC)",
                "experiments/data_quality/dq2_stage4.png", True),
    ("stage5",  "Stage 5: Reward SMC, model V\n(best model V + reward SMC)",
                "experiments/data_quality/dq2_stage5.png", True),
    ("stage6",  "Stage 6: Self-consistent best\n(best model V + best model SMC)",
                "experiments/data_quality/dq2_stage6.png", True),
    ("stage7a", "Stage 7a: Self-consistent early\n(early ckpt V + early ckpt SMC)",
                "experiments/data_quality/dq2_stage7a.png", True),
    ("stage7b", "Stage 7b: Self-consistent mid\n(mid ckpt V + mid ckpt SMC)",
                "experiments/data_quality/dq2_stage7b.png", True),
    # --- Reward SMC + raw value at different checkpoints ---
    ("stage8a", "Stage 8a: Reward SMC + early V\n(early ckpt V + reward SMC)",
                "experiments/data_quality/dq2_stage8a.png", True),
    ("stage8b", "Stage 8b: Reward SMC + mid V\n(mid ckpt V + reward SMC)",
                "experiments/data_quality/dq2_stage8b.png", True),
    ("stage8c", "Stage 8c: Reward SMC + best V\n(best model V + reward SMC)",
                "experiments/data_quality/dq2_stage8c.png", True),
    # --- Reward SMC + blended value ---
    ("stage9a", "Stage 9a: Reward SMC + blended early V\n(t*r+(1-t)*V_early + reward SMC)",
                "experiments/data_quality/dq2_stage9a.png", True),
    ("stage9b", "Stage 9b: Reward SMC + blended mid V\n(t*r+(1-t)*V_mid + reward SMC)",
                "experiments/data_quality/dq2_stage9b.png", True),
    ("stage9c", "Stage 9c: Reward SMC + blended best V\n(t*r+(1-t)*V_best + reward SMC)",
                "experiments/data_quality/dq2_stage9c.png", True),
    # --- Blended value for both V and SMC ---
    ("stage10a", "Stage 10a: Blended early for V and SMC\n(blended V_early for both value & smc)",
                 "experiments/data_quality/dq2_stage10a.png", True),
    ("stage10b", "Stage 10b: Blended mid for V and SMC\n(blended V_mid for both value & smc)",
                 "experiments/data_quality/dq2_stage10b.png", True),
    ("stage10c", "Stage 10c: Blended best for V and SMC\n(blended V_best for both value & smc)",
                 "experiments/data_quality/dq2_stage10c.png", True),
    # --- FBRRT-CV with lagged v_policy / live v_target ---
    ("stage11a", "Stage 11a: FBRRT-CV lagged\n(v_policy=early, v_target=mid)",
                 "experiments/data_quality/dq2_stage11a.png", False),
    ("stage11b", "Stage 11b: FBRRT-CV lagged\n(v_policy=mid, v_target=best)",
                 "experiments/data_quality/dq2_stage11b.png", False),
    ("stage11c", "Stage 11c: FBRRT-CV lagged\n(v_policy=best, v_target=oracle)",
                 "experiments/data_quality/dq2_stage11c.png", False),
]


def _load_entry(d):
    return {
        "label":       d["label"],
        "method":      d["method"],
        "lambda_idx":  d.get("lambda_idx"),
        "is_offpolicy":d.get("is_offpolicy", False),
        "stats":       {b: d["stats"][b] for b in BIN_NAMES},
    }


if REPLOT_ONLY:
    print(f"--replot: loading {RESULTS_JSON} and regenerating plots...")
    with open(RESULTS_JSON) as f:
        saved = json.load(f)
    for key, title, path, scatter in STAGE_META:
        if key not in saved:
            print(f"  [skip] {key} not in saved results")
            continue
        entries = [_load_entry(d) for d in saved[key]]
        plot_variance_bias(entries, title, path, include_scatter=scatter)
    print("Done.")
    sys.exit(0)

# ===========================================================================
# STAGE 1: lambda=1, smc=constant (uniform resampling), value=const
# Pure MC baseline: no oracle V boost, no informative SMC.
# value_const ensures the one-step bootstrap plays no role (it is also
# overridden by lam=1, which skips the blend entirely).
# ===========================================================================
if _should_run("stage1"):
    print("\n" + "=" * 70)
    print("STAGE 1: lambda=1, smc=constant (uniform resampling), value=const")
    print("=" * 70)

    stage1_results = []

    # Off-policy baseline
    print("  [off_policy] collecting...")
    x, t, tgt = collect_offpolicy()
    s = binned_stats(x, t, tgt)
    stage1_results.append({
        "label": "Off-Policy (baseline)", "method": "off_policy",
        "stats": s, "is_offpolicy": True
    })
    print_stats_table("off_policy", s)

    # On-policy methods with smc_const, lambda=1, value=const
    for method in ["ancestral_td_lambda", "single_seed_td_lambda",
                   "single_seed_mc", "ancestral_mc_td_lambda", "one_step_bootstrap",
                   "fbrrt", "fbrrt_cv", "fbrrt_mc_z"]:
        lam = 1.0
        label = f"{METHOD_DISPLAY[method]} (λ=1, smc=const)"
        print(f"  [{method}] collecting with λ=1, smc=const, value=const...")
        try:
            x, t, tgt = collect_onpolicy(method, value_const, smc_const, lam)
            s = binned_stats(x, t, tgt)
        except Exception as e:
            print(f"    ERROR: {e}")
            s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"), "var": float("nan")}
                 for b in BIN_NAMES}
        stage1_results.append({
            "label": label, "method": method,
            "lambda_idx": len(LAMBDA_VALUES) - 1,  # λ=1
            "stats": s, "is_offpolicy": False
        })
        print_stats_table(label, s)

    ALL_RESULTS["stage1"] = stage1_results
    _key, _title, _path, _sc = STAGE_META[0]
    plot_variance_bias(stage1_results, _title, _path, include_scatter=_sc)

# ===========================================================================
# STAGE 2: lambda=1, smc=reward, value=const
# Like stage 1 but with reward-based SMC twist (rudimentary importance
# sampling).  Still pure MC (lam=1), still no oracle V bootstrap.
# ===========================================================================
if _should_run("stage2"):
    print("\n" + "=" * 70)
    print("STAGE 2: lambda=1, smc=reward, value=const")
    print("=" * 70)

    stage2_results = []

    # Off-policy baseline (same as stage 1)
    print("  [off_policy] collecting...")
    x, t, tgt = collect_offpolicy()
    s = binned_stats(x, t, tgt)
    stage2_results.append({
        "label": "Off-Policy (baseline)", "method": "off_policy",
        "stats": s, "is_offpolicy": True
    })
    print_stats_table("off_policy", s)

    # On-policy methods with smc_reward, lambda=1, value=const
    for method in ["ancestral_td_lambda", "single_seed_td_lambda",
                   "single_seed_mc", "ancestral_mc_td_lambda", "one_step_bootstrap",
                   "fbrrt", "fbrrt_cv", "fbrrt_mc_z"]:
        lam = 1.0
        label = f"{METHOD_DISPLAY[method]} (λ=1, smc=reward)"
        print(f"  [{method}] collecting with λ=1, smc=reward, value=const...")
        try:
            x, t, tgt = collect_onpolicy(method, value_const, smc_reward, lam)
            s = binned_stats(x, t, tgt)
        except Exception as e:
            print(f"    ERROR: {e}")
            s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"), "var": float("nan")}
                 for b in BIN_NAMES}
        stage2_results.append({
            "label": label, "method": method,
            "lambda_idx": len(LAMBDA_VALUES) - 1,  # λ=1
            "stats": s, "is_offpolicy": False
        })
        print_stats_table(label, s)

    ALL_RESULTS["stage2"] = stage2_results
    _key, _title, _path, _sc = STAGE_META[1]
    plot_variance_bias(stage2_results, _title, _path, include_scatter=_sc)


# ===========================================================================
# Helper: run a lambda sweep for a given (value_fn, smc_value_fn) combo
# ===========================================================================

def run_lambda_sweep(value_fn, smc_fn, stage_name, stage_key,
                     output_path, n_batches=10):
    """
    Runs lambda sweep across all LAMBDA_VALUES for the three TD-lambda methods
    and also runs single_seed_mc once, plus off-policy baseline.

    Returns list of result dicts.
    """
    print(f"\n{'=' * 70}")
    print(stage_name)
    print("=" * 70)

    results = []

    # Off-policy baseline
    print("  [off_policy] collecting...")
    x, t, tgt = collect_offpolicy(n_batches=n_batches)
    s = binned_stats(x, t, tgt)
    results.append({
        "label": "Off-Policy (baseline)",
        "method": "off_policy",
        "stats": s,
        "is_offpolicy": True,
    })
    print_stats_table("off_policy", s)

    # TD-lambda methods with lambda sweep
    td_methods = ["ancestral_td_lambda", "single_seed_td_lambda", "ancestral_mc_td_lambda",
                  "fbrrt_td_lambda"]
    for method in td_methods:
        for lam_idx, (lam, lam_label) in enumerate(zip(LAMBDA_VALUES, LAMBDA_LABELS)):
            label = f"{METHOD_DISPLAY[method]} ({lam_label})"
            print(f"  [{method}] {lam_label}={lam:.5f}  collecting...")
            try:
                x, t, tgt = collect_onpolicy(
                    method, value_fn, smc_fn, lam, n_batches=n_batches
                )
                s = binned_stats(x, t, tgt)
            except Exception as e:
                print(f"    ERROR: {e}")
                s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                          "var": float("nan")} for b in BIN_NAMES}
            results.append({
                "label": label,
                "method": method,
                "lambda_idx": lam_idx,
                "lambda_val": lam,
                "stats": s,
                "is_offpolicy": False,
            })
            print_stats_table(label, s)

    # Single-seed MC (MC equivalent to λ=1)
    print(f"  [single_seed_mc] collecting...")
    try:
        x, t, tgt = collect_onpolicy(
            "single_seed_mc", value_fn, smc_fn, 0.5, n_batches=n_batches
        )
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                  "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"{METHOD_DISPLAY['single_seed_mc']} (MC)",
        "method": "single_seed_mc",
        "lambda_idx": len(LAMBDA_VALUES) - 1,  # visual: treat as λ≈1
        "stats": s,
        "is_offpolicy": False,
    })
    print_stats_table("single_seed_mc", s)

    # One-step bootstrap (no lambda parameter; uses child-averaging)
    print(f"  [one_step_bootstrap] collecting...")
    try:
        x, t, tgt = collect_onpolicy(
            "one_step_bootstrap", value_fn, smc_fn, 0.0, n_batches=n_batches
        )
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                  "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"{METHOD_DISPLAY['one_step_bootstrap']}",
        "method": "one_step_bootstrap",
        "lambda_idx": 0,  # visual: treat as λ≈0 (pure one-step)
        "stats": s,
        "is_offpolicy": False,
    })
    print_stats_table("one_step_bootstrap", s)

    # FBRRT (equivalent to fbrrt_td_lambda at λ=0)
    print(f"  [fbrrt] collecting...")
    try:
        x, t, tgt = collect_onpolicy(
            "fbrrt", value_fn, smc_fn, 0.0, n_batches=n_batches
        )
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                  "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": METHOD_DISPLAY["fbrrt"],
        "method": "fbrrt",
        "lambda_idx": 0,  # visual: treat as λ=0
        "stats": s,
        "is_offpolicy": False,
    })
    print_stats_table("fbrrt", s)

    # FBRRT-CV (residual control variate; v_policy=v_target=value_fn here, so
    # the residual term is identically zero and FBRRT-CV should match FBRRT
    # up to RNG noise.  This collects baseline numbers for the CV variant.)
    print(f"  [fbrrt_cv] collecting...")
    try:
        x, t, tgt = collect_onpolicy(
            "fbrrt_cv", value_fn, smc_fn, 0.0, n_batches=n_batches
        )
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                  "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": METHOD_DISPLAY["fbrrt_cv"],
        "method": "fbrrt_cv",
        "lambda_idx": 0,
        "stats": s,
        "is_offpolicy": False,
    })
    print_stats_table("fbrrt_cv", s)

    # FBRRT-MCZ (MC estimate of Z via Z = (1/dt)*mean[Y dW];
    # v_policy=v_target=value_fn here, so this is the single-V "MC-Z" baseline.)
    print(f"  [fbrrt_mc_z] collecting...")
    try:
        x, t, tgt = collect_onpolicy(
            "fbrrt_mc_z", value_fn, smc_fn, 0.0, n_batches=n_batches
        )
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                  "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": METHOD_DISPLAY["fbrrt_mc_z"],
        "method": "fbrrt_mc_z",
        "lambda_idx": 0,
        "stats": s,
        "is_offpolicy": False,
    })
    print_stats_table("fbrrt_mc_z", s)

    ALL_RESULTS[stage_key] = results
    plot_variance_bias(results, stage_name, output_path, include_scatter=True)

    return results


# ===========================================================================
# STAGE 3: oracle V + oracle SMC, lambda sweep
# ===========================================================================
if _should_run("stage3"):
    stage3_results = run_lambda_sweep(
        value_fn=anal_fn, smc_fn=smc_anal,
        stage_name=STAGE_META[2][1], stage_key="stage3",
        output_path=STAGE_META[2][2],
    )

# ===========================================================================
# STAGE 4: oracle SMC + best model V, lambda sweep
# ===========================================================================
if _should_run("stage4"):
    stage4_results = run_lambda_sweep(
        value_fn=best_model_fn, smc_fn=smc_anal,
        stage_name=STAGE_META[3][1], stage_key="stage4",
        output_path=STAGE_META[3][2],
    )

# ===========================================================================
# STAGE 5: reward SMC + best model V, lambda sweep
# ===========================================================================
if _should_run("stage5"):
    stage5_results = run_lambda_sweep(
        value_fn=best_model_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[4][1], stage_key="stage5",
        output_path=STAGE_META[4][2],
    )

# ===========================================================================
# STAGE 6: best model for both V and SMC, lambda sweep
# ===========================================================================
if _should_run("stage6"):
    stage6_results = run_lambda_sweep(
        value_fn=best_model_fn, smc_fn=best_smc_fn,
        stage_name=STAGE_META[5][1], stage_key="stage6",
        output_path=STAGE_META[5][2],
    )

# ===========================================================================
# STAGE 7a: early model for both V and SMC, lambda sweep
# ===========================================================================
if _should_run("stage7a"):
    stage7a_results = run_lambda_sweep(
        value_fn=early_model_fn, smc_fn=early_smc_fn,
        stage_name=STAGE_META[6][1], stage_key="stage7a",
        output_path=STAGE_META[6][2],
    )

# ===========================================================================
# STAGE 7b: mid model for both V and SMC, lambda sweep
# ===========================================================================
if _should_run("stage7b"):
    stage7b_results = run_lambda_sweep(
        value_fn=mid_model_fn, smc_fn=mid_smc_fn,
        stage_name=STAGE_META[7][1], stage_key="stage7b",
        output_path=STAGE_META[7][2],
    )

# ===========================================================================
# STAGES 8a-c: Reward SMC + raw value at different checkpoints
# Isolates the impact of reward SMC guidance with value functions of
# varying quality.
# ===========================================================================
if _should_run("stage8a"):
    stage8a_results = run_lambda_sweep(
        value_fn=early_model_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[8][1], stage_key="stage8a",
        output_path=STAGE_META[8][2],
    )

if _should_run("stage8b"):
    stage8b_results = run_lambda_sweep(
        value_fn=mid_model_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[9][1], stage_key="stage8b",
        output_path=STAGE_META[9][2],
    )

if _should_run("stage8c"):
    stage8c_results = run_lambda_sweep(
        value_fn=best_model_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[10][1], stage_key="stage8c",
        output_path=STAGE_META[10][2],
    )

# ===========================================================================
# STAGES 9a-c: Reward SMC + blended value (t*r + (1-t)*V)
# Tests whether the reward blend improves target quality.
# ===========================================================================
if _should_run("stage9a"):
    stage9a_results = run_lambda_sweep(
        value_fn=early_blended_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[11][1], stage_key="stage9a",
        output_path=STAGE_META[11][2],
    )

if _should_run("stage9b"):
    stage9b_results = run_lambda_sweep(
        value_fn=mid_blended_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[12][1], stage_key="stage9b",
        output_path=STAGE_META[12][2],
    )

if _should_run("stage9c"):
    stage9c_results = run_lambda_sweep(
        value_fn=best_blended_fn, smc_fn=smc_reward,
        stage_name=STAGE_META[13][1], stage_key="stage9c",
        output_path=STAGE_META[13][2],
    )

# ===========================================================================
# STAGES 10a-c: Blended value for BOTH V and SMC
# Tests whether using the blend for SMC resampling also helps.
# ===========================================================================
if _should_run("stage10a"):
    stage10a_results = run_lambda_sweep(
        value_fn=early_blended_fn, smc_fn=early_blended_fn,
        stage_name=STAGE_META[14][1], stage_key="stage10a",
        output_path=STAGE_META[14][2],
    )

if _should_run("stage10b"):
    stage10b_results = run_lambda_sweep(
        value_fn=mid_blended_fn, smc_fn=mid_blended_fn,
        stage_name=STAGE_META[15][1], stage_key="stage10b",
        output_path=STAGE_META[15][2],
    )

if _should_run("stage10c"):
    stage10c_results = run_lambda_sweep(
        value_fn=best_blended_fn, smc_fn=best_blended_fn,
        stage_name=STAGE_META[16][1], stage_key="stage10c",
        output_path=STAGE_META[16][2],
    )

# ===========================================================================
# STAGE 11: FBRRT-CV with lagged v_policy / live v_target
# We have four value functions of decreasing error:
#   early < mid < best < oracle (anal_fn)
# Treat each as a "lagged" copy of the next:
#   11a: v_policy=early, v_target=mid
#   11b: v_policy=mid,   v_target=best
#   11c: v_policy=best,  v_target=oracle (anal_fn)
# For each pairing we compare:
#   - FBRRT-CV (v_pol=lagged, v_tgt=live)  -- the intended use
#   - FBRRT-CV (v_pol=v_tgt=live)          -- collapses to FBRRT (sanity)
#   - FBRRT (v=v_target only, no lag)      -- naive baseline
#   - FBRRT (v=v_policy only, lagged)      -- naive baseline w/ stable drift
# ===========================================================================

def run_fbrrt_cv_lagged_stage(v_policy_fn, v_policy_label,
                               v_target_fn, v_target_label,
                               stage_name, stage_key, output_path,
                               n_calls=10, n_batches=10):
    print(f"\n{'=' * 70}")
    print(stage_name)
    print("=" * 70)

    results = []

    # 1) FBRRT-CV with lagged v_policy / live v_target  (intended use)
    print(f"  [fbrrt_cv  v_pol={v_policy_label}, v_tgt={v_target_label}] collecting...")
    try:
        x, t, tgt = collect_fbrrt_cv_direct(v_policy_fn, v_target_fn, n_calls=n_calls)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT-CV (v_pol={v_policy_label}, v_tgt={v_target_label})",
        "method": "fbrrt_cv", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt_cv  v_pol={v_policy_label}, v_tgt={v_target_label}", s)

    # 2) FBRRT-CV with v_policy = v_target = live (collapse: should match FBRRT)
    print(f"  [fbrrt_cv  v_pol=v_tgt={v_target_label}] collecting...")
    try:
        x, t, tgt = collect_fbrrt_cv_direct(v_target_fn, v_target_fn, n_calls=n_calls)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT-CV (v_pol=v_tgt={v_target_label})",
        "method": "fbrrt_cv", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt_cv  v_pol=v_tgt={v_target_label}", s)

    # 3) FBRRT with v=v_target only (live network alone)
    print(f"  [fbrrt  v={v_target_label}] collecting...")
    try:
        x, t, tgt = collect_onpolicy("fbrrt", v_target_fn, smc_reward, 0.0,
                                       n_batches=n_batches)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT (v={v_target_label})",
        "method": "fbrrt", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt v={v_target_label}", s)

    # 4) FBRRT with v=v_policy only (lagged network alone)
    print(f"  [fbrrt  v={v_policy_label}] collecting...")
    try:
        x, t, tgt = collect_onpolicy("fbrrt", v_policy_fn, smc_reward, 0.0,
                                       n_batches=n_batches)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT (v={v_policy_label})",
        "method": "fbrrt", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt v={v_policy_label}", s)

    # 5) FBRRT-MCZ with lagged v_policy / live v_target (intended use)
    print(f"  [fbrrt_mc_z  v_pol={v_policy_label}, v_tgt={v_target_label}] collecting...")
    try:
        x, t, tgt = collect_fbrrt_mc_z_direct(v_policy_fn, v_target_fn, n_calls=n_calls)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT-MCZ (v_pol={v_policy_label}, v_tgt={v_target_label})",
        "method": "fbrrt_mc_z", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt_mc_z  v_pol={v_policy_label}, v_tgt={v_target_label}", s)

    # 6) FBRRT-MCZ with v_policy = v_target = live (single-V baseline of MCZ)
    print(f"  [fbrrt_mc_z  v_pol=v_tgt={v_target_label}] collecting...")
    try:
        x, t, tgt = collect_fbrrt_mc_z_direct(v_target_fn, v_target_fn, n_calls=n_calls)
        s = binned_stats(x, t, tgt)
    except Exception as e:
        print(f"    ERROR: {e}")
        s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                 "var": float("nan")} for b in BIN_NAMES}
    results.append({
        "label": f"FBRRT-MCZ (v_pol=v_tgt={v_target_label})",
        "method": "fbrrt_mc_z", "lambda_idx": 0,
        "stats": s, "is_offpolicy": False,
    })
    print_stats_table(f"fbrrt_mc_z  v_pol=v_tgt={v_target_label}", s)

    ALL_RESULTS[stage_key] = results
    plot_variance_bias(results, stage_name, output_path, include_scatter=False)
    return results


if _should_run("stage11a"):
    stage11a_results = run_fbrrt_cv_lagged_stage(
        v_policy_fn=early_model_fn, v_policy_label="early",
        v_target_fn=mid_model_fn,   v_target_label="mid",
        stage_name=STAGE_META[17][1], stage_key="stage11a",
        output_path=STAGE_META[17][2],
    )

if _should_run("stage11b"):
    stage11b_results = run_fbrrt_cv_lagged_stage(
        v_policy_fn=mid_model_fn,  v_policy_label="mid",
        v_target_fn=best_model_fn, v_target_label="best",
        stage_name=STAGE_META[18][1], stage_key="stage11b",
        output_path=STAGE_META[18][2],
    )

if _should_run("stage11c"):
    stage11c_results = run_fbrrt_cv_lagged_stage(
        v_policy_fn=best_model_fn, v_policy_label="best",
        v_target_fn=anal_fn,       v_target_label="oracle",
        stage_name=STAGE_META[19][1], stage_key="stage11c",
        output_path=STAGE_META[19][2],
    )

# ===========================================================================
# STAGE 12: Branch-factor sweep for FBRRT and FBRRT-CV
# Theory says Var[Z_RCV] ~ |eps|^2 / (B * dt). Stage 11 used B=4 and saw a
# 5-40x variance penalty for FBRRT-CV.  Sweep B in {4, 10, 30, 100} for the
# same three pairings and check whether the residual term attenuates.
# ===========================================================================
if _should_run("stage12"):
    print(f"\n{'=' * 70}")
    print("STAGE 12: Branch-factor sweep B in {4, 10, 30, 100}")
    print("=" * 70)

    BRANCHES = [4, 10, 30, 100]
    PAIRINGS = [
        ("11a", early_model_fn, "early", mid_model_fn,  "mid"),
        ("11b", mid_model_fn,   "mid",   best_model_fn, "best"),
        ("11c", best_model_fn,  "best",  anal_fn,       "oracle"),
    ]
    N_CALLS_S12 = 5

    stage12_results = []
    for pair_label, v_pol_fn, pol_label, v_tgt_fn, tgt_label in PAIRINGS:
        print(f"\n--- Pairing {pair_label}: v_pol={pol_label}, v_tgt={tgt_label} ---")
        for B in BRANCHES:
            # FBRRT-CV with lagged v_policy / live v_target
            tag_cv = f"{pair_label} CV B={B:3d}"
            print(f"  [{tag_cv}] collecting...")
            try:
                x, t, tgt = collect_fbrrt_cv_direct(
                    v_pol_fn, v_tgt_fn, n_calls=N_CALLS_S12, branch=B,
                )
                s = binned_stats(x, t, tgt)
            except Exception as e:
                print(f"    ERROR: {e}")
                s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                         "var": float("nan")} for b in BIN_NAMES}
            stage12_results.append({
                "label": f"FBRRT-CV {pair_label} (v_pol={pol_label}, v_tgt={tgt_label}) B={B}",
                "method": "fbrrt_cv", "pairing": pair_label, "branch": B,
                "stats": s, "is_offpolicy": False, "lambda_idx": 0,
            })
            print_stats_table(tag_cv, s)

            # FBRRT with v=v_target only
            tag_fb = f"{pair_label} FB B={B:3d}"
            print(f"  [{tag_fb}] collecting...")
            try:
                x, t, tgt = collect_fbrrt_direct(
                    v_tgt_fn, n_calls=N_CALLS_S12, branch=B,
                )
                s = binned_stats(x, t, tgt)
            except Exception as e:
                print(f"    ERROR: {e}")
                s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                         "var": float("nan")} for b in BIN_NAMES}
            stage12_results.append({
                "label": f"FBRRT {pair_label} (v={tgt_label}) B={B}",
                "method": "fbrrt", "pairing": pair_label, "branch": B,
                "stats": s, "is_offpolicy": False, "lambda_idx": 0,
            })
            print_stats_table(tag_fb, s)

            # FBRRT-MCZ with lagged v_policy / live v_target
            tag_mcz = f"{pair_label} MCZ B={B:3d}"
            print(f"  [{tag_mcz}] collecting...")
            try:
                x, t, tgt = collect_fbrrt_mc_z_direct(
                    v_pol_fn, v_tgt_fn, n_calls=N_CALLS_S12, branch=B,
                )
                s = binned_stats(x, t, tgt)
            except Exception as e:
                print(f"    ERROR: {e}")
                s = {b: {"n": 0, "mean": float("nan"), "std": float("nan"),
                         "var": float("nan")} for b in BIN_NAMES}
            stage12_results.append({
                "label": f"FBRRT-MCZ {pair_label} (v_pol={pol_label}, v_tgt={tgt_label}) B={B}",
                "method": "fbrrt_mc_z", "pairing": pair_label, "branch": B,
                "stats": s, "is_offpolicy": False, "lambda_idx": 0,
            })
            print_stats_table(tag_mcz, s)

    ALL_RESULTS["stage12"] = stage12_results

    # ----- Plot stage 12 -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    pair_titles = {
        "11a": "11a (early → mid)",
        "11b": "11b (mid → best)",
        "11c": "11c (best → oracle)",
    }
    for col, (pair_label, _, pol_label, _, tgt_label) in enumerate(PAIRINGS):
        rows = [r for r in stage12_results if r.get("pairing") == pair_label]
        cv_rows  = sorted([r for r in rows if r["method"] == "fbrrt_cv"],   key=lambda r: r["branch"])
        fb_rows  = sorted([r for r in rows if r["method"] == "fbrrt"],      key=lambda r: r["branch"])
        mcz_rows = sorted([r for r in rows if r["method"] == "fbrrt_mc_z"], key=lambda r: r["branch"])
        Bs        = [r["branch"] for r in cv_rows]
        cv_var    = [avg_stats(r["stats"])[0] for r in cv_rows]
        fb_var    = [avg_stats(r["stats"])[0] for r in fb_rows]
        mcz_var   = [avg_stats(r["stats"])[0] for r in mcz_rows]
        cv_bias   = [avg_stats(r["stats"])[1] for r in cv_rows]
        fb_bias   = [avg_stats(r["stats"])[1] for r in fb_rows]
        mcz_bias  = [avg_stats(r["stats"])[1] for r in mcz_rows]

        ax = axes[0, col]
        ax.loglog(Bs, fb_var, "o-", color=METHOD_COLORS["fbrrt"],
                  label=f"FBRRT (v={tgt_label})", markersize=7)
        ax.loglog(Bs, cv_var, "s-", color=METHOD_COLORS["fbrrt_cv"],
                  label=f"FBRRT-CV (v_pol={pol_label})", markersize=7)
        ax.loglog(Bs, mcz_var, "v-", color=METHOD_COLORS["fbrrt_mc_z"],
                  label=f"FBRRT-MCZ (v_pol={pol_label})", markersize=7)
        # 1/B reference line (theory for residual)
        ref = cv_var[0] * (Bs[0] / np.array(Bs))
        ax.loglog(Bs, ref, "k:", alpha=0.4, label="∝ 1/B")
        ax.set_xlabel("branch B"); ax.set_ylabel("avg variance")
        ax.set_title(f"{pair_titles[pair_label]} — variance")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0: ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.loglog(Bs, fb_bias, "o-", color=METHOD_COLORS["fbrrt"],
                  label=f"FBRRT (v={tgt_label})", markersize=7)
        ax.loglog(Bs, cv_bias, "s-", color=METHOD_COLORS["fbrrt_cv"],
                  label=f"FBRRT-CV (v_pol={pol_label})", markersize=7)
        ax.loglog(Bs, mcz_bias, "v-", color=METHOD_COLORS["fbrrt_mc_z"],
                  label=f"FBRRT-MCZ (v_pol={pol_label})", markersize=7)
        ax.set_xlabel("branch B"); ax.set_ylabel("avg |bias|")
        ax.set_title(f"{pair_titles[pair_label]} — |bias|")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0: ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("experiments/data_quality/dq2_stage12_branch_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved plot: experiments/data_quality/dq2_stage12_branch_sweep.png")

# ===========================================================================
# Save all results to JSON
# ===========================================================================

def results_to_json(all_results):
    """Convert results to JSON-serializable form."""
    out = {}
    for stage_key, results in all_results.items():
        out[stage_key] = []
        for entry in results:
            rec = {
                "label":       entry["label"],
                "method":      entry["method"],
                "is_offpolicy": entry.get("is_offpolicy", False),
                "lambda_idx":  entry.get("lambda_idx", None),
                "lambda_val":  entry.get("lambda_val", None),
                "pairing":     entry.get("pairing", None),
                "branch":      entry.get("branch", None),
                "stats":       entry["stats"],
            }
            avg_var, avg_bias = avg_stats(entry["stats"])
            rec["avg_var"]  = avg_var
            rec["avg_bias"] = avg_bias
            out[stage_key].append(rec)
    return out


json_out = results_to_json(ALL_RESULTS)

# When running with DQ2_ONLY, merge new stage results into the existing JSON
# instead of clobbering stages we deliberately skipped.
if ONLY_STAGES is not None and os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON) as f:
        existing = json.load(f)
    existing.update(json_out)
    json_out = existing
    print(f"\nMerged {len(ALL_RESULTS)} stage(s) into existing JSON.")

with open(RESULTS_JSON, "w") as f:
    json.dump(json_out, f, indent=2)
print("\nAll results saved to experiments/data_quality/data_quality_v2_results.json")

# ===========================================================================
# Comprehensive Summary
# ===========================================================================

def print_summary_table(stage_key, title, results):
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    print(f"  {'Method / Config':<45}  {'avg_var':>10}  {'avg|bias|':>10}  {'rel_var':>8}")
    print(f"  {'-' * 77}")

    # Find off-policy baseline for relative comparison
    op_var = None
    for entry in results:
        if entry.get("is_offpolicy"):
            op_var, _ = avg_stats(entry["stats"])
            break

    for entry in results:
        avg_var, avg_bias = avg_stats(entry["stats"])
        rel_var = (avg_var / op_var) if (op_var and op_var > 0) else float("nan")
        label = entry["label"][:45]
        var_str  = f"{avg_var:.4f}"  if not np.isnan(avg_var)  else "    nan"
        bias_str = f"{avg_bias:.4f}" if not np.isnan(avg_bias) else "    nan"
        rel_str  = f"{rel_var:.3f}"  if not np.isnan(rel_var)  else "  nan"
        print(f"  {label:<45}  {var_str:>10}  {bias_str:>10}  {rel_str:>8}")


print("\n\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY")
print("=" * 80)

stage_titles = {
    "stage1":  "Stage 1: lambda~=1, smc=const (uniform resampling)",
    "stage2":  "Stage 2: lambda~=1, smc=reward",
    "stage3":  "Stage 3: Oracle V + Oracle SMC, lambda sweep",
    "stage4":  "Stage 4: Oracle SMC + Best Model V, lambda sweep",
    "stage5":  "Stage 5: Reward SMC + Best Model V, lambda sweep",
    "stage6":  "Stage 6: Best Model V + Best Model SMC, lambda sweep",
    "stage7a": "Stage 7a: Early Model (V + SMC), lambda sweep",
    "stage7b": "Stage 7b: Mid Model (V + SMC), lambda sweep",
    "stage11a": "Stage 11a: FBRRT-CV lagged (v_policy=early, v_target=mid)",
    "stage11b": "Stage 11b: FBRRT-CV lagged (v_policy=mid, v_target=best)",
    "stage11c": "Stage 11c: FBRRT-CV lagged (v_policy=best, v_target=oracle)",
}

for stage_key, title in stage_titles.items():
    if stage_key in ALL_RESULTS:
        print_summary_table(stage_key, title, ALL_RESULTS[stage_key])

# ---------------------------------------------------------------------------
# Key findings narrative
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("""
Stage 1 (λ≈1, smc=const):
  - Compares all methods at the MC limit (lambda~1) with uniform SMC resampling.
  - With smc=const, resampling provides no quality improvement over raw IS.
  - Off-policy baseline shows the variance floor from reward-only targets.
  - Methods with high lambda accumulate more trajectory variance.

Stage 2 (λ≈1, smc=reward):
  - Same as Stage 1 but SMC resamples toward high-reward regions.
  - Reward-based SMC should reduce variance for ancestral methods.
  - Single-seed methods unaffected by SMC choice (no resampling between seeds).

Stage 3 (Oracle V + Oracle SMC, lambda sweep):
  - Oracle setting: both V and SMC are the analytical truth.
  - Shows the INTRINSIC variance/bias of each method at optimality.
  - Lower lambda → more TD-like → lower variance, higher bias from bootstrapping.
  - Higher lambda → more MC-like → higher variance, lower bias.
  - Off-policy has zero bias (targets are V_anal by construction) but high variance.

Stage 4 (Oracle SMC + Best Model V, lambda sweep):
  - Replaces oracle V with the best trained model.
  - Shows how model error propagates through each method's target construction.
  - Higher lambda methods accumulate more model error along trajectories.
  - Oracle SMC still guides sampling toward high-value regions.

Stage 5 (Reward SMC + Best Model V, lambda sweep):
  - SMC now uses reward instead of oracle V for resampling.
  - Reward-based SMC less effective than oracle but more practical.
  - Compares the tradeoff between SMC quality and lambda.

Stage 6 (Best Model V + Best Model SMC, lambda sweep):
  - Fully self-consistent: model used for both V-targets and SMC resampling.
  - Most realistic training scenario (no oracle information).
  - Shows the bias-variance tradeoff under model imperfection.

Stage 7a (Early Model, lambda sweep):
  - Early training snapshot (step~3600): substantial model error expected.
  - High bias across all methods; variance dominated by model quality.

Stage 7b (Mid Model, lambda sweep):
  - Mid training snapshot (step~10400): model improving but not converged.
  - Transition behavior between early and best model performance.
  - Shows whether optimal lambda changes as training progresses.
""")

print(f"\nE_OPT (analytical) = {E_OPT:.4f}")
print("All plots saved to experiments/data_quality/dq2_stage*.png")
print("All data saved to experiments/data_quality/data_quality_v2_results.json")
print("\nDone.")
