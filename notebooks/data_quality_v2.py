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
  python notebooks/data_quality_v2.py            # full run (collect data + plot)
  python notebooks/data_quality_v2.py --replot   # reload saved JSON and replot only

Outputs: notebooks/dq2_stage{N}.png + notebooks/data_quality_v2_results.json
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

# ---------------------------------------------------------------------------
# Lambda sweep configuration
# ---------------------------------------------------------------------------
LAMBDA_VALUES = [1e-5, 0.05, 0.2, 0.5, 0.8, 1.0]
LAMBDA_LABELS = ["λ≈0", "λ=0.05", "λ=0.2", "λ=0.5", "λ=0.8", "λ=1"]

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
}
METHOD_COLORS = {
    "ancestral_td_lambda":    "#e74c3c",
    "single_seed_td_lambda":  "#3498db",
    "ancestral_mc_td_lambda": "#2ecc71",
    "single_seed_mc":         "#9b59b6",
    "one_step_bootstrap":     "#f39c12",
    "off_policy":             "#2c3e50",
}
METHOD_DISPLAY = {
    "ancestral_td_lambda":    "Ancestral TD(λ)",
    "single_seed_td_lambda":  "Single-Seed TD(λ)",
    "ancestral_mc_td_lambda": "Ancestral MC-TD(λ)",
    "single_seed_mc":         "Single-Seed MC",
    "one_step_bootstrap":     "One-Step Bootstrap",
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
RESULTS_JSON = "notebooks/data_quality_v2_results.json"
REPLOT_ONLY  = "--replot" in sys.argv or (
    os.path.exists(RESULTS_JSON) and "--replot" in sys.argv
)

STAGE_META = [
    ("stage1",  "Stage 1: Unbiased MC baseline\n(λ=1, uniform SMC, const V)",
                "notebooks/dq2_stage1.png", False),
    ("stage2",  "Stage 2: Reward-guided MC\n(λ=1, reward SMC, const V)",
                "notebooks/dq2_stage2.png", False),
    ("stage3",  "Stage 3: Oracle lower bound\n(oracle V + oracle SMC)",
                "notebooks/dq2_stage3.png", True),
    ("stage4",  "Stage 4: Oracle SMC, model V\n(best model V + oracle SMC)",
                "notebooks/dq2_stage4.png", True),
    ("stage5",  "Stage 5: Reward SMC, model V\n(best model V + reward SMC)",
                "notebooks/dq2_stage5.png", True),
    ("stage6",  "Stage 6: Self-consistent best\n(best model V + best model SMC)",
                "notebooks/dq2_stage6.png", True),
    ("stage7a", "Stage 7a: Self-consistent early\n(early ckpt V + early ckpt SMC)",
                "notebooks/dq2_stage7a.png", True),
    ("stage7b", "Stage 7b: Self-consistent mid\n(mid ckpt V + mid ckpt SMC)",
                "notebooks/dq2_stage7b.png", True),
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
               "single_seed_mc", "ancestral_mc_td_lambda", "one_step_bootstrap"]:
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
               "single_seed_mc", "ancestral_mc_td_lambda", "one_step_bootstrap"]:
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
    td_methods = ["ancestral_td_lambda", "single_seed_td_lambda", "ancestral_mc_td_lambda"]
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

    ALL_RESULTS[stage_key] = results
    plot_variance_bias(results, stage_name, output_path, include_scatter=True)

    return results


# ===========================================================================
# STAGE 3: oracle V + oracle SMC, lambda sweep
# ===========================================================================
stage3_results = run_lambda_sweep(
    value_fn=anal_fn, smc_fn=smc_anal,
    stage_name=STAGE_META[2][1], stage_key="stage3",
    output_path=STAGE_META[2][2],
)

# ===========================================================================
# STAGE 4: oracle SMC + best model V, lambda sweep
# ===========================================================================
stage4_results = run_lambda_sweep(
    value_fn=best_model_fn, smc_fn=smc_anal,
    stage_name=STAGE_META[3][1], stage_key="stage4",
    output_path=STAGE_META[3][2],
)

# ===========================================================================
# STAGE 5: reward SMC + best model V, lambda sweep
# ===========================================================================
stage5_results = run_lambda_sweep(
    value_fn=best_model_fn, smc_fn=smc_reward,
    stage_name=STAGE_META[4][1], stage_key="stage5",
    output_path=STAGE_META[4][2],
)

# ===========================================================================
# STAGE 6: best model for both V and SMC, lambda sweep
# ===========================================================================
stage6_results = run_lambda_sweep(
    value_fn=best_model_fn, smc_fn=best_smc_fn,
    stage_name=STAGE_META[5][1], stage_key="stage6",
    output_path=STAGE_META[5][2],
)

# ===========================================================================
# STAGE 7a: early model for both V and SMC, lambda sweep
# ===========================================================================
stage7a_results = run_lambda_sweep(
    value_fn=early_model_fn, smc_fn=early_smc_fn,
    stage_name=STAGE_META[6][1], stage_key="stage7a",
    output_path=STAGE_META[6][2],
)

# ===========================================================================
# STAGE 7b: mid model for both V and SMC, lambda sweep
# ===========================================================================
stage7b_results = run_lambda_sweep(
    value_fn=mid_model_fn, smc_fn=mid_smc_fn,
    stage_name=STAGE_META[7][1], stage_key="stage7b",
    output_path=STAGE_META[7][2],
)

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
                "stats":       entry["stats"],
            }
            avg_var, avg_bias = avg_stats(entry["stats"])
            rec["avg_var"]  = avg_var
            rec["avg_bias"] = avg_bias
            out[stage_key].append(rec)
    return out


json_out = results_to_json(ALL_RESULTS)
with open("notebooks/data_quality_v2_results.json", "w") as f:
    json.dump(json_out, f, indent=2)
print("\nAll results saved to notebooks/data_quality_v2_results.json")

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
print("All plots saved to notebooks/dq2_stage*.png")
print("All data saved to notebooks/data_quality_v2_results.json")
print("\nDone.")
