"""
Lambda training sweep with duplicate-averaging fix.

Compare ancestral_td_lambda and ancestral_mc_td_lambda at λ = 0, 0.03, 0.1
against one_step_bootstrap and off-policy baselines. All use smc=reward.
No warm-start — training from scratch.

Evaluation:
  1. Terminal reward vs training steps
  2. Value function error vs analytical V on two distributions:
     (a) off-policy (forward noising from p1)
     (b) smc_reward trajectories
"""

import json
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM Setup
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
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

# Analytical value function
class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        cc = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * cc).sum(-1)
               + 200.0 * v * (cc ** 2).sum()) / denom
            - 10.0 * (cc ** 2).sum()
        )

    def forward(self, x, t):
        x = x.double()
        t = t.double().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        m = self.means.double()
        s2 = self.sigma2.double()
        w = self.weights.double()
        dk = t_ * s2[None, :] + 2 * self.a * (1 - t_)
        mm = t_[:, :, None] * m[None, :, :]
        d2 = ((x[:, None, :] - mm) ** 2).sum(-1)
        t_safe = t_ + 1e-40
        lg = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - d2 / (2 * t_safe * dk)
        )
        lw = torch.log(w)[None, :]
        lpw = lw + lg - torch.logsumexp(lw + lg, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


# Drift & reward
def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    olw = torch.log(weights_)
    denom = 2 * a * (1 - ts) + ts * sigmas_ ** 2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a * (1 - ts) / denom) * d / 2
    lrw = olw + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), olw, lw)
    nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
    return {"log_weights": lw, "means": nm}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt, ts,
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a
    )
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(torch.exp(cond["log_weights"])[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward(x)


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("notebooks/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# Reward-tilted distribution q*(x1)
tau2 = _sigma2 / (1.0 + 20.0 * _sigma2)
nu = (_means + 20.0 * _sigma2[:, None] * c.double()[None, :]) / (1.0 + 20.0 * _sigma2[:, None])
log_Zk = (
    -(D / 2.0) * torch.log(1.0 + 20.0 * _sigma2)
    + (nu ** 2).sum(dim=1) / (2.0 * tau2)
    - (_means ** 2).sum(dim=1) / (2.0 * _sigma2)
    - 10.0 * (c.double() ** 2).sum()
)
log_w_gmm = torch.log(_weights)
log_Z_total = torch.logsumexp(log_w_gmm + log_Zk, dim=0)
w_star = torch.exp(log_w_gmm + log_Zk - log_Z_total)

print(f"E_OPT = {E_OPT:.4f}  V(0,0) = {V_0_0:.4f}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
MAX_STEPS = 6000
LOG_DIR = "lightning_logs/lambda_training"
CKPT_DIR = "checkpoints/lambda_training"

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

LAMBDA_VALUES = [0.0, 0.03, 0.1]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RUNS = []

# Off-policy baseline
RUNS.append(dict(name="offpolicy", method="offpolicy", lam=None))

# One-step bootstrap baseline
RUNS.append(dict(name="osb", method="one_step_bootstrap", lam=0.0))

# Ancestral TD(lambda) sweep
for lam in LAMBDA_VALUES:
    RUNS.append(dict(name=f"atd_lam{lam}", method="ancestral_td_lambda", lam=lam))

# Ancestral MC-TD(lambda) sweep
for lam in LAMBDA_VALUES:
    RUNS.append(dict(name=f"amctd_lam{lam}", method="ancestral_mc_td_lambda", lam=lam))


def run_training(cfg):
    run_name = cfg["name"]
    method = cfg["method"]
    lam = cfg["lam"]

    print(f"\n{'='*70}")
    print(f"  {run_name} ({method}, λ={lam})")
    print(f"{'='*70}")

    # Check if done
    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= MAX_STEPS - 1:
            print("  Already complete, skipping.")
            return

    # Clean stale
    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    if method == "offpolicy":
        ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OffPolicyValue(
            base_score_module=base_drift, reward_function=reward,
            value_module=vm, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
    else:
        ds = OnPolicySMCDataset(
            dim=D, drift=base_drift, value=vm, smc_value=smc_reward,
            reward=reward, device=DEVICE, a=a,
            batch_size=32, n_steps=100, mc_samples_per_step=10,
            sampling_method=method, lambda_eff=lam if lam is not None else 0.0,
        )
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OnPolicyValue(
            base_score_module=base_drift, value_module=vm,
            reward_function=reward, dim=D, a=a, lr=LR,
            loss_type="quad", analytical_value_fn=anal_fn,
        )

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        val_check_interval=max(1, MAX_STEPS // 60),
        callbacks=[ckpt_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed/60:.1f} min")


for cfg in RUNS:
    run_training(cfg)


# ---------------------------------------------------------------------------
# Value function evaluation
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print("  VALUE FUNCTION EVALUATION")
print(f"{'='*70}")

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_LABELS = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]
N_EVAL = 5000


def sample_p1(n, rng=None):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return torch.from_numpy(means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)).float()


def sample_qstar(n, rng=None):
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    idx = torch.multinomial(w_star.float(), n, replacement=True, generator=rng)
    eps = torch.randn(n, D, dtype=torch.float64, generator=rng)
    return (nu[idx] + torch.sqrt(tau2[idx, None]) * eps).float()


def forward_noise(x1, t):
    t_ = t.unsqueeze(-1)
    return t_ * x1 + torch.sqrt(2 * a * t_ * (1 - t_)) * torch.randn_like(x1)


def load_value_fn(ckpt_path):
    vm = ValueNetwork(D, bias=bias_val)
    dummy_drift = lambda x, t: torch.zeros_like(x)
    model = OnPolicyValue(
        base_score_module=dummy_drift, value_module=vm,
        dim=D, a=a, lr=1e-3, loss_type="quad",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.value_module


def evaluate_vf(value_fn, x1_sampler, n=N_EVAL, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator().manual_seed(seed)
    x1 = x1_sampler(n, rng=rng)
    t = torch.rand(n)
    xt = forward_noise(x1, t)
    with torch.no_grad():
        v_pred = value_fn(xt, t)
        v_anal = anal_fn(xt, t)
    err = v_pred - v_anal
    stats = {}
    for bname, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (t >= lo) & (t < hi)
        n_bin = mask.sum().item()
        if n_bin < 2:
            stats[bname] = {"mae": float("nan"), "bias": float("nan")}
            continue
        e = err[mask]
        stats[bname] = {"mae": e.abs().mean().item(), "bias": e.mean().item()}
    avg_mae = np.mean([s["mae"] for s in stats.values() if not np.isnan(s["mae"])])
    avg_bias = np.mean([s["bias"] for s in stats.values() if not np.isnan(s["bias"])])
    return stats, avg_mae, avg_bias


# Evaluate all runs
eval_results = {}
for cfg in RUNS:
    run_name = cfg["name"]
    ckpt = f"{CKPT_DIR}/{run_name}/best.ckpt"
    if not os.path.exists(ckpt):
        ckpt = f"{CKPT_DIR}/{run_name}/last.ckpt"
    if not os.path.exists(ckpt):
        print(f"  {run_name}: no checkpoint found, skipping eval")
        continue

    vf = load_value_fn(ckpt)
    p1_stats, p1_mae, p1_bias = evaluate_vf(vf, sample_p1)
    qs_stats, qs_mae, qs_bias = evaluate_vf(vf, sample_qstar)
    eval_results[run_name] = {
        "p1_mae": p1_mae, "p1_bias": p1_bias, "p1_stats": p1_stats,
        "qs_mae": qs_mae, "qs_bias": qs_bias, "qs_stats": qs_stats,
    }
    print(f"  {run_name:<20}  p1_mae={p1_mae:.4f}  q*_mae={qs_mae:.4f}  p1_bias={p1_bias:.4f}  q*_bias={qs_bias:.4f}")


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'='*70}")

print(f"\n  {'Run':<20} {'Best Rwd':>9} {'Final Rwd':>10} {'Gap':>7} {'p1 MAE':>8} {'q* MAE':>8}")
print(f"  {'-'*68}")

run_data = {}
for cfg in RUNS:
    rn = cfg["name"]
    csv = f"{LOG_DIR}/{rn}/version_0/metrics.csv"
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    er = eval_results.get(rn, {})
    p1m = er.get("p1_mae", float("nan"))
    qsm = er.get("qs_mae", float("nan"))
    print(f"  {rn:<20} {best:>9.4f} {final:>10.4f} {gap:>7.4f} {p1m:>8.4f} {qsm:>8.4f}")
    run_data[rn] = (val["step"].values, val["val_reward_mean"].values)


# ---------------------------------------------------------------------------
# Plot 1: Terminal reward vs steps
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title("Lambda Training Sweep (no warm-start, smc=reward)", fontsize=14, fontweight="bold")

cmap = plt.cm.tab10
colors = {}
for i, cfg in enumerate(RUNS):
    colors[cfg["name"]] = cmap(i / len(RUNS))
colors["offpolicy"] = "black"
colors["osb"] = "gray"

ls_map = {"offpolicy": "--", "osb": ":"}

for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    steps, rewards = run_data[rn]
    ls = ls_map.get(rn, "-")
    lw = 2.5 if rn in ("offpolicy", "osb") else 1.5
    label = rn
    ax.plot(steps, rewards, color=colors[rn], linestyle=ls, linewidth=lw, label=label)

ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"E_opt={E_OPT:.3f}")
ax.set_xlabel("Training Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/lambda_training_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/lambda_training_reward.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Value function MAE by t-bin
# ---------------------------------------------------------------------------
bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Value Function Error: |V_model - V_analytical| by t-bin", fontsize=14, fontweight="bold")

for ax_idx, (dist_name, dist_label) in enumerate([("p1", "x₁ ~ p₁ (off-policy)"),
                                                    ("qs", "x₁ ~ q* (reward-tilted)")]):
    ax = axes[ax_idx]
    ax.set_title(dist_label, fontsize=12)

    for cfg in RUNS:
        rn = cfg["name"]
        er = eval_results.get(rn)
        if er is None:
            continue
        stats = er[f"{dist_name}_stats"]
        maes = [stats[b]["mae"] for b in BIN_NAMES]
        ls = ls_map.get(rn, "-")
        lw = 2.5 if rn in ("offpolicy", "osb") else 1.5
        ax.plot(bin_centers, maes, "o-", color=colors[rn], linestyle=ls,
                linewidth=lw, markersize=5, label=rn)

    ax.set_xlabel("t")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/lambda_training_mae.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/lambda_training_mae.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Summary bar chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Summary: Terminal Reward & Value Function Error", fontsize=14, fontweight="bold")

run_names = [cfg["name"] for cfg in RUNS if cfg["name"] in run_data]
x_pos = np.arange(len(run_names))
bar_colors = [colors[rn] for rn in run_names]

# Best reward
ax = axes[0]
best_rwds = [run_data[rn][1].max() for rn in run_names]
ax.bar(x_pos, best_rwds, color=bar_colors)
ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Best Terminal Reward")
ax.set_title("Best Reward")
ax.grid(True, alpha=0.3, axis="y")

# p1 MAE
ax = axes[1]
p1_maes = [eval_results.get(rn, {}).get("p1_mae", 0) for rn in run_names]
ax.bar(x_pos, p1_maes, color=bar_colors)
ax.set_xticks(x_pos)
ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Avg MAE (p₁)")
ax.set_title("Value Error on p₁")
ax.grid(True, alpha=0.3, axis="y")

# q* MAE
ax = axes[2]
qs_maes = [eval_results.get(rn, {}).get("qs_mae", 0) for rn in run_names]
ax.bar(x_pos, qs_maes, color=bar_colors)
ax.set_xticks(x_pos)
ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Avg MAE (q*)")
ax.set_title("Value Error on q*")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("notebooks/lambda_training_summary.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/lambda_training_summary.png")
plt.close()

# Save JSON
out = {
    "E_OPT": E_OPT,
    "runs": {},
}
for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    steps, rewards = run_data[rn]
    er = eval_results.get(rn, {})
    out["runs"][rn] = {
        "method": cfg["method"],
        "lambda": cfg["lam"],
        "best_reward": float(rewards.max()),
        "final_reward": float(rewards[-1]),
        "p1_mae": er.get("p1_mae"),
        "qs_mae": er.get("qs_mae"),
        "p1_bias": er.get("p1_bias"),
        "qs_bias": er.get("qs_bias"),
    }
with open("notebooks/lambda_training_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: notebooks/lambda_training_results.json")

print(f"\nDone. E_OPT = {E_OPT:.4f}")
