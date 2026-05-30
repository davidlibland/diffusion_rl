"""
FBRRT hyperparameter sweep.

Structured one-at-a-time sweep from a default config, plus baselines.
Each axis is swept while other params are at default.

Default: FBRRT(λ=0), n_steps=100, EMA=0.999, grad_decay=None,
         branch=4, ws=3000, LR=1e-3

Sweep axes:
  - branch: 2, 4, 8, 16
  - lambda: 0, λ_s=0.1, λ_s=0.5
  - n_steps: 30, 100
  - EMA decay: None, 0.99, 0.999, 0.9999
  - grad_decay: None, 1e-8, 1e-6
  - LR: 3e-4, 1e-3, 3e-3
  - warm-start: 0, 1000, 3000

Baselines: off-policy, osb_blend (ws=3000), atd0 (ws=3000)
"""

import gc
import json
import os
import shutil
import time
from functools import partial
from math import sqrt

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
        d = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(d)
            + (-10 * (m**2).sum(-1) + 20 * (m * cc).sum(-1) + 200 * v * (cc**2).sum())
            / d
            - 10 * (cc**2).sum()
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
        ts = t_ + 1e-40
        lg = -self.D / 2.0 * torch.log(2 * torch.pi * ts * dk) - d2 / (2 * ts * dk)
        lw = torch.log(w)[None, :]
        lpw = lw + lg - torch.logsumexp(lw + lg, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_anal_vm = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm(x.cpu(), t.cpu()).to(x.device)


def get_cond_mix(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    olw = torch.log(weights_)
    denom = 2 * a * (1 - ts) + ts * sigmas_**2
    le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
    lsf = torch.log(2 * a * (1 - ts) / denom) * d / 2
    lrw = olw + le + lsf
    lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
    lw = torch.where((ts == 0), olw, lw)
    nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
    return {"log_weights": lw, "means": nm}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_cond_mix(
        xt, ts, _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a
    )
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(torch.exp(cond["log_weights"])[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(
    x, t if t.ndim >= 1 else t.unsqueeze(0), a
).to(dtype=torch.float)
_reward_c = c.clone()


def reward_fn(x):
    return -10 * (x - _reward_c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward_fn(x)


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("experiments/common/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
print(f"E_OPT={E_OPT:.4f}")

LOADER_BATCH_SIZE = 256
TOTAL_STEPS = 6000
LOG_DIR = "lightning_logs/fbrrt_sweep"
CKPT_DIR = "checkpoints/fbrrt_sweep"
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ---------------------------------------------------------------------------
# Trajectory eval callback
# ---------------------------------------------------------------------------
class TrajCB(Callback):
    def __init__(self, anal_fn, n_traj=256, n_steps=100):
        super().__init__()
        self.anal_fn = anal_fn
        self.n_traj = n_traj
        self.n_steps = n_steps

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx > 0:
            return
        dim = pl_module.hparams.dim
        device = pl_module.device
        n = self.n_traj
        dt = 1.0 / self.n_steps
        for beta, label in [(0, "base"), (1, "guided")]:
            x = torch.zeros(n, dim, device=device)
            all_x, all_t = [x], [torch.zeros(n, device=device)]
            drift_fn = partial(pl_module.drift, beta=beta)
            for st in torch.linspace(0, 1, self.n_steps + 1, device=device)[:-1]:
                tv = st.expand(n)
                dx = drift_fn(x, tv) * dt
                db = sqrt(2 * pl_module.a * dt) * torch.randn_like(x)
                x = x + dx + db
                all_x.append(x)
                all_t.append(torch.full((n,), float(st) + dt, device=device))
            all_x = torch.cat(all_x)
            all_t = torch.cat(all_t)
            with torch.no_grad():
                vp = pl_module.value_module(all_x, all_t)
                va = self.anal_fn(all_x, all_t)
            err = vp - va
            pl_module.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)
            pl_module.log(f"traj_avg_bias_{label}", err.mean(), prog_bar=False)


# ---------------------------------------------------------------------------
# Generic training function
# ---------------------------------------------------------------------------
def run_config(name, method, lr, ws_steps, n_steps, mc_samples, lambda_eff,
               ema_decay, grad_decay, branch=4, entropy_lambda=1.0):
    """Run a single training configuration.

    method: 'offpolicy', 'osb_blend', 'atd', 'fbrrt', 'fbrrt_td_lambda'
    """
    on_steps = TOTAL_STEPS - ws_steps
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  method={method} lr={lr} ws={ws_steps} n_steps={n_steps} mc={mc_samples}")
    print(f"  λ_eff={lambda_eff} ema={ema_decay} grad_decay={grad_decay} branch={branch} ent_λ={entropy_lambda}")
    print(f"{'='*70}")

    csv_path = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if method in ("offpolicy",):
        csv_check = csv_path
    elif ws_steps > 0:
        csv_check = f"{LOG_DIR}/{name}/version_1/metrics.csv"
    else:
        csv_check = csv_path

    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        expected = TOTAL_STEPS if method == "offpolicy" else on_steps
        if len(val) > 0 and val["step"].max() >= expected - 1:
            print("  Already complete, skipping.")
            return

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    # --- Off-policy baseline ---
    if method == "offpolicy":
        ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OffPolicyValue(
            base_score_module=base_drift, reward_function=reward_fn,
            value_module=vm, dim=D, a=a, lr=lr,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        traj_cb = TrajCB(anal_fn)
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 60),
            callbacks=[traj_cb], logger=logger,
            enable_checkpointing=False, enable_progress_bar=True,
        )
        t0 = time.perf_counter()
        trainer.fit(model, loader, val_dataloaders=val_loader)
        print(f"  Elapsed: {(time.perf_counter()-t0)/60:.1f} min")
        del trainer, model, vm, ds, loader
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return

    # --- Phase 1: off-policy warm-start ---
    if ws_steps > 0:
        print(f"  Phase 1: Off-policy for {ws_steps} steps...")
        ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
        loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
        model = OffPolicyValue(
            base_score_module=base_drift, reward_function=reward_fn,
            value_module=vm, dim=D, a=a, lr=lr,
            loss_type="quad", analytical_value_fn=anal_fn,
        )
        traj_cb = TrajCB(anal_fn)
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=ws_steps, val_check_interval=max(1, ws_steps // 30),
            callbacks=[traj_cb], logger=logger,
            enable_checkpointing=False, enable_progress_bar=True,
        )
        t0 = time.perf_counter()
        trainer.fit(model, loader, val_dataloaders=val_loader)
        print(f"  Phase 1 done: {(time.perf_counter()-t0)/60:.1f} min")
        del trainer, model, ds, loader
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # --- Phase 2: on-policy ---
    ema_kw = {"ema_decay": ema_decay} if ema_decay is not None else {"ema_decay": 0.99}
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward_fn, dim=D, a=a, lr=lr,
        loss_type="quad", analytical_value_fn=anal_fn,
        grad_decay=grad_decay, **ema_kw,
    )

    # Determine sampling method and value fn
    if method == "osb_blend":
        sampling_method = "one_step_bootstrap"
        smc_fn = smc_reward
        value_for_ds = vm
    elif method == "atd":
        sampling_method = "ancestral_td_lambda"
        smc_fn = smc_reward
        value_for_ds = model.ema if ema_decay is not None else vm
    elif method in ("fbrrt", "fbrrt_td_lambda"):
        sampling_method = method
        smc_fn = smc_reward
        value_for_ds = model.ema if ema_decay is not None else vm
    else:
        raise ValueError(f"Unknown method: {method}")

    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=value_for_ds, smc_value=smc_fn,
        reward=reward_fn, device=DEVICE, a=a,
        batch_size=32, n_steps=n_steps, mc_samples_per_step=mc_samples,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
        branch=branch, entropy_lambda=entropy_lambda,
    )

    if method == "osb_blend":
        ds.raw_value_fn = ds.value_fn  # monkey-patch blend

    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    traj_cb = TrajCB(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    version = 1 if ws_steps > 0 else 0
    logger = CSVLogger(LOG_DIR, name=name, version=version)
    trainer = L.Trainer(
        max_steps=on_steps, val_check_interval=max(1, on_steps // 60),
        callbacks=[ckpt_cb, traj_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )
    print(f"  Phase 2: {method} for {on_steps} steps...")
    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Phase 2 done: {(time.perf_counter()-t0)/60:.1f} min")
    del trainer, model, vm, ds, loader
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Define all configurations
# ---------------------------------------------------------------------------
# Default FBRRT config
DEF = dict(method="fbrrt", lr=1e-3, ws_steps=3000, n_steps=100,
           mc_samples=10, lambda_eff=0.0, ema_decay=0.999,
           grad_decay=None, branch=4, entropy_lambda=1.0)

CONFIGS = []

# Baselines
CONFIGS.append(dict(name="offpolicy", method="offpolicy", lr=1e-3, ws_steps=6000,
                    n_steps=100, mc_samples=10, lambda_eff=0.0, ema_decay=None,
                    grad_decay=None, branch=4, entropy_lambda=1.0))
CONFIGS.append(dict(name="osb_blend_ws3k", method="osb_blend", lr=1e-3, ws_steps=3000,
                    n_steps=100, mc_samples=10, lambda_eff=0.0, ema_decay=None,
                    grad_decay=None, branch=4, entropy_lambda=1.0))
CONFIGS.append(dict(name="atd0_ema999_ws3k", method="atd", lr=1e-3, ws_steps=3000,
                    n_steps=100, mc_samples=10, lambda_eff=0.0, ema_decay=0.999,
                    grad_decay=None, branch=4, entropy_lambda=1.0))

# Default FBRRT
CONFIGS.append(dict(name="fbrrt_default", **DEF))

# Branch sweep (fix everything else at default)
for br in [2, 8, 16]:
    CONFIGS.append(dict(name=f"fbrrt_br{br}", **{**DEF, "branch": br}))

# Lambda sweep
for lam_s, lam_eff in [(0.1, 0.1**100), (0.5, 0.5**100)]:
    CONFIGS.append(dict(name=f"fbrrt_td_lams{lam_s}",
                        **{**DEF, "method": "fbrrt_td_lambda", "lambda_eff": lam_eff}))

# n_steps sweep
CONFIGS.append(dict(name="fbrrt_ns30", **{**DEF, "n_steps": 30}))

# EMA sweep
for ema in [None, 0.99, 0.9999]:
    label = str(ema).replace(".", "").replace("None", "none")
    CONFIGS.append(dict(name=f"fbrrt_ema{label}", **{**DEF, "ema_decay": ema}))

# grad_decay sweep
for gd in [1e-8, 1e-6]:
    CONFIGS.append(dict(name=f"fbrrt_gd{gd:.0e}", **{**DEF, "grad_decay": gd}))

# LR sweep
for lr in [3e-4, 3e-3]:
    CONFIGS.append(dict(name=f"fbrrt_lr{lr:.0e}", **{**DEF, "lr": lr}))

# Warm-start sweep
for ws in [0, 1000]:
    CONFIGS.append(dict(name=f"fbrrt_ws{ws}", **{**DEF, "ws_steps": ws}))

# entropy_lambda sweep (higher values stabilize exploration)
for el in [2.0, 5.0, 10.0]:
    CONFIGS.append(dict(name=f"fbrrt_ent{el:.0f}", **{**DEF, "entropy_lambda": el}))

print(f"\nTotal configs: {len(CONFIGS)}")
for c in CONFIGS:
    print(f"  {c['name']}")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
for cfg in CONFIGS:
    name = cfg.pop("name")
    try:
        run_config(name, **cfg)
    except Exception as e:
        print(f"  ERROR: {e}")
    cfg["name"] = name  # restore for results


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT={E_OPT:.4f})")
print(f"{'='*70}")


def load_metrics(name, ws_steps, method):
    dfs = []
    csv0 = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if os.path.exists(csv0):
        dfs.append(pd.read_csv(csv0))
    if method != "offpolicy" and ws_steps > 0:
        csv1 = f"{LOG_DIR}/{name}/version_1/metrics.csv"
        if os.path.exists(csv1):
            d = pd.read_csv(csv1).copy()
            d["step"] = d["step"] + ws_steps
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else None


print(f"\n  {'Name':<25} {'Best':>8} {'Final':>8} {'Gap':>7} {'B-MAE':>7} {'G-MAE':>7} {'Stable':>7}")
print(f"  {'-'*73}")

run_data = {}
for cfg in CONFIGS:
    name = cfg["name"]
    df = load_metrics(name, cfg["ws_steps"], cfg["method"])
    if df is None:
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    bm = df.dropna(subset=["traj_avg_mae_base"])
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    bf = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    stable = "YES" if abs(final - best) < 5 else "no"
    print(f"  {name:<25} {best:>8.3f} {final:>8.3f} {gap:>7.3f} {bf:>7.3f} {gf:>7.3f} {stable:>7}")
    run_data[name] = df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Plot 1: Terminal reward for all runs
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("FBRRT Sweep: Terminal Reward vs Steps", fontsize=14, fontweight="bold")
cmap = plt.cm.tab20
for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    if name not in run_data:
        continue
    df = run_data[name]
    val = df.dropna(subset=["val_reward_mean"])
    color = "black" if "offpolicy" in name else ("red" if "osb" in name else cmap(i / len(CONFIGS)))
    ls = "--" if "offpolicy" in name else (":" if "osb" in name else "-")
    lw = 2.5 if name in ("offpolicy", "osb_blend_ws3k") else 1.0
    ax.plot(val["step"].values, val["val_reward_mean"].values,
            color=color, linestyle=ls, linewidth=lw, label=name, alpha=0.8)
ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.5, label=f"E_opt={E_OPT:.3f}")
ax.set_xlabel("Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=5, loc="lower right", ncol=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_reward.png", dpi=150, bbox_inches="tight")
print("\nSaved: experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_reward.png")
plt.close()

# Plot 2: Guided MAE for all runs
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("FBRRT Sweep: Guided Trajectory MAE vs Steps", fontsize=14, fontweight="bold")
for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    if name not in run_data:
        continue
    df = run_data[name]
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) == 0:
        continue
    color = "black" if "offpolicy" in name else ("red" if "osb" in name else cmap(i / len(CONFIGS)))
    ls = "--" if "offpolicy" in name else (":" if "osb" in name else "-")
    lw = 2.5 if name in ("offpolicy", "osb_blend_ws3k") else 1.0
    ax.plot(sub["step"].values, sub["traj_avg_mae_guided"].values,
            color=color, linestyle=ls, linewidth=lw, label=name, alpha=0.8)
ax.set_xlabel("Steps")
ax.set_ylabel("Avg MAE (guided)")
ax.legend(fontsize=5, loc="upper right", ncol=3)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_mae.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_mae.png")
plt.close()

# Plot 3: Summary bar chart — best reward by config
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_title("FBRRT Sweep: Best Terminal Reward by Config", fontsize=14, fontweight="bold")
names = [c["name"] for c in CONFIGS if c["name"] in run_data]
bests = [run_data[n].dropna(subset=["val_reward_mean"])["val_reward_mean"].max() for n in names]
colors = ["black" if "offpolicy" in n else ("red" if "osb" in n else ("blue" if "atd" in n else "green"))
          for n in names]
ax.barh(range(len(names)), bests, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.axvline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
ax.set_xlabel("Best Terminal Reward")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_summary.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_summary.png")
plt.close()

# Save JSON
out = {"E_OPT": E_OPT, "configs": [], "results": {}}
for cfg in CONFIGS:
    out["configs"].append(cfg)
    name = cfg["name"]
    if name in run_data:
        df = run_data[name]
        val = df.dropna(subset=["val_reward_mean"])
        bm = df.dropna(subset=["traj_avg_mae_base"])
        gm = df.dropna(subset=["traj_avg_mae_guided"])
        out["results"][name] = {
            "best_reward": float(val["val_reward_mean"].max()),
            "final_reward": float(val["val_reward_mean"].iloc[-1]),
            "final_base_mae": float(bm["traj_avg_mae_base"].iloc[-1]) if len(bm) > 0 else None,
            "final_guided_mae": float(gm["traj_avg_mae_guided"].iloc[-1]) if len(gm) > 0 else None,
        }
with open("experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: experiments/misc/2026-05-06_fbrrt_sweeps/fbrrt_sweep_results.json")

print(f"\nDone. E_OPT={E_OPT:.4f}")
