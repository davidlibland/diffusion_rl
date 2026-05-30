"""
Mixed off-policy/on-policy training with proper memory cleanup.

Uses OnPolicySMCDataset.off_policy_frac with FracScheduleCallback.
Runs each config as a separate function with explicit cleanup between runs.

All runs use LR=1e-3, EMA decay=0.999, warm-start 3000 steps.
"""

import gc
import json
import os
import shutil
import sys
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

# GMM Setup
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
            + (-10 * (m**2).sum(-1) + 20 * (m * cc).sum(-1) + 200 * v * (cc**2).sum()) / d
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
            s2[None, :, None] * x[:, None, :] + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        return torch.logsumexp(lpw + self._log_Z(tmu, tV), dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)
def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


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
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward_fn = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
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

LR = 1e-3
EMA_DECAY = 0.999
LOADER_BATCH_SIZE = 256
WS_STEPS = 3000
ON_STEPS = 3000
TOTAL_STEPS = WS_STEPS + ON_STEPS
LOG_DIR = "lightning_logs/mixed_final"
CKPT_DIR = "checkpoints/mixed_final"
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class TrajectoryEvalCallback(Callback):
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


class FracScheduleCallback(Callback):
    def __init__(self, dataset, switch_step, new_frac):
        super().__init__()
        self.dataset = dataset
        self.switch_step = switch_step
        self.new_frac = new_frac
        self.switched = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.switched and trainer.global_step >= self.switch_step:
            self.dataset.off_policy_frac = self.new_frac
            self.switched = True
            print(f"  [step {trainer.global_step}] off_policy_frac → {self.new_frac}")


def cleanup():
    """Aggressive cleanup between runs."""
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()


def run_config(run_name, on_frac):
    """Run a single config: ws with frac=1.0 then on-policy with on_frac."""
    print(f"\n{'='*70}")
    print(f"  {run_name}: LR={LR} EMA={EMA_DECAY} frac={on_frac}")
    print(f"{'='*70}")

    csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
            print("  Already complete.")
            return

    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward_fn, dim=D, a=a, lr=LR,
        loss_type="quad", analytical_value_fn=anal_fn,
        ema_decay=EMA_DECAY,
    )

    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.ema, smc_value=smc_reward,
        reward=reward_fn, device=DEVICE, a=a,
        batch_size=4, n_steps=100, mc_samples_per_step=10,
        sampling_method="ancestral_td_lambda", lambda_eff=0.0,
        off_policy_frac=1.0,
        generating_function=gmm_sample,
        off_policy_batch_size=1024,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    frac_cb = FracScheduleCallback(ds, switch_step=WS_STEPS, new_frac=on_frac)
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)

    trainer = L.Trainer(
        max_steps=TOTAL_STEPS,
        val_check_interval=max(1, TOTAL_STEPS // 60),
        callbacks=[ckpt_cb, traj_cb, frac_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Elapsed: {(time.perf_counter()-t0)/60:.1f} min")

    # Explicit cleanup
    del trainer, model, vm, ds, loader, logger
    cleanup()


def run_offpolicy():
    rn = "offpolicy"
    print(f"\n{'='*70}\n  {rn}\n{'='*70}")
    csv_check = f"{LOG_DIR}/{rn}/version_0/metrics.csv"
    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
            print("  Already complete.")
            return
    for v in range(3):
        p = f"{LOG_DIR}/{rn}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)
    vm = ValueNetwork(D, bias=bias_val)
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(
        base_score_module=base_drift, reward_function=reward_fn,
        value_module=vm, dim=D, a=a, lr=LR,
        loss_type="quad", analytical_value_fn=anal_fn,
    )
    traj_cb = TrajectoryEvalCallback(anal_fn)
    logger = CSVLogger(LOG_DIR, name=rn, version=0)
    trainer = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 60),
        callbacks=[traj_cb], logger=logger,
        enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    del trainer, model, vm, ds, loader, logger
    cleanup()


def run_osb_blend():
    rn = "osb_blend"
    print(f"\n{'='*70}\n  {rn}\n{'='*70}")
    csv_check = f"{LOG_DIR}/{rn}/version_0/metrics.csv"
    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= TOTAL_STEPS - 1:
            print("  Already complete.")
            return
    for v in range(3):
        p = f"{LOG_DIR}/{rn}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)
    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward_fn, dim=D, a=a, lr=LR,
        loss_type="quad", analytical_value_fn=anal_fn,
    )
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=vm, smc_value=smc_reward,
        reward=reward_fn, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="one_step_bootstrap", lambda_eff=0.0,
        off_policy_frac=1.0, generating_function=gmm_sample,
    )
    ds.raw_value_fn = ds.value_fn
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{rn}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    )
    frac_cb = FracScheduleCallback(ds, switch_step=WS_STEPS, new_frac=0.0)
    logger = CSVLogger(LOG_DIR, name=rn, version=0)
    trainer = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 60),
        callbacks=[ckpt_cb, traj_cb, frac_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    del trainer, model, vm, ds, loader, logger
    cleanup()


# ---------------------------------------------------------------------------
# Execute each run with cleanup between
# ---------------------------------------------------------------------------
run_offpolicy()
run_osb_blend()
run_config("frac0", on_frac=0.0)
cleanup()
run_config("frac25", on_frac=0.25)
cleanup()
run_config("frac50", on_frac=0.5)
cleanup()
run_config("frac75", on_frac=0.75)
cleanup()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT={E_OPT:.4f}, LR={LR})")
print(f"{'='*70}")

RUNS = ["offpolicy", "osb_blend", "frac0", "frac25", "frac50", "frac75"]

print(f"\n  {'Run':<15} {'Best':>8} {'Final':>8} {'Gap':>7} {'B-MAE':>7} {'G-MAE':>7}")
print(f"  {'-'*56}")
run_data = {}
for rn in RUNS:
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
    bm = df.dropna(subset=["traj_avg_mae_base"])
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    bf = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    print(f"  {rn:<15} {best:>8.3f} {final:>8.3f} {gap:>7.3f} {bf:>7.3f} {gf:>7.3f}")
    run_data[rn] = df

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title(f"Mixed Training (LR={LR}, EMA={EMA_DECAY}): Reward vs Steps", fontsize=13, fontweight="bold")
cmap = plt.cm.tab10
colors = {rn: cmap(i / len(RUNS)) for i, rn in enumerate(RUNS)}
colors["offpolicy"] = "black"
colors["osb_blend"] = "red"
for rn in RUNS:
    if rn not in run_data:
        continue
    df = run_data[rn]
    val = df.dropna(subset=["val_reward_mean"])
    ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
    lw = 2.5 if rn in ("offpolicy", "osb_blend") else 1.5
    ax.plot(val["step"].values, val["val_reward_mean"].values,
            color=colors[rn], linestyle=ls, linewidth=lw, label=rn)
ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.5, label=f"E_opt={E_OPT:.3f}")
ax.axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
ax.set_xlabel("Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_mixed_training/mixed_final_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: experiments/misc/2026-05-06_mixed_training/mixed_final_reward.png")
plt.close()

print(f"\nDone.")
