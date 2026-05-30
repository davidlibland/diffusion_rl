"""
Final warm-start experiment: ATD(λ=0) + smc=reward + raw V.

Configurations:
  - off-policy baseline (6000 steps)
  - ATD(λ=0) + smc=reward + raw V, warm-start durations: 0, 500, 1000, 2000, 3000
  - OSB + smc=reward + blended V (old best), warm-start 2000 (previous best config)

Metrics (all tracked vs training steps):
  1. Average terminal reward
  2. Value function error along base-drift trajectories (beta=0)
  3. Value function error along guided trajectories (beta=1)
"""

import json
import os
import shutil
import time
from functools import partial
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
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
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10 * (m**2).sum(-1) + 20 * (m * cc).sum(-1) + 200 * v * (cc**2).sum())
            / denom
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


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
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
    nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[
        :, None, :
    ]
    return {"log_weights": lw, "means": nm}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt,
        ts,
        _means.float().to(xt),
        _sigmas.float().to(xt),
        _weights_col.float().to(xt),
        a,
    )
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(
        torch.exp(cond["log_weights"])[:, None, :] * us, "n d m -> n d", "sum"
    )


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(
    dtype=torch.float
)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward(x)


def gmm_sample(n):
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)


with open("experiments/common/analytical_target.json") as f:
    _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()
print(f"E_OPT = {E_OPT:.4f}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
TOTAL_STEPS = 6000
LOG_DIR = "lightning_logs/warmstart_final"
CKPT_DIR = "checkpoints/warmstart_final"

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


# ---------------------------------------------------------------------------
# Custom validation callback: evaluate V error along trajectories
# ---------------------------------------------------------------------------
class TrajectoryEvalCallback(Callback):
    """At each validation step, sample trajectories with beta=0 and beta=1,
    evaluate V_model vs V_analytical along them, and log per-bin MAE."""

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
            for step_t in torch.linspace(0, 1, self.n_steps + 1, device=device)[:-1]:
                t_vec = step_t.expand(n)
                dx = drift_fn(x, t_vec) * dt
                db = sqrt(2 * pl_module.a * dt) * torch.randn_like(x)
                x = x + dx + db
                all_x.append(x)
                all_t.append(torch.full((n,), float(step_t) + dt, device=device))

            all_x = torch.cat(all_x, dim=0)
            all_t = torch.cat(all_t, dim=0)

            with torch.no_grad():
                v_pred = pl_module.value_module(all_x, all_t)
                v_anal = self.anal_fn(all_x, all_t)
            err = v_pred - v_anal

            # Per-bin MAE
            for bname, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:]):
                mask = (all_t >= lo) & (all_t < hi)
                if mask.sum() > 0:
                    pl_module.log(
                        f"traj_mae_{label}_{bname}",
                        err[mask].abs().mean(),
                        prog_bar=False,
                    )
                    pl_module.log(
                        f"traj_bias_{label}_{bname}", err[mask].mean(), prog_bar=False
                    )

            # Average across bins
            avg_mae = err.abs().mean()
            avg_bias = err.mean()
            pl_module.log(f"traj_avg_mae_{label}", avg_mae, prog_bar=False)
            pl_module.log(f"traj_avg_bias_{label}", avg_bias, prog_bar=False)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_offpolicy(vm, max_steps, run_name, version=0):
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(
        base_score_module=base_drift,
        reward_function=reward,
        value_module=vm,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )
    traj_cb = TrajectoryEvalCallback(anal_fn)
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[traj_cb],
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_onpolicy_atd(vm, max_steps, run_name, version=1):
    """ATD(λ=0) + smc=reward + raw V."""
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_reward,
        reward=reward,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method="ancestral_td_lambda",
        lambda_eff=0.0,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )
    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb, traj_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_onpolicy_osb_blend(vm, max_steps, run_name, version=1):
    """OSB + smc=reward + blended V (the old best config)."""
    # The old OSB used value_fn (blend) — we replicate this by using
    # one_step_bootstrap which now uses raw_value_fn, but we override
    # the dataset to use the old blend wrapper.
    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_reward,
        reward=reward,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method="one_step_bootstrap",
        lambda_eff=0.0,
    )

    # Patch by setting the raw_value_fn to value_fn before iterating
    ds.raw_value_fn = ds.value_fn

    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OnPolicyValue(
        base_score_module=base_drift,
        value_module=vm,
        reward_function=reward,
        dim=D,
        a=a,
        lr=LR,
        loss_type="quad",
        analytical_value_fn=anal_fn,
    )
    traj_cb = TrajectoryEvalCallback(anal_fn)
    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=max(1, max_steps // 60),
        callbacks=[ckpt_cb, traj_cb],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------
WARMSTART_DURATIONS = [0, 500, 1000, 2000, 3000]

RUNS = []

# Off-policy baseline
RUNS.append(dict(name="offpolicy", ws=TOTAL_STEPS, method="offpolicy"))

# ATD(λ=0) + smc=reward + raw V at various warm-start durations
for ws in WARMSTART_DURATIONS:
    RUNS.append(dict(name=f"atd0_ws{ws}", ws=ws, method="atd"))

# OSB + blended V + smc=reward, warm-start 2000 (previous best)
RUNS.append(dict(name="osb_blend_ws2000", ws=2000, method="osb_blend"))


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
for cfg in RUNS:
    run_name = cfg["name"]
    ws = cfg["ws"]
    method = cfg["method"]
    on_steps = TOTAL_STEPS - ws if method != "offpolicy" else 0

    print(f"\n{'=' * 70}")
    if method == "offpolicy":
        print(f"  {run_name}: {TOTAL_STEPS} off-policy steps")
    else:
        print(f"  {run_name}: {ws} off → {on_steps} on ({method})")
    print(f"{'=' * 70}")

    # Check completion
    if method == "offpolicy":
        csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    else:
        csv_check = (
            f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
            if ws > 0
            else f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        )

    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        expected = TOTAL_STEPS if method == "offpolicy" else on_steps
        if len(val) > 0 and val["step"].max() >= expected - 1:
            print("  Already complete, skipping.")
            continue

    # Clean stale
    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p):
            shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    if method == "offpolicy":
        t0 = time.perf_counter()
        train_offpolicy(vm, TOTAL_STEPS, run_name)
        print(f"  Elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
        continue

    # Phase 1: off-policy warm-start
    if ws > 0:
        print(f"  Phase 1: Off-policy for {ws} steps...")
        t0 = time.perf_counter()
        train_offpolicy(vm, ws, run_name, version=0)
        print(f"  Phase 1 done: {(time.perf_counter() - t0) / 60:.1f} min")

    # Phase 2: on-policy
    print(f"  Phase 2: On-policy ({method}) for {on_steps} steps...")
    t0 = time.perf_counter()
    version = 1 if ws > 0 else 0
    if method == "atd":
        train_onpolicy_atd(vm, on_steps, run_name, version=version)
    elif method == "osb_blend":
        train_onpolicy_osb_blend(vm, on_steps, run_name, version=version)
    print(f"  Phase 2 done: {(time.perf_counter() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Load and plot results
# ---------------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print(f"  RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'=' * 70}")


def load_combined(run_name, ws, method):
    dfs = []
    if method == "offpolicy":
        csv = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        if os.path.exists(csv):
            dfs.append(pd.read_csv(csv))
    else:
        csv0 = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
        if ws > 0 and os.path.exists(csv0):
            dfs.append(pd.read_csv(csv0))
        v = 1 if ws > 0 else 0
        csv1 = f"{LOG_DIR}/{run_name}/version_{v}/metrics.csv"
        if os.path.exists(csv1):
            df1 = pd.read_csv(csv1).copy()
            df1["step"] = df1["step"] + ws
            dfs.append(df1)
    return pd.concat(dfs, ignore_index=True) if dfs else None


# Summary table
print(f"\n  {'Run':<25} {'Best Rwd':>9} {'Final Rwd':>10} {'Gap':>7}")
print(f"  {'-' * 55}")

run_data = {}
for cfg in RUNS:
    rn = cfg["name"]
    df = load_combined(rn, cfg["ws"], cfg["method"])
    if df is None:
        continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0:
        continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    print(f"  {rn:<25} {best:>9.4f} {final:>10.4f} {gap:>7.4f}")
    run_data[rn] = df


# ---------------------------------------------------------------------------
# Plot 1: Terminal reward vs steps
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title(
    "Warm-Start Sweep: Terminal Reward vs Training Steps\n(ATD λ=0, smc=reward, raw V)",
    fontsize=13,
    fontweight="bold",
)

cmap = plt.cm.viridis
colors = {"offpolicy": "black", "osb_blend_ws2000": "red"}
for i, ws in enumerate(WARMSTART_DURATIONS):
    colors[f"atd0_ws{ws}"] = cmap(i / max(1, len(WARMSTART_DURATIONS) - 1))

for cfg in RUNS:
    rn = cfg["name"]
    if rn not in run_data:
        continue
    df = run_data[rn]
    val = df.dropna(subset=["val_reward_mean"])
    steps, rewards = val["step"].values, val["val_reward_mean"].values

    ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
    lw = 2.5 if rn in ("offpolicy", "osb_blend_ws2000") else 1.5
    label = (
        rn.replace("atd0_ws", "ATD ws=")
        .replace("offpolicy", "off-policy")
        .replace("osb_blend_ws2000", "OSB+blend ws=2000 (prev best)")
    )
    ax.plot(steps, rewards, color=colors[rn], linestyle=ls, linewidth=lw, label=label)

    if cfg["method"] != "offpolicy" and cfg["ws"] > 0:
        ax.axvline(cfg["ws"], color=colors[rn], linestyle=":", alpha=0.2, linewidth=0.8)

ax.axhline(
    E_OPT,
    color="red",
    linestyle=":",
    linewidth=1.5,
    alpha=0.5,
    label=f"E_opt={E_OPT:.3f}",
)
ax.set_xlabel("Training Steps")
ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_warmstart_final/warmstart_final_reward.png", dpi=150, bbox_inches="tight")
print("\nSaved: experiments/misc/2026-05-06_warmstart_final/warmstart_final_reward.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2: V error along trajectories (base vs guided)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Value Function Error Along Trajectories", fontsize=14, fontweight="bold")

for ax_idx, (metric, title) in enumerate(
    [
        ("traj_avg_mae_base", "MAE on base-drift trajectories (β=0)"),
        ("traj_avg_mae_guided", "MAE on guided trajectories (β=1)"),
    ]
):
    ax = axes[ax_idx]
    ax.set_title(title, fontsize=11)

    for cfg in RUNS:
        rn = cfg["name"]
        if rn not in run_data:
            continue
        df = run_data[rn]
        sub = df.dropna(subset=[metric])
        if len(sub) == 0:
            continue
        steps = sub["step"].values
        vals = sub[metric].values

        ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
        lw = 2.5 if rn in ("offpolicy", "osb_blend_ws2000") else 1.5
        label = (
            rn.replace("atd0_ws", "ATD ws=")
            .replace("offpolicy", "off-policy")
            .replace("osb_blend_ws2000", "OSB+blend ws=2000")
        )
        ax.plot(steps, vals, color=colors[rn], linestyle=ls, linewidth=lw, label=label)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Avg MAE")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_warmstart_final/warmstart_final_mae.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/misc/2026-05-06_warmstart_final/warmstart_final_mae.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Per-bin V error at end of training
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Final V Error by t-bin (end of training)", fontsize=14, fontweight="bold")
bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

for ax_idx, (prefix, title) in enumerate(
    [
        ("traj_mae_base", "Base drift (β=0)"),
        ("traj_mae_guided", "Guided (β=1)"),
    ]
):
    ax = axes[ax_idx]
    ax.set_title(title, fontsize=11)

    for cfg in RUNS:
        rn = cfg["name"]
        if rn not in run_data:
            continue
        df = run_data[rn]
        cols = [f"{prefix}_{b}" for b in BIN_NAMES]
        avail = [c for c in cols if c in df.columns]
        if not avail:
            continue
        sub = df.dropna(subset=avail[:1])
        if len(sub) == 0:
            continue
        last = sub.iloc[-1]
        vals = [last.get(c, np.nan) for c in cols]

        ls = "--" if rn == "offpolicy" else (":" if "osb" in rn else "-")
        lw = 2.5 if rn in ("offpolicy", "osb_blend_ws2000") else 1.5
        label = (
            rn.replace("atd0_ws", "ATD ws=")
            .replace("offpolicy", "off-policy")
            .replace("osb_blend_ws2000", "OSB+blend")
        )
        ax.plot(
            bin_centers,
            vals,
            "o-",
            color=colors[rn],
            linestyle=ls,
            linewidth=lw,
            markersize=4,
            label=label,
        )

    ax.set_xlabel("t")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_warmstart_final/warmstart_final_bins.png", dpi=150, bbox_inches="tight")
print("Saved: experiments/misc/2026-05-06_warmstart_final/warmstart_final_bins.png")
plt.close()

print(f"\nDone. E_OPT = {E_OPT:.4f}")
