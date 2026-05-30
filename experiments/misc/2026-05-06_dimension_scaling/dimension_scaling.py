"""
Dimension scaling experiment: off-policy vs FBRRT mixed as d increases.

Setup for each dimension d:
  - Base distribution: random GMM with K=20 components, random means/scales
  - Reward: r(x) = -10 * ||x - c||² where c is a random point
  - Analytical drift and value available (same structure as moons)
  - Network: ValueNetwork with hidden_dim scaled to min(256, 64*d)

Dimensions: 2, 5, 10, 20, 50
Training: 1500 steps per run (quick, ~2 min each)
Methods: off-policy, FBRRT mixed 50% (ws=500 then mix)

Total: 5 dims × 2 methods = 10 runs, ~20-25 min.
"""

import gc, json, os, shutil, time
from functools import partial
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

DEVICE = "mps"
TOTAL_STEPS = 1500
WS_STEPS = 500
LOADER_BATCH_SIZE = 256
LR = 1e-3
LOG_DIR = "lightning_logs/dim_scaling"
CKPT_DIR = "checkpoints/dim_scaling"
DIMS = [50, 20, 10]
a = 1.0

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


def make_problem(d, seed=42):
    """Create a random GMM + quadratic reward problem in d dimensions.

    Returns: drift_fn, reward_fn, gmm_sample_fn, anal_value_fn, E_OPT, bias_val
    """
    rng = np.random.RandomState(seed)
    K = 20  # number of components

    # Random GMM: means in [-2, 2]^d, spherical variance in [0.01, 0.5]
    means = rng.uniform(-2, 2, size=(K, d)).astype(np.float64)
    sigma2 = rng.uniform(0.01, 0.5, size=K).astype(np.float64)
    weights = rng.dirichlet(np.ones(K)).astype(np.float64)

    # Reward center: random point near origin
    c_np = rng.uniform(-1, 1, size=d).astype(np.float64)
    reward_scale = 10.0 / d  # scale with dimension to keep reward magnitude bounded

    means_t = torch.from_numpy(means).double()
    sigma2_t = torch.from_numpy(sigma2).double()
    weights_t = torch.from_numpy(weights).double()
    c_t = torch.from_numpy(c_np).double()

    # --- Analytical value function (same math as moons) ---
    class AnalyticalValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("means", means_t.float())
            self.register_buffer("sigma2", sigma2_t.float())
            self.register_buffer("weights", weights_t.float())
            self.register_buffer("c", c_t.float())
            self.a = a; self.D = d; self.rs = reward_scale

        def _log_Z(self, m, v):
            cc = self.c.double(); rs2 = 2 * self.rs
            denom = 1.0 + rs2 * v
            return (
                -self.D / 2.0 * torch.log(denom)
                + (-self.rs * (m**2).sum(-1) + rs2 * (m * cc).sum(-1)
                   + rs2 * self.rs * v * (cc**2).sum()) / denom
                - self.rs * (cc**2).sum()
            )

        def forward(self, x, t):
            x = x.double(); t = t.double().reshape(-1)
            if t.numel() == 1: t = t.expand(x.shape[0])
            t_ = t[:, None]
            m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
            dk = t_ * s2[None, :] + 2 * self.a * (1 - t_)
            marg_mean = t_[:, :, None] * m[None, :, :]
            diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
            t_safe = t_ + 1e-40
            log_gauss = (-self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
                         - diff2 / (2 * t_safe * dk))
            log_w = torch.log(w)[None, :]
            log_pw = log_w + log_gauss
            log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
            tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
            tmu = (s2[None, :, None] * x[:, None, :]
                   + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]) / dk[:, :, None]
            log_zk = self._log_Z(tmu, tV)
            return torch.logsumexp(log_pw + log_zk, dim=1).float()

    anal_model = AnalyticalValue()
    def anal_fn(x, t):
        return anal_model(x.cpu(), t.cpu()).to(x.device)

    # --- GMM drift (analytical) ---
    sigmas_col = torch.sqrt(sigma2_t)[:, None]
    weights_col = weights_t[:, None]

    def gmm_drift(xt, ts):
        from einops import reduce
        xt = xt.float(); ts = ts.reshape(-1, 1).float()
        m = means_t.float().to(xt); sig = sigmas_col.float().to(xt); wc = weights_col.float().to(xt)
        xt_ = xt[..., None]; means_ = m.T[None, ...]; ts_ = ts[..., None]
        sigmas_ = sig.T; weights_ = wc.T; olw = torch.log(weights_)
        denom = 2*a*(1-ts) + ts*sigmas_**2
        le = -reduce((xt_ - means_*ts_)**2, "n d m -> n m", "sum") / (2*ts*denom)
        lsf = torch.log(2*a*(1-ts)/denom) * d/2
        lrw = olw + le + lsf
        lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
        lw = torch.where((ts == 0), olw, lw)
        nm = (2*a*(1-ts_)*means_ + xt_*sigmas_[None,...]**2) / denom[:, None, :]
        us = (nm - xt[:, :, None]) / (1 - ts[..., None])
        return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")

    drift_fn = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0)).to(dtype=torch.float)

    # --- Reward ---
    c_float = c_t.float()
    def reward_fn(x):
        return -reward_scale * (x - c_float.to(x)).square().sum(dim=1)

    def smc_reward(x, t):
        return reward_fn(x)

    # --- GMM sampler ---
    def gmm_sample(n):
        k = rng.choice(K, size=n, p=weights)
        return (means[k] + np.sqrt(sigma2[k, np.newaxis]) * rng.randn(n, d)).astype(np.float32)

    # --- Bias value ---
    x_samples = torch.from_numpy(gmm_sample(10000)).float()
    all_r = reward_fn(x_samples)
    max_r = all_r.max()
    bias_val = (torch.log(torch.mean(torch.exp(all_r - max_r))) + max_r).item()

    # E_OPT
    x0 = torch.zeros(1, d)
    t0 = torch.zeros(1)
    V00 = anal_fn(x0, t0).item()

    return drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn, bias_val, V00


class TrajCB(Callback):
    def __init__(self, anal_fn, d, n_traj=128, n_steps=100):
        super().__init__()
        self.anal_fn = anal_fn; self.d = d; self.n = n_traj; self.ns = n_steps

    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi > 0: return
        dev = pl.device; n = self.n; dt = 1.0 / self.ns
        for beta, label in [(1, "guided")]:
            x = torch.zeros(n, self.d, device=dev)
            all_x, all_t = [x], [torch.zeros(n, device=dev)]
            dfn = partial(pl.drift, beta=beta)
            for st in torch.linspace(0, 1, self.ns+1, device=dev)[:-1]:
                tv = st.expand(n)
                dx = dfn(x, tv) * dt
                db = sqrt(2 * pl.a * dt) * torch.randn_like(x)
                x = x + dx + db
                all_x.append(x)
                all_t.append(torch.full((n,), float(st)+dt, device=dev))
            all_x = torch.cat(all_x); all_t = torch.cat(all_t)
            with torch.no_grad():
                vp = pl.value_module(all_x, all_t)
                va = self.anal_fn(all_x, all_t)
            err = vp - va
            pl.log(f"traj_avg_mae_{label}", err.abs().mean(), prog_bar=False)


class FracCB(Callback):
    def __init__(self, ds, ws, frac):
        super().__init__(); self.ds=ds; self.ws=ws; self.frac=frac; self.done=False
    def on_train_batch_start(self, trainer, pl, batch, bi):
        if not self.done and trainer.global_step >= self.ws:
            self.ds.off_policy_frac = self.frac; self.done = True


# --- Run experiment ---
results = {}

for d in DIMS:
    print(f"\n{'#'*70}")
    print(f"  DIMENSION d={d}")
    print(f"{'#'*70}")

    drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn, bias_val, V00 = make_problem(d)
    hidden_dim = min(256, max(64, 32 * d))
    print(f"  V(0,0)={V00:.3f}, bias={bias_val:.3f}, hidden_dim={hidden_dim}")

    for method in ["offpolicy", "fbrrt_mixed"]:
        name = f"d{d}_{method}"
        print(f"\n  --- {name} ---")

        for v in range(3):
            p = f"{LOG_DIR}/{name}/version_{v}"
            if os.path.exists(p): shutil.rmtree(p)

        vm = ValueNetwork(d, hidden_dim=hidden_dim, bias=bias_val)

        if method == "offpolicy":
            ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
            loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
            model = OffPolicyValue(
                base_score_module=drift_fn, reward_function=reward_fn,
                value_module=vm, dim=d, a=a, lr=LR,
                loss_type="quad", analytical_value_fn=anal_fn,
            )
            traj_cb = TrajCB(anal_fn, d)
            logger = CSVLogger(LOG_DIR, name=name, version=0)
            trainer = L.Trainer(
                max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS//20),
                callbacks=[traj_cb], logger=logger,
                enable_checkpointing=False, enable_progress_bar=True,
            )
            t0 = time.perf_counter()
            trainer.fit(model, loader, val_dataloaders=val_loader)
            elapsed = time.perf_counter() - t0

        else:  # fbrrt_mixed
            model = OnPolicyValue(
                base_score_module=drift_fn, value_module=vm,
                reward_function=reward_fn, dim=d, a=a, lr=LR,
                loss_type="quad", analytical_value_fn=anal_fn,
                ema_decay=0.999, grad_decay=1e-6,
            )
            ds = OnPolicySMCDataset(
                dim=d, drift=drift_fn, value=model.ema, smc_value=smc_reward,
                reward=reward_fn, device=DEVICE, a=a,
                batch_size=32, n_steps=100, mc_samples_per_step=10,
                sampling_method="fbrrt_td_lambda", lambda_eff=0.1**100,
                branch=4, entropy_lambda=2.0, fbrrt_alpha=1.0,
                off_policy_frac=1.0,
                generating_function=gmm_sample,
            )
            loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
            traj_cb = TrajCB(anal_fn, d)
            frac_cb = FracCB(ds, WS_STEPS, 0.5)
            logger = CSVLogger(LOG_DIR, name=name, version=0)
            trainer = L.Trainer(
                max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS//20),
                callbacks=[traj_cb, frac_cb], logger=logger,
                enable_checkpointing=False, enable_progress_bar=True,
            )
            t0 = time.perf_counter()
            trainer.fit(model, loader, val_dataloaders=val_loader)
            elapsed = time.perf_counter() - t0

        csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            val = df.dropna(subset=["val_reward_mean"])
            gm = df.dropna(subset=["traj_avg_mae_guided"])
            final_rwd = val["val_reward_mean"].iloc[-1] if len(val) > 0 else float("nan")
            best_rwd = val["val_reward_mean"].max() if len(val) > 0 else float("nan")
            final_mae = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
            results[(d, method)] = dict(best=best_rwd, final=final_rwd, mae=final_mae, elapsed=elapsed)
            print(f"  best={best_rwd:.3f} final={final_rwd:.3f} mae={final_mae:.3f} ({elapsed/60:.1f} min)")

        del trainer, model, vm, ds, loader
        gc.collect()
        if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


# --- Summary ---
print(f"\n\n{'='*70}")
print(f"  DIMENSION SCALING RESULTS")
print(f"{'='*70}")

print(f"\n  {'d':>3} {'Off-pol best':>13} {'FBRRT best':>11} {'Off-pol MAE':>12} {'FBRRT MAE':>10} {'Ratio best':>11}")
print(f"  {'-'*63}")
for d in DIMS:
    off = results.get((d, "offpolicy"), {})
    on = results.get((d, "fbrrt_mixed"), {})
    ob = off.get("best", float("nan")); fb = on.get("best", float("nan"))
    om = off.get("mae", float("nan")); fm = on.get("mae", float("nan"))
    ratio = fb / ob if ob != 0 and not (np.isnan(ob) or np.isnan(fb)) else float("nan")
    print(f"  {d:>3} {ob:>13.3f} {fb:>11.3f} {om:>12.3f} {fm:>10.3f} {ratio:>11.3f}")


# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Dimension Scaling: Off-Policy vs FBRRT Mixed", fontsize=14, fontweight="bold")

# Gather data
off_bests = [results.get((d, "offpolicy"), {}).get("best", np.nan) for d in DIMS]
on_bests = [results.get((d, "fbrrt_mixed"), {}).get("best", np.nan) for d in DIMS]
off_maes = [results.get((d, "offpolicy"), {}).get("mae", np.nan) for d in DIMS]
on_maes = [results.get((d, "fbrrt_mixed"), {}).get("mae", np.nan) for d in DIMS]

axes[0].plot(DIMS, off_bests, "ko--", lw=2, markersize=8, label="Off-Policy")
axes[0].plot(DIMS, on_bests, "bs-", lw=2, markersize=8, label="FBRRT Mixed")
axes[0].set_xlabel("Dimension d"); axes[0].set_ylabel("Best Terminal Reward")
axes[0].set_title("Best Reward vs Dimension"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(DIMS, off_maes, "ko--", lw=2, markersize=8, label="Off-Policy")
axes[1].plot(DIMS, on_maes, "bs-", lw=2, markersize=8, label="FBRRT Mixed")
axes[1].set_xlabel("Dimension d"); axes[1].set_ylabel("Guided MAE")
axes[1].set_title("V Error vs Dimension"); axes[1].legend()
axes[1].grid(True, alpha=0.3); axes[1].set_yscale("log")

# Ratio plot
ratios = [on_bests[i] / off_bests[i] if off_bests[i] != 0 else np.nan for i in range(len(DIMS))]
axes[2].plot(DIMS, ratios, "gs-", lw=2, markersize=8)
axes[2].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
axes[2].set_xlabel("Dimension d"); axes[2].set_ylabel("FBRRT / Off-Policy (reward ratio)")
axes[2].set_title("Relative Performance"); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/misc/2026-05-06_dimension_scaling/dimension_scaling.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: experiments/misc/2026-05-06_dimension_scaling/dimension_scaling.png")
plt.close()
print("Done.")
