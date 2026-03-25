"""
Two-part experiment:

Part 1: Lambda sweep for ancestral_mc_td_lambda training stability.
  Geometric distribution of lambdas: 0, 0.01, 0.05, 0.1, 0.3
  All warm-started from 2000 off-policy steps, then 4000 on-policy.
  Uses smc=reward (best from prior experiments).

Part 2: Investigate logZ ratios vs exp(V) contributions to targets.
  For anc_mctd, target ≈ A + B where:
    A = log_mean_exp(V_last_generation) (bootstrap/reward at leaves)
    B = sum of logZ_ratios along ancestry (resampling corrections)
  Investigate: which term dominates? Which is the variance source?
"""

import json
import os
import shutil
import time
import copy
from math import ceil, sqrt, log

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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import (
    OnPolicySMCDataset, OnPolicyValue,
    _sde_step, _resample, _log_mean_exp_by_ancestor, _log_td_blend,
)
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
D = 2; a = 1.0; c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_


class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None: c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a; self.D = D; self.register_buffer("c", c.float())
    def _log_Z(self, m, v):
        cc = self.c.double(); denom = 1.0 + 20.0 * v
        return (-self.D/2.0*torch.log(denom) + (-10*(m**2).sum(-1)+20*(m*cc).sum(-1)+200*v*(cc**2).sum())/denom - 10*(cc**2).sum())
    def forward(self, x, t):
        x = x.double(); t = t.double().reshape(-1)
        if t.numel() == 1: t = t.expand(x.shape[0])
        t_ = t[:, None]; m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
        dk = t_*s2[None,:]+2*self.a*(1-t_); mm = t_[:,:,None]*m[None,:,:]
        d2 = ((x[:,None,:]-mm)**2).sum(-1); lg = (-self.D/2.0*torch.log(2*torch.pi*(t_+1e-40)*dk)-d2/(2*(t_+1e-40)*dk))
        lw = torch.log(w)[None,:]; lpw = lw+lg-torch.logsumexp(lw+lg,dim=1,keepdim=True)
        tV = 2*self.a*(1-t_)*s2[None,:]/dk
        tmu = (s2[None,:,None]*x[:,None,:]+2*self.a*(1-t_)[:,:,None]*m[None,:,:])/dk[:,:,None]
        return torch.logsumexp(lpw+self._log_Z(tmu,tV),dim=1).float()

_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)
def anal_fn(x, t): return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)

def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape; xt_ = xt[...,None]; means_ = means.T[None,...]; ts_ = ts[...,None]
    sigmas_ = sigmas.T; weights_ = weights.T; olw = torch.log(weights_)
    denom = 2*a*(1-ts)+ts*sigmas_**2
    le = -reduce((xt_-means_*ts_)**2,"n d m -> n m","sum")/(2*ts*denom)
    lsf = torch.log(2*a*(1-ts)/denom)*d/2
    lrw = olw+le+lsf; lw = lrw-torch.logsumexp(lrw,dim=1,keepdim=True)
    lw = torch.where((ts==0),olw,lw)
    nm = (2*a*(1-ts_)*means_+xt_*sigmas_[None,...]**2)/denom[:,None,:]
    return {"log_weights": lw, "means": nm}

def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1,1)
    cond = get_conditional_mixture(xt,ts,_means.float().to(xt),_sigmas.float().to(xt),_weights_col.float().to(xt),a)
    us = (cond["means"]-xt[:,:,None])/(1-ts[...,None])
    return reduce(torch.exp(cond["log_weights"])[:,None,:]*us,"n d m -> n d","sum")

DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim>=1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10*(x-c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward(x)
def gmm_sample(n):
    k = np.random.choice(len(weights_np),size=n,p=weights_np)
    return means_np[k]+sigmas_np[k,np.newaxis]*np.random.randn(n,D)

with open("notebooks/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards-max_r)))+max_r).item()
print(f"E_OPT = {E_OPT:.4f}")

# ---------------------------------------------------------------------------
# Part 2: Investigate logZ ratios in ancestral_mc_td_lambda targets
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("  PART 2: logZ ratio analysis")
print("="*70)


def decompose_anc_mctd_targets(
    drift, value, log_tau, h, a, lambda_eff,
    batch_size, mc_samples, dim, n_steps, device, dtype=torch.float32,
):
    """
    Run ancestral_mc_td_lambda forward+backward, but return decomposed targets:
      - A_term: log_mean_exp of leaf R values (reward/bootstrap contribution)
      - B_term: cumulative sum of logZ ratios along ancestry
      - logZ_per_step: per-step logZ values
      - full_target: the actual target (should equal A + B + log_tau approximately)
    """
    lam = lambda_eff ** (1.0 / n_steps) if lambda_eff > 0 else 0.0
    dt = 1.0 / n_steps
    N = mc_samples
    BN = batch_size * N

    def flat(z): return z.reshape(BN, dim)
    def tvec(t_scalar): return torch.full((BN, 1), t_scalar, dtype=dtype, device=device)

    # Forward pass
    x = torch.zeros(batch_size, N, dim, dtype=dtype, device=device)
    log_tau_x = log_tau(flat(x), tvec(0.0)).reshape(batch_size, N, 1)

    fwd_x_post = []
    fwd_log_v = []
    fwd_log_w = []
    fwd_ix = []
    fwd_ts = []
    fwd_logZ = []  # per-step logZ values

    for step_idx, _t in enumerate(torch.linspace(0, 1, n_steps + 1, dtype=dtype)[:-1]):
        t_curr = float(_t); t_next = t_curr + dt
        x_flat = flat(x)
        x_next_flat = _sde_step(x_flat, drift, a, t_curr, dt, batch_size, N, dim, device)
        x_next = x_next_flat.reshape(batch_size, N, dim)
        log_tau_next = log_tau(x_next_flat, tvec(t_next)).reshape(batch_size, N, 1)
        log_w = log_tau_next - log_tau_x

        is_terminal = step_idx == n_steps - 1
        if is_terminal and h is not None:
            log_v = h(x_next_flat).reshape(batch_size, N, 1)
        else:
            log_v = value(x_next_flat, tvec(t_next)).reshape(batch_size, N, 1)

        fwd_log_w.append(log_w)
        fwd_log_v.append(log_v)

        x_post, log_tau_x, ix = _resample(log_w, x_next, log_tau_next, batch_size, N, dim)
        fwd_x_post.append(x_post.clone())
        fwd_ix.append(ix)
        fwd_ts.append(t_next)
        x = x_post

    # Backward pass - decompose targets
    x_final = fwd_x_post[-1]
    log_tau_final = log_tau(flat(x_final), tvec(fwd_ts[-1])).reshape(batch_size, N, 1)
    log_h_final = h(flat(x_final)).reshape(batch_size, N, 1)
    R = log_h_final - log_tau_final

    # For the final generation: A = h(x), B = 0, target = h(x)
    step_logZ_values = []  # (gen, B, N, 1) logZ at each backward step
    step_A_values = []     # leaf/bootstrap contribution
    step_B_cumulative = [] # cumulative logZ sum

    # Track per-particle cumulative logZ
    cumB = torch.zeros(batch_size, N, 1, dtype=dtype, device=device)

    all_A = []
    all_B = []
    all_logZ = []
    all_t = []
    all_tgt = []

    # Final step: pure reward, no logZ
    all_A.append(log_h_final.reshape(BN))
    all_B.append(torch.zeros(BN, dtype=dtype, device=device))
    all_logZ.append(torch.zeros(BN, dtype=dtype, device=device))
    all_t.append(torch.full((BN,), fwd_ts[-1], dtype=dtype, device=device))
    all_tgt.append((R + log_tau_final).reshape(BN))

    for gen in range(n_steps - 1, 0, -1):
        ix_gen = fwd_ix[gen]
        log_w_gen = fwd_log_w[gen]
        x_post_prev = fwd_x_post[gen - 1]

        log_tau_pp = log_tau(flat(x_post_prev), tvec(fwd_ts[gen-1])).reshape(batch_size, N, 1)
        V = fwd_log_v[gen]
        O = V - log_tau_pp

        log_Z, has_children = _log_mean_exp_by_ancestor(log_w_gen, ix_gen)
        log_mean_R, _ = _log_mean_exp_by_ancestor(R, ix_gen)
        M = log_Z + log_mean_R

        R = torch.where(has_children > 0, _log_td_blend(O, M, lam), O)
        target = (R + log_tau_pp).reshape(BN)

        # Decomposition for particles WITH children:
        # M = log_Z + log_mean_R
        # At the leaf, R starts as h - log_tau. As we recurse back,
        # the logZ terms accumulate.
        # For a pure MC (lam=1) particle chain of length L:
        #   target ≈ logZ_1 + logZ_2 + ... + logZ_L + h(x_leaf)
        # A = contribution from leaf values (h or V bootstrap)
        # B = sum of logZ ratios

        # Approximate: for particles with children, decompose M
        safe_logZ = torch.where(has_children > 0, log_Z, torch.zeros_like(log_Z))

        all_logZ.append(safe_logZ.reshape(BN))
        all_A.append(O.reshape(BN))  # one-step bootstrap (V value)
        all_B.append(safe_logZ.reshape(BN))  # this step's logZ contribution
        all_t.append(torch.full((BN,), fwd_ts[gen-1], dtype=dtype, device=device))
        all_tgt.append(target)

    # Reverse to chronological
    all_A = all_A[::-1]
    all_B = all_B[::-1]
    all_logZ = all_logZ[::-1]
    all_t = all_t[::-1]
    all_tgt = all_tgt[::-1]

    return {
        "A": torch.cat(all_A),      # one-step bootstrap / leaf values
        "logZ": torch.cat(all_logZ), # per-step logZ ratios
        "t": torch.cat(all_t),
        "target": torch.cat(all_tgt),
    }


LR = 3e-3

# Load the best off-policy warm-started model for analysis
vm_for_analysis = ValueNetwork(D, bias=bias_val)
# Train briefly off-policy to get a reasonable model
print("  Training analysis model (2000 off-policy steps)...")
ds_off = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
loader_off = DataLoader(ds_off, batch_size=256)
model_off = OffPolicyValue(
    base_score_module=base_drift, reward_function=reward, value_module=vm_for_analysis,
    dim=D, a=a, lr=LR, loss_type="quad",
)
trainer_off = L.Trainer(max_steps=2000, enable_checkpointing=False,
                        enable_progress_bar=True, logger=False)
trainer_off.fit(model_off, loader_off,
                val_dataloaders=DataLoader(TensorDataset(torch.zeros(1)), batch_size=1))
print("  Analysis model trained.")

# Now decompose targets for different lambda values
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_LABELS = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]

analysis_lambdas = [0.0, 0.01, 0.05, 0.1, 0.3, 0.8, 1.0]
decomp_results = {}

for lam_eff in analysis_lambdas:
    print(f"\n  Decomposing targets for lambda_eff={lam_eff}...")
    torch.manual_seed(42)
    with torch.no_grad():
        res = decompose_anc_mctd_targets(
            drift=base_drift, value=vm_for_analysis, log_tau=smc_reward, h=reward,
            a=a, lambda_eff=lam_eff,
            batch_size=32, mc_samples=10, dim=D, n_steps=100, device="cpu",
        )

    t = res["t"]
    A = res["A"]
    logZ = res["logZ"]
    target = res["target"]

    # Compute analytical value for comparison
    # (need x for this, but we don't return x from decompose - use target stats instead)

    print(f"    Overall: |A| mean={A.abs().mean():.4f} std={A.std():.4f}  "
          f"|logZ| mean={logZ.abs().mean():.4f} std={logZ.std():.4f}  "
          f"|target| mean={target.abs().mean():.4f} std={target.std():.4f}")

    stats = {}
    for bname, blabel, lo, hi in zip(BIN_NAMES, BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (t >= lo) & (t < hi)
        n = mask.sum().item()
        if n < 2:
            stats[bname] = {"n": 0}
            continue
        stats[bname] = {
            "n": n,
            "A_mean": A[mask].mean().item(),
            "A_std": A[mask].std().item(),
            "A_abs_mean": A[mask].abs().mean().item(),
            "logZ_mean": logZ[mask].mean().item(),
            "logZ_std": logZ[mask].std().item(),
            "logZ_abs_mean": logZ[mask].abs().mean().item(),
            "target_mean": target[mask].mean().item(),
            "target_std": target[mask].std().item(),
            "ratio_logZ_over_A": logZ[mask].abs().mean().item() / max(A[mask].abs().mean().item(), 1e-10),
        }
        print(f"    {blabel}: A_std={stats[bname]['A_std']:.4f}  logZ_std={stats[bname]['logZ_std']:.4f}  "
              f"|logZ|/|A|={stats[bname]['ratio_logZ_over_A']:.4f}  target_std={stats[bname]['target_std']:.4f}")

    decomp_results[lam_eff] = stats


# Plot decomposition
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Target Decomposition: A (bootstrap) vs logZ (resampling correction)", fontsize=14, fontweight="bold")

bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
cmap = plt.cm.viridis
lam_colors = {l: cmap(i / (len(analysis_lambdas)-1)) for i, l in enumerate(analysis_lambdas)}

# Plot 1: |A| std by t-bin
ax = axes[0, 0]
ax.set_title("Std of A (bootstrap/value term)")
for lam_eff in analysis_lambdas:
    vals = [decomp_results[lam_eff].get(b, {}).get("A_std", np.nan) for b in BIN_NAMES]
    ax.plot(bin_centers, vals, "o-", color=lam_colors[lam_eff], label=f"λ={lam_eff}")
ax.set_xlabel("t"); ax.set_ylabel("Std(A)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Plot 2: logZ std by t-bin
ax = axes[0, 1]
ax.set_title("Std of logZ (resampling correction)")
for lam_eff in analysis_lambdas:
    vals = [decomp_results[lam_eff].get(b, {}).get("logZ_std", np.nan) for b in BIN_NAMES]
    ax.plot(bin_centers, vals, "o-", color=lam_colors[lam_eff], label=f"λ={lam_eff}")
ax.set_xlabel("t"); ax.set_ylabel("Std(logZ)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Plot 3: |logZ|/|A| ratio by t-bin
ax = axes[1, 0]
ax.set_title("|logZ| / |A| ratio (>1 means logZ dominates)")
for lam_eff in analysis_lambdas:
    vals = [decomp_results[lam_eff].get(b, {}).get("ratio_logZ_over_A", np.nan) for b in BIN_NAMES]
    ax.plot(bin_centers, vals, "o-", color=lam_colors[lam_eff], label=f"λ={lam_eff}")
ax.axhline(1.0, color="red", linestyle=":", alpha=0.5)
ax.set_xlabel("t"); ax.set_ylabel("|logZ|/|A|"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Plot 4: target std by t-bin
ax = axes[1, 1]
ax.set_title("Std of full target")
for lam_eff in analysis_lambdas:
    vals = [decomp_results[lam_eff].get(b, {}).get("target_std", np.nan) for b in BIN_NAMES]
    ax.plot(bin_centers, vals, "o-", color=lam_colors[lam_eff], label=f"λ={lam_eff}")
ax.set_xlabel("t"); ax.set_ylabel("Std(target)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/logz_decomposition.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/logz_decomposition.png")
plt.close()


# ---------------------------------------------------------------------------
# Part 1: Lambda sweep training
# ---------------------------------------------------------------------------
print("\n\n" + "="*70)
print("  PART 1: Lambda sweep training")
print("="*70)

LR = 3e-3
LOADER_BATCH_SIZE = 256
WS_STEPS = 2000
ON_STEPS = 4000
TOTAL_STEPS = WS_STEPS + ON_STEPS
LOG_DIR = "lightning_logs/lambda_sweep_mctd"
CKPT_DIR = "checkpoints/lambda_sweep_mctd"

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

SWEEP_LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.3]


def train_offpolicy(vm, max_steps, run_name):
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(
        base_score_module=base_drift, reward_function=reward, value_module=vm,
        dim=D, a=a, lr=LR, loss_type="quad", analytical_value_fn=anal_fn,
    )
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)
    trainer = L.Trainer(
        max_steps=max_steps, val_check_interval=max(1, max_steps//60),
        logger=logger, enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


def train_onpolicy_mctd(vm, lambda_eff, max_steps, run_name, version=1):
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=vm, smc_value=smc_reward,
        reward=reward, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="ancestral_mc_td_lambda", lambda_eff=lambda_eff,
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
    logger = CSVLogger(LOG_DIR, name=run_name, version=version)
    trainer = L.Trainer(
        max_steps=max_steps, val_check_interval=max(1, max_steps//60),
        callbacks=[ckpt_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    return vm


# Off-policy baseline
run_name = "pure_offpolicy"
csv_check = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
if not os.path.exists(csv_check):
    print(f"\n  {run_name}: {TOTAL_STEPS} off-policy steps")
    vm = ValueNetwork(D, bias=bias_val)
    train_offpolicy(vm, TOTAL_STEPS, run_name)

# Lambda sweep
for lam_eff in SWEEP_LAMBDAS:
    run_name = f"mctd_lam{lam_eff}"
    print(f"\n{'='*70}")
    print(f"  {run_name}: {WS_STEPS} off → {ON_STEPS} on (anc_mctd, λ={lam_eff}, smc=reward)")
    print(f"{'='*70}")

    csv_on = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
    if os.path.exists(csv_on):
        df = pd.read_csv(csv_on)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= ON_STEPS - 1:
            print("  Already complete, skipping.")
            continue

    for v in range(3):
        p = f"{LOG_DIR}/{run_name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)
    print(f"  Phase 1: Off-policy for {WS_STEPS} steps...")
    t0 = time.perf_counter()
    train_offpolicy(vm, WS_STEPS, run_name)
    print(f"  Phase 1 done: {(time.perf_counter()-t0)/60:.1f} min")

    print(f"  Phase 2: On-policy (anc_mctd, λ={lam_eff}) for {ON_STEPS} steps...")
    t0 = time.perf_counter()
    train_onpolicy_mctd(vm, lam_eff, ON_STEPS, run_name, version=1)
    print(f"  Phase 2 done: {(time.perf_counter()-t0)/60:.1f} min")


# Results
print(f"\n\n{'='*70}")
print(f"  TRAINING RESULTS (E_OPT = {E_OPT:.4f})")
print(f"{'='*70}")

def load_combined(run_name, ws, has_on):
    dfs = []
    csv0 = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv0): dfs.append(pd.read_csv(csv0))
    if has_on:
        csv1 = f"{LOG_DIR}/{run_name}/version_1/metrics.csv"
        if os.path.exists(csv1):
            d = pd.read_csv(csv1).copy(); d["step"] = d["step"] + ws; dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else None

all_cfgs = [("pure_offpolicy", TOTAL_STEPS, False)] + [(f"mctd_lam{l}", WS_STEPS, True) for l in SWEEP_LAMBDAS]

print(f"\n  {'Run':<25} {'Best Reward':>12} {'Final Reward':>13} {'Gap':>8}")
print(f"  {'-'*62}")

run_data = {}
for rn, ws, has_on in all_cfgs:
    df = load_combined(rn, ws, has_on)
    if df is None: continue
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max()
    final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    print(f"  {rn:<25} {best:>12.4f} {final:>13.4f} {gap:>8.4f}")
    run_data[rn] = (val["step"].values, val["val_reward_mean"].values, ws)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_title("Lambda Sweep: anc_mctd + smc=reward (warm-started)", fontsize=14, fontweight="bold")

colors_sweep = {f"mctd_lam{l}": cmap(i/(len(SWEEP_LAMBDAS)-1)) for i, l in enumerate(SWEEP_LAMBDAS)}
colors_sweep["pure_offpolicy"] = "black"

for rn, ws, has_on in all_cfgs:
    if rn not in run_data: continue
    steps, rewards, wsl = run_data[rn]
    ls = "--" if "pure" in rn else "-"
    lw = 2.5 if "pure" in rn else 1.5
    ax.plot(steps, rewards, color=colors_sweep.get(rn, "gray"), linestyle=ls, linewidth=lw, label=rn)
    if has_on: ax.axvline(wsl, color=colors_sweep.get(rn, "gray"), linestyle=":", alpha=0.3, linewidth=0.8)

ax.axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"E_opt={E_OPT:.3f}")
ax.set_xlabel("Training Steps"); ax.set_ylabel("Avg Terminal Reward")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/lambda_sweep_mctd_reward.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/lambda_sweep_mctd_reward.png")
plt.close()

print("\nDone.")
