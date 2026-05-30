"""
Lambda sweep for quad on-policy td-lambda methods.

lambda_eff is the weight of the terminal reward (t=1) in the TD(lambda) target.
  lambda_eff = 0  →  pure one-step bootstrap
  lambda_eff = 1  →  pure MC

Methods swept: single_seed_td_lambda, ancestral_td_lambda (both use lambda_eff)
Loss: quad only (best from prior sweep)

Phase 1 values: [0.01, 0.05, 0.15, 0.35, 0.6, 0.9]
Phase 2: refine around best region from phase 1

Results logged to lightning_logs/moons_lambda_sweep/
"""

import glob
import time

import numpy as np
import pandas as pd
import torch
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# Dataset / GMM (copied from moons_sweep.py)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)


def moons_generator(batch_size):
    X_, _ = make_moons(n_samples=batch_size, noise=0.05, random_state=42)
    return scalar.transform(X_)


clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means = torch.from_numpy(clf.means_)
sigmas = torch.sqrt(torch.from_numpy(clf.covariances_))[:, None]
weights = torch.from_numpy(clf.weights_)[:, None]


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
    cond = get_conditional_mixture(xt, ts, means.to(xt), sigmas.to(xt), weights.to(xt), a)
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


a = 1
DIM = 2
DEVICE = "mps"
BATCH_SIZE = 256

reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)
all_rewards = reward(torch.from_numpy(X).float())
max_reward = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

# ---------------------------------------------------------------------------
# Calibrate MAX_STEPS against single_seed_td_lambda (the slowest method here)
# ---------------------------------------------------------------------------
MAX_MINUTES = 10
CALIBRATION_STEPS = 20

print("Calibrating MAX_STEPS against single_seed_td_lambda...")
_cv = ValueNetwork(DIM, bias=bias)
_calib_dataset = OnPolicySMCDataset(
    dim=DIM, drift=base_drift, value=_cv, smc_value=_cv, reward=reward,
    device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_td_lambda", lambda_eff=0.1,
)
_calib_loader = DataLoader(_calib_dataset, batch_size=BATCH_SIZE)
_calib_model = OnPolicyValue(base_score_module=base_drift, value_module=_cv, dim=DIM, a=a, lr=1e-2)
_calib_trainer = L.Trainer(max_steps=CALIBRATION_STEPS, enable_checkpointing=False,
                            enable_progress_bar=False, logger=False)
t0 = time.perf_counter()
_calib_trainer.fit(_calib_model, _calib_loader)
elapsed = time.perf_counter() - t0

steps_per_second = CALIBRATION_STEPS / elapsed
MAX_STEPS = int(steps_per_second * MAX_MINUTES * 60)
VAL_STEPS = max(1, MAX_STEPS // 3)
print(f"  {CALIBRATION_STEPS} steps in {elapsed:.1f}s → {steps_per_second:.2f} steps/s")
print(f"  MAX_STEPS={MAX_STEPS}  VAL_STEPS={VAL_STEPS}")


# ---------------------------------------------------------------------------
# Run a single config and return its final val_reward_mean
# ---------------------------------------------------------------------------
def run_config(sampling_method, lambda_eff, run_name, log_dir):
    value_module = ValueNetwork(DIM, bias=bias)
    train_loader = DataLoader(
        OnPolicySMCDataset(
            dim=DIM, drift=base_drift, value=value_module, smc_value=value_module,
            reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
            sampling_method=sampling_method, lambda_eff=lambda_eff,
        ),
        batch_size=BATCH_SIZE,
    )
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=value_module,
        reward_function=reward, dim=DIM, a=a, lr=1e-2,
        loss_type="quad", grad_decay=1e-8,
    )
    logger = CSVLogger(log_dir, name=run_name)
    trainer = L.Trainer(
        max_steps=MAX_STEPS, val_check_interval=VAL_STEPS,
        logger=logger, enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)


def load_results(log_dir):
    all_paths = glob.glob(f"{log_dir}/*/version_*/metrics.csv")
    latest = {}
    for path in all_paths:
        parts = path.split("/")
        run = parts[-3]
        version = int(parts[-2].split("_")[1])
        if run not in latest or version > latest[run][0]:
            latest[run] = (version, path)
    rows = []
    for run, (_, path) in sorted(latest.items()):
        df = pd.read_csv(path)
        val_rows = df.dropna(subset=["val_reward_mean"])
        if val_rows.empty:
            continue
        last = val_rows.iloc[-1]
        rows.append({
            "run": run,
            "val_reward_mean": last["val_reward_mean"],
            "val_reward_std": last["val_reward_std"],
            "val_reward_max": last["val_reward_max"],
        })
    return pd.DataFrame(rows).sort_values("val_reward_mean", ascending=False)


# ---------------------------------------------------------------------------
# Phase 1: initial sweep
# ---------------------------------------------------------------------------
PHASE1_LAMBDAS = [0.01, 0.05, 0.15, 0.35, 0.6, 0.9]
METHODS = ["single_seed_td_lambda", "ancestral_td_lambda"]
LOG_DIR = "lightning_logs/moons_lambda_sweep"

print(f"\n{'='*60}")
print(f"PHASE 1: lambda_eff in {PHASE1_LAMBDAS}")
print(f"{'='*60}")

for method in METHODS:
    for lam in PHASE1_LAMBDAS:
        run_name = f"{method}_lam{lam:.3f}"
        print(f"\n--- {run_name} ---")
        run_config(method, lam, run_name, LOG_DIR)

print("\n\nPHASE 1 RESULTS:")
p1_results = load_results(LOG_DIR)
print(p1_results.to_string(index=False))

# ---------------------------------------------------------------------------
# Phase 2: geometric sweep of small lambda_eff for single_seed_td_lambda
# ---------------------------------------------------------------------------
# Best from phase 1 was the lowest tested value (0.010), so sweep small values
# using a geometric progression: ~[0.001, 0.002, 0.004, 0.007, 0.02, 0.05]
# (excludes values already in PHASE1_LAMBDAS)
METHOD_P2 = "single_seed_td_lambda"
PHASE2_CANDIDATE_LAMBDAS = sorted(set(
    round(v, 4) for v in np.geomspace(0.001, 0.05, 8)
    if round(v, 4) not in PHASE1_LAMBDAS
))

print(f"\n{'='*60}")
print("PHASE 2: small-lambda geometric sweep for single_seed_td_lambda")
print(f"  values: {PHASE2_CANDIDATE_LAMBDAS}")
print(f"{'='*60}")

for lam in PHASE2_CANDIDATE_LAMBDAS:
    run_name = f"p2_{METHOD_P2}_lam{lam:.4f}"
    print(f"\n--- {run_name} ---")
    run_config(METHOD_P2, lam, run_name, LOG_DIR)

print("\n\nFINAL RESULTS (all phases):")
final = load_results(LOG_DIR)
print(final.to_string(index=False))
