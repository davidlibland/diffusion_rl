"""
Systematic experiment comparing value function learning methods on the moons dataset.

Grid:
  - policy:           off, on
  - loss_type:        mse, quad
  - sampling_method:  one_step_bootstrap, ancestral_td_lambda,
                      single_seed_td_lambda, single_seed_mc  (on-policy only)

Total: 2 (off) + 8 (on) = 10 runs.

MAX_STEPS is calibrated automatically: we time the slowest on-policy method
for CALIBRATION_STEPS steps, then set MAX_STEPS so that run takes ≤ MAX_MINUTES.
All runs use the same MAX_STEPS so total samples seen is identical across runs.

Results logged to lightning_logs/moons_sweep/<run_name>/
"""

import time

import numpy as np
import torch
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)


def moons_generator(batch_size):
    X_, _ = make_moons(n_samples=batch_size, noise=0.05, random_state=42)
    return scalar.transform(X_)


# ---------------------------------------------------------------------------
# GMM base model
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Reward and bias
# ---------------------------------------------------------------------------
a = 1
DIM = 2
DEVICE = "mps"  # change to "cuda" or "cpu" as needed
BATCH_SIZE = 256

reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)

all_rewards = reward(torch.from_numpy(X).float())
max_reward = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

# ---------------------------------------------------------------------------
# Calibrate MAX_STEPS against the slowest on-policy sampling method
# ---------------------------------------------------------------------------
MAX_MINUTES = 10
CALIBRATION_STEPS = 20  # short timing run

print("Calibrating MAX_STEPS against slowest on-policy method (single_seed_mc)...")

_calib_value = ValueNetwork(DIM, bias=bias)
_calib_dataset = OnPolicySMCDataset(
    dim=DIM,
    drift=base_drift,
    value=_calib_value,
    smc_value=_calib_value,
    reward=reward,
    device=DEVICE,
    a=a,
    batch_size=32,
    n_steps=100,
    sampling_method="single_seed_mc",
)
_calib_loader = DataLoader(_calib_dataset, batch_size=BATCH_SIZE)
_calib_model = OnPolicyValue(
    base_score_module=base_drift,
    value_module=_calib_value,
    # no reward_function: skips validation_step during calibration
    dim=DIM,
    a=a,
    lr=1e-2,
)
_calib_trainer = L.Trainer(
    max_steps=CALIBRATION_STEPS,
    enable_checkpointing=False,
    enable_progress_bar=False,
    logger=False,
)
t0 = time.perf_counter()
_calib_trainer.fit(_calib_model, _calib_loader)
elapsed = time.perf_counter() - t0

steps_per_second = CALIBRATION_STEPS / elapsed
MAX_STEPS = int(steps_per_second * MAX_MINUTES * 60)
VAL_STEPS = max(1, MAX_STEPS // 3)  # validate ~3 times per run

print(f"  {CALIBRATION_STEPS} steps in {elapsed:.1f}s → {steps_per_second:.2f} steps/s")
print(f"  MAX_STEPS = {MAX_STEPS}  (≈{MAX_MINUTES} min for slowest run)")
print(f"  VAL_STEPS = {VAL_STEPS}")

# ---------------------------------------------------------------------------
# Dummy validation dataloader (one trigger per check)
# ---------------------------------------------------------------------------
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
ON_POLICY_SAMPLING_METHODS = [
    "one_step_bootstrap",
    "ancestral_td_lambda",
    "single_seed_td_lambda",
    "single_seed_mc",
]
LOSS_TYPES = ["mse", "quad"]

configs = []
for loss_type in LOSS_TYPES:
    configs.append({"policy": "off", "loss_type": loss_type, "sampling_method": None})
for loss_type in LOSS_TYPES:
    for sampling_method in ON_POLICY_SAMPLING_METHODS:
        configs.append({"policy": "on", "loss_type": loss_type, "sampling_method": sampling_method})

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
for cfg in configs:
    policy = cfg["policy"]
    loss_type = cfg["loss_type"]
    sampling_method = cfg["sampling_method"]
    run_name = f"{policy}_{loss_type}" + (f"_{sampling_method}" if sampling_method else "")
    print(f"\n{'='*60}\nRun: {run_name}\n{'='*60}")

    logger = CSVLogger("lightning_logs/moons_sweep", name=run_name)

    if policy == "off":
        value_module = ValueNetwork(DIM, bias=bias)
        train_loader = DataLoader(
            InterpolatingNumpyDataset(moons_generator, a=a, batch_size=1024),
            batch_size=BATCH_SIZE,
        )
        model = OffPolicyValue(
            base_score_module=base_drift,
            reward_function=reward,
            value_module=value_module,
            a=a,
            lr=1e-2,
            dim=DIM,
            loss_type=loss_type,
            grad_decay=1e-8,
        )
    else:
        value_module = ValueNetwork(DIM, bias=bias)
        train_loader = DataLoader(
            OnPolicySMCDataset(
                dim=DIM,
                drift=base_drift,
                value=value_module,
                smc_value=value_module,
                reward=reward,
                device=DEVICE,
                a=a,
                batch_size=32,
                n_steps=100,
                sampling_method=sampling_method,
            ),
            batch_size=BATCH_SIZE,
        )
        model = OnPolicyValue(
            base_score_module=base_drift,
            value_module=value_module,
            reward_function=reward,
            dim=DIM,
            a=a,
            lr=1e-2,
            loss_type=loss_type,
            grad_decay=1e-8,
        )

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        val_check_interval=VAL_STEPS,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
import glob
import pandas as pd

dfs = []
# For each run, use the highest version (latest successful run)
all_paths = glob.glob("lightning_logs/moons_sweep/*/version_*/metrics.csv")
latest = {}
for path in all_paths:
    run = path.split("/")[2]
    version = int(path.split("/")[3].split("_")[1])
    if run not in latest or version > latest[run][0]:
        latest[run] = (version, path)

for run, (_, path) in sorted(latest.items()):
    df = pd.read_csv(path)
    val_rows = df.dropna(subset=["val_reward_mean"])
    if val_rows.empty:
        continue
    last = val_rows.iloc[-1]
    dfs.append({
        "run": run,
        "val_reward_mean": last["val_reward_mean"],
        "val_reward_std": last["val_reward_std"],
        "val_reward_max": last["val_reward_max"],
        "val_value_at_t0": last["val_value_at_t0"],
    })

results = pd.DataFrame(dfs).sort_values("val_reward_mean", ascending=False)
print("\n" + "="*60)
print("RESULTS (sorted by val_reward_mean)")
print("="*60)
print(results.to_string(index=False))
