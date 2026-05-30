"""
Targeted training runs:
  1. anc_mctd_smc_reward_lam1.0  — most promising anc_mctd + reward SMC
  2. anc_mctd_smc_model_lam0.8   — most promising anc_mctd + model SMC
  3. ss_mc_smc_reward             — single_seed_mc + reward SMC (no bootstrap)
"""

import json
import os
import time
from math import ceil

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
        means = self.means.double()
        sigma2 = self.sigma2.double()
        weights = self.weights.double()
        dk = t_ * sigma2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * means[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + 1e-40
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(weights)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * sigma2[None, :] / dk
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        return torch.logsumexp(log_pw + log_zk, dim=1).float()


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
    orig_log_weights = torch.log(weights_)
    denominator = 2 * a * (1 - ts) + ts * sigmas_ ** 2
    likelihood_exp_numerator = reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum")
    likelihood_exp = -likelihood_exp_numerator / (2 * ts * denominator)
    log_std_factor = torch.log(2 * a * (1 - ts) / denominator) * d / 2
    log_rel_weights = orig_log_weights + likelihood_exp + log_std_factor
    normalization = torch.logsumexp(log_rel_weights, dim=1, keepdim=True)
    log_weights = log_rel_weights - normalization
    log_weights = torch.where((ts == 0), orig_log_weights, log_weights)
    new_means = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denominator[:, None, :]
    new_sigmas = torch.sqrt(0.5 * torch.exp(log_std_factor) * sigmas_ ** 2)
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


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)
smc_reward = lambda x, t: reward(x)

with open("experiments/common/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LR = 3e-3
LOADER_BATCH_SIZE = 256
MAX_STEPS = 3000
LOG_DIR = "lightning_logs/ancestral_sweep"
CKPT_DIR = "checkpoints/ancestral_sweep"


class SampleCounter(Callback):
    def __init__(self, samples_per_step):
        super().__init__()
        self.samples_per_step = samples_per_step
        self.total_samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_samples += self.samples_per_step


RUNS = [
    dict(
        name="anc_mctd_smc_reward_lam1.0",
        sampling_method="ancestral_mc_td_lambda",
        lambda_eff=1.0,
        smc_mode="reward",
    ),
    dict(
        name="anc_mctd_smc_model_lam0.8",
        sampling_method="ancestral_mc_td_lambda",
        lambda_eff=0.8,
        smc_mode="model",
    ),
    dict(
        name="ss_mc_smc_reward",
        sampling_method="single_seed_mc",
        lambda_eff=0.1,  # unused for single_seed_mc
        smc_mode="reward",
    ),
]

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

for run_cfg in RUNS:
    run_name = run_cfg["name"]
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    print(f"{'='*70}")

    csv_path = f"{LOG_DIR}/{run_name}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= MAX_STEPS - 1:
            print(f"  Already complete, skipping.")
            continue

    vm = ValueNetwork(D, bias=bias_val)

    if run_cfg["smc_mode"] == "model":
        smc_fn = vm
    else:
        smc_fn = smc_reward

    ds = OnPolicySMCDataset(
        dim=D,
        drift=base_drift,
        value=vm,
        smc_value=smc_fn,
        reward=reward,
        device=DEVICE,
        a=a,
        batch_size=32,
        n_steps=100,
        mc_samples_per_step=10,
        sampling_method=run_cfg["sampling_method"],
        lambda_eff=run_cfg["lambda_eff"],
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

    # More frequent validation for single_seed_mc (user requested)
    if "ss_mc" in run_name:
        val_interval = max(1, MAX_STEPS // 120)  # ~120 checkpoints
    else:
        val_interval = max(1, MAX_STEPS // 60)

    ckpt_cb = ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{run_name}",
        save_last=True,
        save_top_k=1,
        monitor="val_reward_mean",
        mode="max",
        filename="best",
    )
    counter = SampleCounter(LOADER_BATCH_SIZE)
    logger = CSVLogger(LOG_DIR, name=run_name, version=0)

    trainer = L.Trainer(
        max_steps=MAX_STEPS,
        val_check_interval=val_interval,
        callbacks=[ckpt_cb, counter],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    elapsed = time.perf_counter() - t0

    # Report
    df = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_reward_mean"])
    steps = val["step"].values
    rewards = val["val_reward_mean"].values
    best_r = rewards.max()
    final_r = rewards[-1]

    print(f"\n  Elapsed: {elapsed/60:.1f} min, samples: {counter.total_samples:,}")
    print(f"  Best:  {best_r:.4f}  (gap {E_OPT - best_r:.4f})")
    print(f"  Final: {final_r:.4f}")

    # Print trajectory
    indices = np.linspace(0, len(steps) - 1, min(15, len(steps)), dtype=int)
    print(f"\n  {'step':>8}  {'reward':>10}  {'gap':>8}")
    for i in indices:
        marker = "  <-- best" if rewards[i] == best_r else ""
        print(f"  {int(steps[i]):>8}  {rewards[i]:>10.4f}  {E_OPT - rewards[i]:>8.4f}{marker}")

print(f"\nDone. E_OPT = {E_OPT:.4f}")
