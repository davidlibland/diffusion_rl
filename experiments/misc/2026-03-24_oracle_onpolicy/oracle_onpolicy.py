"""
Oracle on-policy experiment.

Uses OnPolicySMCDataset with value=anal_vm and smc_value=anal_vm — the
analytical value function provides both the TD-λ bootstrap targets and the
SMC importance weights.  The SDE trajectories still come from the base
(uncontrolled) drift; the oracle only eliminates bootstrapping error.

Sampling method: single_seed_td_lambda with λ=0.2  (lowest target variance
from data_quality.py analysis).

Steps:
1. Quick LR sweep (SWEEP_MINUTES each).
2. Run best LR to convergence.
3. Save CSV log to lightning_logs/oracle_onpolicy/ for the comparison plot.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import reduce
from scipy.optimize import curve_fit
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM + analytical value function
# ---------------------------------------------------------------------------
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


# CPU-only analytical value (float64); wraps to any device
_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D).cpu()


def anal_fn(x, t):
    """Callable usable from any device (MPS / CPU / CUDA)."""
    result = _anal_vm_cpu(x.cpu(), t.cpu())
    return result.to(x.device)


# ---------------------------------------------------------------------------
# GMM base drift
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
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a,
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
E_OPT = json.loads(open("experiments/common/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/oracle_onpolicy"

LAMBDA = 0.2   # lowest variance from data_quality analysis
BATCH_SIZE = 256
LRS = [1e-3, 3e-3, 1e-2]
SWEEP_MINUTES = 2

# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------
print("Calibrating steps/sec...")
CALIB_STEPS = 20
_cv = ValueNetwork(D, bias=bias_val)
_ds = OnPolicySMCDataset(
    dim=D, drift=base_drift, value=anal_fn, smc_value=anal_fn,
    reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_td_lambda", lambda_eff=LAMBDA,
)
_loader  = DataLoader(_ds, batch_size=BATCH_SIZE)
_model   = OnPolicyValue(base_score_module=base_drift, value_module=_cv, dim=D, a=a, lr=1e-3)
_trainer = L.Trainer(max_steps=CALIB_STEPS, enable_checkpointing=False,
                     enable_progress_bar=False, logger=False)
t0 = time.perf_counter()
_trainer.fit(_model, _loader)
elapsed = time.perf_counter() - t0
steps_per_sec = CALIB_STEPS / elapsed
SWEEP_STEPS = int(steps_per_sec * SWEEP_MINUTES * 60)
VAL_STEPS   = max(1, SWEEP_STEPS // 15)
print(f"  {steps_per_sec:.2f} steps/s → {SWEEP_STEPS} steps for {SWEEP_MINUTES} min")


# ---------------------------------------------------------------------------
# LR sweep
# ---------------------------------------------------------------------------
def build(lr):
    vm = ValueNetwork(D, bias=bias_val)
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=anal_fn, smc_value=anal_fn,
        reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method="single_seed_td_lambda", lambda_eff=LAMBDA,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    model  = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward, dim=D, a=a, lr=lr,
        loss_type="quad", analytical_value_fn=anal_fn,
    )
    return model, loader


print(f"\nLR sweep (single_seed_td_lambda λ={LAMBDA}, oracle V)")
print(f"{'='*55}")
results = {}
all_curves = {}

for lr in LRS:
    print(f"\n  lr={lr:.0e} ...")
    model, loader = build(lr)
    logger = CSVLogger(LOG_DIR, name=f"sweep/lr_{lr:.0e}", version=0)
    trainer = L.Trainer(
        max_steps=SWEEP_STEPS,
        val_check_interval=VAL_STEPS,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    csv_path = f"{LOG_DIR}/sweep/lr_{lr:.0e}/version_0/metrics.csv"
    df  = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_reward_mean"])
    steps   = val["step"].values.astype(float)
    rewards = val["val_reward_mean"].values
    final   = float(rewards[-1]) if len(rewards) else float("nan")
    results[lr] = final
    all_curves[lr] = (steps, rewards)
    print(f"    final = {final:.4f}  (gap {E_OPT - final:.4f})")

best_lr = max(LRS, key=lambda lr: results.get(lr, -np.inf))
print(f"\nBest lr = {best_lr:.0e}")
print(f"  {'lr':>8}  {'final':>10}  {'gap':>8}")
for lr in LRS:
    r = results.get(lr, float("nan"))
    marker = "  ← best" if lr == best_lr else ""
    print(f"  {lr:>8.1e}  {r:>10.4f}  {E_OPT - r:>8.4f}{marker}")


# ---------------------------------------------------------------------------
# Fit convergence and run to convergence
# ---------------------------------------------------------------------------
def fit_convergence(steps, rewards, e_opt, target_gap=0.5):
    gap = e_opt - np.array(rewards)
    valid = gap > 0
    if valid.sum() < 3:
        return None
    try:
        def mdl(s, A, tau):
            return A * np.exp(-s / tau)
        popt, _ = curve_fit(mdl, steps[valid], gap[valid],
                            p0=[gap[valid][0], float(steps[valid][-1])],
                            bounds=([0, 1], [np.inf, np.inf]), maxfev=10000)
        A, tau = popt
        t_conv = tau * np.log(A / target_gap) if A > target_gap else 0.0
        return A, tau, t_conv
    except Exception as e:
        print(f"  [fit failed: {e}]")
        return None


steps_b, rewards_b = all_curves[best_lr]
fit = fit_convergence(steps_b, rewards_b, E_OPT)
if fit:
    A, tau, t_conv = fit
    total_min = t_conv / (steps_per_sec * 60)
    print(f"\nConvergence estimate: {t_conv:.0f} steps = {total_min:.1f} min")
    max_steps_conv = int(t_conv * 1.5)
else:
    print("\nCould not fit; using 3× sweep steps as budget")
    max_steps_conv = 3 * SWEEP_STEPS

max_steps_conv = max(max_steps_conv, SWEEP_STEPS * 3)
print(f"Running to convergence with lr={best_lr:.0e}, max_steps={max_steps_conv} ...")

model, loader = build(best_lr)
val_interval  = max(1, max_steps_conv // 50)

ckpt_dir = "checkpoints/oracle_onpolicy"
ckpt_cb  = ModelCheckpoint(
    dirpath=ckpt_dir, save_last=True,
    every_n_train_steps=max(1, max_steps_conv // 5),
    save_top_k=1, monitor="val_reward_mean", mode="max", filename="best",
)
logger  = CSVLogger(LOG_DIR, name="convergence", version=0)
trainer = L.Trainer(
    max_steps=max_steps_conv,
    val_check_interval=val_interval,
    callbacks=[ckpt_cb],
    logger=logger,
    enable_checkpointing=True,
    enable_progress_bar=True,
)
t0 = time.perf_counter()
trainer.fit(model, loader, val_dataloaders=val_loader)
elapsed = time.perf_counter() - t0

csv_path = f"{LOG_DIR}/convergence/version_0/metrics.csv"
df  = pd.read_csv(csv_path)
val = df.dropna(subset=["val_reward_mean"])
steps_c   = val["step"].values
rewards_c = val["val_reward_mean"].values
best_r    = rewards_c.max()

print(f"\nElapsed: {elapsed/60:.1f} min")
print(f"Best:  {best_r:.4f}  (gap {E_OPT - best_r:.4f})")
print(f"Final: {rewards_c[-1]:.4f}")
print(f"E_opt: {E_OPT:.4f}")
print(f"Checkpoint: {ckpt_cb.best_model_path}")
