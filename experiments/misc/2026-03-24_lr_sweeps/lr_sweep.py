"""
LR sweep for single_seed_TD(λ=0.6) and single_seed_MC.

Runs each (method, lr) for SWEEP_MINUTES, then:
  - picks the best LR per method
  - fits an exponential decay to val_reward_mean vs step
  - estimates steps to within 0.5 of E_opt
  - reports estimated wall-clock time at the best LR
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from einops import reduce
from scipy.optimize import curve_fit
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means   = torch.from_numpy(clf.means_)
sigmas  = torch.sqrt(torch.from_numpy(clf.covariances_))[:, None]
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
max_reward  = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

E_OPT = json.loads(open("experiments/common/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/lr_sweep"

LRS = [3e-3, 1e-2, 3e-2]
METHODS = {
    "single_seed_td_lam0.6": dict(sampling_method="single_seed_td_lambda", lambda_eff=0.6),
    "single_seed_mc":        dict(sampling_method="single_seed_mc",         lambda_eff=0.1),
}
SWEEP_MINUTES = 3

# ---------------------------------------------------------------------------
# Calibrate steps/sec
# ---------------------------------------------------------------------------
print("Calibrating steps/sec...")
CALIBRATION_STEPS = 20
_cv = ValueNetwork(DIM, bias=bias)
_ds = OnPolicySMCDataset(
    dim=DIM, drift=base_drift, value=_cv, smc_value=_cv, reward=reward,
    device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_mc",
)
_loader  = DataLoader(_ds, batch_size=BATCH_SIZE)
_model   = OnPolicyValue(base_score_module=base_drift, value_module=_cv, dim=DIM, a=a, lr=1e-2)
_trainer = L.Trainer(max_steps=CALIBRATION_STEPS, enable_checkpointing=False,
                     enable_progress_bar=False, logger=False)
t0 = time.perf_counter()
_trainer.fit(_model, _loader)
elapsed = time.perf_counter() - t0
steps_per_sec = CALIBRATION_STEPS / elapsed
SWEEP_STEPS = int(steps_per_sec * SWEEP_MINUTES * 60)
VAL_STEPS   = max(1, SWEEP_STEPS // 20)
print(f"  {steps_per_sec:.2f} steps/s → {SWEEP_STEPS} steps for {SWEEP_MINUTES} min sweep")


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
def build(sampling_method, lambda_eff, lr):
    vm = ValueNetwork(DIM, bias=bias)
    ds = OnPolicySMCDataset(
        dim=DIM, drift=base_drift, value=vm, smc_value=vm, reward=reward,
        device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    model  = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward, dim=DIM, a=a, lr=lr,
        loss_type="quad",
    )
    return model, loader


results = {}   # (method, lr) -> final_val_reward_mean
all_curves = {}

for method, cfg in METHODS.items():
    print(f"\n{'='*60}\n{method}\n{'='*60}")
    for lr in LRS:
        run_key = f"{method}/lr_{lr:.0e}"
        print(f"\n  lr={lr:.0e} ...")
        model, loader = build(lr=lr, **cfg)
        logger = CSVLogger(LOG_DIR, name=run_key, version=0)
        trainer = L.Trainer(
            max_steps=SWEEP_STEPS,
            val_check_interval=VAL_STEPS,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader, val_dataloaders=val_loader)
        # load curve
        csv_path = f"{LOG_DIR}/{run_key}/version_0/metrics.csv"
        df = pd.read_csv(csv_path)
        val = df.dropna(subset=["val_reward_mean"])
        steps   = val["step"].values.astype(float)
        rewards = val["val_reward_mean"].values
        final   = float(rewards[-1]) if len(rewards) else float("nan")
        results[(method, lr)] = final
        all_curves[(method, lr)] = (steps, rewards)
        print(f"    final val_reward_mean = {final:.4f}  (gap {E_OPT - final:.4f})")


# ---------------------------------------------------------------------------
# Pick best LR per method and estimate convergence
# ---------------------------------------------------------------------------
def fit_convergence(steps, rewards, e_opt, target_gap=0.5):
    """
    Fit r(s) = E_opt - A*exp(-s/tau).
    Returns (A, tau, steps_to_convergence).
    """
    gap = e_opt - np.array(rewards)
    valid = gap > 0
    if valid.sum() < 3:
        return None
    try:
        def model(s, A, tau):
            return A * np.exp(-s / tau)
        popt, _ = curve_fit(
            model, steps[valid], gap[valid],
            p0=[gap[valid][0], float(steps[valid][-1])],
            bounds=([0, 1], [np.inf, np.inf]),
            maxfev=10000,
        )
        A, tau = popt
        if A <= target_gap:
            return A, tau, 0.0
        t_conv = tau * np.log(A / target_gap)
        return A, tau, t_conv
    except Exception as e:
        print(f"    [fit failed: {e}]")
        return None


print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
best_lrs = {}
estimated_steps = {}

for method in METHODS:
    best_lr = max(LRS, key=lambda lr: results.get((method, lr), -np.inf))
    best_lrs[method] = best_lr

    print(f"\n{method}:")
    print(f"  {'lr':>8}  {'final':>10}  {'gap':>8}")
    for lr in LRS:
        r = results.get((method, lr), float("nan"))
        marker = "  ← best" if lr == best_lr else ""
        print(f"  {lr:>8.1e}  {r:>10.4f}  {E_OPT - r:>8.4f}{marker}")

    # Fit convergence curve for best LR
    steps, rewards = all_curves[(method, best_lr)]
    fit = fit_convergence(steps, rewards, E_OPT)
    if fit is not None:
        A, tau, t_conv = fit
        t_min = t_conv / (steps_per_sec * 60)
        already = steps[-1] / (steps_per_sec * 60)
        extra_min = max(0.0, t_min - already)
        print(f"\n  Best lr={best_lr:.0e}:  A={A:.3f}, tau={tau:.0f} steps")
        print(f"  Extrapolated convergence (gap<0.5): {t_conv:.0f} steps = {t_min:.1f} min total")
        print(f"  Already ran {steps[-1]:.0f} steps ({already:.1f} min)")
        print(f"  Extra needed: {extra_min:.1f} min")
        estimated_steps[method] = (int(t_conv), t_min, best_lr)
    else:
        print(f"  [Could not fit convergence curve for best lr={best_lr:.0e}]")
        estimated_steps[method] = (None, None, best_lr)
