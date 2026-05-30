"""
LR sweep for OffPolicyValue on single_seed_mc and single_seed_TD(λ=0.6) targets.

Off-policy uses InterpolatingNumpyDataset: samples (x1, x, t) from the fixed
base distribution — no policy dependence. Target = r(x1).

Runs each lr for SWEEP_MINUTES, picks best, fits convergence curve.
If estimated convergence < 30 min, runs to convergence.
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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means_np   = clf.means_
sigmas_np  = np.sqrt(clf.covariances_)
weights_np = clf.weights_

means   = torch.from_numpy(means_np)
sigmas  = torch.from_numpy(sigmas_np)[:, None]
weights = torch.from_numpy(weights_np)[:, None]

a = 1
DIM = 2
DEVICE = "mps"
BATCH_SIZE = 256

reward_fn = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)
all_rewards = reward_fn(torch.from_numpy(X).float())
max_reward  = all_rewards.max()
bias = (torch.log(torch.mean(torch.exp(all_rewards - max_reward))) + max_reward).item()


def gmm_sample(n):
    """Sample n points from the fitted GMM."""
    k = np.random.choice(len(weights_np), size=n, p=weights_np)
    return means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, DIM)


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


base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

E_OPT = json.loads(open("experiments/common/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/offpolicy_lr_sweep"

LRS = [1e-4, 3e-4, 1e-3, 3e-3]
SWEEP_MINUTES = 3

# ---------------------------------------------------------------------------
# Calibrate steps/sec for off-policy
# ---------------------------------------------------------------------------
print("Calibrating off-policy steps/sec...")
CALIBRATION_STEPS = 50
_cv = ValueNetwork(DIM, bias=bias)
_ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=BATCH_SIZE)
_loader = DataLoader(_ds, batch_size=BATCH_SIZE)
_model  = OffPolicyValue(
    base_score_module=base_drift, reward_function=reward_fn,
    value_module=_cv, a=a, lr=1e-3, dim=DIM,
)
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
# LR sweep
# ---------------------------------------------------------------------------
def build(lr):
    vm = ValueNetwork(DIM, bias=bias)
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=BATCH_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    model  = OffPolicyValue(
        base_score_module=base_drift, reward_function=reward_fn,
        value_module=vm, a=a, lr=lr, dim=DIM, loss_type="quad",
    )
    return model, loader


results = {}
all_curves = {}

print(f"\n{'='*60}\nOff-policy LR sweep\n{'='*60}")
for lr in LRS:
    run_key = f"offpolicy/lr_{lr:.0e}"
    print(f"\n  lr={lr:.0e} ...")
    model, loader = build(lr=lr)
    logger = CSVLogger(LOG_DIR, name=run_key, version=0)
    trainer = L.Trainer(
        max_steps=SWEEP_STEPS,
        val_check_interval=VAL_STEPS,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader, val_dataloaders=val_loader)
    csv_path = f"{LOG_DIR}/{run_key}/version_0/metrics.csv"
    df  = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_reward_mean"])
    steps   = val["step"].values.astype(float)
    rewards = val["val_reward_mean"].values
    final   = float(rewards[-1]) if len(rewards) else float("nan")
    results[lr] = final
    all_curves[lr] = (steps, rewards)
    print(f"    final val_reward_mean = {final:.4f}  (gap {E_OPT - final:.4f})")


# ---------------------------------------------------------------------------
# Pick best LR and estimate convergence
# ---------------------------------------------------------------------------
def fit_convergence(steps, rewards, e_opt, target_gap=0.5):
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
        t_conv = tau * np.log(A / target_gap) if A > target_gap else 0.0
        return A, tau, t_conv
    except Exception as e:
        print(f"    [fit failed: {e}]")
        return None


print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"\n  {'lr':>8}  {'final':>10}  {'gap':>8}")
for lr in LRS:
    r = results.get(lr, float("nan"))
    print(f"  {lr:>8.1e}  {r:>10.4f}  {E_OPT - r:>8.4f}")

best_lr = max(LRS, key=lambda lr: results.get(lr, -np.inf))
print(f"\n  Best lr = {best_lr:.0e}")

steps, rewards = all_curves[best_lr]
fit = fit_convergence(steps, rewards, E_OPT)

if fit is None:
    print("  Could not fit convergence curve.")
    print("  Stopping — cannot estimate convergence time.")
else:
    A, tau, t_conv = fit
    total_min = t_conv / (steps_per_sec * 60)
    already_min = steps[-1] / (steps_per_sec * 60)
    extra_min = max(0.0, total_min - already_min)
    print(f"  A={A:.3f}, tau={tau:.0f} steps")
    print(f"  Convergence (gap<0.5): {t_conv:.0f} steps = {total_min:.1f} min total")
    print(f"  Already ran {steps[-1]:.0f} steps ({already_min:.1f} min)")
    print(f"  Extra needed: {extra_min:.1f} min")

    if total_min <= 30:
        print(f"\n  {'='*50}")
        print(f"  Estimated {total_min:.1f} min ≤ 30 — running to convergence now.")
        print(f"  {'='*50}")

        max_steps_conv = int(t_conv * 1.5)
        model, loader = build(lr=best_lr)
        val_interval  = max(1, max_steps_conv // 40)

        ckpt_dir = f"checkpoints/offpolicy_convergence"
        ckpt_cb  = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_last=True,
            every_n_train_steps=max(1, max_steps_conv // 5),
            save_top_k=1,
            monitor="val_reward_mean",
            mode="max",
            filename="best",
        )
        logger  = CSVLogger("lightning_logs/offpolicy_convergence", name="offpolicy", version=0)
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

        csv_path = "lightning_logs/offpolicy_convergence/offpolicy/version_0/metrics.csv"
        df2  = pd.read_csv(csv_path)
        val2 = df2.dropna(subset=["val_reward_mean"])
        steps2   = val2["step"].values
        rewards2 = val2["val_reward_mean"].values
        best_r   = rewards2.max()

        print(f"\n  Elapsed: {elapsed/60:.1f} min")
        print(f"  Best val_reward_mean:  {best_r:.4f}  (gap {E_OPT - best_r:.4f})")
        print(f"  Final val_reward_mean: {rewards2[-1]:.4f}")
        print(f"  Optimal target:        {E_OPT:.4f}")
        print(f"\n  {'step':>8}  {'reward':>10}  {'gap':>8}")
        for s, r in zip(steps2, rewards2):
            marker = "  ← best" if r == best_r else ""
            print(f"  {int(s):>8}  {r:>10.4f}  {E_OPT - r:>8.4f}{marker}")
        print(f"\n  Checkpoint: {ckpt_cb.best_model_path}")
    else:
        print(f"\n  Estimated {total_min:.1f} min > 30 — not running automatically.")
        print(f"  Best lr={best_lr:.0e}, run with max_steps={int(t_conv)} to converge.")
