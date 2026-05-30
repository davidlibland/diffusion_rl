"""Higher-dimensional functionality + numerical-safety check.

Goal
----
Before running a full dimension-scaling *comparison*, verify that the four
methods we care about
    single_seed_mc, single_seed_td_lambda, ancestral_mc_td_lambda, off-policy
all FUNCTION at higher d, and that NEITHER loss (quad / mse) errors or
overflows with the reward at those dimensions.

How to scale the reward with dimension (the careful part)
---------------------------------------------------------
The reward is  r(x) = -s * ||x - c||^2.  Under the base GMM, ||x-c||^2 grows
~linearly in d (a sum of d per-coordinate squared distances), so the choice of
the scale s(d) controls both the numerics and the difficulty:

  * Fix the MEAN reward  (s = const/d, what dimension_scaling.py does):
    E[r] is O(1) but std(r) ~ s*std(||.||^2) ~ const/sqrt(d) -> 0.  At high d
    exp(r) collapses to a constant: the SMC twist becomes uninformative and the
    control problem trivialises.
  * Fix the STD of r  (s ~ 1/sqrt(d)):  keeps the dynamic range but E[r] ~
    -const*sqrt(d) -> -inf, so exp(r) UNDERFLOWS and off-policy's exp(reward)
    targets become 0.

Neither is right.  What actually matters for a *fair* comparison is the
difficulty of the value-learning problem, measured in nats by the control gap

    gap(d) = r_max - V(0,0) = -V(0,0) = -log E_base[exp(r(X_1))]     (r_max = 0)

so we CALIBRATE s(d) per dimension to hit a FIXED target gap (default 6 nats).
This simultaneously:
  - fixes the difficulty across d (the thing we want to compare against), and
  - bounds everything numerically: V(0,0) = -gap is fixed, so exp(value) in
    [e^-gap, 1]; and exp(r) stays in a safe fp32 range (no under/overflow),
    because the bulk of r sits in [-O(gap), 0] and the tails in ~[-O(3*gap),0].

The resulting s(d) decays between 1/d and 1/sqrt(d) (geometry dependent); we
print it next to the naive 10/d for reference.

What this script does
---------------------
Part A — per-dimension numeric report: calibrated s, V(0,0) (empirical &
  analytical), reward range, exp(reward) range, and a direct finiteness probe
  of BOTH losses (value + gradient) over the real reward distribution, incl. a
  deliberately-diverged prediction to confirm how each loss degrades.
Part B — a short training run for every (dimension, method, loss): a finite-
  loss guard + post-hoc scan + a value-range probe, reported as a PASS/FAIL
  matrix with the achieved validation reward.

Run (GPU, a few minutes):  uv run python experiments/misc/2026-05-29_dim_scaling_methods_check/dim_scaling_methods_check.py
Env: DIMS="2,5,10,20,50,100"  STEPS=60  TARGET_GAP=6.0
"""

import gc
import math
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.losses.log_quadratic_bregman import log_quadratic_bregman_divergence
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork


# ── device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


a = 1.0
DIMS = [int(x) for x in os.environ.get("DIMS", "2,5,10,20,50,100").split(",")]
STEPS = int(os.environ.get("STEPS", 60))
TARGET_GAP = float(os.environ.get("TARGET_GAP", 6.0))
DS_BATCH = 64
LOADER_BATCH = 32
N_STEPS = 20
MC = 8
LAMBDA = 0.5
LR = 1e-3
LOG_DIR = "lightning_logs/dim_check"
FP32_LOG_MAX = 88.0  # exp overflows in float32 above ~88.7

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class OnPolicyValueLive(OnPolicyValue):
    """Drift uses the LIVE network (the BS=4 default)."""

    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


# ── problem builder with calibrated reward scale ───────────────────────────
def make_problem(d, target_gap=TARGET_GAP, seed=42):
    """Random GMM base + quadratic reward, with s calibrated to a fixed gap.

    Returns drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn, bias_val,
    V00, reward_scale, diag(dict of numeric-safety stats).
    """
    rng = np.random.RandomState(seed)
    K = 20
    means = rng.uniform(-2, 2, size=(K, d)).astype(np.float64)
    sigma2 = rng.uniform(0.01, 0.5, size=K).astype(np.float64)
    weights = rng.dirichlet(np.ones(K)).astype(np.float64)
    c_np = rng.uniform(-1, 1, size=d).astype(np.float64)

    means_t = torch.from_numpy(means).double()
    sigma2_t = torch.from_numpy(sigma2).double()
    weights_t = torch.from_numpy(weights).double()
    c_t = torch.from_numpy(c_np).double()
    c_float = c_t.float()

    def gmm_sample(n):
        k = rng.choice(K, size=n, p=weights)
        return (means[k] + np.sqrt(sigma2[k, np.newaxis]) * rng.randn(n, d)).astype(
            np.float32
        )

    # --- Calibrate s so that -log E_base[exp(-s ||x-c||^2)] == target_gap. ---
    # gap(s) is monotone increasing in s, so bisect geometrically.
    x_cal = torch.from_numpy(gmm_sample(40_000)).float()
    dist2 = (x_cal - c_float).square().sum(1)  # ||x-c||^2 per sample, ~O(d)

    def gap_of(s):
        b = torch.logsumexp(-s * dist2, 0) - math.log(dist2.numel())
        return -b.item()  # = -V(0,0) estimate

    lo, hi = 1e-8, 1e4
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if gap_of(mid) < target_gap:
            lo = mid
        else:
            hi = mid
    reward_scale = math.sqrt(lo * hi)

    def reward_fn(x):
        return -reward_scale * (x - c_float.to(x)).square().sum(dim=1)

    def smc_reward(x, t):
        return reward_fn(x)

    # --- Analytical value (same math as moons / dimension_scaling.py) ---
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
            if t.numel() == 1:
                t = t.expand(x.shape[0])
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
        m = means_t.float().to(xt); sig = sigmas_col.float().to(xt)
        wc = weights_col.float().to(xt)
        xt_ = xt[..., None]; means_ = m.T[None, ...]; ts_ = ts[..., None]
        sigmas_ = sig.T; weights_ = wc.T; olw = torch.log(weights_)
        denom = 2 * a * (1 - ts) + ts * sigmas_**2
        le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
        lsf = torch.log(2 * a * (1 - ts) / denom) * d / 2
        lrw = olw + le + lsf
        lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
        lw = torch.where((ts == 0), olw, lw)
        nm = (2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
        us = (nm - xt[:, :, None]) / (1 - ts[..., None])
        return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")

    drift_fn = lambda x, t: gmm_drift(
        x, t if t.ndim >= 1 else t.unsqueeze(0)).to(dtype=torch.float)

    # --- bias value (network init) = log E_base[exp(r)] ≈ V(0,0) ---
    r_cal = -reward_scale * dist2
    bias_val = (torch.logsumexp(r_cal, 0) - math.log(r_cal.numel())).item()
    V00 = anal_fn(torch.zeros(1, d), torch.zeros(1)).item()

    diag = {
        "reward_scale": reward_scale,
        "naive_10_over_d": 10.0 / d,
        "gap_target": target_gap,
        "gap_empirical": -bias_val,
        "bias_val": bias_val,
        "V00_analytical": V00,
        "r_min": float(r_cal.min()),
        "r_max": float(r_cal.max()),
        "expr_min": float(torch.exp(r_cal.min()).item()),
        "mean_dist2": float(dist2.mean()),
    }
    return drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn, bias_val, V00, \
        reward_scale, diag


# ── Part A: direct loss-finiteness probe ───────────────────────────────────
def probe_losses(reward_fn, gmm_sample, bias_val, d):
    """Evaluate quad & mse loss + gradients on the real reward distribution.

    `target` = terminal log-value = reward(x1) (what off-policy regresses on,
    and the t=1 target for the on-policy methods).  We test predictions offset
    from the target by a range of deltas, including a large positive delta that
    deliberately diverges the prediction, to expose any overflow.
    """
    x1 = torch.from_numpy(gmm_sample(4096)).float().to(DEVICE)
    target = reward_fn(x1).detach()  # (N,) log-space, <= 0
    out = {}
    for delta in (-10.0, -3.0, 0.0, 3.0, 10.0, 40.0):
        pred = (target + delta).clone().requires_grad_(True)
        rec = {}
        for name, fn in (
            ("mse", lambda p, t: nn.functional.mse_loss(p.exp(), t.exp())),
            ("quad", lambda p, t: log_quadratic_bregman_divergence(
                p[:, None], t[:, None]).mean()),
        ):
            try:
                loss = fn(pred, target)
                g = torch.autograd.grad(loss, pred, retain_graph=False)[0]
                rec[name] = (bool(torch.isfinite(loss).all()),
                             bool(torch.isfinite(g).all()))
            except Exception as e:  # noqa: BLE001
                rec[name] = (f"ERR:{type(e).__name__}", False)
            pred = (target + delta).clone().requires_grad_(True)
        out[delta] = rec
    return out


# ── Part B: short training run per (dim, method, loss) ──────────────────────
class FiniteGuard(Callback):
    def __init__(self):
        super().__init__(); self.bad = False; self.max_abs = 0.0; self.n = 0

    def on_train_batch_end(self, trainer, pl, outputs, batch, bi):
        l = trainer.callback_metrics.get("train_loss")
        if l is None:
            return
        v = float(l); self.n += 1; self.max_abs = max(self.max_abs, abs(v))
        if not math.isfinite(v):
            self.bad = True


def build(method, loss_type, prob, hidden_dim):
    drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn, bias_val, V00, rs, _ = prob
    vm = ValueNetwork(DIM, hidden_dim=hidden_dim, bias=bias_val)
    if method == "offpolicy":
        model = OffPolicyValue(
            base_score_module=drift_fn, reward_function=reward_fn, value_module=vm,
            dim=DIM, a=a, lr=LR, loss_type=loss_type).to(DEVICE)
        ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a,
                                       batch_size=1024)
        loader = DataLoader(ds, batch_size=LOADER_BATCH)
        return model, vm, ds, loader
    # on-policy family: base drift + reward twist (matches the BS=4 winners)
    model = OnPolicyValueLive(
        base_score_module=drift_fn, value_module=vm, reward_function=reward_fn,
        dim=DIM, a=a, lr=LR, loss_type=loss_type, ema_decay=0.99).to(DEVICE)
    ds = OnPolicySMCDataset(
        dim=DIM, drift=drift_fn, value=model.value_module, smc_value=smc_reward,
        reward=reward_fn, device=DEVICE, a=a, batch_size=DS_BATCH, n_steps=N_STEPS,
        mc_samples_per_step=MC, sampling_method=method, lambda_eff=LAMBDA,
        off_policy_frac=0.0, include_t_zero=False, random_t=False,
        generating_function=gmm_sample)
    loader = DataLoader(ds, batch_size=LOADER_BATCH)
    return model, vm, ds, loader


def run_one(method, loss_type, prob, hidden_dim, gmm_sample):
    t0 = time.perf_counter()
    name = f"d{DIM}_{method}_{loss_type}"
    guard = FiniteGuard()
    err = None
    try:
        model, vm, ds, loader = build(method, loss_type, prob, hidden_dim)
        logger = CSVLogger(LOG_DIR, name=name, version=0)
        trainer = L.Trainer(
            max_steps=STEPS, val_check_interval=max(1, STEPS // 2),
            callbacks=[guard], logger=logger, enable_checkpointing=False,
            enable_progress_bar=False, num_sanity_val_steps=0)
        trainer.fit(model, loader, val_dataloaders=val_loader)
        # value-range probe: predictions must stay exp-safe.  Use the model's
        # own device (Lightning may leave it on CPU after fit()).
        with torch.no_grad():
            dev = next(model.value_module.parameters()).device
            xs = torch.from_numpy(gmm_sample(2048)).float().to(dev)
            ts = torch.rand(2048, device=dev)
            vp = model.value_module(xs, ts)
        max_abs_pred = float(vp.abs().max())
        pred_finite = bool(torch.isfinite(vp).all())
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:80]}"
        max_abs_pred, pred_finite = float("nan"), False
        try:
            del model, vm, ds, loader, trainer
        except Exception:
            pass
    gc.collect(); empty_cache()
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    val_reward = float("nan"); loss_curve_finite = True
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if "val_reward_mean" in df.columns:
            vr = df["val_reward_mean"].dropna()
            if len(vr):
                val_reward = float(vr.iloc[-1])
        if "train_loss" in df.columns:
            tl = df["train_loss"].dropna().to_numpy()
            loss_curve_finite = bool(np.isfinite(tl).all()) if len(tl) else True
    ok = (err is None and not guard.bad and loss_curve_finite
          and pred_finite and max_abs_pred < FP32_LOG_MAX)
    return {
        "ok": ok, "err": err, "val_reward": val_reward,
        "max_train_loss": guard.max_abs, "loss_finite": (not guard.bad)
        and loss_curve_finite, "max_abs_pred": max_abs_pred,
        "elapsed": time.perf_counter() - t0,
    }


# ── main ───────────────────────────────────────────────────────────────────
METHODS = ["single_seed_mc", "single_seed_td_lambda", "ancestral_mc_td_lambda",
           "offpolicy"]
LOSSES = ["quad", "mse"]

print(f"device={DEVICE}  dims={DIMS}  steps={STEPS}  target_gap={TARGET_GAP}\n")
print("=" * 96)
print("PART A — reward scaling & loss-finiteness numerics")
print("=" * 96)
probes = {}
problems = {}
for DIM in DIMS:
    prob = make_problem(DIM)
    problems[DIM] = prob
    diag = prob[-1]
    print(f"\nd={DIM:>3}  s={diag['reward_scale']:.4g} (vs 10/d={diag['naive_10_over_d']:.4g})"
          f"  gap≈{diag['gap_empirical']:.2f} (target {diag['gap_target']:.1f})"
          f"  V00: emp={diag['bias_val']:.2f} anal={diag['V00_analytical']:.2f}")
    print(f"       reward∈[{diag['r_min']:.2f},{diag['r_max']:.2f}]  "
          f"min exp(r)={diag['expr_min']:.1e}  mean‖x-c‖²={diag['mean_dist2']:.1f}")
    pr = probe_losses(prob[1], prob[3], prob[5], DIM)
    probes[DIM] = pr
    # compact finiteness line: for each delta, OK if (loss & grad finite) both losses
    flags = []
    for delta, rec in pr.items():
        q = rec["quad"]; m = rec["mse"]
        qf = q[0] is True and q[1]
        mf = m[0] is True and m[1]
        tag = {True: "·"}.get(qf and mf, None)
        if tag is None:
            tag = f"Δ{int(delta):+d}[q={'ok' if qf else 'X'},m={'ok' if mf else 'X'}]"
        flags.append(tag if tag != "·" else f"Δ{int(delta):+d}:ok")
    print("       loss probe: " + "  ".join(flags))

print("\n" + "=" * 96)
print("PART B — short training run per (dim, method, loss):  '✓'=ok  'X'=fail")
print("=" * 96)
header = f"{'dim':>4} | " + " | ".join(f"{m[:14]:>14}" for m in METHODS)
results = {}
for DIM in DIMS:
    prob = problems[DIM]
    _, _, _, gmm_sample = prob[0], prob[1], prob[2], prob[3]
    hidden_dim = min(256, max(64, 32 * DIM))
    for method in METHODS:
        for loss_type in LOSSES:
            r = run_one(method, loss_type, prob, hidden_dim, gmm_sample)
            results[(DIM, method, loss_type)] = r
            status = "✓" if r["ok"] else "X"
            extra = "" if r["ok"] else f"  <-- {r['err'] or 'nonfinite/overflow'}"
            print(f"  d={DIM:>3} {method:>24} {loss_type:>4}: {status}  "
                  f"val_r={r['val_reward']:>8.3f}  maxloss={r['max_train_loss']:.2e}  "
                  f"max|V|={r['max_abs_pred']:.1f}  ({r['elapsed']:.1f}s){extra}",
                  flush=True)

# summary matrices (quad / mse), val reward per dim×method
print("\n" + "=" * 96)
print("SUMMARY — PASS/FAIL and final validation reward")
print("=" * 96)
for loss_type in LOSSES:
    print(f"\n[{loss_type}]  " + header)
    print("  " + "-" * (7 + 17 * len(METHODS)))
    for DIM in DIMS:
        cells = []
        for method in METHODS:
            r = results[(DIM, method, loss_type)]
            mark = "✓" if r["ok"] else "X"
            cells.append(f"{mark} {r['val_reward']:>10.2f}".rjust(14))
        print(f"{DIM:>4} | " + " | ".join(cells))

n_fail = sum(1 for r in results.values() if not r["ok"])
print(f"\n{len(results)} runs, {n_fail} failures.")
print("Done." if n_fail == 0 else f"{n_fail} FAILURES — see rows marked X above.")
