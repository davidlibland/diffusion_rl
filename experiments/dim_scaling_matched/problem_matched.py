"""Difficulty-matched variant of the d-dim GMM/reward problem (suggestion (b)).

Identical to ../dim_scaling_bs4/problem.py EXCEPT the reward scale is calibrated
to fix the **mean base reward** rather than the log-partition gap:

    E_base[r] = -s * E_base[||x-c||^2] = MEAN_REWARD_TARGET   ->   s = -target / E[||x-c||^2]

(default target = -10).  Rationale: the gap-calibration (-V(0,0)=6) makes the
optimal tilt vanish at high d (KL -> 0), so high-d problems trivialise and the
regret-vs-d trend mostly reflects the calibration.  Fixing the *mean reward*
keeps the reward magnitude comparable across d (and to moons, which used s=10),
so the difficulty axis better reflects dimension.  The blobs and reward centre
are the SAME as the gap-version at a given seed (same RNG draw order); only the
scale `s` differs.

`optimal_terminal_and_reward` is calibration-agnostic, so we reuse it from the
gap-version module.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

A = 1.0

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "dim_scaling_bs4"))
from problem import optimal_terminal_and_reward  # noqa: E402,F401  (re-exported)

MEAN_REWARD_TARGET = float(os.environ.get("MM_MEAN_REWARD", -10.0))


def make_problem(d, mean_reward_target=MEAN_REWARD_TARGET, seed=42):
    """Build the difficulty-matched d-dim GMM/reward problem (E_base[r]=target)."""
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

    # --- Matched calibration: s so that E_base[r] = -s*E[||x-c||^2] = target. ---
    x_cal = torch.from_numpy(gmm_sample(40_000)).float()
    dist2 = (x_cal - c_float).square().sum(1)
    mean_dist2 = float(dist2.mean())
    reward_scale = (-mean_reward_target) / mean_dist2

    def reward_fn(x):
        return -reward_scale * (x - c_float.to(x)).square().sum(dim=1)

    def smc_reward(x, t):
        return reward_fn(x)

    class AnalyticalValue(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("means", means_t.float())
            self.register_buffer("sigma2", sigma2_t.float())
            self.register_buffer("weights", weights_t.float())
            self.register_buffer("c", c_t.float())
            self.a = A; self.D = d; self.rs = reward_scale

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

    sigmas_col = torch.sqrt(sigma2_t)[:, None]
    weights_col = weights_t[:, None]

    def gmm_drift(xt, ts):
        from einops import reduce
        xt = xt.float(); ts = ts.reshape(-1, 1).float()
        m = means_t.float().to(xt); sig = sigmas_col.float().to(xt)
        wc = weights_col.float().to(xt)
        xt_ = xt[..., None]; means_ = m.T[None, ...]; ts_ = ts[..., None]
        sigmas_ = sig.T; weights_ = wc.T; olw = torch.log(weights_)
        denom = 2 * A * (1 - ts) + ts * sigmas_**2
        le = -reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum") / (2 * ts * denom)
        lsf = torch.log(2 * A * (1 - ts) / denom) * d / 2
        lrw = olw + le + lsf
        lw = lrw - torch.logsumexp(lrw, dim=1, keepdim=True)
        lw = torch.where((ts == 0), olw, lw)
        nm = (2 * A * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2) / denom[:, None, :]
        us = (nm - xt[:, :, None]) / (1 - ts[..., None])
        return reduce(torch.exp(lw)[:, None, :] * us, "n d m -> n d", "sum")

    drift_fn = lambda x, t: gmm_drift(
        x, t if t.ndim >= 1 else t.unsqueeze(0)).to(dtype=torch.float)

    r_cal = -reward_scale * dist2
    bias_val = (torch.logsumexp(r_cal, 0) - math.log(r_cal.numel())).item()
    V00 = anal_fn(torch.zeros(1, d), torch.zeros(1)).item()

    diag = {
        "d": d, "calibration": "matched_mean_reward",
        "mean_reward_target": mean_reward_target, "reward_scale": reward_scale,
        "mean_dist2": mean_dist2, "E_base_r": -reward_scale * mean_dist2,
        "gap_emergent": -bias_val, "bias_val": bias_val, "V00_analytical": V00,
        "r_min": float(r_cal.min()), "r_max": float(r_cal.max()),
    }
    return {
        "drift_fn": drift_fn, "reward_fn": reward_fn, "smc_reward": smc_reward,
        "gmm_sample": gmm_sample, "anal_fn": anal_fn, "bias_val": bias_val,
        "V00": V00, "reward_scale": reward_scale, "diag": diag,
        "means": means, "sigma2": sigma2, "weights": weights, "c": c_np,
    }
