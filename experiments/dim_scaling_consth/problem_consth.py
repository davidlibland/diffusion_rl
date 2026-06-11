"""Constant-HEADROOM, coordinate-NESTED variant of the d-dim GMM/reward problem.

Two changes vs ../dim_scaling_matched/problem_matched.py:

1. **Nested instances.** Each seed's GMM/reward is drawn ONCE at D_MAX=512 and
   the d-dimensional instance is its truncation (means[:, :d], c[:d]; same
   spherical sigma^2 and weights).  Spherical components make the truncation
   exactly the MARGINAL of the 512-d instance, and the quadratic reward
   truncates consistently -- so within a seed, dimension is the ONLY thing
   that varies across the d-axis (no per-dimension instance redraw noise).

2. **Constant-headroom calibration.** The reward scale s is set per (seed, d)
   by analytic bisection so that the control headroom is fixed:

       E_{p*}[r] - E_base[r] = HEADROOM_TARGET   (default 6 nats)

   i.e. every instance at every dimension offers the SAME total reward
   improvement to a perfect controller; regret/HEADROOM is directly
   comparable across dims and seeds.  (The earlier gap-calibration fixed
   -V(0,0)=6, which made high-d problems trivial -- headroom 0.26 nats at
   d=512 -- and d=2 catastrophic, s~92.  The implied -V(0,0) here grows to
   ~30-40 nats at high d, which the numerically-stabilized losses now
   handle; bias_val therefore uses the ANALYTIC V00, not a 40k-sample MC
   logsumexp, which would be hopeless at such gaps.)
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

HEADROOM_TARGET = float(os.environ.get("CH_HEADROOM", 6.0))
D_MAX = 512


def make_problem(d, headroom_target=HEADROOM_TARGET, seed=42):
    """Build the nested, constant-headroom d-dim GMM/reward problem."""
    assert d <= D_MAX, f"d={d} exceeds D_MAX={D_MAX}"
    rng = np.random.RandomState(seed)
    K = 20
    # Draw at D_MAX and truncate: the d-dim instance is the exact marginal of
    # the (seed-determined) 512-d instance.
    means_full = rng.uniform(-2, 2, size=(K, D_MAX)).astype(np.float64)
    sigma2 = rng.uniform(0.01, 0.5, size=K).astype(np.float64)
    weights = rng.dirichlet(np.ones(K)).astype(np.float64)
    c_full = rng.uniform(-1, 1, size=D_MAX).astype(np.float64)
    means = np.ascontiguousarray(means_full[:, :d])
    c_np = np.ascontiguousarray(c_full[:d])

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

    # --- Constant-headroom calibration: bisect s so E_opt - E_base = target.
    # All quantities analytic (Gaussian algebra); headroom is monotone in s. ---
    comp_d2 = ((means - c_np[None, :]) ** 2).sum(1) + d * sigma2  # E||x-c||^2 / comp

    def _headroom(s):
        _, e_opt, _ = optimal_terminal_and_reward(
            means, sigma2, weights, c_np, s, d)
        e_base = float(-s * (weights * comp_d2).sum())
        return e_opt - e_base

    lo, hi = 1e-8, 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if _headroom(mid) < headroom_target:
            lo = mid
        else:
            hi = mid
    reward_scale = math.sqrt(lo * hi)

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

    V00_exact, e_opt, _ = optimal_terminal_and_reward(
        means, sigma2, weights, c_np, reward_scale, d)
    e_base = float(-reward_scale * (weights * comp_d2).sum())
    # bias for the value-net output head: the ANALYTIC V(0,0) (a 40k-sample MC
    # logsumexp underflows hopelessly at 30+ nat gaps).
    bias_val = float(V00_exact)
    V00 = anal_fn(torch.zeros(1, d), torch.zeros(1)).item()

    diag = {
        "d": d, "calibration": "constant_headroom_nested",
        "headroom_target": headroom_target, "headroom": e_opt - e_base,
        "reward_scale": reward_scale, "E_base_r": e_base, "E_opt_r": e_opt,
        "gap": -V00_exact, "bias_val": bias_val, "V00_analytical": V00,
    }
    return {
        "drift_fn": drift_fn, "reward_fn": reward_fn, "smc_reward": smc_reward,
        "gmm_sample": gmm_sample, "anal_fn": anal_fn, "bias_val": bias_val,
        "V00": V00, "reward_scale": reward_scale, "diag": diag,
        "means": means, "sigma2": sigma2, "weights": weights, "c": c_np,
    }
