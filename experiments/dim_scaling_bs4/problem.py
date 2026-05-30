"""Shared d-dimensional benchmark problem for the dimension-scaling study.

`make_moons` is intrinsically 2-D, so to scale dimension we use the random-GMM
+ quadratic-reward family from ``notebooks/dimension_scaling.py``, with one
addition: the reward scale ``s`` is CALIBRATED per dimension so the control gap

    gap(d) = -V(0,0) = -log E_base[exp(r(X_1))]      (reward max is 0)

is a fixed number of nats (default 6).  ``r(x) = -s ||x - c||^2`` and
``||x-c||^2 ~ O(d)``, so a naive fixed ``s`` makes ``exp(r)`` either collapse to
a constant (uninformative twist) or underflow.  Pinning the gap instead fixes
both the *difficulty* of the value-learning problem across d AND the numerics
(``V(0,0) = -gap`` ⇒ ``exp(value) ∈ [e^-gap, 1]``; ``exp(r)`` stays in fp32
range).  See ``notebooks/dim_scaling_methods_check.py`` for the validation that
all four methods + both losses are finite/overflow-free under this scaling.
"""

import math

import numpy as np
import torch
import torch.nn as nn

A = 1.0  # diffusion coefficient (matches the BS=4 study)


def make_problem(d, target_gap=6.0, seed=42):
    """Build the calibrated d-dim GMM/reward problem.

    Returns a dict with: drift_fn, reward_fn, smc_reward, gmm_sample, anal_fn,
    bias_val (≈V(0,0)), V00 (analytical), reward_scale, diag.
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
    # gap(s) increases monotonically in s; bisect geometrically.
    x_cal = torch.from_numpy(gmm_sample(40_000)).float()
    dist2 = (x_cal - c_float).square().sum(1)

    def gap_of(s):
        b = torch.logsumexp(-s * dist2, 0) - math.log(dist2.numel())
        return -b.item()

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

    # --- Analytical value (same math as moons) — used only for V00 / checks ---
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
        "d": d, "reward_scale": reward_scale, "target_gap": target_gap,
        "gap_empirical": -bias_val, "bias_val": bias_val, "V00_analytical": V00,
        "mean_dist2": float(dist2.mean()),
        "r_min": float(r_cal.min()), "r_max": float(r_cal.max()),
    }
    return {
        "drift_fn": drift_fn, "reward_fn": reward_fn, "smc_reward": smc_reward,
        "gmm_sample": gmm_sample, "anal_fn": anal_fn, "bias_val": bias_val,
        "V00": V00, "reward_scale": reward_scale, "diag": diag,
        # raw GMM + reward params (purely additive — for the analytical baseline)
        "means": means, "sigma2": sigma2, "weights": weights, "c": c_np,
    }


def optimal_terminal_and_reward(means, sigma2, weights, c, reward_scale, d):
    r"""Closed-form optimal-policy baseline.

    The optimal controlled diffusion (control = 2a·∇log v, v = log E[exp(r)|·])
    produces the terminal law obtained by tilting the base path measure by
    exp(r(X_1)); its terminal marginal is the base GMM tilted pointwise by
    exp(r) = exp(-s||x-c||^2)  (a Gaussian factor — NOTE the tilt is exp(+r),
    *toward* c, since r is already negative).  A Gaussian × Gaussian is Gaussian,
    so the optimum is again a GMM:

        denom_k = 1 + 2 s sigma_k^2
        v_k     = sigma_k^2 / denom_k                     (posterior variance)
        m_k     = (mu_k + 2 s sigma_k^2 c) / denom_k       (posterior mean)
        logZ_k  = log w_k - (d/2) log denom_k - s ||mu_k - c||^2 / denom_k
        pi_k    = softmax(logZ_k),   V(0,0) = logsumexp(logZ_k)

    Returns (V00, E_opt_reward, tilted) where
        V00          = log E_base[exp(r)]  (== the optimal value at the origin),
        E_opt_reward = E_{p*}[r] = -s sum_k pi_k (||m_k - c||^2 + d v_k)
                       — the expected reward of the optimal model (the baseline),
        tilted       = dict(weights=pi, means=m_k, var=v_k).

    By the Gibbs variational identity E_opt_reward = V00 + KL(p* || p_base) >= V00,
    which is why a trained policy's reward can exceed V(0,0): V(0,0) is a
    log-partition value, not the achievable reward.
    """
    s = float(reward_scale)
    mu = np.asarray(means, dtype=np.float64)       # (K, d)
    s2 = np.asarray(sigma2, dtype=np.float64)      # (K,)
    w = np.asarray(weights, dtype=np.float64)      # (K,)
    cc = np.asarray(c, dtype=np.float64)           # (d,)

    denom = 1.0 + 2.0 * s * s2                      # (K,)
    v_k = s2 / denom                               # (K,)
    m_k = (mu + (2.0 * s * s2)[:, None] * cc[None, :]) / denom[:, None]  # (K, d)
    d2_muc = ((mu - cc[None, :]) ** 2).sum(1)      # (K,)
    logZ_k = np.log(w) - 0.5 * d * np.log(denom) - s * d2_muc / denom    # (K,)

    M = logZ_k.max()
    V00 = float(M + np.log(np.exp(logZ_k - M).sum()))
    pi = np.exp(logZ_k - V00)                      # (K,) normalised

    d2_mc = ((m_k - cc[None, :]) ** 2).sum(1)      # (K,)
    E_dist2 = float((pi * (d2_mc + d * v_k)).sum())
    E_opt_reward = -s * E_dist2
    return V00, E_opt_reward, {"weights": pi, "means": m_k, "var": v_k,
                              "E_dist2": E_dist2}
