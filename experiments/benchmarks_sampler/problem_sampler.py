"""Boltzmann sampler benchmarks (GMM-40, Many-Well-32) mapped onto our
value-learning framework.

Fixed base = driftless Brownian motion from X_0=0, dX = sqrt(2a) dW on [0,1],
so the base terminal is p_base = N(0, 2a I) (a=1 here -> N(0, 2I), which keeps
the whole existing build()/value-net machinery unchanged).  The optimal
controlled terminal is p_base * e^r / Z; setting that proportional to a target
pi(x) ∝ e^{-E(x)} gives the tilt-correction reward

    r(y) = log pi_unnorm(y) - log p_base(y)            (normalized coords y)

so that V(y,t) = log E_base[e^{r(Y_1)} | Y_t=y] is our learned object and
V(0,0) = log Z_target.  Coordinates are normalized by L (energy evaluated at
L*y) so far-flung targets sit at O(1) inside the N(0,2I) reference.

Returns the SAME dict interface as dim_scaling_consth.make_problem (drift_fn,
reward_fn, smc_reward, gmm_sample, anal_fn, bias_val, V00, reward_scale, diag)
plus benchmark extras (energy_fn, true_sample, log_Z, mode_coverage,
energy_w1, L, a).  See PLAN.md for definitions and references.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A = 1.0                       # base diffusion coeff: p_base = N(0, 2A I)
_TWO_A = 2.0 * A


def _log_p_base(y):
    """log N(y; 0, 2A I), y: (n,d)."""
    d = y.shape[1]
    return -(y * y).sum(1) / (2.0 * _TWO_A) - 0.5 * d * math.log(2.0 * math.pi * _TWO_A)


def _base_sample_np(n, d, rng):
    return (math.sqrt(_TWO_A) * rng.standard_normal((n, d))).astype(np.float32)


# ------------------------------------------------------------------ GMM-40 ---
def make_gmm40(n_mixes=40, loc_scaling=40.0, log_var_scaling=1.0, L=50.0,
               seed=0, cov_check=True):
    """FAB/DEM GMM: dim=2, means ~ U[-loc,loc]^2 (torch seed 0), equal weights,
    isotropic cov = softplus(log_var_scaling)^2 I.  Normalized by L."""
    d = 2
    g = torch.Generator().manual_seed(seed)
    means_raw = (torch.rand((n_mixes, d), generator=g) - 0.5) * 2 * loc_scaling
    scale_raw = float(F.softplus(torch.tensor(log_var_scaling)))       # std in raw coords
    # normalized coords
    m = (means_raw / L).double()                                       # (K,d)
    s = torch.full((n_mixes,), scale_raw / L, dtype=torch.float64)     # (K,) isotropic std
    w = torch.full((n_mixes,), 1.0 / n_mixes, dtype=torch.float64)
    rng = np.random.RandomState(1234)

    def _log_target(y):                    # log pi_norm(y) (normalized mixture, log Z = 0)
        y = y.double()
        m_ = m.to(y); s_ = s.to(y); logw = torch.log(w.to(y))
        diff2 = ((y[:, None, :] - m_[None]) ** 2).sum(-1)              # (n,K)
        log_comp = (logw[None] - 0.5 * d * torch.log(2 * math.pi * s_[None] ** 2)
                    - diff2 / (2 * s_[None] ** 2))
        return torch.logsumexp(log_comp, dim=1)

    def reward_fn(y):
        return (_log_target(y) - _log_p_base(y)).float()

    class AnalyticV(nn.Module):
        """Closed-form V(y,t) for driftless-BM base + Gaussian-mixture target."""
        def __init__(self):
            super().__init__()
            self.register_buffer("m", m); self.register_buffer("s", s)
            self.register_buffer("logw", torch.log(w))

        def forward(self, y, t):
            y = y.double(); t = t.double().reshape(-1)
            if t.numel() == 1:
                t = t.expand(y.shape[0])
            beta = (_TWO_A * (1.0 - t)).clamp_min(1e-12)[:, None]      # (n,1) bridge var
            gamma = _TWO_A
            s2 = (self.s ** 2)[None]                                   # (1,K)
            P = 1.0 / beta + 1.0 / s2 - 1.0 / gamma                    # (n,K)
            # b = y/beta + m/s2 ; |b|^2 summed over d
            b = y[:, None, :] / beta[:, :, None] + self.m[None] / s2[:, :, None]  # (n,K,d)
            b2 = (b * b).sum(-1)                                       # (n,K)
            C = (y * y).sum(1)[:, None] / beta + (self.m ** 2).sum(1)[None] / s2  # (n,K)
            logint = (self.logw[None]
                      - 0.5 * d * torch.log(beta) - d * torch.log(self.s)[None]
                      + 0.5 * d * math.log(gamma) - 0.5 * d * torch.log(P)
                      - 0.5 * (C - b2 / P))
            return torch.logsumexp(logint, dim=1).float()

    anal = AnalyticV()

    def anal_fn(x, t):
        return anal(x.cpu(), t.cpu()).to(x.device)

    def true_sample(n):
        k = rng.randint(0, n_mixes, size=n)
        mm = m.numpy()[k]; ss = s.numpy()[k][:, None]
        return (mm + ss * rng.randn(n, d)).astype(np.float32)

    def mode_coverage(samples, radius=None):
        if radius is None:
            radius = 4.0 * float(s[0])                                 # ~4 sigma in norm coords
        sm = torch.as_tensor(samples).float()
        mm = m.float()
        dist = torch.cdist(mm, sm)                                     # (K, n)
        hit = (dist.min(1).values < radius).float().mean().item()
        return hit

    extra = dict(energy_fn=lambda y: (-_log_target(y)).float(),
                 true_sample=true_sample, log_Z=0.0,
                 mode_coverage=mode_coverage, n_modes=n_mixes,
                 name="gmm40", L=L, means_norm=m.float(), std_norm=float(s[0]))
    return _assemble(d, reward_fn, anal_fn, bias_val=0.0, name="gmm40",
                     rng=rng, extra=extra)


# ------------------------------------------------------------ Many-Well-32 ---
# FAB DoubleWellEnergy (a=-0.5,b=-6,c=1): E_2d = a x1 + b x1^2 + c x1^4 + 0.5 x2^2
_DW_A, _DW_B, _DW_C = -0.5, -6.0, 1.0
_DW_TARGET_Z1D = 11784.50927     # FAB constant: int exp(-x^4+6x^2+0.5x) dx


def make_manywell(dim=32):
    assert dim % 2 == 0
    d = dim; n_wells = dim // 2
    rng = np.random.RandomState(1234)

    def _energy(y):                     # FAB unnormalized energy, normalized coords L=1
        y = y.double()
        x1 = y[:, 0::2]; x2 = y[:, 1::2]                               # (n, n_wells) each
        e = (_DW_A * x1 + _DW_B * x1 ** 2 + _DW_C * x1 ** 4 + 0.5 * x2 ** 2).sum(1)
        return e

    def reward_fn(y):
        return (-_energy(y) - _log_p_base(y)).float()

    # closed-form log Z: Z_2d = Z_1d * sqrt(2 pi); log Z = n_wells * log Z_2d
    log_Z2d = math.log(_DW_TARGET_Z1D) + 0.5 * math.log(2 * math.pi)
    log_Z = n_wells * log_Z2d

    # 1D rejection sampler for the double-well coordinate (FAB scheme)
    def _sample_dw1(n):
        target_lp = lambda x: -x ** 4 + 6 * x ** 2 + 0.5 * x
        mix = np.array([0.2, 0.8]); mu = np.array([-1.7, 1.7]); sc = 0.5
        k_const = _DW_TARGET_Z1D * 3
        out = np.empty(0)
        while out.shape[0] < n:
            comp = (rng.rand(n) < mix[1]).astype(int)
            prop = mu[comp] + sc * rng.randn(n)
            log_prop = np.log(mix[0] * np.exp(-0.5 * ((prop - mu[0]) / sc) ** 2)
                              + mix[1] * np.exp(-0.5 * ((prop - mu[1]) / sc) ** 2)
                              ) - math.log(sc * math.sqrt(2 * math.pi))
            log_accept = target_lp(prop) - (math.log(k_const) + log_prop)
            acc = np.log(rng.rand(n)) < log_accept
            out = np.concatenate([out, prop[acc]])
        return out[:n]

    def true_sample(n):
        y = np.empty((n, d), dtype=np.float32)
        for j in range(n_wells):
            y[:, 2 * j] = _sample_dw1(n)
            y[:, 2 * j + 1] = rng.randn(n)
        return y

    def mode_coverage(samples):
        # mean over wells of P(deep/positive well): the target sets this by the
        # double-well weights; a diffuse sampler gives ~0.5, so |gen-ref| is a
        # well-balance error (distinct-pattern count is NOT used -- a random
        # sampler maximizes it).
        sm = torch.as_tensor(samples).float()
        return float((sm[:, 0::2] > 0).float().mean())

    extra = dict(energy_fn=lambda y: _energy(y).float(),
                 true_sample=true_sample, log_Z=log_Z,
                 mode_coverage=mode_coverage, n_modes=2 ** n_wells,
                 name="manywell", L=1.0, n_wells=n_wells)
    # bias the value head with the known log Z (= V(0,0)); loss_shift uses it too
    return _assemble(d, reward_fn, anal_fn=None, bias_val=log_Z,
                     name="manywell", rng=rng, extra=extra)


# ------------------------------------------------------------------ shared ---
def _assemble(d, reward_fn, anal_fn, bias_val, name, rng, extra):
    def drift_fn(x, t):                          # driftless base
        return torch.zeros_like(x)

    def smc_reward(x, t):
        return reward_fn(x)

    def gmm_sample(n):                           # base terminal sampler N(0,2A I)
        return _base_sample_np(n, d, rng)

    # V(0,0) = log Z; energy-histogram W1 helper (1-D Wasserstein via sorted quantiles)
    def energy_w1(gen_samples, ref_samples):
        eg = np.sort(extra["energy_fn"](torch.as_tensor(gen_samples).float()).numpy())
        er = np.sort(extra["energy_fn"](torch.as_tensor(ref_samples).float()).numpy())
        m = min(len(eg), len(er))
        qg = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(eg)), eg)
        qr = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(er)), er)
        return float(np.abs(qg - qr).mean())

    diag = dict(name=name, d=d, calibration="boltzmann_tilt", a=A,
                log_Z=extra["log_Z"], bias_val=bias_val, n_modes=extra["n_modes"])
    prob = dict(drift_fn=drift_fn, reward_fn=reward_fn, smc_reward=smc_reward,
                gmm_sample=gmm_sample, anal_fn=anal_fn, bias_val=bias_val,
                V00=extra["log_Z"], reward_scale=1.0, diag=diag,
                means=None, sigma2=None, weights=None, c=None,
                energy_w1=energy_w1, a=A)
    prob.update(extra)
    return prob
