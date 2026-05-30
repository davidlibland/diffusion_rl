"""
Sanity check: plug the analytical value function into the optimal-control
drift and verify that the resulting SDE samples achieve the optimal expected
reward E_opt = -2.587 computed in analytical_target.json.

Pipeline (mirrors OnPolicyValue.drift + validation_step exactly):
  1. Build the analytical V(x_t, t) as a differentiable nn.Module.
  2. Instantiate OnPolicyValue with it, using the GMM base drift.
  3. Call trainer.validate() to run the same validation_step used in training.
"""

import json
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from diffusion_rl.algorithms.integration import integrate_sde

# ---------------------------------------------------------------------------
# Load targets
# ---------------------------------------------------------------------------
targets = json.loads(open("experiments/common/analytical_target.json").read())
E_OPT = targets["E_opt"]   # -2.587
V_0_0 = targets["V_0_0"]   # -5.085
print(f"Target  E_opt = {E_OPT:.4f}")
print(f"Target  V_0_0 = {V_0_0:.4f}")

# ---------------------------------------------------------------------------
# GMM (same as all other scripts)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

_means   = torch.from_numpy(clf.means_).double()        # (K, D)
_sigma2  = torch.from_numpy(clf.covariances_).double()  # (K,)
_weights = torch.from_numpy(clf.weights_).double()      # (K,)

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])

# ---------------------------------------------------------------------------
# Analytical value function as a differentiable nn.Module
# ---------------------------------------------------------------------------

class AnalyticalValue(nn.Module):
    """
    Wraps the closed-form V(x_t, t) so it can be used anywhere a
    value_module(x, t) -> (N,) is expected, including autograd through x.
    """
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        # Register as buffers so they move with .to(device)
        self.register_buffer("means",   means.float())
        self.register_buffer("sigma2",  sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        """
        log integral of N(x; m, v*I) * exp(-10*||x-c||^2) dx
        Numerically stable; valid at v=0 (gives r(m)).

        m : (..., D),  v : (...)
        """
        c = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * c).sum(-1)
               + 200.0 * v * (c ** 2).sum()) / denom
            - 10.0 * (c ** 2).sum()
        )

    def forward(self, x, t):
        """
        Args:
            x : (N, D)  -- supports grad
            t : scalar tensor or (N,) tensor
        Returns:
            V : (N,)
        """
        # Broadcast t to (N,)
        x = x.double()
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.double()
        t_ = t[:, None]          # (N, 1)

        means   = self.means.double()    # (K, D)
        sigma2  = self.sigma2.double()   # (K,)
        weights = self.weights.double()  # (K,)
        eps = 1e-40

        # d_k = t*sigma_k^2 + 2a*(1-t)   -- always > 0
        dk = t_ * sigma2[None, :] + 2 * self.a * (1 - t_)     # (N, K)

        # Posterior weights: tilde_w_k ∝ w_k * N(x_t; t*mu_k, t*dk*I)
        marg_mean = t_[:, :, None] * means[None, :, :]         # (N, K, D)
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)     # (N, K)
        t_safe = t_ + eps
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )                                                        # (N, K)

        log_w = torch.log(weights)[None, :]                     # (1, K)
        log_pw = log_w + log_gauss                              # (N, K)
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)

        # Posterior parameters
        tV  = 2 * self.a * (1 - t_) * sigma2[None, :] / dk    # (N, K)
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]                                      # (N, K, D)

        log_zk = self._log_Z(tmu, tV)                          # (N, K)

        V = torch.logsumexp(log_pw + log_zk, dim=1)            # (N,)
        return V


# ---------------------------------------------------------------------------
# Base drift (GMM score)
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

_sigmas  = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(xt, ts, _means.to(xt), _sigmas.to(xt), _weights_col.to(xt), a)
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")

base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)

reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)

# ---------------------------------------------------------------------------
# Sanity-check 1: V(0, 0) from analytical module
# ---------------------------------------------------------------------------
anal_vm = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)

with torch.no_grad():
    x0_test = torch.zeros(1, D)
    t0_test = torch.zeros(1)
    v00 = anal_vm(x0_test, t0_test).item()
print(f"\nAnalyticalValue(0, 0) = {v00:.6f}   (target {V_0_0:.6f}   Δ={v00-V_0_0:.2e})")

# ---------------------------------------------------------------------------
# Sanity-check 2: use OnPolicyValue.drift with analytical V, run SDE,
# check that rewards match E_opt
# ---------------------------------------------------------------------------
DEVICE = "cpu"
anal_vm = anal_vm.to(DEVICE)
N_SAMPLES = 2048
n_steps_list = [20, 50, 100, 200]

N_SAMPLES = 2048
n_steps_list = [20, 50, 100, 200]

def make_drift(coeff_fn):
    def d(x, t):
        with torch.inference_mode(False):
            x_clone = x.clone()
            x_clone.requires_grad_(True)
            value = anal_vm(x_clone, t).sum()
            value.backward()
            value_grad = x_clone.grad
        base_score = base_drift(x, t)
        return base_score + coeff_fn(t) * value_grad
    return d

def run(drift_fn, label):
    print(f"\n--- {label} ---")
    print(f"{'n_steps':>8}  {'E[r(x1)]':>12}  {'gap':>10}  {'std':>8}")
    print("-" * 46)
    for n_steps in n_steps_list:
        x0 = torch.zeros(N_SAMPLES, D, device=DEVICE)
        x_final = integrate_sde(x0, drift=drift_fn, a=a, n_steps=n_steps)
        rewards = reward(x_final)
        mean_r = rewards.mean().item()
        print(f"{n_steps:>8}  {mean_r:>12.4f}  {E_OPT - mean_r:>10.4f}  {rewards.std().item():>8.4f}")

# Base-only (no guidance) – should be far from E_opt
run(lambda x, t: base_drift(x, t), "base only (no guidance)")

# sigma(t) * grad V  — currently in on_policy.py
drift_s1 = make_drift(lambda t: (2 * a * t * (1 - t)).sqrt())
run(drift_s1, "sigma(t) * grad V  [current on_policy.py]")

# sigma(t)^2 * grad V — time-varying Doob
drift_s2 = make_drift(lambda t: 2 * a * t * (1 - t))
run(drift_s2, "sigma(t)^2 * grad V  [time-varying Doob]")

# 2a * grad V — constant-noise Doob (matches integrate_sde noise)
drift_2a = make_drift(lambda t: torch.tensor(2 * a, dtype=torch.float))
run(drift_2a, "2a * grad V  [constant-noise Doob]")

print(f"\nOptimal target E_opt = {E_OPT:.4f}")
