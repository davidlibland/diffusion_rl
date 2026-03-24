"""
Compute the analytical optimal expected reward for the moons experiment.

Setup:
  - base distribution p(x_1) = GMM(means, sigmas, weights) fitted to moons
  - reward r(x) = -10 * ||x - [1,0]||^2
  - optimal policy samples from q*(x_1) ∝ p(x_1) * exp(r(x_1))

Key quantities:
  V(0,0) = log E_p[exp(r(x_1))]          -- value function at x_0=0, t=0
  E_opt  = E_{q*}[r(x_1)]                -- optimal expected reward (target for val_reward_mean)

For each GMM component k (mean mu_k, spherical var sigma_k^2, weight w_k):

  q* component:
    tau_k^2  = sigma_k^2 / (1 + 20*sigma_k^2)
    nu_k     = (mu_k + 20*sigma_k^2 * c) / (1 + 20*sigma_k^2)

  log Z_k  = -(d/2)*log(1 + 20*sigma_k^2)
             + ||nu_k||^2 / (2*tau_k^2)
             - ||mu_k||^2 / (2*sigma_k^2)
             - 10*||c||^2

  V(0,0) = logsumexp_k( log w_k + log Z_k )

  w_k* = w_k * exp(log Z_k) / exp(V(0,0))   (normalized)

  E_{q*}[r(x_1)] = sum_k w_k* * (-10*(||nu_k - c||^2 + d*tau_k^2))
"""

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reproduce the GMM fit from moons_sweep.py
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical", random_state=42)
clf.fit(X)

means   = torch.from_numpy(clf.means_).double()          # (K, D)
sigma2  = torch.from_numpy(clf.covariances_).double()    # (K,)  spherical variance
weights = torch.from_numpy(clf.weights_).double()        # (K,)

D = 2
K = means.shape[0]
c = torch.tensor([1.0, 0.0], dtype=torch.float64)       # reward centre

# ---------------------------------------------------------------------------
# Twisted GMM parameters
# ---------------------------------------------------------------------------
# tau_k^2 = sigma_k^2 / (1 + 20*sigma_k^2)
tau2 = sigma2 / (1.0 + 20.0 * sigma2)                   # (K,)

# nu_k = (mu_k + 20*sigma_k^2 * c) / (1 + 20*sigma_k^2)
nu = (means + 20.0 * sigma2[:, None] * c[None, :]) / (1.0 + 20.0 * sigma2[:, None])  # (K, D)

# ---------------------------------------------------------------------------
# log Z_k  (log normalizing constant per component)
# ---------------------------------------------------------------------------
log_Zk = (
    -(D / 2.0) * torch.log(1.0 + 20.0 * sigma2)           # shape factor
    + (nu ** 2).sum(dim=1) / (2.0 * tau2)                  # new mean energy
    - (means ** 2).sum(dim=1) / (2.0 * sigma2)             # old mean energy
    - 10.0 * (c ** 2).sum()                                 # reward centre energy
)                                                           # (K,)

# ---------------------------------------------------------------------------
# V(0, 0) = log E_p[exp(r(x_1))]
# ---------------------------------------------------------------------------
log_w  = torch.log(weights)                               # (K,)
log_Z  = torch.logsumexp(log_w + log_Zk, dim=0)          # scalar

print(f"V(0, 0)  =  log E_p[exp(r(x))]  =  {log_Z.item():.6f}")

# ---------------------------------------------------------------------------
# Optimal expected reward  E_{q*}[r(x_1)]
# ---------------------------------------------------------------------------
log_w_star = log_w + log_Zk - log_Z                      # normalised log weights
w_star     = torch.exp(log_w_star)                        # (K,)

# E_{N(nu_k, tau_k^2 I)}[-10||x-c||^2]  =  -10*(||nu_k - c||^2 + D*tau_k^2)
E_r_per_comp = (
    -10.0 * ((nu - c[None, :]) ** 2).sum(dim=1)
    - 10.0 * D * tau2
)                                                          # (K,)
E_opt = (w_star * E_r_per_comp).sum()

print(f"E_opt    =  E_q*[r(x_1)]        =  {E_opt.item():.6f}")

# ---------------------------------------------------------------------------
# Cross-check: empirical estimate from exact samples of q*
# ---------------------------------------------------------------------------
# Sample from q* by importance weighting:
#   1. Sample from each component proportionally to w_k*
#   2. Draw x ~ N(nu_k, tau_k^2 I)
n_samples = 200_000
rng = torch.Generator().manual_seed(42)
component_idx = torch.multinomial(w_star.float(), n_samples, replacement=True, generator=rng)
eps = torch.randn(n_samples, D, dtype=torch.float64, generator=rng)
x_samples = nu[component_idx] + torch.sqrt(tau2[component_idx, None]) * eps

reward = lambda x: -10.0 * ((x - c) ** 2).sum(dim=1)
rewards_empirical = reward(x_samples)
print(f"Empirical check (n={n_samples:,}):  E[r] = {rewards_empirical.mean().item():.6f}  "
      f"±  {rewards_empirical.std().item() / (n_samples ** 0.5):.6f}")

# ---------------------------------------------------------------------------
# Also report the empirical estimate that the script computes at startup
# (the 'bias' variable) vs our analytical V(0,0)
# ---------------------------------------------------------------------------
all_rewards_data = reward(torch.from_numpy(X).double())
max_r = all_rewards_data.max()
bias_empirical = (torch.log(torch.mean(torch.exp(all_rewards_data - max_r))) + max_r).item()
print(f"\nbias (empirical from 10k data points): {bias_empirical:.6f}")
print(f"V(0,0) (analytical):                   {log_Z.item():.6f}")

# ---------------------------------------------------------------------------
# Save for use in plotting
# ---------------------------------------------------------------------------
import json, pathlib
out = {
    "V_0_0":  log_Z.item(),
    "E_opt":  E_opt.item(),
}
pathlib.Path("notebooks/analytical_target.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved to notebooks/analytical_target.json: {out}")
