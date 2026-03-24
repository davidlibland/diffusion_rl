"""
Compute and visualise the analytical value function V(x_t, t).

V(x_t, t) = log E_{p(x1|xt)} [exp(r(x1))]

where:
  - p(x1) = GMM fitted to moons data
  - p(x1|xt) = posterior GMM (see analytical_value_function.md)
  - r(x1) = -10 * ||x1 - c||^2,  c = [1, 0]

The function is computed analytically using the formulas derived in
analytical_value_function.md.  No sampling required.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Fit the GMM (same as all other scripts)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means   = torch.from_numpy(clf.means_).double()       # (K, D)
sigma2  = torch.from_numpy(clf.covariances_).double() # (K,)  spherical variance
weights = torch.from_numpy(clf.weights_).double()     # (K,)

D   = 2
a   = 1.0
c   = torch.tensor([1.0, 0.0], dtype=torch.float64)
K   = means.shape[0]

# ---------------------------------------------------------------------------
# 2. Core analytical computation
# ---------------------------------------------------------------------------

def log_Z(m, v):
    """
    log integral of N(x; m, v*I) * exp(-10*||x-c||^2) dx

    Numerically stable form (no 1/(2*tau^2) singularity at v->0):
      log Z = -D/2 * log(1 + 20v)
              + (-10*||m||^2 + 20*(m·c) + 200*v*||c||^2) / (1 + 20v)
              - 10*||c||^2

    Args:
        m : (..., D)
        v : (...)      -- scalar variance per point/component
    Returns:
        (...,)
    """
    c_ = c.to(m)
    denom = 1.0 + 20.0 * v                          # (...)
    return (
        -D / 2.0 * torch.log(denom)
        + (-10.0 * (m ** 2).sum(-1)
           + 20.0 * (m * c_).sum(-1)
           + 200.0 * v * (c_ ** 2).sum()) / denom
        - 10.0 * (c_ ** 2).sum()
    )


def analytical_value(xt, t):
    """
    Compute V(x_t, t) analytically for a batch of (x_t, t) pairs.

    Key reparametrisation for numerical stability:
        d_k(t) = t * sigma_k^2 + 2a*(1-t)          -- always > 0
        s_k^2  = t * d_k(t)                          -- marginal variance
        tV_k   = 2a*(1-t)*sigma_k^2 / d_k            -- posterior variance
        tmu_k  = (sigma_k^2 * x_t + 2a*(1-t)*mu_k) / d_k  -- posterior mean

    These are finite at t=0 (d_k -> 2a) and t=1 (d_k -> sigma_k^2).

    Args:
        xt : (N, D)  positions at time t
        t  : (N,)    times in [0, 1]

    Returns:
        V  : (N,)    value function V(x_t, t)
    """
    xt = xt.double()
    t  = t.double()
    t_ = t[:, None]               # (N, 1)
    eps = 1e-40                   # only used inside log to avoid log(0) at t=0

    # ------------------------------------------------------------------
    # Step 1: d_k = t * sigma_k^2 + 2a*(1-t)   shape (N, K)
    # ------------------------------------------------------------------
    dk = t_ * sigma2[None, :] + 2 * a * (1 - t_)          # (N, K), always > 0

    # ------------------------------------------------------------------
    # Step 2: posterior weights  tilde_w_k ∝ w_k * N(x_t; t*mu_k, s_k^2 I)
    #   s_k^2 = t * d_k
    #   log N(x; m, t*dk) = -D/2*log(2pi*t*dk) - ||x-t*mu_k||^2/(2*t*dk)
    #
    # At t=0 with x_t=0: the Gaussian is a delta at 0 for all k equally,
    # so differences between log-weights come only from log w_k.
    # We write s2 = t*dk and use (t+eps) to protect log.
    # ------------------------------------------------------------------
    marg_mean = t_[:, :, None] * means[None, :, :]        # (N, K, D)
    diff2 = ((xt[:, None, :] - marg_mean) ** 2).sum(-1)   # (N, K)

    t_safe = t_ + eps                                       # (N, 1)
    log_gauss_marg = (
        -D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
        - diff2 / (2 * t_safe * dk)
    )                                                       # (N, K)

    log_w = torch.log(weights)[None, :]                    # (1, K)
    log_post_w_unnorm = log_w + log_gauss_marg             # (N, K)
    log_norm = torch.logsumexp(log_post_w_unnorm, dim=1, keepdim=True)
    log_post_w = log_post_w_unnorm - log_norm              # (N, K)

    # ------------------------------------------------------------------
    # Step 3: posterior parameters (both stable at t=0 and t=1)
    #   tV_k  = 2a*(1-t)*sigma_k^2 / d_k
    #   tmu_k = (sigma_k^2 * x_t + 2a*(1-t)*mu_k) / d_k
    # ------------------------------------------------------------------
    tV = 2 * a * (1 - t_) * sigma2[None, :] / dk          # (N, K)

    tmu = (
        sigma2[None, :, None] * xt[:, None, :]             # (N, K, D)
        + 2 * a * (1 - t_)[:, :, None] * means[None, :, :]# (N, K, D)
    ) / dk[:, :, None]                                     # (N, K, D)

    # ------------------------------------------------------------------
    # Step 4: log Z for each posterior component
    # ------------------------------------------------------------------
    log_zk = log_Z(tmu, tV)                               # (N, K)

    # ------------------------------------------------------------------
    # Step 5: V(x_t, t) = logsumexp_k [ log_post_w_k + log_Z_k ]
    # ------------------------------------------------------------------
    V = torch.logsumexp(log_post_w + log_zk, dim=1)       # (N,)
    return V.float()


# ---------------------------------------------------------------------------
# 3. Verify boundary conditions
# ---------------------------------------------------------------------------
print("=== Boundary checks ===")

# V(0, 0) should equal V_0_0 = -5.085
x0  = torch.zeros(1, D, dtype=torch.float64)
t0  = torch.tensor([0.0], dtype=torch.float64)
v00 = analytical_value(x0, t0).item()
ref = json.loads(open("notebooks/analytical_target.json").read())["V_0_0"]
print(f"V(0, 0) analytical: {v00:.6f}   reference: {ref:.6f}   Δ={v00-ref:.2e}")

# V(x, 1) should equal r(x) = -10*||x-c||^2
test_pts = torch.tensor([[1.0, 0.0], [0.0, 0.0], [-1.0, 1.0]], dtype=torch.float64)
t1  = torch.ones(3, dtype=torch.float64)
vt1 = analytical_value(test_pts, t1)
rt1 = -10.0 * ((test_pts - c.double()) ** 2).sum(-1).float()
for i, (x, vv, rr) in enumerate(zip(test_pts, vt1, rt1)):
    print(f"V({x.tolist()}, 1) = {vv:.4f}   r(x) = {rr:.4f}   Δ={vv-rr:.2e}")


# ---------------------------------------------------------------------------
# 4. Visualise V(x_t, t) at several times as a 2-D heatmap
# ---------------------------------------------------------------------------
grid_n = 200
xs = np.linspace(-3, 3, grid_n)
ys = np.linspace(-3, 3, grid_n)
XX, YY = np.meshgrid(xs, ys)
grid = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1)).float()

times_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]
n_t = len(times_to_plot)

fig, axes = plt.subplots(2, n_t, figsize=(4 * n_t, 8))
fig.suptitle("Analytical value function  V(x_t, t)", fontsize=14)

with torch.no_grad():
    for col, tv in enumerate(times_to_plot):
        t_batch = torch.full((grid_n ** 2,), tv)
        V_grid  = analytical_value(grid, t_batch).reshape(grid_n, grid_n).numpy()

        # Top row: V
        ax = axes[0, col]
        im = ax.contourf(XX, YY, V_grid, levels=50, cmap="RdYlGn")
        ax.set_title(f"t = {tv:.2f}")
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        plt.colorbar(im, ax=ax)

        # Bottom row: exp(V) = E[exp(r(x1))|xt]  (proportional to optimal policy density)
        ax2 = axes[1, col]
        im2 = ax2.contourf(XX, YY, np.exp(V_grid), levels=50, cmap="hot")
        ax2.set_title(f"exp(V),  t = {tv:.2f}")
        ax2.set_aspect("equal")
        ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
        plt.colorbar(im2, ax=ax2)

axes[0, 0].set_ylabel("$V(x_t, t)$")
axes[1, 0].set_ylabel("$\\exp V(x_t, t)$")

fig.tight_layout()
fig.savefig("notebooks/analytical_value_heatmaps.png", dpi=150)
print("\nSaved analytical_value_heatmaps.png")


# ---------------------------------------------------------------------------
# 5. Compare network predictions vs analytical V at t = 0, 0.25, 0.5
# ---------------------------------------------------------------------------
from diffusion_rl.modules.resnet_mlp import ValueNetwork
from diffusion_rl.models.on_policy import OnPolicyValue

all_rewards_np = -10 * ((torch.from_numpy(X).float() - c.float()) ** 2).sum(-1)
max_r = all_rewards_np.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards_np - max_r))) + max_r).item()

ckpt_path = "checkpoints/long_run/single_seed_td_lam0.6/last.ckpt"
vm = ValueNetwork(D, bias=bias_val)

dummy_drift = lambda x, t: torch.zeros_like(x)
model = OnPolicyValue(base_score_module=dummy_drift, value_module=vm,
                      dim=D, a=a, lr=1e-2, loss_type="quad")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
vm = model.value_module

fig2, axes2 = plt.subplots(3, n_t, figsize=(4 * n_t, 10))
fig2.suptitle("Analytical  vs  Network predictions", fontsize=14)

vmin_global, vmax_global = -30, 0

with torch.no_grad():
    for col, tv in enumerate(times_to_plot):
        t_batch  = torch.full((grid_n ** 2,), tv)
        V_true   = analytical_value(grid, t_batch).reshape(grid_n, grid_n).numpy()

        t_net    = t_batch.float()
        V_net    = vm(grid, t_net).reshape(grid_n, grid_n).numpy()
        V_err    = V_net - V_true

        kw = dict(levels=np.linspace(vmin_global, vmax_global, 51), cmap="RdYlGn",
                  vmin=vmin_global, vmax=vmax_global)

        ax0 = axes2[0, col]
        im0 = ax0.contourf(XX, YY, V_true, **kw)
        ax0.set_title(f"Analytical  t={tv:.2f}")
        ax0.set_aspect("equal")
        plt.colorbar(im0, ax=ax0)

        ax1 = axes2[1, col]
        im1 = ax1.contourf(XX, YY, V_net, **kw)
        ax1.set_title(f"Network  t={tv:.2f}")
        ax1.set_aspect("equal")
        plt.colorbar(im1, ax=ax1)

        err_abs = np.abs(V_err)
        ax2 = axes2[2, col]
        im2 = ax2.contourf(XX, YY, err_abs,
                           levels=np.linspace(0, err_abs.max(), 51), cmap="Reds")
        ax2.set_title(f"|Error|  t={tv:.2f}\nmax={err_abs.max():.2f} mean={err_abs.mean():.2f}")
        ax2.set_aspect("equal")
        plt.colorbar(im2, ax=ax2)

for ax in axes2.flat:
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
axes2[0, 0].set_ylabel("Analytical $V$")
axes2[1, 0].set_ylabel("Network $V$")
axes2[2, 0].set_ylabel("$|V_{net} - V_{true}|$")

fig2.tight_layout()
fig2.savefig("notebooks/analytical_vs_network.png", dpi=150)
print("Saved analytical_vs_network.png")

plt.show()
