"""
Shared setup for the 2026-06-10 FBRRT data-quality re-run.

Faithful copy of the GMM / analytical-value / drift / reward / metric code from
the original `data_quality_v2.py` (now archived under ../2026-05-28/), trimmed to
exactly what the FBRRT comparison needs.  Keeping it identical means the
bias/variance numbers are directly comparable to the archived run.

  - same moons->GMM (100 spherical comps, random_state=42)
  - same analytical V(x,t) = log E[exp r(X_1) | X_t = x],  reward r = -10||x-c||^2
  - same base diffusion drift (gmm_drift), a=1, D=2
  - same per-t-bin metric: bias = mean(target - V_anal), var = var(target - V_anal)
"""

import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from diffusion_rl.models.off_policy import OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM setup (identical to data_quality_v2.py)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical", random_state=0)
clf.fit(X)

_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Analytical value function V(x,t) = log E[exp r(X_1) | X_t = x]
# ---------------------------------------------------------------------------
class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float())
        self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float())
        self.a = a
        self.D = D
        self.register_buffer("c", c.float())

    def _log_Z(self, m, v):
        c = self.c.double()
        denom = 1.0 + 20.0 * v
        return (
            -self.D / 2.0 * torch.log(denom)
            + (-10.0 * (m**2).sum(-1) + 20.0 * (m * c).sum(-1)
               + 200.0 * v * (c**2).sum()) / denom
            - 10.0 * (c**2).sum()
        )

    def forward(self, x, t):
        x = x.double()
        t = t.double().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        means = self.means.double()
        sigma2 = self.sigma2.double()
        weights = self.weights.double()
        eps = 1e-40
        dk = t_ * sigma2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * means[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + eps
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(weights)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * sigma2[None, :] / dk
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        return torch.logsumexp(log_pw + log_zk, dim=1).float()


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm_cpu(x.cpu(), t.cpu()).to(x.device)


# ---------------------------------------------------------------------------
# Base diffusion drift + reward
# ---------------------------------------------------------------------------
def get_conditional_mixture(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape
    xt_ = xt[..., None]
    means_ = means.T[None, ...]
    ts_ = ts[..., None]
    sigmas_ = sigmas.T
    weights_ = weights.T
    orig_log_weights = torch.log(weights_)
    denominator = 2 * a * (1 - ts) + ts * sigmas_**2
    likelihood_exp_numerator = reduce((xt_ - means_ * ts_) ** 2, "n d m -> n m", "sum")
    likelihood_exp = -likelihood_exp_numerator / (2 * ts * denominator)
    log_std_factor = torch.log(2 * a * (1 - ts) / denominator) * d / 2
    log_rel_weights = orig_log_weights + likelihood_exp + log_std_factor
    normalization = torch.logsumexp(log_rel_weights, dim=1, keepdim=True)
    log_weights = log_rel_weights - normalization
    log_weights = torch.where((ts == 0), orig_log_weights, log_weights)
    std_factor = torch.exp(log_std_factor)
    new_means = (
        2 * a * (1 - ts_) * means_ + xt_ * sigmas_[None, ...] ** 2
    ) / denominator[:, None, :]
    new_sigmas = torch.sqrt(0.5 * std_factor * sigmas_**2)
    return {"log_weights": log_weights, "means": new_means, "sigmas": new_sigmas}


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt, ts, _means.float().to(xt), _sigmas.float().to(xt),
        _weights_col.float().to(xt), a,
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


def base_drift(x, t):
    return gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)


def reward(x):
    return -10 * (x - c.to(x)).square().sum(dim=1)


# ---------------------------------------------------------------------------
# Checkpoint loading (model value functions)
# ---------------------------------------------------------------------------
_all_rewards = reward(torch.from_numpy(X).float())
_max_r = _all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(_all_rewards - _max_r))) + _max_r).item()


def load_value_fn(ckpt_path, model_class, hidden_dim=64, num_blocks=4):
    # The dim_scaling_bs4 D=2 checkpoints used hidden_dim=64, num_blocks=4
    # (the default ValueNetwork is hidden_dim=256).
    vm = ValueNetwork(D, hidden_dim=hidden_dim, num_blocks=num_blocks, bias=bias_val)
    dummy_drift = lambda x, t: torch.zeros_like(x)
    if model_class == "on_policy":
        tmp = OnPolicyValue(
            base_score_module=dummy_drift, value_module=vm,
            dim=D, a=a, lr=1e-3, loss_type="quad",
        )
    else:
        tmp = OffPolicyValue(
            base_score_module=dummy_drift, reward_function=reward,
            value_module=vm, a=a, lr=1e-3, dim=D,
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tmp.load_state_dict(ckpt["state_dict"])
    vm_loaded = tmp.value_module.eval()
    return lambda x, t: vm_loaded(x.cpu(), t.cpu().flatten())


# ---------------------------------------------------------------------------
# Per-t-bin bias/variance metric (identical to data_quality_v2.py)
# ---------------------------------------------------------------------------
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_MIDS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_PER_BIN = 1000


def binned_stats(x, t, target, n_per_bin=N_PER_BIN, seed=42):
    """bin_name -> {n, mean, std, var} for err = target - V_anal(x,t)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        v_anal = anal_fn(x, t)
    err = target - v_anal
    finite = torch.isfinite(err)
    stats = {}
    for name, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (t >= lo) & (t < hi) & finite
        n_avail = int(mask.sum().item())
        if n_avail < 2:
            stats[name] = {"n": n_avail, "mean": float("nan"),
                           "std": float("nan"), "var": float("nan")}
            continue
        e = err[mask]
        if n_avail > n_per_bin:
            idx = torch.randperm(n_avail, generator=rng)[:n_per_bin]
            e = e[idx]
        stats[name] = {
            "n": len(e),
            "mean": e.mean().item(),
            "std": e.std().item(),
            "var": e.var().item(),
            "n_nonfinite": int((~torch.isfinite(target[(t >= lo) & (t < hi)])).sum()),
        }
    return stats


def avg_stats(stats):
    vars_ = [v["var"] for v in stats.values() if not np.isnan(v["var"])]
    biases = [abs(v["mean"]) for v in stats.values() if not np.isnan(v["mean"])]
    avg_var = float(np.mean(vars_)) if vars_ else float("nan")
    avg_bias = float(np.mean(biases)) if biases else float("nan")
    return avg_var, avg_bias
