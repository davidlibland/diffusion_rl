"""
Evaluate learned value functions against the analytical V(x_t, t).

Two evaluation distributions for x_1:
  (A) Off-policy: x_1 ~ p_1 (base GMM distribution)
  (B) Reward-tilted: x_1 ~ q*(x_1) ∝ p_1(x_1) * exp(h(x_1))

For each, we noise to x_t = t*x_1 + sqrt(2*a*t*(1-t)) * eps, evaluate
V_model(x_t, t) vs V_analytical(x_t, t), and report per-t-bin statistics.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from diffusion_rl.models.on_policy import OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM Setup
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

means = torch.from_numpy(clf.means_).double()
sigma2 = torch.from_numpy(clf.covariances_).double()
weights = torch.from_numpy(clf.weights_).double()

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0], dtype=torch.float64)
K = means.shape[0]

means_np = clf.means_
sigmas_np = np.sqrt(clf.covariances_)
weights_np = clf.weights_

with open("notebooks/analytical_target.json") as f:
    _anal_tgt = json.load(f)
E_OPT = _anal_tgt["E_opt"]
V_0_0 = _anal_tgt["V_0_0"]

all_rewards = -10 * (torch.from_numpy(X).float() - c.float()).square().sum(dim=1)
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

# ---------------------------------------------------------------------------
# Analytical value function
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
            + (-10.0 * (m ** 2).sum(-1) + 20.0 * (m * c).sum(-1)
               + 200.0 * v * (c ** 2).sum()) / denom
            - 10.0 * (c ** 2).sum()
        )

    def forward(self, x, t):
        x = x.double()
        t = t.double().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        m = self.means.double()
        s2 = self.sigma2.double()
        w = self.weights.double()
        dk = t_ * s2[None, :] + 2 * self.a * (1 - t_)
        marg_mean = t_[:, :, None] * m[None, :, :]
        diff2 = ((x[:, None, :] - marg_mean) ** 2).sum(-1)
        t_safe = t_ + 1e-40
        log_gauss = (
            -self.D / 2.0 * torch.log(2 * torch.pi * t_safe * dk)
            - diff2 / (2 * t_safe * dk)
        )
        log_w = torch.log(w)[None, :]
        log_pw = log_w + log_gauss
        log_pw = log_pw - torch.logsumexp(log_pw, dim=1, keepdim=True)
        tV = 2 * self.a * (1 - t_) * s2[None, :] / dk
        tmu = (
            s2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * m[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        return torch.logsumexp(log_pw + log_zk, dim=1).float()


_anal_vm = AnalyticalValue(means, sigma2, weights, a=a, c=c, D=D)


def anal_fn(x, t):
    return _anal_vm(x, t)


# ---------------------------------------------------------------------------
# Reward-tilted distribution q*(x_1) ∝ p_1(x_1) * exp(h(x_1))
# Parameters: nu_k, tau_k^2, w_k* (from analytical_target.py)
# ---------------------------------------------------------------------------
tau2 = sigma2 / (1.0 + 20.0 * sigma2)
nu = (means + 20.0 * sigma2[:, None] * c[None, :]) / (1.0 + 20.0 * sigma2[:, None])

log_Zk = (
    -(D / 2.0) * torch.log(1.0 + 20.0 * sigma2)
    + (nu ** 2).sum(dim=1) / (2.0 * tau2)
    - (means ** 2).sum(dim=1) / (2.0 * sigma2)
    - 10.0 * (c ** 2).sum()
)
log_w = torch.log(weights)
log_Z = torch.logsumexp(log_w + log_Zk, dim=0)
log_w_star = log_w + log_Zk - log_Z
w_star = torch.exp(log_w_star)

print(f"V(0,0) = {log_Z.item():.4f}, E_opt = {E_OPT:.4f}")


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------
def sample_p1(n, rng=None):
    """Sample x_1 ~ p_1 (base GMM)."""
    k = np.random.choice(K, size=n, p=weights_np)
    x1 = means_np[k] + sigmas_np[k, np.newaxis] * np.random.randn(n, D)
    return torch.from_numpy(x1).float()


def sample_qstar(n, rng=None):
    """Sample x_1 ~ q*(x_1) ∝ p_1(x_1) * exp(h(x_1))."""
    if rng is None:
        rng = torch.Generator().manual_seed(0)
    idx = torch.multinomial(w_star.float(), n, replacement=True, generator=rng)
    eps = torch.randn(n, D, dtype=torch.float64, generator=rng)
    x1 = nu[idx] + torch.sqrt(tau2[idx, None]) * eps
    return x1.float()


def forward_noise(x1, t, a=1.0):
    """Forward noising: x_t = t*x1 + sqrt(2*a*t*(1-t)) * eps."""
    t_ = t.unsqueeze(-1)
    noise_scale = torch.sqrt(2 * a * t_ * (1 - t_))
    eps = torch.randn_like(x1)
    return t_ * x1 + noise_scale * eps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]
BIN_LABELS = ["[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0)"]
N_EVAL = 5000  # samples per evaluation


def load_value_fn(ckpt_path):
    """Load a ValueNetwork from a Lightning checkpoint."""
    vm = ValueNetwork(D, bias=bias_val)
    dummy_drift = lambda x, t: torch.zeros_like(x)
    model = OnPolicyValue(
        base_score_module=dummy_drift, value_module=vm,
        dim=D, a=a, lr=1e-3, loss_type="quad",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.value_module


def evaluate_value_fn(value_fn, x1_sampler, n=N_EVAL, seed=42):
    """
    Evaluate V_model vs V_analytical on forward-noised samples.

    Returns dict of per-bin {mae, bias, rmse, n}.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator().manual_seed(seed)

    x1 = x1_sampler(n, rng=rng)
    t = torch.rand(n)
    xt = forward_noise(x1, t, a=a)

    with torch.no_grad():
        v_pred = value_fn(xt, t)
        v_anal = anal_fn(xt, t)

    err = v_pred - v_anal

    stats = {}
    for name, label, lo, hi in zip(BIN_NAMES, BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (t >= lo) & (t < hi)
        n_bin = mask.sum().item()
        if n_bin < 2:
            stats[name] = {"n": n_bin, "mae": float("nan"), "bias": float("nan"),
                           "rmse": float("nan"), "label": label}
            continue
        e = err[mask]
        stats[name] = {
            "n": n_bin,
            "mae": e.abs().mean().item(),
            "bias": e.mean().item(),
            "rmse": (e ** 2).mean().sqrt().item(),
            "label": label,
        }
    return stats


# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------
MODELS = {
    "offpolicy": "checkpoints/ancestral_sweep/offpolicy/best.ckpt",
    "anc_td_smc_model_lam0.8": "checkpoints/ancestral_sweep/anc_td_smc_model_lam0.8/best.ckpt",
    "anc_td_smc_model_lam0.5": "checkpoints/ancestral_sweep/anc_td_smc_model_lam0.5/best.ckpt",
    "anc_td_smc_reward_lam0.5": "checkpoints/ancestral_sweep/anc_td_smc_reward_lam0.5/best.ckpt",
    "anc_mctd_smc_reward_lam1.0": "checkpoints/ancestral_sweep/anc_mctd_smc_reward_lam1.0/best.ckpt",
    "anc_mctd_smc_model_lam0.8": "checkpoints/ancestral_sweep/anc_mctd_smc_model_lam0.8/best.ckpt",
    "ss_mc_smc_reward": "checkpoints/ancestral_sweep/ss_mc_smc_reward/best.ckpt",
}

# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------
results = {}

for model_name, ckpt_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    vf = load_value_fn(ckpt_path)

    for dist_name, sampler in [("p1_offpolicy", sample_p1), ("qstar_reward", sample_qstar)]:
        stats = evaluate_value_fn(vf, sampler)
        results[(model_name, dist_name)] = stats

        avg_mae = np.mean([s["mae"] for s in stats.values() if not np.isnan(s["mae"])])
        avg_bias = np.mean([s["bias"] for s in stats.values() if not np.isnan(s["bias"])])
        avg_rmse = np.mean([s["rmse"] for s in stats.values() if not np.isnan(s["rmse"])])

        print(f"\n  x1 ~ {dist_name}:  avg_mae={avg_mae:.4f}  avg_bias={avg_bias:.4f}  avg_rmse={avg_rmse:.4f}")
        print(f"    {'bin':<12} {'n':>5} {'MAE':>8} {'bias':>8} {'RMSE':>8}")
        for name in BIN_NAMES:
            s = stats[name]
            print(f"    {s['label']:<12} {s['n']:>5} {s['mae']:>8.4f} {s['bias']:>8.4f} {s['rmse']:>8.4f}")


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------
print(f"\n\n{'='*90}")
print(f"  SUMMARY: Avg MAE by model and evaluation distribution")
print(f"{'='*90}")
print(f"  {'Model':<35} {'p1 (off-policy)':>16} {'q* (reward)':>16}")
print(f"  {'-'*69}")

for model_name in MODELS:
    p1_stats = results[(model_name, "p1_offpolicy")]
    qs_stats = results[(model_name, "qstar_reward")]
    p1_mae = np.mean([s["mae"] for s in p1_stats.values() if not np.isnan(s["mae"])])
    qs_mae = np.mean([s["mae"] for s in qs_stats.values() if not np.isnan(s["mae"])])
    print(f"  {model_name:<35} {p1_mae:>16.4f} {qs_mae:>16.4f}")

print()
print(f"  {'Model':<35} {'p1 (off-policy)':>16} {'q* (reward)':>16}")
print(f"  SUMMARY: Avg BIAS")
print(f"  {'-'*69}")

for model_name in MODELS:
    p1_stats = results[(model_name, "p1_offpolicy")]
    qs_stats = results[(model_name, "qstar_reward")]
    p1_bias = np.mean([s["bias"] for s in p1_stats.values() if not np.isnan(s["bias"])])
    qs_bias = np.mean([s["bias"] for s in qs_stats.values() if not np.isnan(s["bias"])])
    print(f"  {model_name:<35} {p1_bias:>16.4f} {qs_bias:>16.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Value Function Error: V_model - V_analytical (binned by t)", fontsize=14, fontweight="bold")

bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

for col, (dist_name, dist_label) in enumerate([("p1_offpolicy", r"$x_1 \sim p_1$"),
                                                 ("qstar_reward", r"$x_1 \sim q^*$")]):
    # MAE plot
    ax_mae = axes[0, col]
    ax_mae.set_title(f"MAE,  {dist_label}", fontsize=13)

    for model_name in MODELS:
        stats = results[(model_name, dist_name)]
        maes = [stats[b]["mae"] for b in BIN_NAMES]
        ax_mae.plot(bin_centers, maes, "o-", label=model_name, linewidth=1.5, markersize=5)

    ax_mae.set_xlabel("t")
    ax_mae.set_ylabel("MAE")
    ax_mae.legend(fontsize=7, loc="upper left")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.set_yscale("log")

    # Bias plot
    ax_bias = axes[1, col]
    ax_bias.set_title(f"Bias,  {dist_label}", fontsize=13)

    for model_name in MODELS:
        stats = results[(model_name, dist_name)]
        biases = [stats[b]["bias"] for b in BIN_NAMES]
        ax_bias.plot(bin_centers, biases, "o-", label=model_name, linewidth=1.5, markersize=5)

    ax_bias.axhline(0, color="black", linestyle=":", linewidth=0.5)
    ax_bias.set_xlabel("t")
    ax_bias.set_ylabel("Bias (V_model - V_anal)")
    ax_bias.legend(fontsize=7, loc="upper left")
    ax_bias.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/eval_value_functions.png", dpi=150, bbox_inches="tight")
print("\nSaved: notebooks/eval_value_functions.png")


# Per-bin detailed plot
fig2, axes2 = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
fig2.suptitle(r"MAE per t-bin, $x_1 \sim p_1$ (off-policy distribution)", fontsize=14)

model_names = list(MODELS.keys())
x_pos = np.arange(len(model_names))

for i, bname in enumerate(BIN_NAMES):
    ax = axes2[i]
    maes = [results[(m, "p1_offpolicy")][bname]["mae"] for m in model_names]
    bars = ax.bar(x_pos, maes, color=plt.cm.Set2(np.linspace(0, 1, len(model_names))))
    ax.set_title(BIN_LABELS[i], fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("anc_", "a_").replace("smc_", "s") for m in model_names],
                       rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    if i == 0:
        ax.set_ylabel("MAE")

plt.tight_layout()
plt.savefig("notebooks/eval_value_functions_bars.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/eval_value_functions_bars.png")

# Save JSON
out = {}
for (model_name, dist_name), stats in results.items():
    key = f"{model_name}__{dist_name}"
    out[key] = {b: {k: v for k, v in s.items() if k != "label"} for b, s in stats.items()}
with open("notebooks/eval_value_functions.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: notebooks/eval_value_functions.json")
