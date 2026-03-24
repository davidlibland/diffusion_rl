"""
Data quality analysis: compare sampling methods using the analytical value function.

For each sampling method (single_seed_mc, single_seed_td_lambda, one_step_bootstrap,
ancestral_td_lambda) we generate batches (x, t, target) in two settings:

  Setting A — Oracle: value=anal_vm, smc_value=anal_vm
    → measures the INTRINSIC variance of each method's targets vs V_true, binned by t
    → shows which method produces lowest-noise estimates at optimality

  Setting B — Best model + Oracle SMC: value=best_ckpt_model, smc_value=anal_vm
    → measures variance AND bias of the targets against V_true
    → shows how model error interacts with each method's target construction

Results are printed in tables and saved to notebooks/data_quality_results.json.
"""

import json

import numpy as np
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# ---------------------------------------------------------------------------
# GMM setup (same as all other scripts)
# ---------------------------------------------------------------------------
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler()
X = scalar.fit_transform(X)

clf = GaussianMixture(n_components=100, covariance_type="spherical")
clf.fit(X)

_means   = torch.from_numpy(clf.means_).double()
_sigma2  = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas  = torch.sqrt(_sigma2)[:, None]
_weights_col = _weights[:, None]

D = 2
a = 1.0
c = torch.tensor([1.0, 0.0])

# ---------------------------------------------------------------------------
# Analytical value function
# ---------------------------------------------------------------------------
class AnalyticalValue(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None:
            c = torch.tensor([1.0, 0.0])
        self.register_buffer("means",   means.float())
        self.register_buffer("sigma2",  sigma2.float())
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
        t = t.double().reshape(-1)   # handles scalar, (N,), and (N,1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        t_ = t[:, None]
        means   = self.means.double()
        sigma2  = self.sigma2.double()
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
        tV  = 2 * self.a * (1 - t_) * sigma2[None, :] / dk
        tmu = (
            sigma2[None, :, None] * x[:, None, :]
            + 2 * self.a * (1 - t_)[:, :, None] * means[None, :, :]
        ) / dk[:, :, None]
        log_zk = self._log_Z(tmu, tV)
        V = torch.logsumexp(log_pw + log_zk, dim=1)
        return V.float()


anal_vm = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D)

# ---------------------------------------------------------------------------
# GMM drift
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


def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1, 1)
    cond = get_conditional_mixture(
        xt, ts,
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "cpu"   # analytical V needs cpu (float64)
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - c.to(x)).square().sum(dim=1)

# Analytical value callable (cpu, handles float32 input)
anal_vm = anal_vm.to(DEVICE)
anal_fn = lambda x, t: anal_vm(x.cpu(), t.cpu())

E_OPT = json.loads(open("notebooks/analytical_target.json").read())["E_opt"]

# ---------------------------------------------------------------------------
# Load best on-policy checkpoint (TD λ=0.6)
# ---------------------------------------------------------------------------
CKPT_PATH = "checkpoints/convergence_run/single_seed_td_lam0.6/best.ckpt"
all_rewards_np = reward(torch.from_numpy(X).float())
max_r = all_rewards_np.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards_np - max_r))) + max_r).item()

vm_best = ValueNetwork(D, bias=bias_val)
dummy_drift = lambda x, t: torch.zeros_like(x)
_tmp_model = OnPolicyValue(base_score_module=dummy_drift, value_module=vm_best,
                            dim=D, a=a, lr=1e-3, loss_type="quad")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
_tmp_model.load_state_dict(ckpt["state_dict"])
vm_best = _tmp_model.value_module.eval()
vm_best_fn = lambda x, t: vm_best(x.cpu(), t.cpu())

print(f"Loaded best TD(λ=0.6) checkpoint from {CKPT_PATH}")

# ---------------------------------------------------------------------------
# Collect (x, t, target) batches from a dataset
# ---------------------------------------------------------------------------
N_BATCHES = 20
BATCH_SIZE = 512

T_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
BIN_NAMES = ["t00_20", "t20_40", "t40_60", "t60_80", "t80_100"]


def collect_batches(sampling_method, value_fn, smc_value_fn, lambda_eff=0.6):
    """Run N_BATCHES through OnPolicySMCDataset, return (x, t, target) stacked."""
    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=value_fn, smc_value=smc_value_fn,
        reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
        sampling_method=sampling_method, lambda_eff=lambda_eff,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    xs, ts, tgts = [], [], []
    for i, (y, x, t) in enumerate(loader):
        xs.append(x); ts.append(t.flatten()); tgts.append(y.flatten())
        if i + 1 >= N_BATCHES:
            break
    return torch.cat(xs), torch.cat(ts), torch.cat(tgts)


def binned_stats(x, t, target, value_fn, label):
    """Compute per-bin variance and bias of (target - V_analytical(x,t))."""
    with torch.no_grad():
        v_anal = anal_fn(x, t)
    err = target - v_anal
    stats = {}
    print(f"\n  {label}")
    print(f"  {'bin':>10}  {'N':>6}  {'mean(err)':>12}  {'std(err)':>10}  {'var(err)':>10}")
    print(f"  {'-'*55}")
    for name, (lo, hi) in zip(BIN_NAMES, T_BINS):
        mask = (t >= lo) & (t < hi)
        n = mask.sum().item()
        if n > 1:
            e = err[mask]
            stats[name] = {"n": n, "mean": e.mean().item(), "std": e.std().item(),
                           "var": e.var().item()}
            print(f"  {name:>10}  {n:>6}  {e.mean().item():>12.4f}  {e.std().item():>10.4f}  {e.var().item():>10.4f}")
        else:
            stats[name] = {"n": n, "mean": float("nan"), "std": float("nan"), "var": float("nan")}
            print(f"  {name:>10}  {n:>6}  {'—':>12}  {'—':>10}  {'—':>10}")
    return stats


# ---------------------------------------------------------------------------
# SETTING A: Oracle (value=anal, smc_value=anal)
# ---------------------------------------------------------------------------
METHODS = {
    "single_seed_mc":       dict(sampling_method="single_seed_mc",         lambda_eff=0.6),
    "single_seed_td_lam06": dict(sampling_method="single_seed_td_lambda",  lambda_eff=0.6),
    "single_seed_td_lam02": dict(sampling_method="single_seed_td_lambda",  lambda_eff=0.2),
    "one_step_bootstrap":   dict(sampling_method="one_step_bootstrap",     lambda_eff=0.6),
}

print(f"\n{'='*70}")
print("SETTING A: Oracle  (value=analytical, smc_value=analytical)")
print(f"{'='*70}")
print("Measures INTRINSIC variance of each method's targets at optimality.\n")

results_A = {}
for method_name, cfg in METHODS.items():
    print(f"\n--- {method_name} ---")
    try:
        x, t, tgt = collect_batches(value_fn=anal_fn, smc_value_fn=anal_fn, **cfg)
        results_A[method_name] = binned_stats(x, t, tgt, anal_fn, "target - V_anal")
    except Exception as e:
        print(f"  ERROR: {e}")
        results_A[method_name] = {}


# ---------------------------------------------------------------------------
# SETTING B: Best model + Oracle SMC (value=best_model, smc_value=anal)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SETTING B: Best model value + Oracle SMC reweighting")
print(f"{'='*70}")
print("value=best_TD_checkpoint  smc_value=analytical")
print("Measures variance AND bias when value function is imperfect.\n")

results_B = {}
for method_name, cfg in METHODS.items():
    print(f"\n--- {method_name} ---")
    try:
        x, t, tgt = collect_batches(value_fn=vm_best_fn, smc_value_fn=anal_fn, **cfg)
        results_B[method_name] = binned_stats(x, t, tgt, anal_fn, "target - V_anal")
    except Exception as e:
        print(f"  ERROR: {e}")
        results_B[method_name] = {}


# ---------------------------------------------------------------------------
# Summary comparison: total variance across all bins
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMMARY: Total variance (oracle setting)")
print(f"{'='*70}")
print(f"{'method':>30}  {'total_var (A)':>14}  {'total_var (B)':>14}")
print("-" * 65)
for m in METHODS:
    def total_var(stats):
        vals = [v["var"] for v in stats.values() if not np.isnan(v.get("var", np.nan))]
        return np.mean(vals) if vals else float("nan")
    vA = total_var(results_A.get(m, {}))
    vB = total_var(results_B.get(m, {}))
    print(f"  {m:>30}  {vA:>14.4f}  {vB:>14.4f}")

# Save
output = {"setting_A": results_A, "setting_B": results_B}
with open("notebooks/data_quality_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to notebooks/data_quality_results.json")
