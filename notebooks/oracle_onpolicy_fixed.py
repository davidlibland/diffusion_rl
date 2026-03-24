"""
Oracle on-policy: fixed 5000-step run at best lr=1e-2.

Shorter version of oracle_onpolicy.py that doesn't try to fit convergence.
LR was chosen from sweep (lr=1e-2 was best at 2 min).
"""

import json
import torch
import torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

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
        t = t.double().reshape(-1)
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


_anal_vm_cpu = AnalyticalValue(_means, _sigma2, _weights, a=a, c=c, D=D).cpu()


def anal_fn(x, t):
    result = _anal_vm_cpu(x.cpu(), t.cpu())
    return result.to(x.device)


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
        _means.float().to(xt), _sigmas.float().to(xt), _weights_col.float().to(xt), a,
    )
    new_weights = torch.exp(cond["log_weights"])
    us = (cond["means"] - xt[:, :, None]) / (1 - ts[..., None])
    return reduce(new_weights[:, None, :] * us, "n d m -> n d", "sum")


DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim >= 1 else t.unsqueeze(0), a).to(dtype=torch.float)
reward = lambda x: -10 * (x - torch.tensor([[1.0, 0.0]]).to(x)).square().sum(dim=1)

all_rewards = reward(torch.from_numpy(X).float())
max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards - max_r))) + max_r).item()

E_OPT = json.loads(open("notebooks/analytical_target.json").read())["E_opt"]
LOG_DIR = "lightning_logs/oracle_onpolicy_fixed"
LAMBDA = 0.2
BATCH_SIZE = 256
MAX_STEPS = 5000
LR = 1e-2
VAL_INTERVAL = 100

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

vm = ValueNetwork(D, bias=bias_val)
ds = OnPolicySMCDataset(
    dim=D, drift=base_drift, value=anal_fn, smc_value=anal_fn,
    reward=reward, device=DEVICE, a=a, batch_size=32, n_steps=100,
    sampling_method="single_seed_td_lambda", lambda_eff=LAMBDA,
)
loader = DataLoader(ds, batch_size=BATCH_SIZE)
model = OnPolicyValue(
    base_score_module=base_drift, value_module=vm,
    reward_function=reward, dim=D, a=a, lr=LR,
    loss_type="quad", analytical_value_fn=anal_fn,
)

ckpt_cb = ModelCheckpoint(
    dirpath="checkpoints/oracle_onpolicy_fixed",
    save_last=True, every_n_train_steps=1000,
    save_top_k=1, monitor="val_reward_mean", mode="max", filename="best",
)
logger = CSVLogger(LOG_DIR, name="oracle_onpolicy", version=0)
trainer = L.Trainer(
    max_steps=MAX_STEPS,
    val_check_interval=VAL_INTERVAL,
    callbacks=[ckpt_cb],
    logger=logger,
    enable_checkpointing=True,
    enable_progress_bar=True,
)

print(f"Running oracle on-policy: lr={LR}, max_steps={MAX_STEPS}, lambda={LAMBDA}")
trainer.fit(model, loader, val_dataloaders=val_loader)

import pandas as pd
csv_path = f"{LOG_DIR}/oracle_onpolicy/version_0/metrics.csv"
df = pd.read_csv(csv_path)
val = df.dropna(subset=["val_reward_mean"])
best_r = val["val_reward_mean"].max()
print(f"\nBest: {best_r:.4f}  (gap {E_OPT - best_r:.4f})")
print(f"E_opt: {E_OPT:.4f}")
