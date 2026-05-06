"""Quick comparison: quad vs mse loss for off-policy on 2D moons."""

import gc, json, os, shutil, time
from functools import partial
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch, torch.nn as nn
from einops import reduce
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler(); X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]; _weights_col = _weights[:, None]
D = 2; a = 1.0; _c = torch.tensor([1.0, 0.0])
means_np = clf.means_; sigmas_np = np.sqrt(clf.covariances_); weights_np = clf.weights_

class AV(nn.Module):
    def __init__(self, means, sigma2, weights, a=1.0, c=None, D=2):
        super().__init__()
        if c is None: c = torch.tensor([1.0, 0.0])
        self.register_buffer("means", means.float()); self.register_buffer("sigma2", sigma2.float())
        self.register_buffer("weights", weights.float()); self.a = a; self.D = D
        self.register_buffer("c", c.float())
    def _log_Z(self, m, v):
        cc = self.c.double(); d = 1+20*v
        return (-self.D/2*torch.log(d)+(-10*(m**2).sum(-1)+20*(m*cc).sum(-1)+200*v*(cc**2).sum())/d-10*(cc**2).sum())
    def forward(self, x, t):
        x = x.double(); t = t.double().reshape(-1)
        if t.numel() == 1: t = t.expand(x.shape[0])
        t_ = t[:, None]; m = self.means.double(); s2 = self.sigma2.double(); w = self.weights.double()
        dk = t_*s2[None,:]+2*self.a*(1-t_); mm = t_[:,:,None]*m[None,:,:]
        d2 = ((x[:,None,:]-mm)**2).sum(-1); ts = t_+1e-40
        lg = (-self.D/2*torch.log(2*torch.pi*ts*dk)-d2/(2*ts*dk))
        lw = torch.log(w)[None,:]; lpw = lw+lg-torch.logsumexp(lw+lg,dim=1,keepdim=True)
        tV = 2*self.a*(1-t_)*s2[None,:]/dk
        tmu = (s2[None,:,None]*x[:,None,:]+2*self.a*(1-t_)[:,:,None]*m[None,:,:])/dk[:,:,None]
        return torch.logsumexp(lpw+self._log_Z(tmu,tV),dim=1).float()

_anal_vm = AV(_means, _sigma2, _weights, a=a, c=_c, D=D)
def anal_fn(x, t): return _anal_vm(x.cpu(), t.cpu()).to(x.device)

def get_cond_mix(xt, ts, means, sigmas, weights, a):
    n, d = xt.shape; xt_ = xt[...,None]; means_ = means.T[None,...]; ts_ = ts[...,None]
    sigmas_ = sigmas.T; weights_ = weights.T; olw = torch.log(weights_)
    denom = 2*a*(1-ts)+ts*sigmas_**2
    le = -reduce((xt_-means_*ts_)**2,"n d m -> n m","sum")/(2*ts*denom)
    lsf = torch.log(2*a*(1-ts)/denom)*d/2; lrw = olw+le+lsf
    lw = lrw-torch.logsumexp(lrw,dim=1,keepdim=True); lw = torch.where((ts==0),olw,lw)
    nm = (2*a*(1-ts_)*means_+xt_*sigmas_[None,...]**2)/denom[:,None,:]
    return {"log_weights": lw, "means": nm}
def gmm_drift(xt, ts, a):
    ts = ts.reshape(-1,1)
    cond = get_cond_mix(xt,ts,_means.float().to(xt),_sigmas.float().to(xt),_weights_col.float().to(xt),a)
    us = (cond["means"]-xt[:,:,None])/(1-ts[...,None])
    return reduce(torch.exp(cond["log_weights"])[:,None,:]*us,"n d m -> n d","sum")

base_drift = lambda x, t: gmm_drift(x, t if t.ndim>=1 else t.unsqueeze(0), a).to(dtype=torch.float)
_rc = _c.clone()
def reward_fn(x): return -10*(x-_rc.to(x)).square().sum(dim=1)
def gmm_sample(n):
    k = np.random.choice(len(weights_np),size=n,p=weights_np)
    return means_np[k]+sigmas_np[k,np.newaxis]*np.random.randn(n,D)

with open("notebooks/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float()); max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards-max_r)))+max_r).item()

TOTAL_STEPS = 3000; BS = 256; LR = 3e-3
LOG_DIR = "lightning_logs/loss_cmp"
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

class TrajCB(Callback):
    def __init__(self, af, n=256, ns=100): super().__init__(); self.af=af; self.n=n; self.ns=ns
    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi>0: return
        dev=pl.device; n=self.n; dt=1.0/self.ns
        for beta, label in [(0,"base"),(1,"guided")]:
            x=torch.zeros(n,D,device=dev); ax,at=[x],[torch.zeros(n,device=dev)]
            dfn=partial(pl.drift,beta=beta)
            for st in torch.linspace(0,1,self.ns+1,device=dev)[:-1]:
                tv=st.expand(n); dx=dfn(x,tv)*dt; db=sqrt(2*a*dt)*torch.randn_like(x)
                x=x+dx+db; ax.append(x); at.append(torch.full((n,),float(st)+dt,device=dev))
            ax=torch.cat(ax); at=torch.cat(at)
            with torch.no_grad(): vp=pl.value_module(ax,at); va=self.af(ax,at)
            err=vp-va
            pl.log(f"traj_avg_mae_{label}",err.abs().mean(),prog_bar=False)

for loss_type in ["quad", "mse"]:
    name = f"offpol_{loss_type}"
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)
    vm = ValueNetwork(D, bias=bias_val)
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=BS)
    model = OffPolicyValue(base_score_module=base_drift, reward_function=reward_fn,
        value_module=vm, dim=D, a=a, lr=LR, loss_type=loss_type, analytical_value_fn=anal_fn)
    traj_cb = TrajCB(anal_fn)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS//40),
        callbacks=[traj_cb], logger=logger, enable_checkpointing=False, enable_progress_bar=True)
    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Elapsed: {(time.perf_counter()-t0)/60:.1f} min")
    del trainer, model, vm, ds, loader; gc.collect()
    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()

# Results & Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Off-Policy Loss Comparison: Quad vs MSE (2D moons)", fontsize=13, fontweight="bold")

styles = {"offpol_quad": ("blue", "-", "Quad (Bregman)"), "offpol_mse": ("orange", "--", "MSE")}

for name, (color, ls, label) in styles.items():
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    val = df.dropna(subset=["val_reward_mean"])
    best = val["val_reward_mean"].max(); final = val["val_reward_mean"].iloc[-1]
    print(f"{label}: best={best:.3f} final={final:.3f}")
    axes[0].plot(val["step"].values, val["val_reward_mean"].values, color=color, ls=ls, lw=2, label=label)
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) > 0:
        axes[1].plot(sub["step"].values, sub["traj_avg_mae_guided"].values, color=color, ls=ls, lw=2, label=label)

axes[0].axhline(E_OPT, color="red", linestyle=":", linewidth=1, alpha=0.4, label=f"E_opt={E_OPT:.3f}")
axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Avg Terminal Reward")
axes[0].set_title("Terminal Reward"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Avg MAE")
axes[1].set_title("Guided Trajectory MAE"); axes[1].legend()
axes[1].grid(True, alpha=0.3); axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/loss_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/loss_comparison.png")
plt.close()
print("Done.")
