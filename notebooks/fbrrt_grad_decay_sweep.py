"""
Sweep grad_decay on top of the best stable FBRRT config (entropy_lambda=2).
Base: FBRRT λ=0, n_steps=100, EMA=0.999, branch=4, ws=3000, LR=1e-3, ent_λ=2
"""

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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.modules.resnet_mlp import ValueNetwork

# GMM Setup
X, _ = make_moons(n_samples=10_000, noise=0.05, random_state=42)
scalar = StandardScaler(); X = scalar.fit_transform(X)
clf = GaussianMixture(n_components=100, covariance_type="spherical"); clf.fit(X)
_means = torch.from_numpy(clf.means_).double()
_sigma2 = torch.from_numpy(clf.covariances_).double()
_weights = torch.from_numpy(clf.weights_).double()
_sigmas = torch.sqrt(_sigma2)[:, None]; _weights_col = _weights[:, None]
D = 2; a = 1.0
_c = torch.tensor([1.0, 0.0])
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

DEVICE = "mps"
base_drift = lambda x, t: gmm_drift(x, t if t.ndim>=1 else t.unsqueeze(0), a).to(dtype=torch.float)
_rc = _c.clone()
def reward_fn(x): return -10*(x-_rc.to(x)).square().sum(dim=1)
def smc_reward(x, t): return reward_fn(x)
def gmm_sample(n):
    k = np.random.choice(len(weights_np),size=n,p=weights_np)
    return means_np[k]+sigmas_np[k,np.newaxis]*np.random.randn(n,D)

with open("notebooks/analytical_target.json") as f: _at = json.load(f)
E_OPT = _at["E_opt"]
all_rewards = reward_fn(torch.from_numpy(X).float()); max_r = all_rewards.max()
bias_val = (torch.log(torch.mean(torch.exp(all_rewards-max_r)))+max_r).item()
print(f"E_OPT={E_OPT:.4f}")

LOADER_BATCH_SIZE = 256; TOTAL_STEPS = 6000; WS_STEPS = 3000; ON_STEPS = 3000
LOG_DIR = "lightning_logs/fbrrt_gd_sweep"; CKPT_DIR = "checkpoints/fbrrt_gd_sweep"
val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)

class TrajCB(Callback):
    def __init__(self, af, n=256, ns=100): super().__init__(); self.af=af; self.n=n; self.ns=ns
    def on_validation_batch_end(self, trainer, pl, outputs, batch, bi):
        if bi>0: return
        dim=pl.hparams.dim; dev=pl.device; n=self.n; dt=1.0/self.ns
        for beta, label in [(0,"base"),(1,"guided")]:
            x=torch.zeros(n,dim,device=dev); ax,at=[x],[torch.zeros(n,device=dev)]
            dfn=partial(pl.drift,beta=beta)
            for st in torch.linspace(0,1,self.ns+1,device=dev)[:-1]:
                tv=st.expand(n); dx=dfn(x,tv)*dt; db=sqrt(2*pl.a*dt)*torch.randn_like(x)
                x=x+dx+db; ax.append(x); at.append(torch.full((n,),float(st)+dt,device=dev))
            ax=torch.cat(ax); at=torch.cat(at)
            with torch.no_grad(): vp=pl.value_module(ax,at); va=self.af(ax,at)
            err=vp-va
            pl.log(f"traj_avg_mae_{label}",err.abs().mean(),prog_bar=False)
            pl.log(f"traj_avg_bias_{label}",err.mean(),prog_bar=False)

# grad_decay values to sweep
GD_VALUES = [None, 1e-6, 1e-3]

for gd in GD_VALUES:
    gd_label = "none" if gd is None else f"{gd:.0e}"
    name = f"fbrrt_ent2_gd{gd_label}"
    print(f"\n{'='*70}")
    print(f"  {name} (grad_decay={gd})")
    print(f"{'='*70}")

    # Check completion
    csv_check = f"{LOG_DIR}/{name}/version_1/metrics.csv"
    if os.path.exists(csv_check):
        df = pd.read_csv(csv_check)
        val = df.dropna(subset=["val_reward_mean"])
        if len(val) > 0 and val["step"].max() >= ON_STEPS - 1:
            print("  Already complete."); continue

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)

    # Phase 1: off-policy warm-start
    print(f"  Phase 1: Off-policy for {WS_STEPS} steps...")
    ds = InterpolatingNumpyDataset(generating_function=gmm_sample, a=a, batch_size=1024)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    model = OffPolicyValue(base_score_module=base_drift, reward_function=reward_fn,
        value_module=vm, dim=D, a=a, lr=1e-3, loss_type="quad", analytical_value_fn=anal_fn)
    traj_cb = TrajCB(anal_fn)
    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(max_steps=WS_STEPS, val_check_interval=max(1,WS_STEPS//30),
        callbacks=[traj_cb], logger=logger, enable_checkpointing=False, enable_progress_bar=True)
    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Phase 1 done: {(time.perf_counter()-t0)/60:.1f} min")
    del trainer, model, ds, loader; gc.collect()
    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()

    # Phase 2: FBRRT with entropy_lambda=2
    model = OnPolicyValue(base_score_module=base_drift, value_module=vm,
        reward_function=reward_fn, dim=D, a=a, lr=1e-3,
        loss_type="quad", analytical_value_fn=anal_fn,
        ema_decay=0.999, grad_decay=gd)
    ds = OnPolicySMCDataset(dim=D, drift=base_drift, value=model.ema, smc_value=smc_reward,
        reward=reward_fn, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="fbrrt", lambda_eff=0.0,
        branch=4, entropy_lambda=2.0)
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)
    traj_cb = TrajCB(anal_fn)
    ckpt_cb = ModelCheckpoint(dirpath=f"{CKPT_DIR}/{name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best")
    logger = CSVLogger(LOG_DIR, name=name, version=1)
    trainer = L.Trainer(max_steps=ON_STEPS, val_check_interval=max(1,ON_STEPS//60),
        callbacks=[ckpt_cb, traj_cb], logger=logger,
        enable_checkpointing=True, enable_progress_bar=True)
    print(f"  Phase 2: FBRRT (ent_λ=2, gd={gd}) for {ON_STEPS} steps...")
    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Phase 2 done: {(time.perf_counter()-t0)/60:.1f} min")
    del trainer, model, vm, ds, loader; gc.collect()
    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


# Results
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT={E_OPT:.4f})")
print(f"{'='*70}")

print(f"\n  {'Name':<25} {'Best':>8} {'Final':>8} {'Gap':>7} {'B-MAE':>7} {'G-MAE':>7} {'Stable':>7}")
print(f"  {'-'*67}")

run_data = {}
for gd in GD_VALUES:
    gd_label = "none" if gd is None else f"{gd:.0e}"
    name = f"fbrrt_ent2_gd{gd_label}"
    dfs = []
    for v in [0, 1]:
        csv = f"{LOG_DIR}/{name}/version_{v}/metrics.csv"
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            if v == 1: df = df.copy(); df["step"] = df["step"] + WS_STEPS
            dfs.append(df)
    if not dfs: continue
    df = pd.concat(dfs, ignore_index=True)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max(); final = val["val_reward_mean"].iloc[-1]
    gap = E_OPT - best
    bm = df.dropna(subset=["traj_avg_mae_base"]); gm = df.dropna(subset=["traj_avg_mae_guided"])
    bf = bm["traj_avg_mae_base"].iloc[-1] if len(bm) > 0 else float("nan")
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm) > 0 else float("nan")
    stable = "YES" if abs(final-best)<5 else "no"
    print(f"  {name:<25} {best:>8.3f} {final:>8.3f} {gap:>7.3f} {bf:>7.3f} {gf:>7.3f} {stable:>7}")
    run_data[name] = df

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("FBRRT (ent_λ=2) + grad_decay Sweep", fontsize=14, fontweight="bold")
cmap = plt.cm.viridis
for i, gd in enumerate(GD_VALUES):
    gd_label = "none" if gd is None else f"{gd:.0e}"
    name = f"fbrrt_ent2_gd{gd_label}"
    if name not in run_data: continue
    df = run_data[name]
    color = cmap(i / max(1, len(GD_VALUES)-1))

    val = df.dropna(subset=["val_reward_mean"])
    axes[0].plot(val["step"].values, val["val_reward_mean"].values,
                 color=color, linewidth=1.5, label=f"gd={gd_label}")

    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) > 0:
        axes[1].plot(sub["step"].values, sub["traj_avg_mae_guided"].values,
                     color=color, linewidth=1.5, label=f"gd={gd_label}")

axes[0].axhline(E_OPT, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
axes[0].axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Avg Terminal Reward")
axes[0].set_title("Terminal Reward"); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Avg MAE (guided)")
axes[1].set_title("Guided Trajectory MAE"); axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3); axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("notebooks/fbrrt_gd_sweep.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/fbrrt_gd_sweep.png")
plt.close()
print("\nDone.")
