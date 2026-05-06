"""
FBRRT with adaptive TD-error based off-policy mixing.

The off_policy_frac is set adaptively based on comparing trajectory-level
errors from on-policy vs off-policy:

  traj_mse_on = mean over on-policy trajectories of Σ_t (V(x_t) - V(x_{t+dt}))²
  mse_off = mean over off-policy samples of (V(x_t) - r(x1))²
  off_frac = traj_mse_on / (traj_mse_on + mse_off)

When V is bad: on-policy bootstrap errors are large → more off-policy.
As V improves: on-policy errors shrink → more on-policy.

We start with pure off-policy (frac=1.0) for 1500 steps, then switch to
adaptive mixing. Compare against fixed 50% mix and off-policy baseline.

Best config: FBRRT-TD(λ_s=0.1), ent_λ=2, gd=1e-6, EMA=0.999, LR=1e-3
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

TOTAL_STEPS = 6000; WS_STEPS = 1500; LOADER_BATCH_SIZE = 256
LOG_DIR = "lightning_logs/fbrrt_adaptive"; CKPT_DIR = "checkpoints/fbrrt_adaptive"
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


class AdaptiveFracCallback(Callback):
    """Adaptively set off_policy_frac based on trajectory-level TD errors.

    Every `eval_interval` steps, compute:
      traj_mse_on: mean Σ_t (V(x_t) - V(x_{t+dt}))² along on-policy trajectories
      mse_off: mean (V(x_t) - r(x1))² for off-policy samples
      off_frac = clamp(traj_mse_on / (traj_mse_on + mse_off), min_frac, max_frac)
    """

    def __init__(self, dataset, model, ws_step, eval_interval=100,
                 n_eval=128, n_steps=100, min_frac=0.1, max_frac=0.9):
        super().__init__()
        self.dataset = dataset
        self.model = model  # OnPolicyValue, for drift
        self.ws_step = ws_step
        self.eval_interval = eval_interval
        self.n_eval = n_eval
        self.n_steps = n_steps
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.started = False
        self.frac_history = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if step < self.ws_step:
            return
        if not self.started:
            print(f"  [step {step}] Adaptive mixing started")
            self.started = True
        if step % self.eval_interval != 0:
            return

        device = pl_module.device
        dim = pl_module.hparams.dim
        n = self.n_eval
        dt = 1.0 / self.n_steps

        with torch.no_grad():
            # --- On-policy trajectory MSE: Σ_t (V(x_t) - V(x_{t+dt}))² ---
            x = torch.zeros(n, dim, device=device)
            traj_sq_errors = torch.zeros(n, device=device)
            for i in range(self.n_steps):
                t_curr = i * dt
                t_next = (i + 1) * dt
                v_curr = pl_module.value_module(x, torch.full((n,), t_curr, device=device))
                # SDE step with guided drift
                t_vec = torch.full((n,), t_curr, device=device)
                dx = pl_module.drift(x, t_vec, beta=1) * dt
                db = sqrt(2 * pl_module.a * dt) * torch.randn_like(x)
                x_next = x + dx + db
                v_next = pl_module.value_module(x_next, torch.full((n,), t_next, device=device))
                traj_sq_errors += (v_curr - v_next) ** 2
                x = x_next
            traj_mse_on = traj_sq_errors.mean().item()

            # --- Off-policy MSE: (V(x_t) - r(x1))² ---
            x1_np = gmm_sample(n)
            x1 = torch.from_numpy(x1_np).float().to(device)
            t_off = torch.rand(n, device=device)
            eps = torch.randn_like(x1)
            x_off = t_off.unsqueeze(-1) * x1 + torch.sqrt(
                2 * pl_module.a * t_off.unsqueeze(-1) * (1 - t_off.unsqueeze(-1))
            ) * eps
            v_off = pl_module.value_module(x_off, t_off)
            r_x1 = reward_fn(x1)
            mse_off = ((v_off - r_x1) ** 2).mean().item()

        # Compute adaptive fraction
        if traj_mse_on + mse_off > 0:
            raw_frac = traj_mse_on / (traj_mse_on + mse_off)
        else:
            raw_frac = 0.5
        frac = max(self.min_frac, min(self.max_frac, raw_frac))
        self.dataset.off_policy_frac = frac
        self.frac_history.append((step, frac, traj_mse_on, mse_off))

        pl_module.log("adaptive_off_frac", frac, prog_bar=False)
        pl_module.log("traj_mse_on", traj_mse_on, prog_bar=False)
        pl_module.log("mse_off", mse_off, prog_bar=False)

        if step % 500 == 0:
            print(f"  [step {step}] frac={frac:.3f} (mse_on={traj_mse_on:.3f} mse_off={mse_off:.3f})")


def run_config(name, use_adaptive, fixed_frac=None, lam_eff=0.1**100):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    for v in range(3):
        p = f"{LOG_DIR}/{name}/version_{v}"
        if os.path.exists(p): shutil.rmtree(p)

    vm = ValueNetwork(D, bias=bias_val)
    model = OnPolicyValue(
        base_score_module=base_drift, value_module=vm,
        reward_function=reward_fn, dim=D, a=a, lr=1e-3,
        loss_type="quad", analytical_value_fn=anal_fn,
        ema_decay=0.999, grad_decay=1e-6,
    )

    ds = OnPolicySMCDataset(
        dim=D, drift=base_drift, value=model.ema, smc_value=smc_reward,
        reward=reward_fn, device=DEVICE, a=a,
        batch_size=32, n_steps=100, mc_samples_per_step=10,
        sampling_method="fbrrt_td_lambda", lambda_eff=lam_eff,
        branch=4, entropy_lambda=2.0, fbrrt_alpha=1.0,
        off_policy_frac=1.0,  # start pure off-policy
        generating_function=gmm_sample,
    )
    loader = DataLoader(ds, batch_size=LOADER_BATCH_SIZE)

    callbacks = [TrajCB(anal_fn)]
    callbacks.append(ModelCheckpoint(
        dirpath=f"{CKPT_DIR}/{name}", save_last=True, save_top_k=1,
        monitor="val_reward_mean", mode="max", filename="best",
    ))

    if use_adaptive:
        adapt_cb = AdaptiveFracCallback(ds, model, ws_step=WS_STEPS)
        callbacks.append(adapt_cb)
    else:
        # Fixed fraction schedule
        class FixedFracCB(Callback):
            def __init__(self, ds, ws, frac):
                super().__init__(); self.ds=ds; self.ws=ws; self.frac=frac; self.done=False
            def on_train_batch_start(self, trainer, pl, batch, bi):
                if not self.done and trainer.global_step >= self.ws:
                    self.ds.off_policy_frac = self.frac; self.done = True
                    print(f"  [step {trainer.global_step}] frac→{self.frac}")
        callbacks.append(FixedFracCB(ds, WS_STEPS, fixed_frac))

    logger = CSVLogger(LOG_DIR, name=name, version=0)
    trainer = L.Trainer(
        max_steps=TOTAL_STEPS, val_check_interval=max(1, TOTAL_STEPS // 60),
        callbacks=callbacks, logger=logger,
        enable_checkpointing=True, enable_progress_bar=True,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader, val_dataloaders=val_loader)
    print(f"  Elapsed: {(time.perf_counter()-t0)/60:.1f} min")

    if use_adaptive and hasattr(adapt_cb, 'frac_history'):
        print(f"  Adaptive frac history ({len(adapt_cb.frac_history)} updates):")
        for step, frac, mon, moff in adapt_cb.frac_history[-5:]:
            print(f"    step={step}: frac={frac:.3f}")

    del trainer, model, vm, ds, loader
    gc.collect()
    if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()


# --- Execute ---
# Off-policy baseline
run_config("offpolicy_baseline", use_adaptive=False, fixed_frac=1.0, lam_eff=0.0)

# Fixed 50% mix (previous best stable config)
run_config("fixed_50pct", use_adaptive=False, fixed_frac=0.5)

# Adaptive mixing
run_config("adaptive", use_adaptive=True)


# --- Results & Plot ---
print(f"\n\n{'='*70}")
print(f"  RESULTS (E_OPT={E_OPT:.4f})")
print(f"{'='*70}")

runs = {}
for name in ["offpolicy_baseline", "fixed_50pct", "adaptive"]:
    csv = f"{LOG_DIR}/{name}/version_0/metrics.csv"
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    val = df.dropna(subset=["val_reward_mean"])
    if len(val) == 0: continue
    best = val["val_reward_mean"].max(); final = val["val_reward_mean"].iloc[-1]
    gm = df.dropna(subset=["traj_avg_mae_guided"])
    gf = gm["traj_avg_mae_guided"].iloc[-1] if len(gm)>0 else float("nan")
    stable = "YES" if abs(final-best)<5 else "no"
    print(f"  {name:<20} best={best:.3f} final={final:.3f} G-MAE={gf:.3f} stable={stable}")
    runs[name] = df

# Plot
styles = {
    "offpolicy_baseline": dict(color="black", ls="--", lw=2.5),
    "fixed_50pct": dict(color="#1a73e8", ls="-", lw=2.0),
    "adaptive": dict(color="#2ecc71", ls="-", lw=2.5),
}

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("FBRRT-TD(λ_s=0.1) Adaptive Mixing", fontsize=14, fontweight="bold")

for name, df in runs.items():
    s = styles[name]
    val = df.dropna(subset=["val_reward_mean"])
    axes[0].plot(val["step"].values, val["val_reward_mean"].values, label=name, **s)
    sub = df.dropna(subset=["traj_avg_mae_guided"])
    if len(sub) > 0:
        axes[1].plot(sub["step"].values, sub["traj_avg_mae_guided"].values, label=name, **s)

axes[0].axhline(E_OPT, color="red", linestyle=":", linewidth=1, alpha=0.4)
axes[0].axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Avg Terminal Reward")
axes[0].set_title("Terminal Reward"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Avg MAE")
axes[1].set_title("Guided MAE"); axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3); axes[1].set_yscale("log")

# Plot adaptive fraction over time
if "adaptive" in runs:
    df = runs["adaptive"]
    sub = df.dropna(subset=["adaptive_off_frac"])
    if len(sub) > 0:
        axes[2].plot(sub["step"].values, sub["adaptive_off_frac"].values,
                     color="#2ecc71", linewidth=2, label="off_policy_frac")
        axes[2].axhline(0.5, color="#1a73e8", linestyle="--", alpha=0.5, label="fixed 50%")
        axes[2].axvline(WS_STEPS, color="gray", linestyle=":", alpha=0.3)
        axes[2].set_xlabel("Steps"); axes[2].set_ylabel("off_policy_frac")
        axes[2].set_title("Adaptive Fraction"); axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("notebooks/fbrrt_adaptive_mix.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: notebooks/fbrrt_adaptive_mix.png")
plt.close()
print("Done.")
