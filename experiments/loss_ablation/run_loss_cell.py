#!/usr/bin/env python3
"""One (loss, dimension) cell of the off-policy LOSS ablation.

Isolates the divergence from the sampler: every arm trains the same network,
on the same constant-headroom nested GMM problem, from the same off-policy
bridge anchors, with the same budget -- only `loss_type` differs.

Arms
  quad    the Spence loss           (log_quadratic_bregman)
  mse     exp-space squared error   (exp_mse, the STABILISED implementation)
  is      Itakura-Saito             (itakura_saito)
  logmse  log-space squared error   (log_mse; well-conditioned but biased)

Metrics per seed
  frac_closed  (plateau - E_base)/6      -- fraction of the 6-nat headroom closed
  v_rmse       RMSE of V_theta against the ANALYTIC value, over the base
               marginal at uniformly random t (a value-quality axis that does
               not go through the policy)
  v_bias       signed mean (V_theta - V_analytic); the log-space arm should
               show the Jensen deficit here
  skips        non-finite batches skipped (the stability axis of S3)

Usage
  python run_loss_cell.py --loss quad --dim 8 --lr 3e-4 --seeds 20 --tag grid
  python run_loss_cell.py --loss mse  --dim 8 --lr-grid --seeds 3 --steps 5000
"""
import argparse, gc, json, math, os, sys, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import lightning as L
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_bs4"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_lawv2"))
os.environ.setdefault("OPT_SKIP_NONFINITE", "1")   # count skips, don't abort

import sweep_consth as base                                   # noqa: E402
from problem_consth import make_problem                       # noqa: E402
from problem import optimal_terminal_and_reward               # noqa: E402
from diffusion_rl.models.off_policy import InterpolatingNumpyDataset  # noqa: E402
from diffusion_rl.modules.resnet_mlp import ValueNetwork      # noqa: E402

RESULTS = os.path.join(HERE, "results")
A = 1.0
LOSSES = ["quad", "mse", "is", "logmse"]
LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]


class ValCollector(Callback):
    def __init__(self): super().__init__(); self.vals = []
    def on_validation_end(self, trainer, pl):
        m = trainer.callback_metrics.get("val_reward_mean")
        if m is not None: self.vals.append(float(m))


@torch.no_grad()
def value_error(vm, prob, dim, n=20000, seed=0):
    """RMSE / signed bias of V_theta vs the analytic value, over the base
    joint: x1 ~ nu, t ~ U(0,1), x_t on the Brownian bridge (eq. fwd-bridge)."""
    g = torch.Generator().manual_seed(10_000 + seed)
    x1 = torch.from_numpy(prob["gmm_sample"](n)).float()
    t = torch.rand(n, 1, generator=g)
    eps = torch.randn(n, dim, generator=g)
    x = t * x1 + torch.sqrt(2 * A * t * (1 - t)) * eps
    dev = next(vm.parameters()).device
    x, t = x.to(dev), t.to(dev)
    pred = vm(x, t).flatten().double()
    exact = prob["anal_fn"](x, t.flatten()).flatten().double()
    d = (pred - exact)
    ok = torch.isfinite(d)
    return (float(d[ok].pow(2).mean().sqrt()), float(d[ok].mean()),
            int((~ok).sum()))


def run_seed(loss, dim, lr, s, steps, val_every, tail):
    prob = make_problem(dim, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], dim)
    L.seed_everything(s, workers=True)
    hidden = min(256, max(64, 32 * dim))
    vm = ValueNetwork(dim, hidden_dim=hidden, bias=prob["bias_val"])
    model = base.OffPolicyValueGuarded(
        base_score_module=prob["drift_fn"], reward_function=prob["reward_fn"],
        value_module=vm, dim=dim, a=A, lr=lr, loss_type=loss,
        loss_shift=prob["bias_val"], analytical_value_fn=prob["anal_fn"],
    ).to(base.DEVICE)
    ds = InterpolatingNumpyDataset(generating_function=prob["gmm_sample"],
                                   a=A, batch_size=1024)
    loader = DataLoader(ds, batch_size=base.BS)
    vc = ValCollector()
    tr = L.Trainer(max_steps=steps, val_check_interval=val_every, callbacks=[vc],
                   logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader, val_dataloaders=base.val_loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:90]}"
    v = np.array(vc.vals, dtype=float)
    if len(v):
        sm = pd.Series(v).rolling(tail, min_periods=1).mean().to_numpy()
        plateau = float(np.nanmean(sm[-tail:]))
    else:
        plateau = float("nan")
    try:
        v_rmse, v_bias, v_nonfin = value_error(vm, prob, dim, seed=s)
    except Exception as e:                       # a diverged net can be all-NaN
        v_rmse = v_bias = float("nan"); v_nonfin = -1
    out = {"seed": s, "plateau": plateau, "opt_reward": float(e_opt),
           "E_base": prob["diag"]["E_base_r"],
           "frac_closed": ((plateau - prob["diag"]["E_base_r"]) / 6.0
                           if math.isfinite(plateau) else float("nan")),
           "v_rmse": v_rmse, "v_bias": v_bias, "v_nonfinite": v_nonfin,
           "skips": int(getattr(model, "_nonfinite_count_total", 0)),
           "error": err}
    del model, vm, ds, loader, tr
    gc.collect(); base.empty_cache()
    return out


def cell(loss, dim, lr, seeds, steps, val_every, tail, tag):
    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/{tag}_{loss}_d{dim}.json"
    rec = (json.load(open(path)) if os.path.exists(path) else
           {"loss": loss, "dim": dim, "lr": lr, "steps": steps, "seeds": []})
    if rec.get("lr") != lr:                       # lr changed -> fresh cell
        rec = {"loss": loss, "dim": dim, "lr": lr, "steps": steps, "seeds": []}
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== {tag} {loss} d={dim} lr={lr:.1e} done={len(done)}/{seeds}",
          flush=True)
    for s in range(seeds):
        if s in done: continue
        t0 = time.time()
        out = run_seed(loss, dim, lr, s, steps, val_every, tail)
        rec["seeds"].append(out)
        json.dump(rec, open(path, "w"), indent=1)
        print(f"  seed {s:2d}: closed={100*out['frac_closed']:6.1f}% "
              f"v_rmse={out['v_rmse']:7.3f} bias={out['v_bias']:+7.3f} "
              f"skips={out['skips']:4d} err={out['error']} "
              f"({(time.time()-t0)/60:.2f}m)", flush=True)
    fr = np.array([e["frac_closed"] for e in rec["seeds"]
                   if math.isfinite(e.get("frac_closed", float("nan")))])
    if len(fr):
        print(f"  CELL {loss} d={dim} lr={lr:.1e} closed="
              f"{100*fr.mean():.1f}%+-{100*fr.std(ddof=1)/math.sqrt(len(fr)):.1f}"
              f" (n={len(fr)})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=LOSSES)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-grid", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--tag", default="grid")
    a = ap.parse_args()
    if a.lr_grid:
        for lr in LR_GRID:
            cell(a.loss, a.dim, lr, a.seeds, a.steps, a.val_every, a.tail,
                 f"lrscan{lr:.0e}")
    else:
        cell(a.loss, a.dim, a.lr, a.seeds, a.steps, a.val_every, a.tail, a.tag)


if __name__ == "__main__":
    main()
