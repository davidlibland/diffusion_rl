#!/usr/bin/env python3
"""Probe 4: trust-region (TRI-TSMC-style) mechanisms in the value-learning loop.

Adapts the trust-region machinery of Wang et al. 2026 (arXiv:2605.25123,
docs/papers/tri_tsmc.pdf) to our continuous-training regime.  TRI-TSMC's
KL-constrained path-space update has the closed form q* prop R^tau P (escort /
tempered path, their Prop 5.3), found by a 1-D convex dual (their eq 4.7), and
moving along it monotonically reduces chi^2 weight degeneracy (their Thm 5.5).
Their optimal twist IS our value function (their Thm 5.1 = the H-martingale
soft Bellman recursion).  We keep our conditionally-unbiased exp-space
regression targets (valid under ANY sampling measure) and use the trust
region purely as SAMPLING-MEASURE CONTROL — the per-step-normalized
approximation of their path-space escort.

Arms (ssmc, d=512, law-v2 base config, paired seeds):

  tr_twist       : the k_Vema twist is replaced by an ANCHORED twist
                   k*V_anchor.  At each dataset regeneration the anchor moves
                   toward the live value net by the largest weight-space lerp
                   beta whose estimated path-KL (on the fresh epoch's rows) is
                   <= eps nats.  This is exactly an EMA whose decay is SOLVED
                   from a KL budget instead of fixed — the trust region
                   replaces both ema_decay and the cadence knob.  Control:
                   gridv2 ssmc d512 (law-v2, 35.3% on 30 seeds).
  tr_oracle_ramp : twist = beta_i * V* (analytic value), beta_0 = 0, each
                   regeneration takes the largest step delta with estimated
                   KL <= eps, capped at beta=1.  Probe 2 showed instant
                   full-strength oracle concentration destroys learning
                   (16.3% vs 25.9% control); this tests concentration at the
                   rate the fit can absorb.  beta trajectory is recorded.
  tr_unweight    : probe-3 expand_oracle (law-v1 config + oracle twist +
                   k=1 backward-noising expansion + e^{-v} unweighting), with
                   the crude 100x weight clamp replaced by the TRI-TSMC dual:
                   w_hat prop w^tau, tau = 1/(1+lambda_hat),
                   lambda_hat = argmin_l [l*eps + (1+l) log mean w^{1/(1+l)}].
                   Control: probe3 expand_oracle (21.9% on seeds 0-9).

For the tr arms, epoch_rows is overridden to 1216 (fast cadence, ~49
regenerations per run) so the trust region gets enough updates to matter; eps
is then the sole knob governing how fast the sampling measure may move.

Usage: python probe4_cell.py --arm tr_twist --eps 1.0
Env:   DSP_SEEDS=10 DSC_STEPS=15000 DSC_VAL_EVERY=500 DSC_TAIL=8
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_bs4"))
sys.path.insert(0, os.path.join(EXP, "dim_scaling_consth"))   # fit_consth (law-v1)
sys.path.insert(0, os.path.join(EXP, "dim_scaling_lawv2"))    # frozen base + laws v2

import sweep_consth as base  # noqa: E402  (lawv2's frozen copy shadows consth's)
import sweep_lawv2 as sw2  # noqa: E402
from fit_lawv2 import hparams_for_dim as hparams_v2  # noqa: E402
from fit_consth import hparams_for_dim as hparams_v1  # noqa: E402
from problem_consth import make_problem  # noqa: E402
from problem import optimal_terminal_and_reward  # noqa: E402

RESULTS = os.environ.get("DSP_RESULTS_DIR", os.path.join(HERE, "results"))
N_SEEDS = int(os.environ.get("DSP_SEEDS", 10))
STEPS = int(os.environ.get("DSC_STEPS", 15000))
VAL_EVERY = int(os.environ.get("DSC_VAL_EVERY", 500))
TAIL = int(os.environ.get("DSC_TAIL", 8))
METHOD = "single_seed_mc"
DIM = 512
T_MIN = 0.05
KL_ROWS = 2048          # max rows used per trust-region KL estimate
TR_EPOCH_ROWS = 1216    # fast cadence for the tr arms (~49 regens / 15k steps)


def _kl_of_logratio(z):
    """KL(q_new || q_old) for q_new prop e^z q_old, from samples of q_old.

    KL = E_new[z] - log E_old[e^z]; with samples z_j from q_old this is
    sum(softmax(z) * z) - (logsumexp(z) - log n).  Centering z first is a
    no-op mathematically and keeps the exps sane.
    """
    z = z - z.mean()
    w = torch.softmax(z, 0)
    n = z.numel()
    return float((w * z).sum() - (torch.logsumexp(z, 0) - math.log(n)))


def _bisect_scale(dl, eps, hi=1.0):
    """Largest s in [0, hi] with KL(s*dl) <= eps (KL monotone in s)."""
    if _kl_of_logratio(hi * dl) <= eps:
        return hi
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _kl_of_logratio(mid * dl) <= eps:
            lo = mid
        else:
            hi = mid
    return lo


def temper_weights(w, eps):
    """TRI-TSMC dual (eq 4.7): w_hat prop w^tau, tau = 1/(1+lambda_hat)."""
    lw = torch.log(w.clamp_min(1e-300))
    lw = lw - lw.max()
    n = lw.numel()

    def phi(lam):
        tau = 1.0 / (1.0 + lam)
        return lam * eps + (1.0 + lam) * float(
            torch.logsumexp(tau * lw, 0) - math.log(n))

    lo, hi = 0.0, 1e4
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if phi(m1) < phi(m2):
            hi = m2
        else:
            lo = m1
    tau = 1.0 / (1.0 + 0.5 * (lo + hi))
    wt = torch.exp(tau * lw)
    return wt * (n / wt.sum()), tau


class TrustRegionTwist:
    """Anchored twist k*V_anchor; anchor lerps toward the live net under a
    per-regeneration KL budget (adaptive-decay EMA)."""

    def __init__(self, k, source_net, eps):
        self.k, self.eps, self.source = float(k), float(eps), source_net
        self.anchor = copy.deepcopy(source_net).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.betas = []

    def __call__(self, x, t):
        with torch.no_grad():
            return self.k * self.anchor(x, t.reshape(-1))

    def update(self, x, t):
        n = x.shape[0]
        if n > KL_ROWS:
            idx = torch.randperm(n, device=x.device)[:KL_ROWS]
            x, t = x[idx], t[idx]
        with torch.no_grad():
            dl = (self.k * (self.source(x, t.reshape(-1))
                            - self.anchor(x, t.reshape(-1)))).flatten()
        beta = _bisect_scale(dl, self.eps)
        with torch.no_grad():
            for pa, ps in zip(self.anchor.parameters(),
                              self.source.parameters()):
                pa.lerp_(ps, beta)
        self.betas.append(beta)


class OracleRampTwist:
    """Twist beta*V*; beta grows per regeneration under the KL budget."""

    def __init__(self, anal_fn, eps):
        self.anal_fn, self.eps, self.beta = anal_fn, float(eps), 0.0
        self.betas = []

    def __call__(self, x, t):
        if self.beta == 0.0:
            return torch.zeros(x.shape[0], device=x.device)
        return self.beta * self.anal_fn(x, t)

    def update(self, x, t):
        if self.beta >= 1.0:
            self.betas.append(self.beta)
            return
        n = x.shape[0]
        if n > KL_ROWS:
            idx = torch.randperm(n, device=x.device)[:KL_ROWS]
            x, t = x[idx], t[idx]
        with torch.no_grad():
            v = self.anal_fn(x, t).flatten()
        delta = _bisect_scale(v, self.eps, hi=1.0 - self.beta)
        self.beta = min(1.0, self.beta + delta)
        self.betas.append(self.beta)


def make_expand_temper(a, eps, t_min=T_MIN):
    """Probe-3 expand_oracle augmentation with dual-tempered unweighting."""
    taus = []

    def expand(all_x, all_t, all_tgt, all_w):
        src = torch.isfinite(all_tgt) & (all_t > t_min)
        if not src.any():
            return all_x, all_t, all_tgt, all_w
        xs, ss = all_x[src], all_t[src]
        ys, ws = all_tgt[src], all_w[src]
        tp = t_min + torch.rand_like(ss) * (ss - t_min)
        var = (2.0 * a * tp * (ss - tp) / ss).clamp_min(0.0)
        xp = ((tp / ss).unsqueeze(-1) * xs
              + torch.sqrt(var).unsqueeze(-1) * torch.randn_like(xs))
        lw = torch.nan_to_num(-ys, nan=-torch.inf)
        raw = ws * torch.exp(lw - lw.max())
        if raw.sum() > 0:
            wp, tau = temper_weights(raw, eps)
            taus.append(tau)
        else:
            wp = torch.ones_like(raw)
        return (torch.cat([all_x, xp]), torch.cat([all_t, tp]),
                torch.cat([all_tgt, ys]), torch.cat([all_w, wp]))

    expand.taus = taus
    return expand


class ValCollector(Callback):
    def __init__(self):
        super().__init__(); self.vals = []

    def on_validation_end(self, trainer, pl):
        m = trainer.callback_metrics.get("val_reward_mean")
        if m is not None:
            self.vals.append(float(m))


def run_seed(arm, eps, s, hidden):
    prob = make_problem(DIM, seed=s)
    _, e_opt, _ = optimal_terminal_and_reward(
        prob["means"], prob["sigma2"], prob["weights"], prob["c"],
        prob["reward_scale"], DIM)

    twist = None
    expand_fn = None
    if arm in ("tr_twist", "tr_oracle_ramp"):
        params = hparams_v2(METHOD, DIM)
        params["epoch_rows"] = TR_EPOCH_ROWS
        model, vm, ds, loader = sw2.build(METHOD, params, prob, DIM, hidden, s)
        if arm == "tr_twist":
            twist = TrustRegionTwist(params["k"], model.value_module, eps)
        else:
            twist = OracleRampTwist(prob["anal_fn"], eps)
        ds.smc_value = twist
        base_aug = ds.augment_fn

        def aug(x, t, y, w):
            twist.update(x, t)      # rows sampled under the CURRENT twist
            return base_aug(x, t, y, w)

        ds.augment_fn = aug
    else:  # tr_unweight
        params = hparams_v1(METHOD, DIM)   # law-v1 config = probe-3 control
        model, vm, ds, loader = sw2.build(METHOD, params, prob, DIM, hidden, s)
        ds.smc_value = prob["anal_fn"]     # oracle twist, full strength
        expand_fn = make_expand_temper(ds.a, eps)
        ds.augment_fn = expand_fn

    vc = ValCollector()
    tr = L.Trainer(max_steps=STEPS, val_check_interval=VAL_EVERY,
                   callbacks=[vc], logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    err = None
    try:
        tr.fit(model, loader, val_dataloaders=base.val_loader)
    except (RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"
    v = np.array(vc.vals, dtype=float)
    if len(v):
        sm = pd.Series(v).rolling(TAIL, min_periods=1).mean().to_numpy()
        plateau = float(np.nanmean(sm[-TAIL:])); best = float(np.nanmax(sm))
    else:
        plateau = best = float("nan")
    out = {"seed": s, "plateau": plateau, "best": best,
           "opt_reward": float(e_opt), "E_base": prob["diag"]["E_base_r"],
           "regret": plateau - float(e_opt),
           "frac_closed": ((plateau - prob["diag"]["E_base_r"]) / 6.0
                           if math.isfinite(plateau) else float("nan")),
           "nonfinite_skips": int(getattr(model, "_nonfinite_count_total", 0)),
           "error": err}
    if twist is not None and twist.betas:
        b = np.array(twist.betas, dtype=float)
        out["n_updates"] = int(len(b))
        out["beta_mean"] = float(b.mean())
        out["beta_final"] = float(b[-1])
        out["betas"] = [round(float(x), 4) for x in b]
    if expand_fn is not None and expand_fn.taus:
        tau = np.array(expand_fn.taus, dtype=float)
        out["tau_mean"] = float(tau.mean())
        out["tau_final"] = float(tau[-1])
    del model, vm, ds, loader, tr
    gc.collect(); base.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["tr_twist", "tr_oracle_ramp", "tr_unweight"])
    ap.add_argument("--eps", type=float, required=True)
    args = ap.parse_args()
    arm, eps = args.arm, args.eps
    os.makedirs(RESULTS, exist_ok=True)
    out_path = f"{RESULTS}/probe4_{arm}_eps{eps:g}.json"
    hidden = min(256, max(64, 32 * DIM))
    rec = (json.load(open(out_path)) if os.path.exists(out_path) else
           {"arm": arm, "eps": eps, "method": METHOD, "dim": DIM,
            "steps": STEPS, "seeds": []})
    done = {e["seed"] for e in rec["seeds"]}
    print(f"=== PROBE4 {arm} eps={eps:g} done={len(done)}/{N_SEEDS}",
          flush=True)
    for s in range(N_SEEDS):
        if s in done:
            continue
        t0 = time.time()
        out = run_seed(arm, eps, s, hidden)
        rec["seeds"].append(out)
        json.dump(rec, open(out_path, "w"), indent=1)
        extra = ""
        if "beta_final" in out:
            extra = f" beta={out['beta_final']:.3f}"
        if "tau_mean" in out:
            extra = f" tau={out['tau_mean']:.3f}"
        print(f"  seed {s:2d}: plateau={out['plateau']:8.3f} "
              f"closed={100*out['frac_closed']:5.1f}%{extra} "
              f"err={out['error']} ({(time.time()-t0)/60:.1f}m)", flush=True)
    fr = np.array([e["frac_closed"] for e in rec["seeds"]
                   if math.isfinite(e.get("frac_closed", float("nan")))])
    if len(fr):
        se = fr.std(ddof=1) / math.sqrt(len(fr)) if len(fr) > 1 else 0.0
        print(f"  DONE frac_closed={100*fr.mean():.1f}%±{100*se:.1f} "
              f"(n={len(fr)})", flush=True)


if __name__ == "__main__":
    main()
