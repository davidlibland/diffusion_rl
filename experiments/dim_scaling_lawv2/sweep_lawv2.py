#!/usr/bin/env python3
"""Anchor Optuna sweep for the law-v2 study: the consth SMC search space plus
the recipe knobs (n_steps 12..120 log, expand_frac, epoch_rows) swept jointly
so the hyperparameter laws can move them with dimension.

Anchors now INCLUDE d=512: the consth anchors (d <= 128) could not see the
high-d integrator bias, which is how n_steps=19 got locked into the old laws.

Methods: single_seed_mc, single_seed_td_lambda (the two that reach/beat
off-policy). Everything else (problem family, build, objective, LCB
selection) is the frozen consth machinery in this directory.

Usage:  python sweep_lawv2.py --method single_seed_mc --dim 512
Env:    DSC_N_TRIALS=60 DSC_MAX_STEPS=5000 DSC_N_VAL=50 DSC_TOPK=3
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "dim_scaling_bs4"))
sys.path.insert(0, HERE)

import sweep_consth as base  # noqa: E402  (frozen local copy)
from problem_consth import make_problem  # noqa: E402

RESULTS = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
LOGROOT = "lightning_logs/dim_scaling_lawv2"
MAX_STEPS = int(os.environ.get("DSC_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("DSC_N_VAL", 50))
N_TRIALS = int(os.environ.get("DSC_N_TRIALS", 60))
TOPK = int(os.environ.get("DSC_TOPK", 3))

METHODS = ("single_seed_mc", "single_seed_td_lambda")
T_MIN = 0.05


def make_augment(a, expand_frac, epoch_rows, t_min=T_MIN):
    """Backward-noising expansion of a fraction of rows + epoch-row cap.

    Expansion: each selected source row (x_s, s, v-hat, w) with s > t_min
    contributes one extra row at t' ~ U(t_min, s),
    x' ~ N((t'/s) x_s, 2a t'(s-t')/s I), same target and weight (exact base
    backward kernel; consistent by the harmonic property for base-marginal
    sources -- see ../dim_scaling_recipe/run_recipe_cell.py).
    Cap: epochs longer than epoch_rows are uniformly subsampled, controlling
    the dataset regeneration cadence.
    """
    def fn(all_x, all_t, all_tgt, all_w):
        if expand_frac > 0:
            src = (torch.isfinite(all_tgt) & (all_t > t_min)
                   & (torch.rand_like(all_t) < expand_frac))
            if src.any():
                xs, ss = all_x[src], all_t[src]
                ys, ws = all_tgt[src], all_w[src]
                tp = t_min + torch.rand_like(ss) * (ss - t_min)
                var = (2.0 * a * tp * (ss - tp) / ss).clamp_min(0.0)
                xp = ((tp / ss).unsqueeze(-1) * xs
                      + torch.sqrt(var).unsqueeze(-1) * torch.randn_like(xs))
                all_x = torch.cat([all_x, xp])
                all_t = torch.cat([all_t, tp])
                all_tgt = torch.cat([all_tgt, ys])
                all_w = torch.cat([all_w, ws.clone()])
        n = all_x.shape[0]
        if epoch_rows and n > epoch_rows:
            idx = torch.randperm(n, device=all_x.device)[:int(epoch_rows)]
            all_x, all_t = all_x[idx], all_t[idx]
            all_tgt, all_w = all_tgt[idx], all_w[idx]
        return all_x, all_t, all_tgt, all_w

    return fn


def build(method, params, prob, dim, hidden_dim, seed):
    """base.build + the recipe augmentation from params."""
    model, vm, ds, loader = base.build(method, params, prob, dim, hidden_dim,
                                       seed)
    ds.augment_fn = make_augment(ds.a, params.get("expand_frac", 0.0),
                                 params.get("epoch_rows", None))
    return model, vm, ds, loader


def sample_params(trial, method):
    assert method in METHODS
    p = {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True)}
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False
    p["n_steps"] = trial.suggest_int("n_steps", 12, 120, log=True)
    p["expand_frac"] = trial.suggest_float("expand_frac", 0.0, 1.0)
    p["epoch_rows"] = trial.suggest_int("epoch_rows", 768, 16384, log=True)
    p["off_policy_frac"] = trial.suggest_float("off_policy_frac", 0.0, 0.5)
    p["mc_samples"] = trial.suggest_int("mc_samples", 1, 24, log=True)
    p["smc_type"] = trial.suggest_categorical(
        "smc_type", ["kt_r", "k_r", "k_Vema", "k_Vnograd", "kV_plus_ltr"])
    p["k"] = trial.suggest_float("k", 1e-3, 1.0, log=True)
    if p["smc_type"] == "kV_plus_ltr":
        p["l"] = trial.suggest_float("l", 1e-3, 1.0, log=True)
    p["use_guidance"] = trial.suggest_categorical("use_guidance", [True, False])
    if p["use_guidance"]:
        p["guidance_scale"] = trial.suggest_float(
            "guidance_scale", 0.05, 1.5, log=True)
        p["guidance_source"] = trial.suggest_categorical(
            "guidance_source", ["ema", "live"])
    if (p["smc_type"] == "k_Vema"
            or (p["use_guidance"] and p.get("guidance_source") == "ema")):
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
    p["random_t"] = trial.suggest_categorical("random_t", [True, False])
    if method == "single_seed_td_lambda":
        p["lambda_eff"] = trial.suggest_float("lambda_eff", 0.0, 1.0)
    return p


def trial_params(t, method):
    pr = dict(t.params)
    p = {"lr": pr["lr"], "use_grad_decay": pr.get("use_grad_decay", False)}
    if p["use_grad_decay"]:
        p["grad_decay"] = pr["grad_decay"]
    for hp in ("n_steps", "expand_frac", "epoch_rows", "off_policy_frac",
               "mc_samples", "smc_type", "k", "random_t"):
        p[hp] = pr[hp]
    if pr["smc_type"] == "kV_plus_ltr":
        p["l"] = pr["l"]
    p["use_guidance"] = pr.get("use_guidance", False)
    if p["use_guidance"]:
        p["guidance_scale"] = pr["guidance_scale"]
        p["guidance_source"] = pr["guidance_source"]
    if "ema_decay" in pr:
        p["ema_decay"] = pr["ema_decay"]
    if method == "single_seed_td_lambda":
        p["lambda_eff"] = pr["lambda_eff"]
    return p


def fmt(method, p):
    s = (f"lr={p['lr']:.1e} ns={p['n_steps']} xf={p['expand_frac']:.2f} "
         f"er={p['epoch_rows']} mc={p['mc_samples']} "
         f"ofp={p['off_policy_frac']:.2f} smc={p['smc_type']} k={p['k']:.3f}")
    if p.get("use_guidance"):
        s += f" GUID={p['guidance_scale']:.2f}/{p['guidance_source']}"
    if "lambda_eff" in p:
        s += f" lam={p['lambda_eff']:.2f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dim", type=int, required=True)
    args = ap.parse_args()
    method, dim = args.method, args.dim
    cell = f"{method}_d{dim}"
    t_start = time.time()
    os.makedirs(RESULTS, exist_ok=True)

    out_path = f"{RESULTS}/anchor_{cell}.json"
    prob = make_problem(dim, seed=0)
    hidden_dim = min(256, max(64, 32 * dim))
    print(f"=== ANCHOR-V2 {cell} device={base.DEVICE} hidden={hidden_dim} "
          f"headroom={prob['diag']['headroom']:.3f} ===", flush=True)

    log_dir = f"{LOGROOT}/{cell}"
    study_db = f"sqlite:///{RESULTS}/study_{cell}.db"

    def objective(trial):
        p = sample_params(trial, method)
        name = f"trial_{trial.number:04d}"
        csv = f"{log_dir}/{name}/version_0/metrics.csv"
        t0 = time.time()

        class Prune(Callback):
            def on_validation_end(self, trainer, pl):
                m = trainer.callback_metrics.get("val_reward_mean")
                if m is None:
                    return
                trial.report(float(m), int(trainer.global_step))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        model = vm = tr = loader = ds = None
        try:
            model, vm, ds, loader = build(method, p, prob, dim, hidden_dim,
                                          1234 + trial.number)
            logger = CSVLogger(log_dir, name=name, version=0)
            tr = L.Trainer(max_steps=MAX_STEPS,
                           val_check_interval=max(1, MAX_STEPS // N_VAL),
                           callbacks=[Prune()], logger=logger,
                           enable_checkpointing=False, enable_progress_bar=False,
                           num_sanity_val_steps=0)
            tr.fit(model, loader, val_dataloaders=base.val_loader)
        except optuna.TrialPruned:
            del model, vm, tr, loader, ds
            gc.collect(); base.empty_cache()
            raise
        except (RuntimeError, ValueError) as e:
            gc.collect(); base.empty_cache()
            print(f"  {name}: ERR {type(e).__name__}: {str(e)[:80]} -> -100",
                  flush=True)
            return -100.0
        st, cv = base.read_curve(csv)
        del model, vm, tr, loader, ds
        gc.collect(); base.empty_cache()
        lcb = base.lcb_of(cv)
        print(f"  {name}: LCB={lcb:.3f} [{fmt(method, p)}] "
              f"{(time.time()-t0)/60:.1f}m", flush=True)
        return lcb

    sampler = TPESampler(multivariate=True, group=True, seed=42)
    pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                             reduction_factor=3)
    study = optuna.create_study(study_name=cell, storage=study_db,
                                load_if_exists=True, direction="maximize",
                                sampler=sampler, pruner=pruner)
    n_done = len([t for t in study.trials if t.state.is_finished()])
    print(f"  sweep: done={n_done} remaining={max(0, N_TRIALS - n_done)}",
          flush=True)
    study.optimize(objective, n_trials=max(0, N_TRIALS - n_done),
                   gc_after_trial=True)

    comp = [t for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None
            and t.value > -99]
    comp.sort(key=lambda t: t.value, reverse=True)
    chosen = comp[:TOPK]
    print("  top: " + ", ".join(f"t{t.number}={t.value:.2f}" for t in chosen),
          flush=True)
    json.dump({
        "method": method, "dim": dim, "problem": prob["diag"],
        "n_trials": len([t for t in study.trials if t.state.is_finished()]),
        "chosen": [{"trial": t.number, "lcb": t.value,
                    "params": trial_params(t, method)} for t in chosen],
        "elapsed_min": (time.time() - t_start) / 60,
    }, open(out_path, "w"), indent=2)
    print(f"  saved {out_path} ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
