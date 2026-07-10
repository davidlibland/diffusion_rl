#!/usr/bin/env python3
"""Anchor-dimension Optuna sweep for one (method, dim) cell on the
constant-headroom nested problem family.

Adapted from ../dim_scaling_bs4/sweep.py with:
  - problem_consth.make_problem (headroom = 6 nats at every dim; nested seeds)
  - loss_shift = V00 wired into both model classes (large-gap conditioning)
  - guided proposals for the four SMC methods: use_guidance / guidance_scale /
    guidance_source {ema, live} (the new gradient-based driver; clipped control,
    exact Girsanov-corrected weights)
  - the two FBRRT methods: fbrrt (generation network = EMA shadow, per the
    bs4_moons stabilization study) and fbrrt_cv (policy = EMA shadow)
  - SWEEP PHASE ONLY: the hparam-law fit consumes the top-K trial params
    (rank-weighted), so the confirm/converge phases of the bs4 pipeline are
    skipped at anchor cells.

Usage:  python sweep_consth.py --method fbrrt --dim 16
Env:    DSC_N_TRIALS=40 DSC_MAX_STEPS=5000 DSC_N_VAL=50 DSC_TOPK=3
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

HERE = os.path.dirname(os.path.abspath(__file__))
BS4 = os.path.join(os.path.dirname(HERE), "dim_scaling_bs4")
sys.path.insert(0, BS4)
sys.path.insert(0, HERE)

import sweep as bs4_sweep  # noqa: E402  (read_curve, lcb_of, make_smc_value, ...)
from problem_consth import make_problem  # noqa: E402

from diffusion_rl.models.off_policy import (  # noqa: E402
    InterpolatingNumpyDataset, OffPolicyValue)
from diffusion_rl.models.on_policy import (  # noqa: E402
    OnPolicySMCDataset, OnPolicyValue)
from diffusion_rl.modules.resnet_mlp import ValueNetwork  # noqa: E402

A = 1.0
DEVICE = bs4_sweep.DEVICE
empty_cache = bs4_sweep.empty_cache
read_curve = bs4_sweep.read_curve
lcb_of = bs4_sweep.lcb_of
make_smc_value = bs4_sweep.make_smc_value
val_loader = bs4_sweep.val_loader

RESULTS = os.environ.get("DSC_RESULTS_DIR", os.path.join(HERE, "results"))
LOGROOT = "lightning_logs/dim_scaling_consth"
BS = 4
DS_BATCH = 64
LOSS = "quad"
MAX_STEPS = int(os.environ.get("DSC_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("DSC_N_VAL", 50))
N_TRIALS = int(os.environ.get("DSC_N_TRIALS", 40))
TOPK = int(os.environ.get("DSC_TOPK", 3))

ON_METHODS = {"single_seed_mc", "single_seed_td_lambda", "ancestral_mc_td_lambda",
              "ancestral_td_lambda"}
TD_METHODS = {"single_seed_td_lambda", "ancestral_mc_td_lambda"}
RANDOMT_METHODS = {"single_seed_mc", "single_seed_td_lambda"}
FBRRT_METHODS = {"fbrrt", "fbrrt_cv"}
SKIP_NONFINITE = os.environ.get("OPT_SKIP_NONFINITE", "0") == "1"


class OnPolicyValueLive(OnPolicyValue):
    """Drift uses the LIVE network; optional skip-on-nonfinite for long runs."""

    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)

    def training_step(self, batch, batch_idx):
        try:
            loss = super().training_step(batch, batch_idx)
        except RuntimeError as e:
            if SKIP_NONFINITE and "not finite" in str(e):
                self._nonfinite_count = getattr(self, "_nonfinite_count", 0) + 1
                self._nonfinite_count_total = getattr(
                    self, "_nonfinite_count_total", 0) + 1
                if self._nonfinite_count > 200:
                    raise
                return None
            raise
        self._nonfinite_count = 0
        return loss


class OffPolicyValueGuarded(OffPolicyValue):
    def training_step(self, batch, batch_idx):
        try:
            loss = super().training_step(batch, batch_idx)
        except RuntimeError as e:
            if SKIP_NONFINITE and "not finite" in str(e):
                self._nonfinite_count = getattr(self, "_nonfinite_count", 0) + 1
                self._nonfinite_count_total = getattr(
                    self, "_nonfinite_count_total", 0) + 1
                if self._nonfinite_count > 200:
                    raise
                return None
            raise
        self._nonfinite_count = 0
        return loss


# ── design spaces ────────────────────────────────────────────────────────────
def sample_params(trial, method):
    p = {"off_policy_frac": 0.0,
         "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True)}
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False

    if method == "off_policy":
        return p

    p["n_steps"] = trial.suggest_int("n_steps", 10, 60)
    p["off_policy_frac"] = trial.suggest_float("off_policy_frac", 0.0, 0.5)
    p["mc_samples"] = trial.suggest_int("mc_samples", 1, 24, log=True)

    if method in FBRRT_METHODS:
        p["branch"] = trial.suggest_int("branch", 2, 16, log=True)
        p["alpha"] = trial.suggest_float("alpha", 0.0, 1.5)
        p["ent_inf"] = trial.suggest_categorical("ent_inf", [True, False])
        if not p["ent_inf"]:
            p["entropy_lambda"] = trial.suggest_float(
                "entropy_lambda", 0.05, 5.0, log=True)
        # fbrrt: generation-network EMA decay; fbrrt_cv: policy-EMA decay.
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
        return p

    # SMC family
    p["smc_type"] = trial.suggest_categorical(
        "smc_type", ["kt_r", "k_r", "k_Vema", "k_Vnograd", "kV_plus_ltr"])
    p["k"] = trial.suggest_float("k", 1e-3, 1.0, log=True)
    if p["smc_type"] == "kV_plus_ltr":
        p["l"] = trial.suggest_float("l", 1e-3, 1.0, log=True)
    # the new gradient-based driver
    p["use_guidance"] = trial.suggest_categorical("use_guidance", [True, False])
    if p["use_guidance"]:
        p["guidance_scale"] = trial.suggest_float(
            "guidance_scale", 0.05, 1.5, log=True)
        p["guidance_source"] = trial.suggest_categorical(
            "guidance_source", ["ema", "live"])
    needs_ema = (p["smc_type"] == "k_Vema"
                 or (p["use_guidance"] and p.get("guidance_source") == "ema"))
    if needs_ema:
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
    if method in RANDOMT_METHODS:
        p["random_t"] = trial.suggest_categorical("random_t", [True, False])
    if method in TD_METHODS:
        p["lambda_eff"] = trial.suggest_float("lambda_eff", 0.0, 1.0)
    return p


def trial_params(t, method):
    pr = dict(t.params)
    p = {"off_policy_frac": 0.0, "lr": pr["lr"],
         "use_grad_decay": pr.get("use_grad_decay", False)}
    if p["use_grad_decay"]:
        p["grad_decay"] = pr["grad_decay"]
    if method == "off_policy":
        return p
    p["n_steps"] = pr["n_steps"]
    p["off_policy_frac"] = pr["off_policy_frac"]
    p["mc_samples"] = pr["mc_samples"]
    if method in FBRRT_METHODS:
        p["branch"] = pr["branch"]; p["alpha"] = pr["alpha"]
        p["ent_inf"] = pr["ent_inf"]
        if not p["ent_inf"]:
            p["entropy_lambda"] = pr["entropy_lambda"]
        p["ema_decay"] = pr["ema_decay"]
        return p
    p["smc_type"] = pr["smc_type"]; p["k"] = pr["k"]
    if pr["smc_type"] == "kV_plus_ltr":
        p["l"] = pr["l"]
    p["use_guidance"] = pr.get("use_guidance", False)
    if p["use_guidance"]:
        p["guidance_scale"] = pr["guidance_scale"]
        p["guidance_source"] = pr["guidance_source"]
    if "ema_decay" in pr:
        p["ema_decay"] = pr["ema_decay"]
    if method in RANDOMT_METHODS:
        p["random_t"] = pr["random_t"]
    if method in TD_METHODS:
        p["lambda_eff"] = pr["lambda_eff"]
    return p


def build(method, params, prob, dim, hidden_dim, seed):
    L.seed_everything(seed, workers=True)
    grad_decay = params["grad_decay"] if params.get("use_grad_decay") else None
    bias_val = prob["bias_val"]
    vm = ValueNetwork(dim, hidden_dim=hidden_dim, bias=bias_val)

    if method == "off_policy":
        model = OffPolicyValueGuarded(
            base_score_module=prob["drift_fn"], reward_function=prob["reward_fn"],
            value_module=vm, dim=dim, a=A, lr=params["lr"], loss_type=LOSS,
            loss_shift=bias_val, grad_decay=grad_decay).to(DEVICE)
        ds = InterpolatingNumpyDataset(generating_function=prob["gmm_sample"],
                                       a=A, batch_size=1024)
        return model, vm, ds, DataLoader(ds, batch_size=BS)

    ema_decay = params.get("ema_decay", 0.99)
    model = OnPolicyValueLive(
        base_score_module=prob["drift_fn"], value_module=vm,
        reward_function=prob["reward_fn"], dim=dim, a=A, lr=params["lr"],
        loss_type=LOSS, loss_shift=bias_val, grad_decay=grad_decay,
        ema_decay=ema_decay).to(DEVICE)

    if method in FBRRT_METHODS:
        ent = float("inf") if params.get("ent_inf") else params["entropy_lambda"]
        if method == "fbrrt":
            # stabilization recipe: generation + targets read the EMA shadow
            gen_value, smc_value = model.ema, model.value_module
        else:  # fbrrt_cv: policy = EMA shadow (v_target stays the live net)
            gen_value, smc_value = model.value_module, model.ema
        ds = OnPolicySMCDataset(
            dim=dim, drift=prob["drift_fn"], value=gen_value,
            smc_value=smc_value, reward=prob["reward_fn"], device=DEVICE, a=A,
            batch_size=DS_BATCH, n_steps=params["n_steps"],
            mc_samples_per_step=params["mc_samples"], sampling_method=method,
            branch=params["branch"], entropy_lambda=ent,
            fbrrt_alpha=params["alpha"],
            off_policy_frac=params["off_policy_frac"], include_t_zero=False,
            generating_function=prob["gmm_sample"])
        return model, vm, ds, DataLoader(ds, batch_size=BS)

    smc_value = make_smc_value(params["smc_type"], params["k"],
                               params.get("l", 0.0), model, prob["reward_fn"])
    gscale = params.get("guidance_scale", 0.0) if params.get("use_guidance") else 0.0
    gsrc = None
    if gscale != 0.0:
        gsrc = (model.ema if params.get("guidance_source") == "ema"
                else model.value_module)
    ds = OnPolicySMCDataset(
        dim=dim, drift=prob["drift_fn"], value=model.value_module,
        smc_value=smc_value, reward=prob["reward_fn"], device=DEVICE, a=A,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"], sampling_method=method,
        lambda_eff=params.get("lambda_eff", 0.1),
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=params.get("random_t", False),
        smc_guidance_scale=gscale, smc_guidance_value=gsrc,
        generating_function=prob["gmm_sample"])
    return model, vm, ds, DataLoader(ds, batch_size=BS)


def fmt(method, p):
    s = f"lr={p['lr']:.1e}"
    if method in FBRRT_METHODS:
        ent = "inf" if p.get("ent_inf") else f"{p.get('entropy_lambda', 0):.2f}"
        s += (f" ns={p['n_steps']} mc={p['mc_samples']} br={p['branch']}"
              f" a={p['alpha']:.2f} ent={ent} ema={p['ema_decay']:.3f}"
              f" ofp={p['off_policy_frac']:.2f}")
    elif method != "off_policy":
        s += (f" ns={p['n_steps']} mc={p['mc_samples']} ofp={p['off_policy_frac']:.2f}"
              f" smc={p['smc_type']} k={p['k']:.3f}")
        if p.get("use_guidance"):
            s += f" GUID={p['guidance_scale']:.2f}/{p['guidance_source']}"
        if "lambda_eff" in p:
            s += f" lam={p['lambda_eff']:.2f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    args = ap.parse_args()
    method, dim = args.method, args.dim
    cell = f"{method}_d{dim}"
    t_start = time.time()
    os.makedirs(RESULTS, exist_ok=True)

    out_path = f"{RESULTS}/anchor_{cell}.json"
    prob = make_problem(dim, seed=0)  # anchor sweeps tune on the seed-0 instance
    hidden_dim = min(256, max(64, 32 * dim))
    print(f"=== ANCHOR {cell} device={DEVICE} hidden={hidden_dim} "
          f"s={prob['reward_scale']:.4g} gap={prob['diag']['gap']:.1f} "
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
            tr.fit(model, loader, val_dataloaders=val_loader)
        except optuna.TrialPruned:
            del model, vm, tr, loader, ds
            gc.collect(); empty_cache()
            raise
        except (RuntimeError, ValueError) as e:
            gc.collect(); empty_cache()
            print(f"  {name}: ERR {type(e).__name__}: {str(e)[:80]} -> -100",
                  flush=True)
            return -100.0
        st, cv = read_curve(csv)
        del model, vm, tr, loader, ds
        gc.collect(); empty_cache()
        lcb = lcb_of(cv)
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
