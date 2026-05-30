#!/usr/bin/env python3
"""One bs4-style Optuna pipeline for a single (method, dimension) cell.

Runs the BS=4 methodology — TPE + Hyperband sweep on the detrended-SEM LCB of
the validation reward, then a multi-seed confirm, then a convergence run —
restricted to the **quad** loss, on the calibrated d-dim GMM problem
(``problem.make_problem``).

Method design spaces mirror the BS=4 study (``optuna_other_onpolicy_pipeline``
/ ``optuna_offpolicy_pipeline``), quad loss only:

  off_policy             : lr, grad_decay(toggle)
  single_seed_mc         : smc_type{kt_r,k_r,k_Vema,k_Vnograd,kV_plus_ltr}, k, l,
                           ema_decay, mc_samples, n_steps, random_t,
                           off_policy_frac, lr, grad_decay
  single_seed_td_lambda  : same + lambda_eff
  ancestral_mc_td_lambda : smc_type space + lambda_eff, mc_samples, n_steps,
                           off_policy_frac, lr, grad_decay  (no random_t)

Usage:
    python sweep.py --method single_seed_mc --dim 8

Budgets (env, Lean defaults):
    DSB_N_TRIALS=40 DSB_MAX_STEPS=5000 DSB_N_VAL=50 DSB_TOPK=3 DSB_N_SEEDS=5
    DSB_CONV_STEPS=30000 DSB_CONV_VAL_EVERY=1000 DSB_TARGET_GAP=6.0
"""

import argparse
import gc
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

from diffusion_rl.models.off_policy import InterpolatingNumpyDataset, OffPolicyValue
from diffusion_rl.models.on_policy import OnPolicySMCDataset, OnPolicyValue

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem import A, make_problem  # noqa: E402

from diffusion_rl.modules.resnet_mlp import ValueNetwork  # noqa: E402


# ── device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOGROOT = "lightning_logs/dim_scaling_bs4"
CKPTROOT = "checkpoints/dim_scaling_bs4"

BS = 4
DS_BATCH = 64
LOSS = "quad"
MAX_STEPS = int(os.environ.get("DSB_MAX_STEPS", 5000))
N_VAL = int(os.environ.get("DSB_N_VAL", 50))
LCB_TAIL = 20
LCB_Z = 1.645
N_TRIALS = int(os.environ.get("DSB_N_TRIALS", 40))
TOPK = int(os.environ.get("DSB_TOPK", 3))
N_SEEDS = int(os.environ.get("DSB_N_SEEDS", 5))
CONV_STEPS = int(os.environ.get("DSB_CONV_STEPS", 30000))
CONV_VAL_EVERY = int(os.environ.get("DSB_CONV_VAL_EVERY", 1000))
TARGET_GAP = float(os.environ.get("DSB_TARGET_GAP", 6.0))

ON_METHODS = {"single_seed_mc", "single_seed_td_lambda", "ancestral_mc_td_lambda"}
TD_METHODS = {"single_seed_td_lambda", "ancestral_mc_td_lambda"}
SMC_METHODS = {"single_seed_mc", "single_seed_td_lambda"}  # support random_t

val_loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)


class OnPolicyValueLive(OnPolicyValue):
    """Drift uses the LIVE network (the BS=4 default)."""

    def drift(self, x, t, beta=1, use_ema=False):
        return super().drift(x, t, beta=beta, use_ema=use_ema)


def make_smc_value(smc_type, k, l, model, reward_fn):
    if smc_type == "kt_r":
        return lambda x, t: k * t.reshape(-1) * reward_fn(x)
    if smc_type == "k_r":
        return lambda x, t: k * reward_fn(x)
    if smc_type == "k_Vema":
        return lambda x, t: k * model.ema(x, t).reshape(-1)
    if smc_type == "k_Vnograd":
        return lambda x, t: k * model.value_module(x, t).reshape(-1)
    if smc_type == "kV_plus_ltr":
        return lambda x, t: (k * model.value_module(x, t).reshape(-1)
                             + l * t.reshape(-1) * reward_fn(x))
    raise ValueError(smc_type)


# ── LCB objective helpers ───────────────────────────────────────────────────
def read_curve(csv_path):
    if not os.path.exists(csv_path):
        return np.array([]), np.array([])
    df = pd.read_csv(csv_path)
    if "val_reward_mean" not in df.columns:
        return np.array([]), np.array([])
    sub = df.dropna(subset=["val_reward_mean"])
    return sub["step"].to_numpy(), sub["val_reward_mean"].to_numpy()


def lcb_of(curve):
    if len(curve) < 5:
        return -100.0
    tail = curve[-LCB_TAIL:]
    n = len(tail)
    xx = np.arange(n, dtype=float)
    A_ = np.vstack([xx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A_, tail, rcond=None)
    resid = tail - A_ @ coef
    sigma = float(np.sqrt((resid ** 2).sum() / max(1, n - 2)))
    return float(tail.mean() - LCB_Z * sigma / np.sqrt(n))


# ── design space (mirrors BS=4, quad only) ─────────────────────────────────
def sample_params(trial, method):
    p = {
        "off_policy_frac": 0.0,
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
    }
    if trial.suggest_categorical("use_grad_decay", [True, False]):
        p["grad_decay"] = trial.suggest_float("grad_decay", 1e-5, 1e-1, log=True)
        p["use_grad_decay"] = True
    else:
        p["use_grad_decay"] = False

    if method == "off_policy":
        return p

    # on-policy family
    p["n_steps"] = trial.suggest_int("n_steps", 10, 60)
    p["off_policy_frac"] = trial.suggest_float("off_policy_frac", 0.0, 0.5)
    p["mc_samples"] = trial.suggest_int("mc_samples", 1, 24, log=True)
    p["smc_type"] = trial.suggest_categorical(
        "smc_type", ["kt_r", "k_r", "k_Vema", "k_Vnograd", "kV_plus_ltr"])
    p["k"] = trial.suggest_float("k", 1e-3, 1.0, log=True)
    if p["smc_type"] == "kV_plus_ltr":
        p["l"] = trial.suggest_float("l", 1e-3, 1.0, log=True)
    if p["smc_type"] == "k_Vema":
        p["ema_decay"] = trial.suggest_float("ema_decay", 0.90, 0.999)
    if method in SMC_METHODS:
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
    p["smc_type"] = pr["smc_type"]; p["k"] = pr["k"]
    if pr["smc_type"] == "kV_plus_ltr":
        p["l"] = pr["l"]
    if pr["smc_type"] == "k_Vema":
        p["ema_decay"] = pr["ema_decay"]
    if method in SMC_METHODS:
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
        model = OffPolicyValue(
            base_score_module=prob["drift_fn"], reward_function=prob["reward_fn"],
            value_module=vm, dim=dim, a=A, lr=params["lr"], loss_type=LOSS,
            grad_decay=grad_decay).to(DEVICE)
        ds = InterpolatingNumpyDataset(generating_function=prob["gmm_sample"], a=A,
                                       batch_size=1024)
        return model, vm, ds, DataLoader(ds, batch_size=BS)

    ema_decay = params.get("ema_decay", 0.99)
    model = OnPolicyValueLive(
        base_score_module=prob["drift_fn"], value_module=vm,
        reward_function=prob["reward_fn"], dim=dim, a=A, lr=params["lr"],
        loss_type=LOSS, grad_decay=grad_decay, ema_decay=ema_decay).to(DEVICE)
    smc_value = make_smc_value(params["smc_type"], params["k"],
                               params.get("l", 0.0), model, prob["reward_fn"])
    ds = OnPolicySMCDataset(
        dim=dim, drift=prob["drift_fn"], value=model.value_module,
        smc_value=smc_value, reward=prob["reward_fn"], device=DEVICE, a=A,
        batch_size=DS_BATCH, n_steps=params["n_steps"],
        mc_samples_per_step=params["mc_samples"], sampling_method=method,
        lambda_eff=params.get("lambda_eff", 0.1),
        off_policy_frac=params["off_policy_frac"], include_t_zero=False,
        random_t=params.get("random_t", False),
        generating_function=prob["gmm_sample"])
    return model, vm, ds, DataLoader(ds, batch_size=BS)


def fmt(method, p):
    s = f"lr={p['lr']:.1e} " + (f"gd={p['grad_decay']:.1e}" if p.get('use_grad_decay')
                                else "gd=off")
    if method != "off_policy":
        s += (f" ns={p['n_steps']} mc={p['mc_samples']} ofp={p['off_policy_frac']:.2f}"
              f" smc={p['smc_type']} k={p['k']:.3f}")
        if "lambda_eff" in p:
            s += f" lam={p['lambda_eff']:.2f}"
        if "random_t" in p:
            s += f" rt={int(p['random_t'])}"
    return s


def detect_convergence(steps, curve, win=8):
    if len(curve) < win + 4:
        return None, float(curve[-1]) if len(curve) else float("nan")
    sm = pd.Series(curve).rolling(win, min_periods=1).mean().to_numpy()
    tail = sm[-max(4, len(sm) // 5):]
    plateau = float(tail.mean())
    noise = float(np.std(curve[len(curve) // 2:] - sm[len(sm) // 2:]))
    conv_step = None
    for i in range(len(sm)):
        if sm[i] >= plateau - 0.5 * noise and np.all(sm[i:] >= plateau - noise):
            conv_step = int(steps[i]); break
    return conv_step, plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dim", type=int, required=True)
    args = ap.parse_args()
    method, dim = args.method, args.dim
    cell = f"{method}_d{dim}"
    t_start = time.time()

    prob = make_problem(dim, target_gap=TARGET_GAP)
    hidden_dim = min(256, max(64, 32 * dim))
    print(f"=== CELL {cell}  device={DEVICE}  hidden={hidden_dim} ===", flush=True)
    print(f"  reward_scale={prob['reward_scale']:.4g} gap≈{prob['diag']['gap_empirical']:.2f}"
          f" V00={prob['V00']:.2f} bias={prob['bias_val']:.2f}", flush=True)

    log_dir = f"{LOGROOT}/{cell}"
    study_db = f"sqlite:///{RESULTS}/study_{cell}.db"

    # ── Phase 1: sweep ──
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
            print(f"  {name}: ERR {type(e).__name__} -> -100", flush=True)
            return -100.0
        st, cv = read_curve(csv)
        del model, vm, tr, loader, ds
        gc.collect(); empty_cache()
        lcb = lcb_of(cv)
        print(f"  {name}: LCB={lcb:.3f} [{fmt(method, p)}] {(time.time()-t0)/60:.1f}m",
              flush=True)
        return lcb

    sampler = TPESampler(multivariate=True, group=True, seed=42)
    pruner = HyperbandPruner(min_resource=500, max_resource=MAX_STEPS,
                             reduction_factor=3)
    study = optuna.create_study(study_name=cell, storage=study_db,
                                load_if_exists=True, direction="maximize",
                                sampler=sampler, pruner=pruner)
    n_done = len([t for t in study.trials if t.state.is_finished()])
    print(f"  PHASE 1 sweep: done={n_done} remaining={max(0, N_TRIALS-n_done)}",
          flush=True)
    study.optimize(objective, n_trials=max(0, N_TRIALS - n_done),
                   gc_after_trial=True)

    comp = [t for t in study.trials
            if t.state == TrialState.COMPLETE and t.value is not None]
    comp.sort(key=lambda t: t.value, reverse=True)
    chosen = comp[:TOPK]
    print(f"  top-{TOPK}: " + ", ".join(f"t{t.number}={t.value:.2f}" for t in chosen),
          flush=True)

    # ── Phase 2: confirm ──
    confirm = {}
    for t in chosen:
        params = trial_params(t, method)
        lcbs, bests = [], []
        for s in range(N_SEEDS):
            name = f"confirm_t{t.number}_s{s}"
            csv = f"{log_dir}/{name}/version_0/metrics.csv"
            if os.path.exists(f"{log_dir}/{name}"):
                shutil.rmtree(f"{log_dir}/{name}")
            try:
                model, vm, ds, loader = build(method, params, prob, dim, hidden_dim,
                                              1000 + s)
                logger = CSVLogger(log_dir, name=name, version=0)
                tr = L.Trainer(max_steps=MAX_STEPS,
                               val_check_interval=max(1, MAX_STEPS // N_VAL),
                               logger=logger, enable_checkpointing=False,
                               enable_progress_bar=False, num_sanity_val_steps=0)
                tr.fit(model, loader, val_dataloaders=val_loader)
                del model, vm, tr, loader, ds
            except (RuntimeError, ValueError) as e:
                print(f"    {name}: ERR {type(e).__name__}", flush=True)
            gc.collect(); empty_cache()
            st, cv = read_curve(csv)
            lcbs.append(lcb_of(cv) if len(cv) else -100.0)
            bests.append(float(cv.max()) if len(cv) else -100.0)
        lcbs = np.array(lcbs); bests = np.array(bests)
        confirm[t.number] = {
            "params": params, "lcb_mean": float(lcbs.mean()),
            "lcb_sd": float(lcbs.std(ddof=1)) if len(lcbs) > 1 else 0.0,
            "lcb_values": lcbs.tolist(), "best_mean": float(bests.mean()),
            "best_sd": float(bests.std(ddof=1)) if len(bests) > 1 else 0.0}
        print(f"  confirm t{t.number}: LCB {lcbs.mean():.3f}±{confirm[t.number]['lcb_sd']:.3f}"
              f"  best {bests.mean():.3f}", flush=True)

    best_trial = max(confirm, key=lambda tr: confirm[tr]["lcb_mean"])
    best_conf = confirm[best_trial]
    print(f"  winner t{best_trial}: LCB {best_conf['lcb_mean']:.3f}  "
          f"[{fmt(method, best_conf['params'])}]", flush=True)

    # ── Phase 3: converge ──
    params = best_conf["params"]
    tag = f"converge_{cell}"
    if os.path.exists(f"{log_dir}/{tag}"):
        shutil.rmtree(f"{log_dir}/{tag}")
    ckdir = f"{CKPTROOT}/{cell}"
    if os.path.exists(ckdir):
        shutil.rmtree(ckdir)
    os.makedirs(ckdir, exist_ok=True)
    t0 = time.time()
    model, vm, ds, loader = build(method, params, prob, dim, hidden_dim, 20240)
    logger = CSVLogger(log_dir, name=tag, version=0)
    ckpt = ModelCheckpoint(dirpath=ckdir, save_last=True, save_top_k=1,
                           monitor="val_reward_mean", mode="max", filename="best")
    tr = L.Trainer(max_steps=CONV_STEPS, val_check_interval=CONV_VAL_EVERY,
                   callbacks=[ckpt], logger=logger, enable_checkpointing=True,
                   enable_progress_bar=False, num_sanity_val_steps=0)
    tr.fit(model, loader, val_dataloaders=val_loader)
    torch.save({"state_dict": model.value_module.state_dict(), "params": params,
                "method": method, "dim": dim, "max_steps": CONV_STEPS},
               f"{ckdir}/value_module.pt")
    st, cv = read_curve(f"{log_dir}/{tag}/version_0/metrics.csv")
    cstep, plateau = detect_convergence(st, cv)
    flcb = lcb_of(cv)
    print(f"  CONVERGED plateau={plateau:.3f} conv@{cstep} final_lcb={flcb:.3f}"
          f" ({(time.time()-t0)/60:.1f}m)", flush=True)
    del model, vm, tr, loader, ds
    gc.collect(); empty_cache()

    out = {
        "method": method, "dim": dim, "loss": LOSS,
        "problem": prob["diag"],
        "winner_trial": best_trial, "winner": best_conf,
        "chosen": [{"trial": t.number, "lcb": t.value,
                    "params": trial_params(t, method)} for t in chosen],
        "confirm": confirm,
        "convergence": {"plateau_reward": plateau, "convergence_step": cstep,
                        "final_lcb": flcb, "ckpt_dir": ckdir},
        "convergence_curve": {"steps": st.tolist(), "val_reward": cv.tolist()},
        "elapsed_min": (time.time() - t_start) / 60,
    }
    out_path = f"{RESULTS}/{cell}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  saved {out_path}  ({out['elapsed_min']:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
