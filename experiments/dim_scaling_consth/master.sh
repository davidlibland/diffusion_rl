#!/usr/bin/env bash
# Master orchestrator for the constant-headroom dim-scaling study.
#   stage A: anchor Optuna sweeps  (6 methods x d in {2,16,128}, parallel pool)
#   stage B: fit hparam-vs-log(d) laws (fit_consth.py)
#   stage C: paired multi-seed grid (6 methods x 8 dims x 30 nested seeds)
# All stages are idempotent/resumable (per-cell optuna DBs, per-seed JSONs).
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../.."}
cd "$REPO"
HERE=experiments/dim_scaling_consth
LOGD=${LOGD:-/tmp/consth_logs}
mkdir -p "$LOGD"
POOL=${POOL:-6}
METHODS="off_policy single_seed_mc single_seed_td_lambda ancestral_mc_td_lambda fbrrt fbrrt_cv"
ANCHORS="2 16 128"
GRID_DIMS=${GRID_DIMS:-"2 8 16 32 64 128 256 512"}
export OPT_SKIP_NONFINITE=${OPT_SKIP_NONFINITE:-0}   # sweeps: hard-fail trials

echo "[$(date +%H:%M:%S)] stage A: anchor sweeps (pool=$POOL)"
for m in $METHODS; do for d in $ANCHORS; do echo "$m $d"; done; done | \
  xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 python experiments/dim_scaling_consth/sweep_consth.py \
       --method $0 --dim $1 > '"$LOGD"'/anchor_$0_d$1.log 2>&1; \
     echo "  done anchor $0 d=$1"'
echo "[$(date +%H:%M:%S)] stage A done"

echo "[$(date +%H:%M:%S)] stage B: fit laws"
python "$HERE/fit_consth.py" > "$LOGD/fit.log" 2>&1 || { echo "FIT FAILED"; exit 1; }
tail -8 "$LOGD/fit.log"

echo "[$(date +%H:%M:%S)] stage C: multi-seed grid (pool=$POOL)"
export OPT_SKIP_NONFINITE=1   # long runs: skip-guard on
# big dims first so the long cells overlap the small ones
for d in 512 256 128 64 32 16 8 2; do
  case " $GRID_DIMS " in *" $d "*) ;; *) continue;; esac
  for m in $METHODS; do echo "$m $d"; done
done | xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 python experiments/dim_scaling_consth/run_cell.py \
       --method $0 --dim $1 > '"$LOGD"'/grid_$0_d$1.log 2>&1; \
     echo "  done grid $0 d=$1"'
echo "[$(date +%H:%M:%S)] ALL DONE"
