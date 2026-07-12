#!/usr/bin/env bash
# Law-v2 pipeline: recipe knobs (n_steps, expand_frac, epoch_rows) in the laws.
#   stage A: anchor Optuna sweeps, 2 methods x d in {2, 16, 128, 512}
#   stage B: fit hparam-vs-log(d) laws (fit_lawv2.py)
#   stage C: paired multi-seed grid, 2 methods x 8 dims x 30 nested seeds
#   stage D: plot vs off-policy / law-v1 / fixed recipe
# All stages idempotent/resumable (per-cell optuna DBs, per-seed JSONs).
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../.."}
cd "$REPO"
HERE=experiments/dim_scaling_lawv2
LOGD="$HERE/logs"
mkdir -p "$LOGD"
POOL=${POOL:-4}
METHODS="single_seed_mc single_seed_td_lambda"
ANCHORS="512 128 16 2"
GRID_DIMS="512 256 128 64 32 16 8 2"
export OPT_SKIP_NONFINITE=0
export DSC_N_TRIALS=${DSC_N_TRIALS:-60}

echo "[$(date +%H:%M:%S)] stage A: anchor sweeps (pool=$POOL, trials=$DSC_N_TRIALS)"
for d in $ANCHORS; do for m in $METHODS; do echo "$m $d"; done; done | \
  xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 python experiments/dim_scaling_lawv2/sweep_lawv2.py \
       --method $0 --dim $1 \
       > experiments/dim_scaling_lawv2/logs/anchor_$0_d$1.log 2>&1; \
     echo "  done anchor $0 d=$1"'
echo "[$(date +%H:%M:%S)] stage A done"

echo "[$(date +%H:%M:%S)] stage B: fit laws"
python "$HERE/fit_lawv2.py" > "$LOGD/fit.log" 2>&1 || { echo "FIT FAILED"; exit 1; }
tail -5 "$LOGD/fit.log"

echo "[$(date +%H:%M:%S)] stage C: multi-seed grid (pool=$POOL)"
export OPT_SKIP_NONFINITE=1
for d in $GRID_DIMS; do for m in $METHODS; do echo "$m $d"; done; done | \
  xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 python experiments/dim_scaling_lawv2/run_lawv2_cell.py \
       --method $0 --dim $1 \
       > experiments/dim_scaling_lawv2/logs/grid_$0_d$1.log 2>&1; \
     echo "  done grid $0 d=$1"'
echo "[$(date +%H:%M:%S)] stage C done"

python "$HERE/plot_lawv2.py" > "$LOGD/plot.log" 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE"
