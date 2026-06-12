#!/usr/bin/env bash
# Queued failure-mode probe: waits for master.sh to finish, then runs the
# probe arms (3 SMC methods x d{128,512} x 3 arms x 10 paired seeds).
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../.."}
cd "$REPO"
LOGD=${LOGD:-/tmp/consth_logs}
until grep -q "ALL DONE" /tmp/consth_master.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] grid done -- starting probe"
export OPT_SKIP_NONFINITE=1
for d in 512 128; do
  for m in single_seed_mc single_seed_td_lambda ancestral_mc_td_lambda; do
    for a in reward_twist mc16 guid_toggle; do echo "$m $d $a"; done
  done
done | xargs -P "${POOL:-6}" -L1 bash -c \
  'OMP_NUM_THREADS=4 python experiments/dim_scaling_consth/probe_cell.py \
     --method $0 --dim $1 --arm $2 > '"$LOGD"'/probe_$2_$0_d$1.log 2>&1; \
   echo "  done probe $2 $0 d=$1"'
echo "[$(date +%H:%M:%S)] PROBE DONE"
