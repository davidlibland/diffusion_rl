#!/usr/bin/env bash
# Probe 4 driver: eps scan (3 seeds) -> pick best eps per arm -> full 10 seeds.
# Resumable at every level (per-seed JSONs; full phase resumes scan seeds).
set -uo pipefail
REPO=${REPO:-"$(dirname "$0")/../.."}
cd "$REPO"
HERE=experiments/probe4_trust_region
LOGD="$HERE/logs"
mkdir -p "$LOGD"
POOL=${POOL:-4}
ARMS="tr_twist tr_oracle_ramp tr_unweight"
EPSILONS="0.3 1.0 3.0"
export OPT_SKIP_NONFINITE=1

echo "[$(date +%H:%M:%S)] phase 1: eps scan (3 seeds each, pool=$POOL)"
for arm in $ARMS; do for e in $EPSILONS; do echo "$arm $e"; done; done | \
  xargs -P "$POOL" -L1 bash -c \
    'OMP_NUM_THREADS=4 DSP_SEEDS=3 python experiments/probe4_trust_region/probe4_cell.py \
       --arm $0 --eps $1 \
       > experiments/probe4_trust_region/logs/scan_$0_eps$1.log 2>&1; \
     echo "  done scan $0 eps=$1"'
echo "[$(date +%H:%M:%S)] phase 1 done"

echo "[$(date +%H:%M:%S)] phase 2: select best eps per arm"
python - <<'EOF'
import json, math, os
import numpy as np
best = {}
for arm in ("tr_twist", "tr_oracle_ramp", "tr_unweight"):
    scores = {}
    for e in ("0.3", "1.0", "3.0"):
        p = f"experiments/probe4_trust_region/results/probe4_{arm}_eps{float(e):g}.json"
        if not os.path.exists(p):
            continue
        fr = [s["frac_closed"] for s in json.load(open(p))["seeds"]
              if isinstance(s.get("frac_closed"), float)
              and math.isfinite(s["frac_closed"])]
        if fr:
            scores[e] = float(np.mean(fr))
    if scores:
        b = max(scores, key=scores.get)
        best[arm] = b
        print(f"  {arm}: " + ", ".join(f"eps={k}:{100*v:.1f}%"
              for k, v in sorted(scores.items())) + f"  -> best eps={b}")
json.dump(best, open("experiments/probe4_trust_region/results/best_eps.json", "w"))
EOF

echo "[$(date +%H:%M:%S)] phase 3: full 10 seeds at best eps (pool=3)"
python -c "
import json
best = json.load(open('experiments/probe4_trust_region/results/best_eps.json'))
print('\n'.join(f'{a} {e}' for a, e in best.items()))" | \
  xargs -P 3 -L1 bash -c \
    'OMP_NUM_THREADS=4 DSP_SEEDS=10 python experiments/probe4_trust_region/probe4_cell.py \
       --arm $0 --eps $1 \
       > experiments/probe4_trust_region/logs/full_$0_eps$1.log 2>&1; \
     echo "  done full $0 eps=$1"'
echo "[$(date +%H:%M:%S)] ALL DONE"
