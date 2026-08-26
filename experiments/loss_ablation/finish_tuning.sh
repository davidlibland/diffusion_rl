#!/bin/bash
# Finish tuning for the two arms added after the first wave, then grid them.
set -u
cd "$(dirname "$0")"
: > joblist_lrscan_rest.txt
for loss in gkl logmse; do for d in 8 32 128; do for lr in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3; do
  echo "$loss $d $lr 3 4000 lr${lr}" >> joblist_lrscan_rest.txt
done; done; done
xargs -a joblist_lrscan_rest.txt -P 10 -L 1 ./job.sh
python pick_lr.py > logs/chosen_lr.txt 2>&1
python - <<'PY' > joblist_grid_rest.txt
import json
c=json.load(open("chosen_lr.json"))
for k,v in sorted(c.items()):
    loss,d = k.rsplit("_d",1)
    if loss in ("gkl","logmse"):
        print(f"{loss} {d} {v['lr']:.6g} 12 8000 grid")
PY
xargs -a joblist_grid_rest.txt -P 10 -L 1 ./job.sh
python analyse_ablation.py | tee logs/ablation_summary.txt
echo "=== FULL ABLATION COMPLETE ==="
