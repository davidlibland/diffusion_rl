#!/bin/bash
set -u
cd "$(dirname "$0")"
# wait for the tuning scan to finish
until [ "$(grep -h CELL logs/lr*.log 2>/dev/null | wc -l)" -ge 118 ]; do sleep 30; done
echo "=== tuning complete: $(grep -h CELL logs/lr*.log | wc -l) cells ==="
python pick_lr.py | tee logs/chosen_lr.txt
: > joblist_grid.txt
python - <<'PY' >> joblist_grid.txt
import json
c = json.load(open("chosen_lr.json"))
for k, v in sorted(c.items()):
    loss, d = k.rsplit("_d", 1)
    print(f"{loss} {d} {v['lr']:.6g} 16 8000 grid")
PY
echo "=== grid: $(wc -l < joblist_grid.txt) cells x 16 seeds ==="
xargs -a joblist_grid.txt -P 26 -L 1 ./job.sh
python analyse_ablation.py | tee logs/ablation_summary.txt
echo "=== GRID COMPLETE ==="
