#!/bin/bash
# End-to-end loss ablation: tune lr per (loss,dim) -> run the seed grid -> analyse.
# Resumable: every stage skips work whose result file already exists.
set -u
cd "$(dirname "$0")"
W=${W:-12}                 # parallel workers
GRID_SEEDS=${GRID_SEEDS:-16}
GRID_STEPS=${GRID_STEPS:-8000}

echo "=== stage 1: lr scan (wave 1) ==="
xargs -a joblist_lrscan.txt  -P $W -L 1 ./job.sh
echo "=== stage 1: lr scan (wave 2, low lrs) ==="
xargs -a joblist_lrscan2.txt -P $W -L 1 ./job.sh
echo "=== stage 1: lr scan (wave 3, generalised KL) ==="
xargs -a joblist_lrscan3.txt -P $W -L 1 ./job.sh

echo "=== stage 2: choose lr ==="
python pick_lr.py | tee logs/chosen_lr.txt

echo "=== stage 3: seed grid ==="
: > joblist_grid.txt
python - <<'PY' >> joblist_grid.txt
import json, os
c = json.load(open("chosen_lr.json"))
for k, v in sorted(c.items()):
    loss, d = k.rsplit("_d", 1)
    print(f"{loss} {d} {v['lr']:.6g} {os.environ.get('GRID_SEEDS','16')} "
          f"{os.environ.get('GRID_STEPS','8000')} grid")
PY
wc -l < joblist_grid.txt
xargs -a joblist_grid.txt -P $W -L 1 ./job.sh

echo "=== stage 4: analyse ==="
python analyse_ablation.py | tee logs/ablation_summary.txt
echo "=== MASTER DONE ==="
