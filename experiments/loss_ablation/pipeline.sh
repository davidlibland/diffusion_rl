#!/bin/bash
# One driver, run to completion. Every stage is resumable (cells skip finished seeds).
set -u
cd "$(dirname "$0")"
W=${W:-26}

echo "=== stage 1: finish tuning (gkl / logmse at d>=8) ==="
: > jl_tune.txt
for loss in gkl logmse; do for d in 8 32 128; do for lr in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3; do
  echo "$loss $d $lr 3 4000 lr${lr}" >> jl_tune.txt
done; done; done
xargs -a jl_tune.txt -P $W -L 1 ./job.sh
echo "tuning cells: $(grep -h CELL logs/lr*.log | wc -l)/120"

echo "=== stage 2: choose lr per (loss, dim) ==="
python pick_lr.py | tee logs/chosen_lr.txt

echo "=== stage 3: seed grid, 12 paired seeds x 8000 steps ==="
python - <<'PY' > jl_grid.txt
import json
c = json.load(open("chosen_lr.json"))
for k, v in sorted(c.items()):
    loss, d = k.rsplit("_d", 1)
    print(f"{loss} {d} {v['lr']:.6g} 12 8000 grid")
PY
echo "grid cells: $(wc -l < jl_grid.txt)"
xargs -a jl_grid.txt -P $W -L 1 ./job.sh

echo "=== stage 4: analyse ==="
python analyse_ablation.py | tee logs/ablation_summary.txt
echo "=== PIPELINE COMPLETE ==="
