#!/usr/bin/env bash
# CATFISH A/B seed replicate on a Kaggle GPU (2026-11-01, user: run experiments on Kaggle to finish sooner).
# Layout after unpacking the bundle into $W (= /kaggle/working/naps):
#   $W/11x11_v2/{alphago_lee*.py}, $W/11x11_v2/v3/{net_v3.py, cpp/mcts_core_solver.cpp},
#   $W/11x11_v2/v3/ab_scratch/{A_nocatfish_sN,B_catfish_sN}/train_v3.py   (same files as on the laptop)
# Usage: bash run_seed_kaggle.sh <seed number> [epochs=60] [hours budget=11.5]
# Each arm stops after <epochs> (STOP file at epoch-1, like run_seeds.ps1). Re-running resumes from model/v3_latest.pth,
# so a session that hits Kaggle's 12 h limit can be continued in a new session from the saved output.
set -euo pipefail
S=${1:?seed number}; EPOCHS=${2:-60}; HOURS=${3:-11.5}
W=${W:-/kaggle/working/naps}
V3=$W/11x11_v2/v3
DEADLINE=$(( $(date +%s) + ${HOURS%.*} * 3600 ))

if ! python -c "import sys; sys.path.insert(0, '$V3/cpp'); import mcts_core" 2>/dev/null; then
  pip install -q pybind11
  ( cd "$V3/cpp" && g++ -O2 -std=c++17 -shared -fPIC $(python -m pybind11 --includes) \
      -DMCTS_MODULE_NAME=mcts_core mcts_core_solver.cpp -o "mcts_core$(python3-config --extension-suffix)" -lpthread )
fi
python -c "import sys; sys.path.insert(0, '$V3/cpp'); import mcts_core; print('mcts_core ok', mcts_core.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc

for arm in A_nocatfish_s$S B_catfish_s$S; do
  d=$V3/ab_scratch/$arm; mkdir -p "$d/model"; cd "$d"
  if grep -q "\] Epoch $EPOCHS " training_report_v3.txt 2>/dev/null; then echo "$arm already done"; continue; fi
  rm -f STOP
  python -u train_v3.py > >(tee -a stdout.log) 2>> stderr.log &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if grep -q "\] Epoch $((EPOCHS - 1)) " training_report_v3.txt 2>/dev/null; then touch STOP; fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "time budget reached: stopping $arm after this epoch"; touch STOP; fi
  done
  tail -n 2 training_report_v3.txt
  if ! grep -q "\] Epoch $EPOCHS " training_report_v3.txt; then echo "$arm not finished; rerun to resume"; exit 0; fi
done
echo "seed $S: both arms done"
