#!/usr/bin/env bash
# CATFISH A/B seed replicate on a Kaggle GPU T4 x2 session (2026-11-01, user: run experiments on Kaggle to finish sooner).
# Layout after kaggle_seed.py sets up $W (= /kaggle/working/naps):
#   $W/11x11_v2/{alphago_lee*.py}, $W/11x11_v2/v3/{net_v3.py, cpp/mcts_core_solver.cpp},
#   $W/11x11_v2/v3/ab_scratch/{A_nocatfish_sN,B_catfish_sN}/train_v3.py   (same files as on the laptop)
# A and B run at the same time, one GPU each (CUDA_VISIBLE_DEVICES 0 / 1), sharing the CPU equally, so both arms see
# the same conditions. Each arm stops after <epochs> (STOP at epoch-1, like run_seeds.ps1) or when the time budget
# runs out; rerunning resumes from model/v3_latest.pth (attach the previous output as input).
# Usage: bash run_seed_kaggle.sh <seed number> [epochs=60] [hours budget=11.5]
set -uo pipefail
S=${1:?seed number}; EPOCHS=${2:-60}; HOURS=${3:-11.5}
W=${W:-/kaggle/working/naps}
V3=$W/11x11_v2/v3
DEADLINE=$(python -c "import time; print(int(time.time() + $HOURS * 3600))")

if ! python -c "import sys; sys.path.insert(0, '$V3/cpp'); import mcts_core" 2>/dev/null; then
  pip install -q pybind11
  ( cd "$V3/cpp" && g++ -O2 -std=c++17 -shared -fPIC $(python -m pybind11 --includes) \
      -DMCTS_MODULE_NAME=mcts_core mcts_core_solver.cpp -o "mcts_core$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")" -lpthread ) || exit 1
fi
python -c "import sys; sys.path.insert(0, '$V3/cpp'); import mcts_core; print('mcts_core ok', mcts_core.__file__)" || exit 1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; echo "cpus: $(nproc)"

gpu=0
for arm in A_nocatfish_s$S B_catfish_s$S; do
  d=$V3/ab_scratch/$arm; mkdir -p "$d/model"
  if grep -q "\] Epoch $EPOCHS " "$d/training_report_v3.txt" 2>/dev/null; then echo "$arm already done"; gpu=$((gpu+1)); continue; fi
  rm -f "$d/STOP"
  ( cd "$d" && CUDA_VISIBLE_DEVICES=$gpu exec python -u train_v3.py > stdout.log 2> stderr.log ) &
  echo "$arm on GPU $gpu (pid $!)"
  gpu=$((gpu+1))
done

while pgrep -f "train_v3.py" > /dev/null; do
  sleep 60
  for arm in A_nocatfish_s$S B_catfish_s$S; do
    d=$V3/ab_scratch/$arm
    if grep -q "\] Epoch $((EPOCHS - 1)) " "$d/training_report_v3.txt" 2>/dev/null; then touch "$d/STOP"; fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then touch "$d/STOP"; fi
  done
  date +%H:%M; for arm in A_nocatfish_s$S B_catfish_s$S; do echo "  $arm: $(tail -n 1 $V3/ab_scratch/$arm/training_report_v3.txt 2>/dev/null | cut -c1-60)"; done
done
for arm in A_nocatfish_s$S B_catfish_s$S; do
  d=$V3/ab_scratch/$arm; tail -n 1 "$d/stderr.log"
  if grep -q "\] Epoch $EPOCHS " "$d/training_report_v3.txt" 2>/dev/null; then echo "$arm done"; else echo "$arm NOT finished (resume next session)"; fi
done
