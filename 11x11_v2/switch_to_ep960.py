"""11x11_v2 (2026-10-21, user): restart the run from NAPS ep960 instead of Gen12.
The Gen12 attempt (epoch 0 not finished) is archived, nothing is deleted. ep960 is only COPIED from the
festival folder (10x10 stays untouched). Epochs restart at 0 (v2 ep N = NAPS lineage ep960 + N).
"""
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "model")
ARCH = os.path.join(M, "pre_ep960_switch_gen12_20261021")
SRC960 = r"C:\Users\My PC\Desktop\final\10x10\model\model_MCTS_ep960_11x11.pth"

os.makedirs(ARCH, exist_ok=False)
moved = []
for f in os.listdir(M):
    p = os.path.join(M, f)
    if os.path.isfile(p) and f not in ("sim_ramp_state_11x11.json",):
        shutil.move(p, os.path.join(ARCH, f)); moved.append(f)
if os.path.isdir(os.path.join(M, "league_anchors")):
    shutil.move(os.path.join(M, "league_anchors"), os.path.join(ARCH, "league_anchors"))
for d in ("gate_candidates", "gate_disagree"):
    if os.path.isdir(os.path.join(M, d)):
        shutil.move(os.path.join(M, d), os.path.join(ARCH, d))
for log in ("train_stdout.log", "train_stderr.log", "start_v2.log", "measure_catfish.log", "watchdog_log.txt", "train.lock"):
    p = os.path.join(HERE, log)
    if os.path.exists(p):
        shutil.move(p, os.path.join(ARCH, log))
for name in ("model_MCTS_ep0_11x11.pth", "alphazero_mcts_11x11.pth", "best_mcts_11x11.pth", "_rapfi_ep0_11x11.pth"):
    shutil.copy(SRC960, os.path.join(M, name))
for f in ("rapfi_ladder_log.jsonl", "rapfi_match_log.jsonl"):
    open(os.path.join(M, f), "w").close()
print(f"archived {len(moved)} files -> {ARCH}; ep960 copied in as epoch 0")
