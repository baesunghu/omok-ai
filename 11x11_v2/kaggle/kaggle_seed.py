"""Kaggle notebook entry for a CATFISH A/B seed replicate (2026-11-01). Paste into one notebook cell
(GPU T4 + Internet on) after setting SEED, then "Save Version -> Save & Run All".

  * clones github.com/baesunghu/omok-ai (public), lays the files out exactly as on the laptop
    (11x11_v2/..., v3/ab_scratch/<arm>/train_v3.py), builds the C++ engine with g++, runs A then B
  * resume: add the previous run's output as an input; its naps/ tree is copied back first
"""
import glob
import os
import shutil
import subprocess

SEED = 3
W = "/kaggle/working/naps"
prev = glob.glob(f"/kaggle/input/*/naps")          # output of an earlier (timed-out) version, if attached
if prev and not os.path.exists(W):
    shutil.copytree(prev[0], W)
if not os.path.exists(f"{W}/11x11_v2/v3/net_v3.py"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/baesunghu/omok-ai.git", "/kaggle/tmp_repo"], check=True)
    shutil.copytree("/kaggle/tmp_repo/11x11_v2", f"{W}/11x11_v2", dirs_exist_ok=True)
    for arm in (f"A_nocatfish_s{SEED}", f"B_catfish_s{SEED}"):
        d = f"{W}/11x11_v2/v3/ab_scratch/{arm}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(f"{W}/11x11_v2/kaggle/train_{arm}.py", f"{d}/train_v3.py")
subprocess.run(["bash", f"{W}/11x11_v2/kaggle/run_seed_kaggle.sh", str(SEED), "60", "11.5"], check=True,
               env=dict(os.environ, W=W))
