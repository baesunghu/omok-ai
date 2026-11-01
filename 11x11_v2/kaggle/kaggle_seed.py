"""Kaggle notebook entry for a CATFISH A/B seed replicate (2026-11-01).
The notebook cell is a single line (typing long code into the Kaggle editor through the browser was unreliable):

    import urllib.request as u; SEED=3; EPOCHS=60; HOURS=11.5; exec(u.urlopen('https://raw.githubusercontent.com/baesunghu/omok-ai/main/11x11_v2/kaggle/kaggle_seed.py').read())

(GPU T4 x2 + Internet on). It clones github.com/baesunghu/omok-ai, lays the files out exactly as on the laptop
(11x11_v2/..., v3/ab_scratch/<arm>/train_v3.py), builds the C++ engine and runs A and B in parallel, streaming the
runner's progress into the cell output. Set TEST=True for a short smoke run in a separate folder.
Resume: attach the previous version's output as an input; its naps/ tree is copied back first.
"""
import glob
import os
import shutil
import subprocess

import time

SEEDS = globals().get("SEEDS") or [globals().get("SEED", 3)]   # several seeds run one after another in one session
EPOCHS = globals().get("EPOCHS", 60)
HOURS = globals().get("HOURS", 11.5)                             # budget for the whole session
T_END = time.time() + HOURS * 3600
W = "/kaggle/working/test/naps" if globals().get("TEST") else "/kaggle/working/naps"
if globals().get("TEST"):
    shutil.rmtree(os.path.dirname(W), ignore_errors=True)
prev = glob.glob("/kaggle/input/*/naps") + glob.glob("/kaggle/input/*/*/naps")
if prev and not os.path.exists(W):
    shutil.copytree(prev[0], W)
    print("resumed from", prev[0], flush=True)
if not os.path.exists(f"{W}/11x11_v2/v3/net_v3.py"):
    shutil.rmtree("/kaggle/tmp_repo", ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/baesunghu/omok-ai.git", "/kaggle/tmp_repo"], check=True)
    shutil.copytree("/kaggle/tmp_repo/11x11_v2", f"{W}/11x11_v2", dirs_exist_ok=True)
for seed in SEEDS:
    left = (T_END - time.time()) / 3600
    if left < 0.5:
        print(f"seed {seed}: not started, {left:.2f} h left", flush=True)
        break
    for arm in (f"A_nocatfish_s{seed}", f"B_catfish_s{seed}"):
        d = f"{W}/11x11_v2/v3/ab_scratch/{arm}"
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(f"{d}/train_v3.py"):
            shutil.copy(f"{W}/11x11_v2/kaggle/train_{arm}.py", f"{d}/train_v3.py")
    print(f"=== seed {seed} ({left:.1f} h left) ===", flush=True)
    p = subprocess.Popen(["bash", f"{W}/11x11_v2/kaggle/run_seed_kaggle.sh", str(seed), str(EPOCHS), f"{left:.2f}"],
                         env=dict(os.environ, W=W), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "warning" not in line and "^" not in line and "|" not in line[:8]:
            print(line, end="", flush=True)
    p.wait()
    print(f"seed {seed}: runner exit {p.returncode}", flush=True)
# keep only what evaluation needs in the output: epoch checkpoints, reports, logs (buffers are large; latest/buffer are
# kept for a resume only if a seed is unfinished)
for seed in SEEDS:
    for arm in (f"A_nocatfish_s{seed}", f"B_catfish_s{seed}"):
        d = f"{W}/11x11_v2/v3/ab_scratch/{arm}"
        rep = f"{d}/training_report_v3.txt"
        if os.path.exists(rep) and f"] Epoch {EPOCHS} " in open(rep).read():
            for f in ("v3_buffer.npz",):
                if os.path.exists(f"{d}/model/{f}"):
                    os.remove(f"{d}/model/{f}")
shutil.rmtree(f"{W}/../tmp_repo", ignore_errors=True)
