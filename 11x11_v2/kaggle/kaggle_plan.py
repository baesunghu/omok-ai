"""Kaggle session runner for a list of steps (2026-11-01, user: use as much Kaggle quota as possible, as fast as possible).
One-line notebook cell (GPU T4 x2 + Internet on; attach earlier sessions' outputs as inputs):

    import urllib.request as u; PLAN='p2'; exec(u.urlopen('https://raw.githubusercontent.com/baesunghu/omok-ai/<commit>/11x11_v2/kaggle/kaggle_plan.py').read())

Steps run in order within HOURS:
  ("train", [armA, armB])         two arms in parallel, one GPU each (run_seed_kaggle.sh), 60 epochs
  ("h2h", [(armA, armB, epoch)])  v3/head2head.py, 100 openings x colour swap = 200 games, 4 shards on 2 GPUs.
                                  The v3 engine build plays the same games as the laptop's v2 engine
                                  (kaggle/engine_parity.py: identical move lists), so results compare with seed 1.
Earlier sessions' naps/ trees found under /kaggle/input are merged in first (training resumes, checkpoints reused).
Results: /kaggle/working/naps/kaggle_results.txt (one line per finished h2h) + every arm's report and epoch checkpoints.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

PLANS = {
    # session 2: verdict games for seeds 2/3 first, then two more seeds and their verdict games
    "p2": [("h2h", [("A_nocatfish_s2", "B_catfish_s2", 60), ("A_nocatfish_s3", "B_catfish_s3", 60),
                    ("A_nocatfish_s2", "B_catfish_s2", 30), ("A_nocatfish_s3", "B_catfish_s3", 30)]),
           ("train", ["A_nocatfish_s4", "B_catfish_s4"]),
           ("h2h", [("A_nocatfish_s4", "B_catfish_s4", 60)]),
           ("h2h", [("A_nocatfish_s2", "E_cf192_s2", 60), ("A_nocatfish_s2", "F_strongcat_s2", 60)]),  # if p3 ran
           ("train", ["A_nocatfish_s5", "B_catfish_s5"]),
           ("h2h", [("A_nocatfish_s5", "B_catfish_s5", 60)])],
    # session 3: why does catfish help/hurt -- data cost (E) and catfish strength (F), both against seed-2 A
    "p3": [("train", ["E_cf192_s2", "F_strongcat_s2"]),
           ("h2h", [("A_nocatfish_s2", "E_cf192_s2", 60), ("A_nocatfish_s2", "F_strongcat_s2", 60)])],
}
plan = PLANS[globals().get("PLAN", "p2")]
HOURS = globals().get("HOURS", 11.6)
T_END = time.time() + HOURS * 3600
W = "/kaggle/working/naps"
V2 = f"{W}/11x11_v2"
V3 = f"{V2}/v3"
RES = f"{W}/kaggle_results.txt"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)


# 1) code: fresh clone (code always from GitHub), then merge earlier outputs (arm folders, results)
if not os.path.exists(f"{V3}/net_v3.py"):
    shutil.rmtree("/kaggle/tmp_repo", ignore_errors=True)
    sh("git clone -q --depth 1 https://github.com/baesunghu/omok-ai.git /kaggle/tmp_repo")
    shutil.copytree("/kaggle/tmp_repo/11x11_v2", V2, dirs_exist_ok=True)
    shutil.rmtree("/kaggle/tmp_repo", ignore_errors=True)
for prev in glob.glob("/kaggle/input/**/naps", recursive=True):
    for d in glob.glob(f"{prev}/11x11_v2/v3/ab_scratch/*/"):
        dst = f"{V3}/ab_scratch/{os.path.basename(os.path.dirname(d))}"
        if not os.path.exists(f"{dst}/training_report_v3.txt"):
            shutil.copytree(d, dst, dirs_exist_ok=True)
            print("merged", dst, "from", prev, flush=True)
    if os.path.exists(f"{prev}/kaggle_results.txt"):
        old = open(f"{prev}/kaggle_results.txt").read()
        have = open(RES).read() if os.path.exists(RES) else ""
        open(RES, "a").write("".join(l for l in old.splitlines(True) if l not in have))

# 2) engine: build the v3 module once; head2head imports mcts_core from 11x11_v2/, training from v3/cpp/
if not glob.glob(f"{V3}/cpp/mcts_core*.so"):
    sh("pip install -q pybind11")
    sh(f'cd {V3}/cpp && g++ -O2 -std=c++17 -shared -fPIC $(python -m pybind11 --includes) -DMCTS_MODULE_NAME=mcts_core '
       f'mcts_core_solver.cpp -o "mcts_core$(python -c "import sysconfig; print(sysconfig.get_config_var(\'EXT_SUFFIX\'))")" '
       f'-lpthread 2>&1 | grep -v -i warning | grep -i error || true')
for so in glob.glob(f"{V3}/cpp/mcts_core*.so"):
    shutil.copy(so, V2)
print("engine:", glob.glob(f"{V2}/mcts_core*.so"), flush=True)


def left():
    return (T_END - time.time()) / 3600


def train(arms):
    for arm in arms:
        d = f"{V3}/ab_scratch/{arm}"
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(f"{d}/train_v3.py"):
            shutil.copy(f"{V2}/kaggle/train_{arm}.py", f"{d}/train_v3.py")
    p = subprocess.Popen(["bash", f"{V2}/kaggle/run_seed_kaggle.sh", "0", "60", f"{max(0.1, left() - 0.3):.2f}"],
                         env=dict(os.environ, W=W, ARMS=" ".join(arms)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "warning" not in line and "^" not in line and "|" not in line[:8]:
            print(line, end="", flush=True)
    p.wait()


def h2h(a, b, ep):
    key = f"{a} vs {b} ep{ep}"
    if os.path.exists(RES) and any(l.startswith(key + ":") for l in open(RES)):
        print("already have", key, flush=True); return
    ca, cb = f"{V3}/ab_scratch/{a}/model/v3_ep{ep}.pth", f"{V3}/ab_scratch/{b}/model/v3_ep{ep}.pth"
    if not (os.path.exists(ca) and os.path.exists(cb)):
        print("skip (no checkpoints):", key, flush=True); return
    out = f"{W}/h2h/{a}__{b}__ep{ep}"
    os.makedirs(out, exist_ok=True)
    procs = [subprocess.Popen([sys.executable, "-u", f"{V3}/head2head.py", ca, cb], cwd=V3,
                              stdout=open(f"{out}/h2h_{k}.log", "w"), stderr=open(f"{out}/h2h_{k}.err", "w"),
                              env=dict(os.environ, H2H_ALL="1", H2H_SHARD=f"{k}/4", CUDA_VISIBLE_DEVICES=str(k % 2)))
             for k in range(4)]
    t0 = time.time()
    for p in procs:
        p.wait()
    w = l = d = 0
    for k in range(4):
        for line in open(f"{out}/h2h_{k}.log"):
            line = line.strip()
            w += line.endswith("-> A"); l += line.endswith("-> B"); d += line.endswith("-> draw")
    n = w + l + d
    msg = f"{key}: A {w} / B {l} / draw {d} (n={n}), B score {(l + 0.5 * d) / max(1, n):.3f} [{(time.time() - t0) / 60:.0f} min]"
    print(msg, flush=True)
    if n == 200:
        open(RES, "a").write(msg + "\n")


for kind, what in plan:
    if left() < 0.4:
        print(f"time budget nearly used ({left():.2f} h left); stopping before {kind} {what}", flush=True)
        break
    print(f"=== {kind} {what} ({left():.1f} h left) ===", flush=True)
    if kind == "train":
        train(what)
    else:
        for a, b, ep in what:
            if left() < 0.4:
                break
            h2h(a, b, ep)
# outputs: drop replay buffers of finished arms (large); unfinished arms keep theirs to resume
for rep in glob.glob(f"{V3}/ab_scratch/*/training_report_v3.txt"):
    if "] Epoch 60 " in open(rep).read():
        buf = os.path.join(os.path.dirname(rep), "model", "v3_buffer.npz")
        if os.path.exists(buf):
            os.remove(buf)
print(open(RES).read() if os.path.exists(RES) else "no results yet")
