"""Rapfi ladder for every 100th 11x11_v3 checkpoint (2026-10-22, user).

Waits for model/v3_ep{N}.pth with N % EVERY == 0, runs ../rapfi_ladder.py on it (same match
engine as v2: noiseless PUCT search with VCF, so v2 and v3 equal points are comparable) and
appends the result to ladder_v3.jsonl. Levels follow the last measured equal point
(eq/3, eq, eq*3, eq*9 -- the first run uses 1000..30000, set from the quick ep70 run). Runs alongside training on the
same GPU. Stop: create the file STOP_LADDER next to this script.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
EVERY = 25          # 10/25 (user): measure every 25 epochs
GAMES = 20          # per level
SIMS = 1100         # same as the v2 ladder
OUT = os.path.join(HERE, "ladder_v3.jsonl")
LOG = os.path.join(V2, "model", "rapfi_ladder_log.jsonl")
TLOG = os.path.join(V2, "model", "tactical_problem_log.jsonl")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def done_epochs():
    if not os.path.exists(OUT):
        return {}
    return {r["epoch"]: r for r in map(json.loads, open(OUT, encoding="utf-8"))}


def levels_from(eq):
    if not isinstance(eq, (int, float)):
        return [30000, 10000, 3000, 1000]   # hardest first (2026-10-22 user: decide sooner); 2026-10-22: ep70 quick ladder already scored 50% at 3000
    base = max(30, eq)
    return sorted({int(round(base * f, -1)) for f in (1 / 3, 1, 3, 9)}, reverse=True)   # hardest first


while not os.path.exists(os.path.join(HERE, "STOP_LADDER")):
    done = done_epochs()
    todo = sorted(int(m.group(1)) for p in glob.glob(os.path.join(HERE, "model", "v3_ep*.pth"))
                  if (m := re.search(r"v3_ep(\d+)\.pth$", p)) and int(m.group(1)) % EVERY == 0
                  and int(m.group(1)) not in done
                  and int(m.group(1)) > max(done, default=0))   # never go back to older unmeasured checkpoints
    if not todo:
        time.sleep(60)
        continue
    ep = todo[-1]                      # newest first; older unmeasured ones are skipped on purpose
    # 10/25 (user, option 1): a flat ladder gives no usable crossing (ep500: ~0 nodes); keep that result as it is, but
    # choose the next levels from the last checkpoint whose equal point is a real number of nodes
    ok_eq = [done[e]["equal_nodes"] for e in sorted(done) if isinstance(done[e]["equal_nodes"], (int, float)) and done[e]["equal_nodes"] >= 1000]
    last = ok_eq[-1] if ok_eq else None
    lv = levels_from(last)
    ck = os.path.join(HERE, "model", f"v3_ep{ep}.pth")
    log(f"ladder v3 ep{ep}: levels {lv}, {GAMES} games each")
    env = dict(os.environ, RAPFI_LEVELS=",".join(map(str, lv)), PYTHONIOENCODING="utf-8")
    n0 = sum(1 for _ in open(LOG, encoding="utf-8")) if os.path.exists(LOG) else 0
    r = subprocess.run([sys.executable, "-u", "rapfi_ladder.py", str(GAMES), str(SIMS), ck], cwd=V2, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], flush=True)
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8").readlines()[n0:]] if os.path.exists(LOG) else []
    if r.returncode != 0 or not rows:
        log(f"ladder failed (rc {r.returncode}): {r.stderr[-800:]}")
        time.sleep(600)
        continue
    res = rows[-1]
    rec = dict(epoch=ep, time=res["time"], equal_nodes=res["equal_nodes"], levels=res["levels"])
    # 2026-10-22 (user): graduation / tier problems right after each ladder
    log(f"problems v3 ep{ep}")
    # 2026-10-22 23:55 (user): stream to the log as it runs -- tier 1 prints first, so it can be read before the rest
    with open(os.path.join(HERE, f"grad_check_ep{ep}.log"), "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u", "tactical_problem_check.py", ck], cwd=V2,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=f, stderr=subprocess.STDOUT)
    try:
        last = json.loads(open(TLOG, encoding="utf-8").readlines()[-1])
        if last.get("checkpoint") == os.path.basename(ck):
            rec["problems"] = {k: last[k] for k in ("passed", "total", "grad_passed", "grad_total", "tiers")}
    except (OSError, ValueError, IndexError):
        pass
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"v3 ep{ep}: equal point {res['equal_nodes']}, problems {rec.get('problems')}")
log("STOP_LADDER found: stopped")
