"""Live progress of the current Rapfi ladder run (2026-10-08).
Counts the games in model/rapfi_match_log.jsonl since the last "running ladder" line
of rapfi_ladder_watch.log, 20 games per level (300 / 1000 / 2000 / 3000 nodes).
Usage:  python rapfi_progress.py          (print once)
        python rapfi_progress.py watch    (refresh every 30 s, Ctrl+C to stop)
        python rapfi_progress.py "2026-10-08 22:39"   (count from a given time)
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [1000, 2000, 3000, 5000, 10000, 20000]  # 2026-10-10 evening: 300 dropped, 10000 added
GAMES = 20
DAWN = {300: 55.0, 1000: 57.5, 2000: 50.0, 3000: 37.5}


def ladder_start():
    start, what = None, ""
    import glob
    for fn in [os.path.join(HERE, "rapfi_ladder_watch.log")] + glob.glob(os.path.join(HERE, "_rapfi_oneoff_*.out")):
        with open(fn, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*running ladder.*)", line)
                if m and (start is None or m.group(1) > start):
                    start, what = m.group(1), m.group(2)
    return start, what


def show(since, what):
    rows = []
    with open(os.path.join(HERE, "model", "rapfi_match_log.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("time", "") >= since:
                rows.append(r)
    print(f"{time.strftime('%H:%M:%S')}  since {since}  {what}")
    print(f"{'nodes':>6} {'games':>6}  W  D  L  {'score':>6}  {'Dawn':>6}")
    # 2026-10-12: group by the run's checkpoint and Rapfi node count (not by position),
    # so one-off runs with other levels (e.g. 1500/3000/5000) and cancelled runs show correctly
    if rows:
        ck = rows[-1].get("checkpoint")
        rows = [r for r in rows if r.get("checkpoint") == ck]
        print(f"  checkpoint: {ck}")
    levels = sorted({r.get("rapfi_nodes") for r in rows if r.get("rapfi_nodes")} | set(LEVELS))
    for nodes in levels:
        seg = [r for r in rows if r.get("rapfi_nodes") == nodes]
        if not seg:
            print(f"{nodes:>6}   waiting")
            continue
        w = sum(r["result"] == "ai_win" for r in seg)
        d = sum(r["result"] == "draw" for r in seg)
        l = sum(r["result"] == "rapfi_win" for r in seg)
        s = (w + d / 2) / len(seg) * 100
        dawn_col = f"{DAWN[nodes]:>5.1f}%" if nodes in DAWN else "    -"
        print(f"{nodes:>6} {len(seg):>3}/{GAMES} {w:>2} {d:>2} {l:>2}  {s:>5.1f}%  {dawn_col}")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg and arg != "watch":
        show(arg, "")
    else:
        while True:
            s, w = ladder_start()
            if s is None:
                print("no ladder run found in rapfi_ladder_watch.log")
                break
            if arg == "watch":
                os.system("cls")
            show(s, w)
            if arg != "watch":
                break
            time.sleep(30)
