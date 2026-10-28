"""15x15 live terminal monitor (2026-10-16, user): one screen, refreshed every second, no Claude needed.
Usage: python monitor.py      (Ctrl+C to quit)
Shows: run-3 chain stage, data/pretraining progress, latest epochs, gates, ladder progress/results, processes, GPU.
"""
import glob
import json
import os
import re
import subprocess
import time
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(D, *a)


def tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def count_lines(pattern):
    t = 0
    for fn in glob.glob(P(pattern)):
        with open(fn, encoding="utf-8", errors="ignore") as f:
            t += sum(1 for _ in f)
    return t


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return "?"


def procs():
    try:
        out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return "?"
    names = {"_restart_train15": "학습", "watchdog_": "워치독", "rapfi_ladder_watch": "래더감시",
             "rapfi_ladder.py": "래더측정", "promotion_sms": "문자", "sl_rapfi_data": "데이터", "sl_pretrain": "지도학습"}
    return " ".join(f"{v}O" if k in out else f"{v}-" for k, v in names.items())


def ladder_progress():
    c = defaultdict(list)
    for line in tail(P("model", "rapfi_match_log.jsonl"), 400):
        try:
            r = json.loads(line.lstrip("﻿"))
        except ValueError:
            continue
        c[(r["checkpoint"], r["rapfi_nodes"])].append({"ai_win": 1, "draw": .5}.get(r["result"], 0))
    if not c:
        return []
    ck = list(c)[-1][0]
    return [f"  {ck}"] + [f"    {n:>6}노드 {len(s):2}판 {100 * sum(s) / len(s):5.1f}%" for (k, n), s in c.items() if k == ck]


import sys
if os.name == "nt":
    os.system("chcp 65001 >nul")                     # Korean in cmd/PowerShell
sys.stdout.reconfigure(encoding="utf-8")
last_gpu, last_proc, t_slow = "", "", 0
while True:
    if time.time() - t_slow > 10:                      # slow probes every 10 s
        last_gpu, last_proc, t_slow = gpu(), procs(), time.time()
    out = [f"=== 15x15 모니터  {time.strftime('%m-%d %H:%M:%S')}   GPU(°C,%) {last_gpu} ===", f"프로세스: {last_proc}", ""]
    out.append(f"[데이터] Rapfi10 대국 {count_lines('sl_data/rapfi10_w*.jsonl'):,} / 8,000")
    out.append("[체인 로그 sl_chain3.log]")
    out += ["  " + l.rstrip()[:150] for l in tail(P("sl_chain3.log"), 6)]
    out.append("")
    out.append("[학습 최근 에포크]")
    for l in tail(P("model", "training_report_15x15.txt"), 4):
        m = re.search(r"\[(\S+ \S+)\] Epoch (\d+)/\d+ \| (.*?)\| Buffer Size: (\d+)", l)
        g = re.search(r"Gate vs best.*", l)
        if m:
            out.append(f"  ep{m.group(2):>3} {m.group(1)[5:16]} {m.group(3).strip()[:70]} buf {m.group(4)}"
                       + (f"\n        {g.group(0)[:110]}" if g else ""))
        elif "Warming up" in l:
            out.append("  " + l.strip()[:110])
    st = tail(P("train_stdout_15.log"), 1)
    out.append("  진행: " + (st[0].strip()[:130] if st else "-"))
    out.append("")
    out.append("[래더 진행 (마지막 체크포인트)]")
    out += ladder_progress()
    out.append("[래더 감시 로그]")
    out += ["  " + l.rstrip()[:150] for l in tail(P("rapfi_ladder_watch.log"), 3)]
    os.system("cls" if os.name == "nt" else "clear")
    print("\n".join(out), flush=True)
    time.sleep(1)
