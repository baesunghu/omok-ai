"""11x11_v3 live terminal monitor (2026-10-22, user): one screen, refreshed every 2 s, no Claude needed.
Usage: python monitor_v3.py      (Ctrl+C to quit)
Shows: latest epochs (losses, game length, X/O/D, speed), loss trend, Rapfi ladder results / progress,
processes, GPU.
"""
import json
import os
import re
import subprocess
import time

D = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(D, *a)
V2 = os.path.dirname(D)
RE = re.compile(r"\[(.+?)\] Epoch (\d+) \| loss ([\d.]+) policy ([\d.]+) value ([\d.]+) opp ([\d.]+) own ([\d.]+) "
                r"surv ([\d.]+) \| target entropy ([\d.]+) \| games \d+: X (\d+) O (\d+) D (\d+), avg len ([\d.]+)"
                r".*?selfplay (\d+)s")


def tail(path, n):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def epochs():
    out = []
    for l in tail(P("training_report_v3.txt"), 100000):
        m = RE.search(l)
        if m:
            g = m.groups()
            out.append(dict(t=g[0][11:], ep=int(g[1]), loss=float(g[2]), pol=float(g[3]), val=float(g[4]),
                            opp=float(g[5]), own=float(g[6]), ent=float(g[8]), x=int(g[9]), o=int(g[10]),
                            d=int(g[11]), len=float(g[12]), sp=int(g[13])))
    return out


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return "?"


def procs():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-c",
                            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | % { $_.CommandLine }"],
                           capture_output=True, text=True, timeout=8,
                           encoding="utf-8", errors="replace").stdout or ""   # 2026-10-23: cp949 decode crash on Korean paths
    except Exception:
        return "?"
    has = lambda s: "O" if s in r else "-"
    return (f"train_v3 {has('train_v3.py')}   ladder_watch_v3 {has('ladder_watch_v3.py')}   "
            f"rapfi_match {has('rapfi_match.py')}")


def spark(vals, w=40):
    if len(vals) < 2:
        return ""
    vals = vals[-w:]
    lo, hi = min(vals), max(vals)
    bars = " ▁▂▃▄▅▆▇█"
    return "".join(bars[int((v - lo) / (hi - lo + 1e-12) * 8)] for v in vals)


def ladder():
    lines = []
    if os.path.exists(P("ladder_v3.jsonl")):
        for r in map(json.loads, open(P("ladder_v3.jsonl"), encoding="utf-8")):
            eq = r["equal_nodes"]
            eqs = f"{eq:,.0f}" if isinstance(eq, (int, float)) else str(eq)
            lv = "  ".join(f"{x['nodes']}:{x['score']:.0%}" for x in r["levels"])
            lines.append(f"  ep{r['epoch']:<5} equal {eqs:>8}   [{lv}]")
    cur = [l.strip() for l in tail(P("ladder_watch_v3.log"), 3) if l.strip()]
    return lines, cur[-1] if cur else ""


prev_proc, proc_t = "", 0.0
while True:
    E = epochs()
    if time.time() - proc_t > 10:
        prev_proc, proc_t = procs(), time.time()
    out = [f"=== 11x11_v3 (Gumbel + aux, 128x8, cold start) monitor  {time.strftime('%H:%M:%S')} ===",
           f"processes: {prev_proc}", f"GPU (temp, util, mem): {gpu()}", ""]
    if E:
        last = E[-1]
        per = (sum(e["sp"] for e in E[-10:]) / len(E[-10:]))
        out.append(f"epoch {last['ep']}  (last {last['t']}, self-play ~{per:.0f}s/epoch)")
        out.append(f"{'ep':>5} {'time':>8} {'loss':>7} {'policy':>7} {'value':>6} {'opp':>6} {'own':>6} {'ent':>5} "
                   f"{'len':>5} {'X/O/D':>11}")
        for e in E[-12:]:
            out.append(f"{e['ep']:>5} {e['t']:>8} {e['loss']:7.3f} {e['pol']:7.3f} {e['val']:6.3f} {e['opp']:6.3f} "
                       f"{e['own']:6.3f} {e['ent']:5.2f} {e['len']:5.1f} {e['x']:>3}/{e['o']:>3}/{e['d']:>3}")
        out.append("")
        out.append(f"policy loss  {spark([e['pol'] for e in E])}")
        out.append(f"value loss   {spark([e['val'] for e in E])}")
        out.append(f"game length  {spark([e['len'] for e in E])}  ({E[0]['len']:.1f} -> {last['len']:.1f})")
        dr = [100 * e["d"] / max(1, e["x"] + e["o"] + e["d"]) for e in E]
        xr = [100 * e["x"] / max(1, e["x"] + e["o"]) for e in E]
        out.append(f"draw %       {spark(dr)}  (last {dr[-1]:.0f}%, avg of last 5 {sum(dr[-5:]) / len(dr[-5:]):.0f}%)")
        out.append(f"X share %    {spark(xr)}  (X wins / decisive games, last {xr[-1]:.0f}%)")
        nxt = (last["ep"] // 100 + 1) * 100
        out.append(f"next ladder checkpoint: ep{nxt} (~{(nxt - last['ep']) * per / 60:.0f} min)")
    else:
        out.append("no epochs yet")
    lad, cur = ladder()
    out += ["", "Rapfi ladder (v3, 20 games/level; v2 ep40 = ~2,500, ep960 = 2,708):"] + (lad or ["  (none yet)"])
    if cur:
        out.append(f"  watcher: {cur[:110]}")
    os.system("cls" if os.name == "nt" else "clear")
    print("\n".join(out), flush=True)
    time.sleep(2)
