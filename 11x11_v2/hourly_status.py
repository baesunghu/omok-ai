"""
Hourly health check (2026-09-28). Detached; appends one block per hour to
model/hourly_status.log: epoch, last gate result, latest Rapfi ladder and
graduation-problem results, GPU temperature, web server and tunnel status.
Run:  python -u hourly_status.py
"""
import json, os, subprocess, time, urllib.request
TENX = r"C:\Users\My PC\Desktop\final\11x11_v2"
M = os.path.join(TENX, "model")
OUT = os.path.join(M, "hourly_status.log")
TUNNEL = "https://calls-berkeley-consequently-utc.trycloudflare.com/api/stats"


def last_line(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def ok(url):
    try:
        return urllib.request.urlopen(url, timeout=15).status == 200
    except Exception:
        return False


def check():
    out = [time.strftime("[%Y-%m-%d %H:%M]")]
    try:
        out.append(f"epoch {json.load(open(os.path.join(M, 'train_state_11x11.json')))['cumulative_epoch']}")
    except Exception as e:
        out.append(f"epoch ? ({e})")
    try:
        with open(max(__import__("glob").glob(os.path.join(TENX, "train_stdout_*.log")), key=os.path.getmtime), encoding="utf-8", errors="ignore") as f:
            gates = [l.strip() for l in f if "Gate vs best" in l]
        if gates:
            out.append("gate: " + gates[-1].split("Gate vs best")[-1][:70])
    except Exception:
        pass
    lad = last_line(os.path.join(M, "rapfi_ladder_log.jsonl"))
    if lad:
        out.append(f"ladder {lad['time']} {lad['checkpoint']}: equal nodes ~{lad.get('equal_nodes')} "
                   + " ".join(f"{l['nodes']}:{round(l['score']*100)}%" for l in lad["levels"]))
    tp = last_line(os.path.join(M, "tactical_problem_log.jsonl"))
    if tp:
        out.append(f"problems {tp['time']}: {tp['passed']}/{tp['total']}")
    try:
        t = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        warn = "  <-- HOT" if t and int(t.split(",")[0]) >= 92 else ""
        out.append(f"gpu {t}{warn}")
    except Exception:
        out.append("gpu ?")
    out.append(f"web local {'OK' if ok('http://127.0.0.1:8765/api/stats') else 'DOWN'}, tunnel {'OK' if ok(TUNNEL) else 'DOWN'}")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(" | ".join(out) + "\n")


while True:
    check()
    time.sleep(3600)
