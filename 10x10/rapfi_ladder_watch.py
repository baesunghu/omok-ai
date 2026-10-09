"""
Runs rapfi_ladder.py automatically every EVERY epochs of the live run
(2026-09-28). Checks train_state_11x11.json once a minute; when the
cumulative epoch reaches the next multiple of EVERY, it snapshots the
tracked best checkpoint and runs the ladder against that snapshot (so a
promotion mid-ladder cannot change the model being measured).
Results accumulate in model/rapfi_ladder_log.jsonl.
Run detached:  python -u rapfi_ladder_watch.py
"""
import json, os, shutil, subprocess, sys, time
TENX = r"C:\Users\My PC\Desktop\final\10x10"
STATE = os.path.join(TENX, "model", "train_state_11x11.json")
BEST = os.path.join(TENX, "model", "best_mcts_11x11.pth")
SNAP = os.path.join(TENX, "model", "_rapfi_ladder_snapshot.pth")
EVERY = 20
GAMES = 20  # per ladder level (2026-09-28: 10 -> 20 for a tighter graduation verdict)


def epoch():
    try:
        return int(json.load(open(STATE))["cumulative_epoch"])
    except Exception:
        return None


# 2026-10-09: the ep820 run died after the problem check -- on Windows, a child output that the
# parent cannot decode with the locale codec leaves r.stdout as None. Force UTF-8 both ways.
RUN_KW = {"encoding": "utf-8", "errors": "replace", "env": dict(os.environ, PYTHONIOENCODING="utf-8")}


def same_weights(a, b):
    try:
        import torch
        x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
        return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
    except Exception:
        return False


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    e = epoch() or 0
    next_at = (e // EVERY + 1) * EVERY
    log(f"rapfi ladder watcher started at epoch {e}; next run at epoch {next_at}")
    while True:
        e = epoch()
        if e is not None and e >= next_at:
            shutil.copy2(BEST, SNAP)
            log(f"epoch {e}: running ladder on best checkpoint snapshot")
            r = subprocess.run([sys.executable, "-u", "rapfi_ladder.py", str(GAMES), "1100", SNAP],
                               cwd=TENX, capture_output=True, text=True, **RUN_KW)
            log((r.stdout or '').strip() or (r.stderr or '').strip()[-500:])
            # 11x11 graduation problems (end criterion 2), same snapshot
            r2 = subprocess.run([sys.executable, "-u", "tactical_problem_check.py", SNAP],
                                cwd=TENX, capture_output=True, text=True, **RUN_KW)
            log(chr(10).join(l for l in (r2.stdout or '').splitlines() if not l.startswith("MESSAGE")) or (r2.stderr or '').strip()[-500:])
            # 2026-09-28: also measure the newest training checkpoint (not just the gated best),
            # so progress is visible even while the gate keeps rejecting candidates
            latest = os.path.join(TENX, "model", f"model_MCTS_ep{next_at}_11x11.pth")
            for _ in range(30):
                if os.path.exists(latest):
                    break
                time.sleep(60)
            if os.path.exists(latest) and same_weights(latest, SNAP):
                log(f"latest checkpoint ep{next_at} is the promoted best (same weights); not measured twice")
            elif os.path.exists(latest):
                snap2 = os.path.join(TENX, "model", f"_rapfi_latest_ep{next_at}.pth")
                shutil.copy2(latest, snap2)
                log(f"running ladder on latest checkpoint ep{next_at}")
                r = subprocess.run([sys.executable, "-u", "rapfi_ladder.py", str(GAMES), "1100", snap2],
                                   cwd=TENX, capture_output=True, text=True, **RUN_KW)
                log((r.stdout or '').strip() or (r.stderr or '').strip()[-500:])
                r2 = subprocess.run([sys.executable, "-u", "tactical_problem_check.py", snap2],
                                    cwd=TENX, capture_output=True, text=True, **RUN_KW)
                log(chr(10).join(l for l in (r2.stdout or '').splitlines() if not l.startswith("MESSAGE")) or (r2.stderr or '').strip()[-500:])
            else:
                log(f"latest checkpoint ep{next_at} not found; skipped")
            next_at = (e // EVERY + 1) * EVERY
            log(f"next run at epoch {next_at}")
        time.sleep(60)


if __name__ == "__main__":
    main()
