"""
Runs rapfi_ladder.py automatically every EVERY epochs of the live run
(2026-09-28). Checks train_state_15x15.json once a minute; when the
cumulative epoch reaches the next multiple of EVERY, it snapshots the
tracked best checkpoint and runs the ladder against that snapshot (so a
promotion mid-ladder cannot change the model being measured).
Results accumulate in model/rapfi_ladder_log.jsonl.
Run detached:  python -u rapfi_ladder_watch.py
"""
import json, os, shutil, subprocess, sys, time
TENX = r"C:\Users\My PC\Desktop\final\15x15"
STATE = os.path.join(TENX, "model", "train_state_15x15.json")
BEST = os.path.join(TENX, "model", "best_mcts_15x15.pth")
SNAP = os.path.join(TENX, "model", "_rapfi_ladder_snapshot.pth")
EVERY = 20
GAMES = 40  # 2026-10-14 (user): 20 -> 40 per level for a tighter Elo estimate.  per ladder level (2026-09-28: 10 -> 20 for a tighter graduation verdict)


def epoch():
    try:
        return int(json.load(open(STATE))["cumulative_epoch"])
    except Exception:
        return None


# 2026-10-09: the ep820 run died after the problem check -- on Windows, a child output that the
# parent cannot decode with the locale codec leaves r.stdout as None. Force UTF-8 both ways.
RUN_KW = {"encoding": "utf-8", "errors": "replace", "env": dict(os.environ, PYTHONIOENCODING="utf-8",
          # 2026-10-13: 25 distinct openings (games of a level no longer share openings), 5 levels
          RAPFI_MORE_OPENINGS="1", RAPFI_LEVELS="30,100,300,1000,3000")}  # 2026-10-14: 15x15 (epoch 0 = ~766). 2026-10-16: run 3 starts from a Rapfi-10 epoch 0 -> 30..3000)}


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
    if len(sys.argv) > 1:   # 2026-10-14: resume a missed run, e.g. "python rapfi_ladder_watch.py 20"
        next_at = int(sys.argv[1])
    log(f"rapfi ladder watcher started at epoch {e}; next run at epoch {next_at}")
    while True:
        # 2026-10-13 (user): gate-passed candidates are measured as soon as they appear
        try:
            cpath = os.path.join(TENX, "model", "gate_candidate.json")
            cand = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else None
        except Exception:
            cand = None
        if cand and cand.get("status") == "pending" and os.path.exists(os.path.join(TENX, cand["ckpt"])):
            snap3 = os.path.join(TENX, "model", "_rapfi_" + os.path.basename(cand["ckpt"]))
            shutil.copy2(os.path.join(TENX, cand["ckpt"]), snap3)
            log(f"gate candidate {cand['ckpt']} (score {cand.get('gate_score')}): running ladder")
            r = subprocess.run([sys.executable, "-u", "rapfi_ladder.py", str(GAMES), "1100", snap3],
                               cwd=TENX, capture_output=True, text=True, **RUN_KW)
            log((r.stdout or '').strip()[-600:] or (r.stderr or '').strip()[-500:])
            r2 = subprocess.run([sys.executable, "-u", "tactical_problem_check.py", snap3],
                                cwd=TENX, capture_output=True, text=True, **RUN_KW)
            log(chr(10).join(l for l in (r2.stdout or '').splitlines() if not l.startswith("MESSAGE")) or (r2.stderr or '').strip()[-500:])
            r3 = subprocess.run([sys.executable, "-u", "ladder_promo.py", os.path.basename(snap3),
                                 os.path.relpath(snap3, TENX), "--candidate"], cwd=TENX, capture_output=True, text=True, **RUN_KW)
            verdict = (r3.stdout or '').strip() or (r3.stderr or '').strip()[-300:]
            log("candidate promotion check: " + verdict)
            cand["status"] = "promoted" if "FLAG WRITTEN" in verdict else "rejected"
            cand["verdict"] = verdict
            json.dump(cand, open(cpath, "w", encoding="utf-8"), indent=1)
        e = epoch()
        if e is not None and e >= next_at:
            # 2026-10-13: ladder + grading are deterministic (node-limited Rapfi, no search noise), so a best
            # whose weights equal the last measured snapshot would replay identical games -- skip it.
            _best_unchanged = os.path.exists(SNAP) and same_weights(BEST, SNAP)
            shutil.copy2(BEST, SNAP)
            if _best_unchanged:
                _ref, _ref_ckpt = {"name": "last best snapshot"}, SNAP
            # 2026-10-12: skip re-measuring a best that the ladder reference already covers (same weights)
            if not _best_unchanged:
                try:
                    _ref = json.load(open(os.path.join(TENX, "model", "ladder_best_ref.json"), encoding="utf-8"))
                    _ref_ckpt = os.path.join(TENX, _ref["ckpt"]) if _ref.get("ckpt") else None
                except Exception:
                    _ref, _ref_ckpt = None, None
            if _ref_ckpt and os.path.exists(_ref_ckpt) and same_weights(_ref_ckpt, SNAP):
                log(f"epoch {e}: best = ladder reference {_ref['name']} (already measured); best ladder skipped")
            else:
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
            latest = os.path.join(TENX, "model", f"model_MCTS_ep{next_at}_15x15.pth")  # 2026-10-14: was the 11x11 name (ep20 latest never found)
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
                # 2026-10-12 (user): Rapfi-ladder promotion path (P >= 80% stronger than the best)
                r3 = subprocess.run([sys.executable, "-u", "ladder_promo.py", os.path.basename(snap2),
                                     os.path.relpath(snap2, TENX)], cwd=TENX, capture_output=True, text=True, **RUN_KW)
                log("ladder promotion check: " + ((r3.stdout or '').strip() or (r3.stderr or '').strip()[-300:]))
            else:
                log(f"latest checkpoint ep{next_at} not found; skipped")
            next_at = (e // EVERY + 1) * EVERY
            log(f"next run at epoch {next_at}")
        time.sleep(60)


if __name__ == "__main__":
    main()
