"""One-off (2026-10-09): the watcher died after the ep820 run, so ep840 (promoted) was never
measured. Same steps as rapfi_ladder_watch.py for one snapshot of the current best."""
import os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rapfi_ladder_watch as w
shutil.copy2(w.BEST, w.SNAP)
w.log("epoch 840 (one-off, watcher was down): running ladder on best checkpoint snapshot")
r = subprocess.run([sys.executable, "-u", "rapfi_ladder.py", str(w.GAMES), "1100", w.SNAP], cwd=w.TENX, capture_output=True, text=True, **w.RUN_KW)
w.log((r.stdout or "").strip() or (r.stderr or "").strip()[-500:])
r2 = subprocess.run([sys.executable, "-u", "tactical_problem_check.py", w.SNAP], cwd=w.TENX, capture_output=True, text=True, **w.RUN_KW)
w.log(chr(10).join(l for l in (r2.stdout or "").splitlines() if not l.startswith("MESSAGE")) or (r2.stderr or "").strip()[-500:])
w.log("one-off ep840 done")
