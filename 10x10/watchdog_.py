"""
Standalone watchdog for the 11x11 hybrid MCTS training process.

Runs forever: every CHECK_INTERVAL_SEC, checks whether a python process
with _restart_train.py in its command line is alive. If not, restarts it
via the same detached-process pattern used manually throughout this project
(PowerShell Start-Process -WindowStyle Hidden), so the restarted trainer
survives independently of this watchdog, this terminal, or any session.

Launch this itself detached so it keeps running unattended:
    powershell -Command "Start-Process -FilePath python -ArgumentList '-u watchdog_.py' -WorkingDirectory 'C:\\Users\\My PC\\Desktop\\final\\10x10' -WindowStyle Hidden"
"""
import os
import subprocess
import time
from datetime import datetime

import psutil

PROJECT_DIR = r"C:\Users\My PC\Desktop\final\10x10"
TRAIN_SCRIPT_MARKER = "_restart_train.py"
TRAIN_TASK_NAME = "AZ11x11_Training"
CHECK_INTERVAL_SEC = 20 * 60
POST_RESTART_WAIT_SEC = 20
WATCHDOG_LOG = os.path.join(PROJECT_DIR, "watchdog_log.txt")


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_training_alive():
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info["name"] or "").lower()
            if "python" not in name:
                continue
            cmdline = proc.info["cmdline"] or []
            if any(TRAIN_SCRIPT_MARKER in part for part in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def restart_training():
    # 2026-09-23: switched OFF the Task Scheduler path. Confirmed via
    # Task Scheduler that "AZ11x11_Training" also carries a LogonTrigger
    # (fires on every Windows logon, independent of whether a trainer is
    # already running) -- after today's reboot this produced two
    # independently-advancing _restart_train.py processes at once (an
    # interleaved, offset-by-~10-epochs duplicate report sequence lasting
    # hours), and this watchdog has no admin rights to remove that trigger
    # itself. Launching directly here (same detached Start-Process pattern
    # used everywhere else in this project) means the ONLY way a trainer
    # ever starts is: (a) manually, or (b) this watchdog when it's
    # genuinely missing -- one code path, no hidden scheduler trigger to
    # race against.
    # No stdout/stderr redirect here -- training_report_11x11.txt (written
    # directly by report_logger, not via stdout) is the authoritative log
    # regardless, and train_stdout.log has an unexplained persistent lock
    # (unrelated process, survives full process kills) not worth fighting
    # inside an unattended restart path.
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Start-Process -FilePath python -ArgumentList '-u _restart_train.py' "
         f"-WorkingDirectory '{PROJECT_DIR}' -WindowStyle Hidden"],
        cwd=PROJECT_DIR,
        check=False,
    )
    log("Restart issued via direct Start-Process (Task Scheduler path disabled).")


def main():
    log(f"Watchdog started (PID {os.getpid()}). "
        f"Checking every {CHECK_INTERVAL_SEC}s for a process containing '{TRAIN_SCRIPT_MARKER}'.")
    while True:
        try:
            pid = is_training_alive()
            if pid is None:
                log("Training process NOT found -- restarting now.")
                restart_training()
                time.sleep(POST_RESTART_WAIT_SEC)
                pid2 = is_training_alive()
                if pid2:
                    log(f"Restart verified: PID {pid2} alive.")
                else:
                    log("WARNING: restart attempted but process still not detected "
                        f"after {POST_RESTART_WAIT_SEC}s.")
            else:
                log(f"OK -- training process alive (PID {pid}).")
        except Exception as e:
            log(f"WATCHDOG ERROR (ignored, continuing): {e!r}")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
