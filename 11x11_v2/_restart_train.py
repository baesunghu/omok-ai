import os
import sys
import psutil

# 2026-10-13 (user, clean rollback to ep960): train only once the emptied buffer holds 40k samples of the
# new best's self-play (harmless afterwards: the buffer stays far above this).
os.environ.setdefault("NAPS_MIN_BUFFER", "40000")
from alphazero_mcts_make_lrfix import train_mcts_alpha_zero

# Single-instance lock (2026-09-24): the AZ11x11_Training scheduled task's
# LogonTrigger fired again at 2026-09-23 14:26 (not from a reboot -- some
# other logon/unlock event) and spawned a SECOND, independent trainer in
# Session 0 that ran for ~18 hours alongside the real one, both writing to
# the same shared checkpoint/buffer/report files -- the same failure mode
# as the earlier multi-hour duplicate-process incident this session. The
# scheduled task itself can't be disabled from here (runs as SYSTEM;
# Set-ScheduledTask/Disable-ScheduledTask/schtasks all fail with "Access is
# denied" without an elevated session) -- so instead of relying on that
# task never firing again, refuse to start a second trainer at all: if
# train.lock names a PID that's still alive, exit immediately instead of
# racing it. This makes ANY duplicate launch (the scheduled task, a manual
# double-start, anything) a harmless no-op rather than silent corruption.
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.lock")


def _acquire_single_instance_lock():
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH) as f:
                existing_pid = int(f.read().strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and psutil.pid_exists(existing_pid):
            print(f"[train.lock] Another trainer is already running (PID {existing_pid}). "
                  f"Refusing to start a second one -- exiting.", flush=True)
            sys.exit(0)
        else:
            print(f"[train.lock] Stale lock (PID {existing_pid} not running) -- taking over.", flush=True)
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))


# Guard required for multiprocessing's "spawn" start method on Windows: a
# spawned child process re-imports this file as a fresh module to reach the
# target function passed to it. Without this guard, that re-import used to
# immediately re-invoke train_mcts_alpha_zero(...) again inside every child
# process (mp_eval's evaluator/worker processes included), which is why the
# evaluator process never reached its own ready_event.set() within the 60s
# timeout -- it was busy re-running this whole call instead. The lock
# acquire must live inside this same guard for the same reason -- otherwise
# every spawned child would also try (and fail) to acquire it.
if __name__ == "__main__":
    _acquire_single_instance_lock()
    # 2026-09-10 intervention. Measured root cause: at 250 sims over ~45
    # radius-2 candidates the search gets ~5 visits/child, below what a 2-ply
    # refutation needs, so the visit distribution recorded as the policy
    # target was FLATTER and less accurate than the network's own prior
    # (block-position top-1: prior 60%, MCTS@250 10%, MCTS@900 67%). Training
    # was descending, not plateauing. Everything below either raises
    # visits-per-child or stops diluting the prior before the search can use it.
    # 2026-09-21 lr-fix restart: tactical_refresh_interval 21->5 so the gate
    # check (which rides this same cadence) samples ~4x more often over the
    # ~80-epoch observation window, instead of only 3-4 checks total.
    # 2026-10-12 17:3x (user): temperature_moves reverted 6 -> 4 (ep980/1000 ladder fell). Earlier: 4 -> 6 (more varied midgames, quiet-threat exposure); workers 10 -> 8 (GPU 92-96C)
    # 2026-10-13 (user): fast_sims 150 -> 300 (less noisy game outcomes = cleaner value targets)
    # 2026-10-13 (user): q weight fixed at 0.2 (value target = 0.8 z + 0.2 q); half-life decay off (halflife=0)
    train_mcts_alpha_zero(num_epochs=20000, num_self_play_games=32,  # 11x11_v2 (2026-10-21): 32 like 15x15
                            num_simulations=900, batch_size=128, self_play_workers=8, eval_interval=0, eval_games=5, eval_simulations=600, num_vs_deterministic_games=0, temperature_moves=4, noise_weight=0.25, dirichlet_alpha=0.3, c_puct=1.6, c_puct_exploit=1.6, vs_det_noise_weight=0.15, vs_det_dirichlet_alpha=0.3, tactical_refresh_interval=5, tactical_refresh_epochs=8, boosted_sims=1800, boost_threshold=5, boost_when_narrow=True, num_curriculum_games=4,
                           num_opening_games=2,
                           num_league_games=12, league_pool_size=40, league_best_checkpoint_prob=0.28,
                           full_search_prob=0.5, fast_sims=300,
                           full_search_prob_other=0.5, use_unified_game_pool=True,
                           value_target_q_weight=0.2,  # 2026-10-10 ~20:50 (user): q weight 0.5 -> 0.2 (value target = 0.8 z + 0.2 q)
                           value_target_q_halflife=0, value_target_q_decay_start=905,  # 2026-10-10 (user): q weight halves every 20 epochs (0.2 @905, 0.1 @925, 0.05 @945, ~0 after ep990)
                           max_game_plies=90, refresh_vsdet_games=0,
                           noise_weight_end=0.10, noise_anneal_epochs=150,
                           dirichlet_alpha_end=0.15,
                           use_prioritized_replay=False)
