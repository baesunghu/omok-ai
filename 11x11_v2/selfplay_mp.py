"""
generate_mcts_self_play_data_mp
===============================
Multi-PROCESS drop-in replacement for
``alphazero_mcts_make_lrfix.generate_mcts_self_play_data`` (the
ThreadPoolExecutor-based self-play game generator).

Derived from the "pipe_batch_local" candidate, which was selected over the
"queue_pool" and "shared_memory" candidates. See WINNER_deploy_notes.md for
the full selection rationale and -- importantly -- for the honest status of
the PERFORMANCE claim, which is summarized here too because nobody should
deploy this without reading it:

    NONE of the three candidates demonstrated a reproducible speedup.
    Every benchmark was taken on a machine whose GPU and all 8 logical
    cores were simultaneously saturated by the live production training
    run, and the run-to-run noise (+-40%) was larger than any effect.
    Independent re-measurements of this candidate landed at 0.78x and
    1.21x -- i.e. indistinguishable from parity. This file is therefore a
    DELIBERATE EXPERIMENT, not a proven optimization. It is built to be
    safe to try in the live loop and trivial to back out, and it
    instruments itself so the A/B answer can finally be measured under
    real single-tenant conditions instead of under benchmark contention.

============================================================
WHY THIS ARCHITECTURE (and why it beat the other two)
============================================================
The diagnosed problem: the "glue" work around each leaf evaluation (numpy
board encoding + the evaluator submit/wait round trip) does NOT release the
GIL, while both the GPU forward pass and the C++ MCTS tree walk do. With
max_workers=10 self-play THREADS in one process, CPython serializes that
glue across all of them.

Corroborating evidence that is NOT wall-clock (and therefore not corrupted
by benchmark contention) -- straight out of the live training log:

    [PooledEvaluator stats] 13500 flushes, avg batch 5.0, max 20,
    target_batch=24, 0/13500 (0%) hit target vs timed out short

Every single flush times out short, at an average batch of 5 against a
target of 24. Ten concurrent self-play threads essentially never have more
than one eval request pending at the same instant. That is the GIL
serialization, visible in production, right now. The shared_memory
candidate independently measured GPU utilization rising from 10-40%
(threaded) to 55-76% (multi-process), confirming the architecture does fix
the batching. What remains unproven is whether fixing it actually buys
wall-clock, since the real bottleneck may simply be the CPU-bound C++ tree
walk on 8 already-oversubscribed cores.

This candidate's structure:

1. Worker processes (persistent pool, spawned ONCE per training run, never
   respawned per epoch). Each is a plain CPU-only OS process with NO CUDA
   context. It plays games by calling the production
   ``_play_one_mcts_game`` COMPLETELY UNMODIFIED, imported read-only from
   the live tree. This is the decisive advantage over the other two
   candidates, both of which hand-ported the game loop, the board encoding
   and/or the augmentation logic into their own files -- creating a second
   copy of production logic that would silently drift the moment anyone
   edits the real one. Here, the only thing that differs from production is
   the *evaluator object* plugged into MCTS; every rule, every curriculum
   branch, every sampling decision is literally the same bytes.

2. The evaluator plugged into each worker's ``MCTS(pooled_evaluator=...)``
   is ``_PipeEvalProxy``, a duck-type of ``PooledEvaluator.submit()``. It
   sends ONE numpy array per already-batched leaf group down a dedicated
   ``multiprocessing.Pipe`` and blocks for the paired result.

   The "batch_local" part comes free: the C++ engine's ``search_batched()``
   already groups eval_batch_size=16 simulations' leaves per round via
   virtual loss BEFORE calling back into Python. So one pipe message
   already carries up to 16 boards -- roughly ceil(sims/16) round trips per
   ply, not one per simulation.

   A dedicated Pipe per worker (rather than one shared Queue) means writers
   never contend on a shared lock; the coordinator reads all of them with a
   single multiplexed ``multiprocessing.connection.wait()``.

3. Exactly ONE thing owns the GPU: ``PipePooledEvaluator``, a background
   THREAD in the MAIN process -- the same process that already holds
   ``model`` on ``device``. No second CUDA context, no checkpoint
   save/reload handshake, no weight-shipping, and therefore ZERO weight
   staleness: workers never hold weights at all. (This is the other reason
   this beat "shared_memory", which needed a separate GPU-owning process
   and a per-call torch.save/torch.load round trip -- extra VRAM on a 4GB
   GTX 1050 and real per-call latency.)

   Its accumulate-until-target_batch-or-silence-gap logic is a direct port
   of ``PooledEvaluator._run``, including that implementation's
   silence-gap-not-deadline quirk, so the two are apples-to-apples at the
   same target_batch/max_wait operating point.

4. Finished games come back over an ordinary Queue -- once per game, not
   once per simulation, so pickling cost there is amortized to nothing.

============================================================
SAFETY DESIGN (what makes this safe to gamble on in a live run)
============================================================
This is going into a 20000-epoch run that must not die. Every known
failure mode degrades to "the original threaded implementation runs this
epoch instead", never to a crash or a hang:

* ANY exception from the MP path (worker death, GPU OOM in the coordinator,
  IPC breakage, timeout) is caught, the pool is torn down, and the epoch is
  transparently re-run on the ORIGINAL production threaded function. The
  epoch still produces correct data; it just doesn't get the experiment.
* After ``MAX_MP_STRIKES`` such failures the MP path disables itself for
  the rest of the process. A recurring problem degrades to "we're just
  running the old code", quietly and permanently.
* Kill switch without editing any code: set ``MP_SELFPLAY_DISABLE=1`` in
  the environment and this becomes a pure pass-through to the original.
* Stall detection is PROGRESS-based, not a fixed deadline: as long as the
  coordinator keeps flushing eval batches or games keep finishing, the call
  is considered healthy. A genuinely wedged pool is caught in ~3 minutes
  and falls back, instead of hanging for the 24 minutes a naive
  games x per-game deadline would have allowed.
* If the coordinator thread ever dies mid-flush, every worker blocked on
  ``recv()`` is explicitly woken with an error sentinel so it fails fast,
  rather than blocking forever on a pipe nobody will ever write to again.
* Workers are daemon processes AND are shut down via atexit, so they cannot
  outlive the trainer.

============================================================
KNOWN, DELIBERATE DEVIATIONS FROM THE ORIGINAL
============================================================
* ``eval_cache`` is accepted but IGNORED. There is no way to share the
  caller's in-process dict with separate processes. Each worker keeps its
  own process-local cache, reset when it sees a new call_id (== a new
  epoch's frozen weights), so cache VALIDITY is identical -- a cached entry
  is still a pure function of frozen weights. What is reduced is the DEGREE
  of cross-game de-dup: in the production shape (num_games=8 spread over 10
  workers) most workers see exactly one game, so they get little cross-game
  reuse, whereas the original's single shared dict saw every game's boards
  AND was additionally shared with the vs-Det / vs-League phases that still
  run threaded. Cost: some extra GPU forward passes. Never a wrong value.
  This is a real, unquantified handicap against the threaded baseline and
  it is included in every measured number.
* ``pooled_evaluator`` is accepted but IGNORED: the MP path must use its
  own cross-process evaluator. The caller's PooledEvaluator is untouched
  and continues serving the still-threaded vs-Det / vs-League phases.
* Per-game randomness comes from each worker's own unseeded RNG stream
  rather than one shared in-process stream. Self-play was never
  reproducible run-to-run either way.
* Only ``generate_mcts_self_play_data`` is replaced. vs-Deterministic and
  vs-League generation stay on the threaded path.
"""
import atexit
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import queue as pyqueue
import sys
import threading
import time

import numpy as np

# ------------------------------------------------------------------
# Make the live production tree importable READ-ONLY. Needed in the main
# process and in every spawned worker (spawn re-imports this module fresh
# in each child), so it happens at import time.
#
# NOTHING in this file ever writes to, restarts, or imports-for-side-effects
# anything under that directory. The production modules imported below have
# no module-level file writes and no module-level checkpoint loads (verified:
# every load_state_dict / open() in them is inside a function body), so
# importing them in a worker allocates one unused CPU nn.Module and nothing
# else.
# ------------------------------------------------------------------
PROD_DIR = r"C:\Users\My PC\Desktop\final\11x11_v2"
if PROD_DIR not in sys.path:
    sys.path.insert(0, PROD_DIR)

# Which production module the workers take _play_one_mcts_game from, and
# which one the threaded fallback calls. Defaults to the module the live
# training run actually executes, so the MP path can NEVER diverge from the
# game logic you are currently editing/training with. Override with
# MP_SELFPLAY_PROD_MODULE=alphazero_mcts_make to point at the older module.
PROD_MODULE = os.environ.get("MP_SELFPLAY_PROD_MODULE", "alphazero_mcts_make_lrfix")

import torch  # noqa: E402  (must follow the sys.path setup above)
from alphago_lee import preferred_memory_format  # noqa: E402
from alphago_lee_v3_survival import SURVIVAL_CAP  # noqa: E402
from alphazero_mcts import BOARD_SIZE  # noqa: E402

NUM_CELLS = BOARD_SIZE * BOARD_SIZE

# Matches the live training loop's own tuned PooledEvaluator operating point
# (target_batch=24, max_wait=0.015 measured +19% self-play games/sec over the
# previous 48/0.005 on this exact machine). Kept identical here so an A/B
# against the threaded path isn't accidentally comparing two different
# batching operating points.
DEFAULT_TARGET_BATCH = 24
DEFAULT_MAX_WAIT = 0.015

# Seconds without ANY progress (no eval flush, no finished game) before the
# pool is declared wedged and the epoch falls back to threaded.
DEFAULT_STALL_TIMEOUT = 180.0

# After this many MP failures, stop trying for the rest of the process.
MAX_MP_STRIKES = 3

# First element of an eval reply is a numpy array on the happy path, so a
# string here is an unambiguous out-of-band error signal.
_EVAL_ERROR_TAG = "__MP_SELFPLAY_EVAL_ERROR__"


# Creating this file turns the experiment off from OUTSIDE the running
# process. An env var can only be set at launch, which is useless for a
# 20000-epoch run already in flight; this is checked once per epoch (a
# single os.path.exists, nothing measurable) so the NEXT epoch reverts to
# the threaded implementation without restarting anything. Delete the file
# to turn it back on.
KILL_SWITCH_FILE = os.path.join(PROD_DIR, "MP_SELFPLAY_OFF")


def _mp_enabled():
    """Kill switches, both of which make this a pure pass-through to the
    original threaded implementation with no pool ever spawned:
      * the file PROD_DIR\\MP_SELFPLAY_OFF exists  (works mid-run)
      * MP_SELFPLAY_DISABLE=1 in the environment   (set at launch)
    """
    try:
        if os.path.exists(KILL_SWITCH_FILE):
            return False
    except OSError:
        pass
    return os.environ.get("MP_SELFPLAY_DISABLE", "").strip() not in ("1", "true", "True", "yes")


# ============================================================
# Worker-side: runs inside each spawned CPU-only process
# ============================================================
def _worker_main(worker_id, task_queue, result_queue, eval_conn, ready_queue, error_queue):
    """Entry point for one persistent worker process.

    Never touches torch.cuda. The only torch usage reachable from this path
    is the non-pooled fallback branch inside the production
    MCTS._cpp_eval_batch_fn, which is dead code here because a
    pooled_evaluator (_PipeEvalProxy) is always supplied -- so the model and
    device arguments below are never actually read, which is why passing
    None for them is safe.
    """
    try:
        prod = __import__(PROD_MODULE)
        _play_one_mcts_game = prod._play_one_mcts_game
    except Exception as e:  # noqa: BLE001
        error_queue.put(f"worker {worker_id} failed to import {PROD_MODULE}: {e!r}")
        ready_queue.put(worker_id)
        return

    proxy = _PipeEvalProxy(eval_conn)
    local_cache = {}
    local_call_id = None

    ready_queue.put(worker_id)

    try:
        while True:
            task = task_queue.get()
            if task is None:  # shutdown sentinel
                break
            call_id, game_idx, start_board, start_player, cfg = task

            if call_id != local_call_id:
                # New generation call == a new epoch's frozen weights. Reset
                # the memoization cache, matching the contract that a shared
                # eval cache is only valid for one epoch's fixed weights.
                local_cache = {}
                local_call_id = call_id

            try:
                history, outcome, ply_indices, total_plies = _play_one_mcts_game(
                    None, None,  # model, device: never read -- see docstring
                    cfg["num_simulations"],
                    boosted_sims=cfg["boosted_sims"],
                    boost_threshold=cfg["boost_threshold"],
                    temperature_moves=cfg["temperature_moves"],
                    dirichlet_alpha=cfg["dirichlet_alpha"],
                    noise_weight=cfg["noise_weight"],
                    c_puct=cfg["c_puct"],
                    eval_cache=local_cache,
                    boost_when_narrow=cfg["boost_when_narrow"],
                    start_board=start_board,
                    start_player=start_player,
                    pooled_evaluator=proxy,
                    full_search_prob=cfg["full_search_prob"],
                    fast_sims=cfg["fast_sims"],
                    return_ply_info=True,
                    max_plies=cfg.get("max_plies"),
                )
                result_queue.put((call_id, game_idx, (history, ply_indices, total_plies), outcome, None))
            except Exception as e:  # noqa: BLE001
                # Includes the coordinator-died error sentinel raised by the
                # proxy. Reported as a per-game failure so the parent can
                # fall back to threaded instead of hanging.
                result_queue.put((call_id, game_idx, None, None, repr(e)))
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    finally:
        try:
            eval_conn.close()
        except Exception:  # noqa: BLE001
            pass


class _PipeEvalProxy:
    """Duck-types ``PooledEvaluator.submit(input_arrays) -> (probs, values)``
    so the unmodified production MCTS/_play_one_mcts_game work inside a
    worker process.

    ``input_arrays`` is already a list of (5, BOARD_SIZE, BOARD_SIZE) float32
    arrays -- the CPU-bound board encoding already happened in THIS process,
    in true parallel with every other worker process and with no GIL
    contention against them. This method's only job is the cross-process
    round trip.
    """
    __slots__ = ("conn",)

    def __init__(self, conn):
        self.conn = conn

    def submit(self, input_arrays):
        n = len(input_arrays)
        if n == 0:
            return np.zeros((0, NUM_CELLS), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        # ONE pipe message for the whole already-batched group, not one per leaf.
        self.conn.send(np.stack(input_arrays))
        probs, values = self.conn.recv()
        if isinstance(probs, str) and probs == _EVAL_ERROR_TAG:
            # The coordinator failed (e.g. CUDA OOM) and deliberately woke us
            # instead of leaving us blocked on a pipe nobody will write to.
            raise RuntimeError(f"MP self-play evaluator failed: {values}")
        return probs, values


# ============================================================
# Coordinator-side: background thread in the MAIN process, owns the GPU
# ============================================================
class PipePooledEvaluator:
    """Cross-process analogue of ``alphazero_mcts.PooledEvaluator``: one
    background thread in the process that already holds ``model`` on
    ``device``, multiplexing every worker's dedicated Pipe with
    ``multiprocessing.connection.wait()`` instead of reading a Queue.

    The flush policy is a direct port of PooledEvaluator._run: the timeout
    re-arms on every arrival, i.e. it is a quiet-period debounce rather than
    a deadline measured from the first pending item.
    """

    def __init__(self, model, device, parent_conns,
                 target_batch=DEFAULT_TARGET_BATCH, max_wait=DEFAULT_MAX_WAIT):
        self.model = model
        self.device = device
        self.parent_conns = list(parent_conns)
        self.target_batch = target_batch
        self.max_wait = max_wait
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._flush_count = 0
        self._total_samples = 0
        self._max_flush = 0
        self._at_target_count = 0
        self._error = None  # first fatal coordinator error, if any
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mp-selfplay-gpu-coordinator")
        self._thread.start()

    # -- health / instrumentation -------------------------------------
    @property
    def flush_count(self):
        """Read by the pool's stall watchdog as a liveness signal."""
        return self._flush_count

    def check_error(self):
        """Returns the coordinator's fatal error, or a synthetic one if the
        thread died without recording anything."""
        if self._error is not None:
            return self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            return "evaluator coordinator thread exited unexpectedly"
        return None

    def stats_summary(self):
        if self._flush_count == 0:
            return "no flushes yet"
        avg = self._total_samples / self._flush_count
        pct = 100.0 * self._at_target_count / self._flush_count
        return (f"{self._flush_count} flushes, avg batch {avg:.1f}, max {self._max_flush}, "
                f"target_batch={self.target_batch}, {self._at_target_count}/{self._flush_count} "
                f"({pct:.0f}%) hit target vs timed out short")

    def update_model(self, model, device):
        """Cheap same-process attribute swap. Workers never hold a reference
        to the model, so there is nothing to resync on their side when the
        weights change from epoch to epoch -- they always, automatically,
        see current weights with zero staleness."""
        with self._lock:
            self.model = model
            self.device = device

    # -- main loop -----------------------------------------------------
    def _run(self):
        pending_conns = []
        pending_arrays = []
        pending_total = 0
        live_conns = list(self.parent_conns)
        try:
            while not self._stop.is_set():
                if not live_conns:
                    time.sleep(0.1)
                    continue
                # Capped at 0.5s when idle purely so this thread notices
                # self._stop promptly; target_batch/max_wait are what
                # actually govern flush timing.
                wait_timeout = self.max_wait if pending_conns else 0.5
                try:
                    ready = mp_connection.wait(live_conns, timeout=wait_timeout)
                except (OSError, ValueError):
                    ready = []
                if self._stop.is_set():
                    break

                if not ready:
                    # Silence gap elapsed -> flush what we have (mirrors
                    # PooledEvaluator's queue.Empty -> 'FLUSH' path).
                    if pending_conns:
                        self._flush(pending_conns, pending_arrays)
                        pending_conns, pending_arrays, pending_total = [], [], 0
                    continue

                for conn in ready:
                    try:
                        if conn.closed:
                            live_conns = [c for c in live_conns if c is not conn]
                            continue
                        arr = conn.recv()
                    except (EOFError, OSError, BrokenPipeError):
                        live_conns = [c for c in live_conns if c is not conn]
                        continue
                    pending_conns.append(conn)
                    pending_arrays.append(arr)
                    pending_total += arr.shape[0]

                if pending_total >= self.target_batch:
                    self._flush(pending_conns, pending_arrays)
                    pending_conns, pending_arrays, pending_total = [], [], 0
                # else: loop again, re-arming a fresh silence window.

            if pending_conns:
                self._flush(pending_conns, pending_arrays)
        except Exception as e:  # noqa: BLE001
            # Never let this thread die silently: record the cause and wake
            # anyone still blocked so the call fails fast and falls back.
            self._error = f"coordinator loop crashed: {e!r}"
            self._wake_with_error(pending_conns, self._error)

    def _wake_with_error(self, conns, msg):
        for conn in conns:
            try:
                conn.send((_EVAL_ERROR_TAG, msg))
            except Exception:  # noqa: BLE001
                pass

    def _flush(self, conns, arrays):
        """One batched GPU forward pass for every currently-pending worker.

        Numerically identical to PooledEvaluator._flush /
        alphago_lee.boards_to_input_tensor: np.stack the groups, one
        host->device transfer, contiguous in the device's preferred memory
        format, inference_mode, exp(log-softmax logits) -> float64 probs,
        squeezed float64 values.
        """
        try:
            batch_np = np.concatenate(arrays, axis=0) if len(arrays) > 1 else np.asarray(arrays[0])
            with self._lock:
                model, device = self.model, self.device
            device = torch.device(device) if not isinstance(device, torch.device) else device

            batch = torch.from_numpy(batch_np).float().to(device)
            batch = batch.contiguous(memory_format=preferred_memory_format(device))
            with torch.inference_mode():
                policy_logits, value_tensor = model(batch)
            probs_batch = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
            if probs_batch.ndim == 1:
                probs_batch = probs_batch.reshape(1, -1)
            values_batch = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
        except Exception as e:  # noqa: BLE001
            # Realistic case: CUDA OOM on the shared 4GB GPU. Do NOT let the
            # thread die and strand every worker on a blocked recv() -- wake
            # them with an error so the epoch fails fast and falls back to
            # the threaded implementation.
            msg = f"GPU forward pass failed: {e!r}"
            if self._error is None:
                self._error = msg
            self._wake_with_error(conns, msg)
            return

        offset = 0
        for conn, arr in zip(conns, arrays):
            k = arr.shape[0]
            try:
                conn.send((probs_batch[offset:offset + k], values_batch[offset:offset + k]))
            except (BrokenPipeError, OSError):
                pass
            offset += k

        n = batch_np.shape[0]
        self._flush_count += 1
        self._total_samples += n
        if n > self._max_flush:
            self._max_flush = n
        if n >= self.target_batch:
            self._at_target_count += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ============================================================
# Persistent worker-process pool
# ============================================================
class MPGamePool:
    """Spawns num_workers persistent CPU-only processes plus one
    PipePooledEvaluator coordinator thread, all reused across every call for
    the lifetime of this pool object.

    Re-spawning ~10 processes (interpreter startup + re-importing
    torch/numpy/mcts_core in each) costs tens of seconds on this machine, so
    paying it once per training run instead of once per epoch is the entire
    point of the persistence.
    """

    def __init__(self, num_workers, model, device, target_batch=DEFAULT_TARGET_BATCH,
                 max_wait=DEFAULT_MAX_WAIT, startup_timeout=180.0):
        self.ctx = mp.get_context("spawn")  # Windows: fork is unavailable
        self.num_workers = num_workers
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.ready_queue = self.ctx.Queue()
        self.error_queue = self.ctx.Queue()
        # 2026-09-26: with more tasks than workers (now the case on the
        # fallback path), a failed call can leave this process's feeder
        # thread blocked on unread tasks, which multiprocessing would join
        # forever at exit. Never join these feeders; shutdown() also drains.
        for q in (self.task_queue, self.result_queue, self.ready_queue, self.error_queue):
            q.cancel_join_thread()
        self.parent_conns = []
        self.workers = []
        self.evaluator = None
        self._next_call_id = 0

        for wid in range(num_workers):
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            self.parent_conns.append(parent_conn)
            p = self.ctx.Process(
                target=_worker_main,
                args=(wid, self.task_queue, self.result_queue, child_conn,
                      self.ready_queue, self.error_queue),
                daemon=True,  # cannot outlive the trainer
            )
            p.start()
            self.workers.append(p)
            child_conn.close()  # parent drops its copy of the worker's end

        got = 0
        deadline = time.monotonic() + startup_timeout
        errors = []
        while got < num_workers and time.monotonic() < deadline:
            try:
                self.ready_queue.get(timeout=1.0)
                got += 1
            except pyqueue.Empty:
                pass
            try:
                while True:
                    errors.append(self.error_queue.get_nowait())
            except pyqueue.Empty:
                pass
            if errors:
                break
        if errors:
            self.shutdown()
            raise RuntimeError(f"MP self-play worker startup failed: {errors}")
        if got < num_workers:
            self.shutdown()
            raise RuntimeError(
                f"only {got}/{num_workers} MP self-play workers became ready within {startup_timeout}s")

        self.evaluator = PipePooledEvaluator(model, device, self.parent_conns, target_batch, max_wait)

    def run_games(self, tasks_meta, cfg, stall_timeout=DEFAULT_STALL_TIMEOUT):
        """tasks_meta: list of (game_idx, start_board, start_player).
        Returns {game_idx: (history, outcome)}.

        Health model is PROGRESS-based rather than a fixed deadline: as long
        as the coordinator keeps flushing eval batches or games keep
        finishing, the call is healthy however long the games take (sims can
        be boosted to 1800+/move in production, so a wall-clock deadline
        would have to be uselessly generous). A truly wedged pool is caught
        in stall_timeout seconds.
        """
        call_id = self._next_call_id
        self._next_call_id += 1
        for (game_idx, start_board, start_player) in tasks_meta:
            self.task_queue.put((call_id, game_idx, start_board, start_player, cfg))

        n = len(tasks_meta)
        results = {}
        last_progress = time.monotonic()
        last_flushes = self.evaluator.flush_count
        last_result_count = 0

        while len(results) < n:
            # 1. Did a worker report a startup/import failure?
            try:
                err = self.error_queue.get_nowait()
                raise RuntimeError(f"MP self-play worker reported a failure: {err}")
            except pyqueue.Empty:
                pass

            # 2. Did the GPU coordinator die (e.g. CUDA OOM)?
            ev_err = self.evaluator.check_error()
            if ev_err is not None:
                raise RuntimeError(f"MP self-play evaluator failed: {ev_err}")

            # 3. Did a worker process die? A worker holding a task takes that
            #    task to the grave, so fail fast rather than wait it out.
            for p in self.workers:
                if not p.is_alive():
                    raise RuntimeError(
                        f"MP self-play worker died unexpectedly (exitcode={p.exitcode})")

            # 4. Collect a finished game.
            try:
                rcall_id, game_idx, history, outcome, err = self.result_queue.get(timeout=1.0)
                if rcall_id == call_id:
                    if err is not None:
                        raise RuntimeError(f"MP self-play worker raised for game {game_idx}: {err}")
                    results[game_idx] = (history, outcome)
                # else: stale result from an earlier call; drop it.
            except pyqueue.Empty:
                pass

            # 5. Stall watchdog. Compare against the LAST observed counts --
            #    testing `len(results) > 0` here instead would re-arm the
            #    timer on every iteration once any game finished, silently
            #    disabling the watchdog for the rest of the call.
            flushes = self.evaluator.flush_count
            if flushes != last_flushes or len(results) != last_result_count:
                last_flushes = flushes
                last_result_count = len(results)
                last_progress = time.monotonic()
            if time.monotonic() - last_progress > stall_timeout:
                raise TimeoutError(
                    f"MP self-play stalled: {len(results)}/{n} games done, no eval flush "
                    f"or finished game in {stall_timeout:.0f}s")

        return results

    def shutdown(self):
        # 2026-09-26: drop still-queued tasks first so workers reach their
        # sentinel after their current game (see __init__ note).
        empty_polls = 0
        while empty_polls < 2:
            try:
                self.task_queue.get(timeout=0.05)
                empty_polls = 0
            except pyqueue.Empty:
                empty_polls += 1
            except Exception:  # noqa: BLE001
                break
        for _ in self.workers:
            try:
                self.task_queue.put(None)
            except Exception:  # noqa: BLE001
                pass
        for p in self.workers:
            try:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            except Exception:  # noqa: BLE001
                pass
        if self.evaluator is not None:
            try:
                self.evaluator.stop()
            except Exception:  # noqa: BLE001
                pass
        for c in self.parent_conns:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


# ============================================================
# Module-level persistent singleton
# ============================================================
_POOL = None
_MP_STRIKES = 0
_MP_PERMANENTLY_DISABLED = False
# Set when a call had to spawn the pool, so that call's timing line can say
# so -- the one-time startup (~10s for 10 processes, each importing torch and
# mcts_core) lands entirely inside the FIRST epoch and would otherwise bias
# an A/B against the multi-process arm. Exclude that epoch, or prewarm.
_POOL_JUST_CREATED = False


def _ensure_pool(num_workers, model, device, target_batch, max_wait):
    global _POOL, _POOL_JUST_CREATED
    if _POOL is not None and _POOL.num_workers != num_workers:
        _POOL.shutdown()
        _POOL = None
    if _POOL is None:
        _POOL = MPGamePool(num_workers, model, device, target_batch, max_wait)
        _POOL_JUST_CREATED = True
    else:
        _POOL.evaluator.update_model(model, device)
    return _POOL


def shutdown_pool():
    """Tear the pool down. Safe to call repeatedly; registered with atexit,
    so explicit calls are optional."""
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _POOL = None


def prewarm_pool(num_workers, model, device,
                 target_batch=DEFAULT_TARGET_BATCH, max_wait=DEFAULT_MAX_WAIT):
    """OPTIONAL. Spawn the pool ahead of time so the first epoch doesn't pay
    the one-time startup cost inside its own timing. Not required -- the
    pool is created lazily on the first generate call either way."""
    return _ensure_pool(num_workers, model, device, target_batch, max_wait)


atexit.register(shutdown_pool)


def _resolve_workers(max_workers, num_games):
    """MP_SELFPLAY_WORKERS overrides ONLY the multi-process pool size,
    leaving the caller's self_play_workers (still used by the threaded
    vs-Det / vs-League phases) alone.

    Worth tuning: this machine has 8 logical cores, and the production
    self_play_workers=10 means 10 worker processes genuinely oversubscribe
    them -- which threads did not, since the GIL kept only a few running
    Python at once. One candidate's exploratory data hinted 8 was better.
    This env var lets that be tested without touching the training script.
    """
    override = os.environ.get("MP_SELFPLAY_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max_workers or max(1, min(num_games, os.cpu_count() or 1))


def _threaded_fallback(reason, kwargs):
    """Run the ORIGINAL production threaded implementation for this epoch.

    Timed and tagged identically to the MP path so the two are directly
    comparable by grepping the training log -- the log has no epoch timings
    of its own, so without this there would be no baseline to A/B against.
    """
    if reason:
        print(f"  [MP self-play] falling back to the threaded implementation: {reason}", flush=True)
    prod = __import__(PROD_MODULE)
    t0 = time.monotonic()
    result = prod.generate_mcts_self_play_data(**kwargs)
    print(f"  [selfplay-timing] threaded {time.monotonic() - t0:.1f}s", flush=True)
    return result


# Paired A/B mode. Set MP_SELFPLAY_AB=1 and the module alternates
# multi-process / threaded on successive epochs, printing a
# "[selfplay-timing]" line for each.
#
# This is the ONLY way to get a trustworthy answer on this machine. Every
# benchmark taken so far ran a second Python process alongside the live
# trainer, so both arms were measured under contention that the deployed
# code will never actually face -- and the resulting noise (+-40%) swamped
# the effect. Alternating INSIDE the real training loop means each arm runs
# on a machine in the same state, with no extra process, and interleaving
# cancels slow drift. Collect ~10 epochs, then compare MEDIANS:
#
#   grep "selfplay-timing" train_stdout.log
#
# Both arms produce equivalent training data, so alternating is harmless to
# the run itself.
_AB_COUNTER = 0


def _ab_mode():
    return os.environ.get("MP_SELFPLAY_AB", "").strip() in ("1", "true", "True", "yes")


# ============================================================
# Drop-in replacement entry point
# ============================================================
def generate_mcts_self_play_data_mp(model, device, num_games=10, num_simulations=40, max_workers=None,
                                    boosted_sims=None, boost_threshold=0,
                                    temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                    c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                    num_curriculum_games=0, num_opening_games=0,
                                    pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                    target_batch=DEFAULT_TARGET_BATCH, max_wait=DEFAULT_MAX_WAIT,
                                    value_q_weight=0.0, max_plies=None):
    """Multi-process drop-in for
    ``alphazero_mcts_make_lrfix.generate_mcts_self_play_data``.

    Identical call signature (plus target_batch/max_wait, which the original
    hard-codes at its PooledEvaluator construction site rather than exposing)
    and identical ``(raw_samples, outcome_counts)`` return contract:
    raw_samples is a list of ``(aug_board_3d, aug_policy_flat, final_val,
    sample_weight)`` rows, 8x dihedral-augmented, and outcome_counts is
    ``{'x_win', 'o_win', 'draw'}`` summing to num_games.

    ``eval_cache`` and ``pooled_evaluator`` are accepted for signature
    compatibility but ignored on the MP path -- see the module docstring.
    They ARE forwarded intact to the threaded implementation whenever this
    falls back, so fallback behavior is byte-for-byte the original.

    On ANY multi-process failure this transparently re-runs the epoch on the
    original threaded implementation, so a bug here can cost time but can
    never cost the training run.
    """
    global _MP_STRIKES, _MP_PERMANENTLY_DISABLED

    # Exact original kwargs, kept so the fallback path is a perfect
    # reproduction of the call the training loop intended to make.
    orig_kwargs = dict(
        model=model, device=device, num_games=num_games, num_simulations=num_simulations,
        max_workers=max_workers, boosted_sims=boosted_sims, boost_threshold=boost_threshold,
        temperature_moves=temperature_moves, dirichlet_alpha=dirichlet_alpha,
        noise_weight=noise_weight, c_puct=c_puct, eval_cache=eval_cache,
        boost_when_narrow=boost_when_narrow, num_curriculum_games=num_curriculum_games,
        num_opening_games=num_opening_games, pooled_evaluator=pooled_evaluator,
        full_search_prob=full_search_prob, fast_sims=fast_sims,
        value_q_weight=value_q_weight, max_plies=max_plies,
    )

    global _AB_COUNTER

    if _MP_PERMANENTLY_DISABLED:
        return _threaded_fallback(None, orig_kwargs)
    if not _mp_enabled():
        if _POOL is not None:
            # Kill switch flipped mid-run: release the ~400MB-per-worker
            # processes instead of leaving them parked and idle.
            shutdown_pool()
        return _threaded_fallback("kill switch is set", orig_kwargs)
    if _ab_mode():
        _AB_COUNTER += 1
        if _AB_COUNTER % 2 == 0:
            print(f"  [A/B] epoch call #{_AB_COUNTER}: threaded arm", flush=True)
            return _threaded_fallback(None, orig_kwargs)
        print(f"  [A/B] epoch call #{_AB_COUNTER}: multi-process arm", flush=True)

    t0 = time.monotonic()
    try:
        # Reference-only imports, done lazily so that (a) importing this
        # module is cheap and (b) there is no circular-import problem when
        # the production module itself imports this one.
        prod = __import__(PROD_MODULE)
        from alphago_lee_make import augment_sample

        workers = _resolve_workers(max_workers, num_games)
        boost_note = (f", boosted to {boosted_sims} while >{boost_threshold} cells empty"
                      if boosted_sims else "")
        print(f"Generating {num_games} MCTS Self-Play games [multi-process] "
              f"(Simulations: {num_simulations}/move{boost_note}, workers: {workers})...", flush=True)

        global _POOL_JUST_CREATED
        _POOL_JUST_CREATED = False
        model.eval()
        pool = _ensure_pool(workers, model, device, target_batch, max_wait)
        spawned_now = _POOL_JUST_CREATED
        flushes_at_start = pool.evaluator.flush_count

        # Curriculum / forced-opening start sampling stays in the calling
        # process, exactly like the original's own loop: these use
        # module-level random state and precomputed window tables that have
        # no business crossing a process boundary.
        tasks_meta = []
        for i in range(num_games):
            start_board, start_player = None, None
            if i < num_curriculum_games:
                sampled = prod._sample_tactical_start()
                if sampled is not None:
                    start_board, start_player = sampled
            elif i < num_curriculum_games + num_opening_games:
                start_board, start_player = prod._sample_forced_opening_start()
            tasks_meta.append((i, start_board, start_player))

        cfg = dict(
            num_simulations=num_simulations, boosted_sims=boosted_sims,
            boost_threshold=boost_threshold, temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha, noise_weight=noise_weight, c_puct=c_puct,
            boost_when_narrow=boost_when_narrow, full_search_prob=full_search_prob,
            fast_sims=fast_sims, max_plies=max_plies,
        )

        game_results = pool.run_games(tasks_meta, cfg)

        # ---- Aggregation: identical to the production implementation ----
        raw_samples = []
        outcome_counts = {'x_win': 0, 'o_win': 0, 'draw': 0}
        opening_rows = []
        for i in range(num_games):
            (history, ply_indices, total_plies), outcome = game_results[i]
            if outcome == 1:
                outcome_counts['x_win'] += 1
            elif outcome == -1:
                outcome_counts['o_win'] += 1
            else:
                outcome_counts['draw'] += 1
            if history and ply_indices and ply_indices[0] == 0:
                first_cell = int(np.argmax(history[0][2]))
                opening_rows.append(first_cell // BOARD_SIZE)
            # 5th element: survival target (remaining plies, capped and
            # normalized) for the V3 auxiliary head -- see
            # alphago_lee_v3_survival.py. Self-play records both sides'
            # plies, so len(history) - index is the true plies-to-end.
            # 2026-09-26: real ply indices (PCR skips recording fast-search
            # plies, so len(history) - k would undercount plies-to-end).
            for (board_3d, player_turn, target_policy_map, q), ply_idx in zip(history, ply_indices):
                final_val = prod.mixed_value_target(float(outcome if player_turn == 'X' else -outcome), q,
                                                    value_q_weight)
                mover_won = ((outcome == 1 and player_turn == 'X')
                             or (outcome == -1 and player_turn == 'O'))
                sample_weight = 1.15 if mover_won else 1.0
                survival = (None if total_plies is None
                            else min(total_plies - ply_idx, SURVIVAL_CAP) / SURVIVAL_CAP)
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val, sample_weight, survival))

        elapsed = time.monotonic() - t0
        flushes = pool.evaluator.flush_count - flushes_at_start
        print(f"Generated {len(raw_samples)} augmented MCTS training samples [multi-process]. "
              f"Outcomes: X {outcome_counts['x_win']} / O {outcome_counts['o_win']} / "
              f"Draw {outcome_counts['draw']} "
              f"| Opening row-coverage this epoch: {len(set(opening_rows))}/{BOARD_SIZE} "
              f"| {flushes} eval flushes | evaluator: {pool.evaluator.stats_summary()}", flush=True)
        # Same "[selfplay-timing]" token as the threaded path, so one grep
        # over the training log yields both arms of the A/B.
        note = " (INCLUDES one-time pool startup -- exclude from A/B)" if spawned_now else ""
        print(f"  [selfplay-timing] multiprocess {elapsed:.1f}s{note}", flush=True)
        return raw_samples, outcome_counts

    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        # Any MP failure costs this epoch some time, never the run itself.
        _MP_STRIKES += 1
        print(f"  [MP self-play] FAILED after {time.monotonic() - t0:.1f}s "
              f"(strike {_MP_STRIKES}/{MAX_MP_STRIKES}): {e!r}", flush=True)
        shutdown_pool()  # next attempt gets a clean pool
        if _MP_STRIKES >= MAX_MP_STRIKES:
            _MP_PERMANENTLY_DISABLED = True
            print("  [MP self-play] too many failures -- permanently reverting to the "
                  "threaded implementation for the rest of this process.", flush=True)
        return _threaded_fallback("re-running this epoch on the threaded path", orig_kwargs)
