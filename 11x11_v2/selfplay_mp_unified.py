"""
selfplay_mp_unified (2026-09-26)
================================
One persistent multi-process pool that plays ALL of an epoch's training
games at once -- self-play (including curriculum/opening starts),
vs-Deterministic and vs-League -- instead of three back-to-back phases.

Built from selfplay_mp.py (kept unchanged as the fallback path). Why:

* The old sequence ran self-play (multi-process), then vs-Det (threaded),
  then vs-League (threaded). Each phase ends only when its slowest game
  does, so the tail of every phase left most workers idle, and the
  2-game vs-League phase kept 8 of 10 workers and most of the GPU idle
  for its whole duration (live log: evaluator avg batch 30 -> 22 during
  it, "hit target" falling to 57%).
* Here every game of the epoch goes onto ONE task queue that 10 worker
  processes pull from (longest games first), so workers stay busy until
  the queue drains and the GPU coordinator keeps getting full batches.

What changed relative to selfplay_mp.py
---------------------------------------
* Workers accept three task kinds ('self', 'vsdet', 'league') and call the
  UNMODIFIED production game functions for each (same principle as before:
  no hand-ported game logic, so nothing can silently drift).
* Eval requests carry a model TAG ('gen' or 'league'). The coordinator
  groups a flush by tag and runs one forward pass per model, so a league
  game's opponent side is evaluated by the league checkpoint while
  everything else uses the generation model. Workers still hold no
  weights; the league model is handed to the coordinator per call.
* Each worker keeps two process-local eval caches (gen / league), both
  reset per call (= per epoch), so cache validity is unchanged.
* Samples are aggregated exactly like the production functions
  (value = outcome from the mover's side, weights 1.15 mover-won /
  1.5 vs-Det / 1.25 vs-League, survival label for self-play only) and in
  the same order (self-play, then vs-Det, then vs-League).

Safety (same model as selfplay_mp.py)
-------------------------------------
Any failure (worker death, coordinator/GPU error, IPC error, stall) tears
the pool down and the epoch is re-run through `fallback()` -- the caller's
original three-phase sequence -- so a bug here costs time, never the run.
After MAX_STRIKES failures this path disables itself for the process.
Kill switch mid-run: create PROD_DIR\\MP_UNIFIED_OFF (checked each epoch).
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

PROD_DIR = r"C:\Users\My PC\Desktop\final\11x11_v2"
if PROD_DIR not in sys.path:
    sys.path.insert(0, PROD_DIR)

PROD_MODULE = os.environ.get("MP_SELFPLAY_PROD_MODULE", "alphazero_mcts_make_lrfix")

import torch  # noqa: E402
from alphago_lee import preferred_memory_format  # noqa: E402
from alphago_lee_v3_survival import SURVIVAL_CAP  # noqa: E402
from alphazero_mcts import BOARD_SIZE  # noqa: E402

NUM_CELLS = BOARD_SIZE * BOARD_SIZE

DEFAULT_TARGET_BATCH = 64   # 2026-10-21 (user): was 24; workers now send 32-leaf requests
DEFAULT_MAX_WAIT = 0.015
DEFAULT_STALL_TIMEOUT = 180.0
MAX_STRIKES = 3

_EVAL_ERROR_TAG = "__MP_UNIFIED_EVAL_ERROR__"
KILL_SWITCH_FILE = os.path.join(PROD_DIR, "MP_UNIFIED_OFF")

TAG_GEN = "gen"
TAG_LEAGUE = "league"
# 2026-09-27: refresh/gate epochs -- the training model being mined with /
# gated, and the tracked best checkpoint it is gated against.
TAG_TRAIN = "train"
TAG_BEST = "best"

# Sample weights -- must match the production aggregation in
# alphazero_mcts_make_lrfix (generate_mcts_self_play_data /
# generate_mcts_vs_deterministic_data / generate_mcts_vs_league_data).
SELFPLAY_MOVER_WON_WEIGHT = 1.15
VSDET_WEIGHT = 1.5
LEAGUE_WEIGHT = 1.25


# The older selfplay_mp switch (PROD_DIR\MP_SELFPLAY_OFF / MP_SELFPLAY_DISABLE)
# is honored here too, so the documented "stop multi-process generation"
# switch keeps working now that this module owns generation.
_OLD_KILL_SWITCH_FILE = os.path.join(PROD_DIR, "MP_SELFPLAY_OFF")
_TRUTHY = ("1", "true", "True", "yes")


def _enabled():
    try:
        if os.path.exists(KILL_SWITCH_FILE) or os.path.exists(_OLD_KILL_SWITCH_FILE):
            return False
    except OSError:
        pass
    if os.environ.get("MP_SELFPLAY_DISABLE", "").strip() in _TRUTHY:
        return False
    return os.environ.get("MP_UNIFIED_DISABLE", "").strip() not in _TRUTHY


# ============================================================
# Worker side
# ============================================================
class _TaggedPipeEvalProxy:
    """Duck-types PooledEvaluator.submit(input_arrays) -> (probs, values).
    Several proxies (one per tag) share the worker's single pipe; a worker
    plays one game at a time and each submit blocks for its reply, so
    requests on the pipe never interleave."""
    __slots__ = ("conn", "tag")

    def __init__(self, conn, tag):
        self.conn = conn
        self.tag = tag

    def submit(self, input_arrays):
        n = len(input_arrays)
        if n == 0:
            return np.zeros((0, NUM_CELLS), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        self.conn.send((self.tag, np.stack(input_arrays)))
        probs, values = self.conn.recv()
        if isinstance(probs, str) and probs == _EVAL_ERROR_TAG:
            raise RuntimeError(f"MP unified evaluator failed: {values}")
        return probs, values


_WORKER_TABLES = {}


def _tactical_tables(prod):
    """Per-worker cache of the (pure, deterministic) tactical window/fork
    geometry tables used by the hard-example classifier."""
    if not _WORKER_TABLES:
        _WORKER_TABLES["windows"] = prod._tactical_windows()
        _WORKER_TABLES["forks"] = prod._fork_geometries()
    return _WORKER_TABLES["windows"], _WORKER_TABLES["forks"]


def _run_task(prod, kind, spec, cfg, proxy, cache):
    """proxy(tag) / cache(tag): this worker's eval proxy and per-call eval
    cache for a model tag."""
    if kind in ("self", "vsdet", "league"):
        common = dict(
            boosted_sims=cfg["boosted_sims"], boost_threshold=cfg["boost_threshold"],
            temperature_moves=cfg["temperature_moves"], dirichlet_alpha=cfg["dirichlet_alpha"],
            noise_weight=cfg["noise_weight"], c_puct=cfg["c_puct"], eval_cache=cache(TAG_GEN),
            boost_when_narrow=cfg["boost_when_narrow"], pooled_evaluator=proxy(TAG_GEN),
            full_search_prob=cfg["full_search_prob"], fast_sims=cfg["fast_sims"],
        )
        if kind == "self":
            start_board, start_player = spec
            history, outcome, ply_indices, total_plies = prod._play_one_mcts_game(
                None, None, cfg["num_simulations"], start_board=start_board, start_player=start_player,
                return_ply_info=True, max_plies=cfg.get("max_plies"), **common)
            return (history, outcome, ply_indices, total_plies)
        if kind == "vsdet":
            mcts_plays_x, opening, depth = spec
            return prod._play_one_mcts_vs_deterministic_game(
                None, None, cfg["num_simulations"], mcts_plays_x, opening=opening, det_max_depth=depth, **common)
        mcts_plays_x, opening = spec[0], spec[1]
        ltag = spec[2] if len(spec) > 2 else TAG_LEAGUE  # 2026-10-08: one tag per league opponent
        return prod._play_one_mcts_vs_league_game(
            None, None, None, cfg["num_simulations"], mcts_plays_x, opening=opening,
            league_pooled_evaluator=proxy(ltag), league_eval_cache=cache(ltag),
            max_plies=cfg.get("max_plies"), **common)

    # ---- refresh epoch: mining games (+ their hard-example classification,
    # which is pure-Python CPU work -- ~0.4 s per model ply -- and so is run
    # here in parallel instead of serially in the trainer afterwards).
    if kind in ("mine_self", "mine_vsdet", "mine_league"):
        if kind == "mine_self":
            (sims,) = spec
            plies, outcome = prod._mine_one_selfplay_game(
                None, None, sims, pooled_evaluator=proxy(TAG_TRAIN), eval_cache=cache(TAG_TRAIN))
        elif kind == "mine_vsdet":
            sims, mcts_plays_x, depth = spec
            plies, outcome = prod._mine_one_vsdet_game(
                None, None, sims, mcts_plays_x, depth, pooled_evaluator=proxy(TAG_TRAIN), eval_cache=cache(TAG_TRAIN))
        else:
            sims, mcts_plays_x = spec
            plies, outcome = prod._mine_one_vsleague_game(
                None, None, None, sims, mcts_plays_x,
                pooled_evaluator=proxy(TAG_TRAIN), eval_cache=cache(TAG_TRAIN),
                league_pooled_evaluator=proxy(TAG_LEAGUE), league_eval_cache=cache(TAG_LEAGUE))
        windows, forks = _tactical_tables(prod)
        flagged = prod._classify_hard_examples([(plies, outcome)], windows, forks)
        return (plies, outcome, flagged)

    if kind == "relabel":
        item, sims = spec
        return prod._relabel_one(item, None, None, sims, pooled_evaluator=proxy(TAG_TRAIN), eval_cache=cache(TAG_TRAIN))

    if kind == "gate":
        opening, a_plays_x = spec
        return prod._play_gate_match_game(
            None, None, None, cfg["num_simulations"], cfg["boosted_sims"], cfg["boost_threshold"],
            cfg["boost_when_narrow"], cfg["c_puct"], a_plays_x, opening,
            cache(TAG_TRAIN), cache(TAG_BEST), proxy(TAG_TRAIN), proxy(TAG_BEST))

    raise ValueError(f"unknown task kind {kind!r}")


def _worker_main(worker_id, task_queue, result_queue, eval_conn, ready_queue, error_queue):
    """One persistent CPU-only worker process. Never touches CUDA: every
    leaf evaluation goes through a tagged pipe proxy, so the model/device
    arguments of the production functions are never read (None)."""
    try:
        prod = __import__(PROD_MODULE)
    except Exception as e:  # noqa: BLE001
        error_queue.put(f"worker {worker_id} failed to import {PROD_MODULE}: {e!r}")
        ready_queue.put(worker_id)
        return

    proxies = {}
    caches = {}

    def proxy(tag):
        p = proxies.get(tag)
        if p is None:
            p = proxies[tag] = _TaggedPipeEvalProxy(eval_conn, tag)
        return p

    def cache(tag):
        c = caches.get(tag)
        if c is None:
            c = caches[tag] = {}
        return c

    local_call_id = None
    ready_queue.put(worker_id)

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            call_id, task_id, kind, spec, cfg = task
            if call_id != local_call_id:
                # New call == new frozen weights / new tag->model mapping:
                # every cache is only valid within one call.
                caches.clear()
                local_call_id = call_id
            try:
                payload = _run_task(prod, kind, spec, cfg, proxy, cache)
                result_queue.put((call_id, task_id, payload, None))
            except Exception as e:  # noqa: BLE001
                result_queue.put((call_id, task_id, None, repr(e)))
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    finally:
        try:
            eval_conn.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# Coordinator: background thread in the main process, owns the GPU
# ============================================================
class TaggedPipePooledEvaluator:
    """PipePooledEvaluator (selfplay_mp.py) with per-request model tags.
    Same accumulate-until-target_batch-or-silence-gap flush policy; a flush
    runs one forward pass per tag present in it."""

    def __init__(self, models, device, parent_conns,
                 target_batch=DEFAULT_TARGET_BATCH, max_wait=DEFAULT_MAX_WAIT):
        self.models = dict(models)
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
        self._tag_forwards = {}  # tag -> [forward passes, boards]
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="mp-unified-gpu-coordinator")
        self._thread.start()

    @property
    def flush_count(self):
        return self._flush_count

    def check_error(self):
        if self._error is not None:
            return self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            return "evaluator coordinator thread exited unexpectedly"
        return None

    def stats_snapshot(self):
        return (self._flush_count, self._total_samples, self._at_target_count,
                {k: list(v) for k, v in self._tag_forwards.items()})

    def stats_since(self, snap):
        """Per-call batching stats (deltas since `snap`), with the
        per-model forward-pass sizes reported separately -- a flush that
        mixes tags runs one forward pass per tag."""
        f0, s0, a0, t0 = snap
        flushes = self._flush_count - f0
        if flushes <= 0:
            return "no flushes"
        boards = self._total_samples - s0
        at_target = self._at_target_count - a0
        per_tag = []
        for tag, (n, b) in sorted(self._tag_forwards.items()):
            n0, b0 = t0.get(tag, [0, 0])
            if n - n0 > 0:
                per_tag.append(f"{tag} {n - n0} fwd avg {(b - b0) / (n - n0):.1f}")
        return (f"{flushes} flushes, avg {boards / flushes:.1f} boards/flush, "
                f"{at_target}/{flushes} ({100.0 * at_target / flushes:.0f}%) hit target_batch={self.target_batch}"
                + (f" | {', '.join(per_tag)}" if per_tag else ""))

    def set_models(self, models, device):
        """Swap the tag -> model map (per call: gen model + this epoch's
        league opponent). Workers hold no weights, so nothing to resync."""
        with self._lock:
            self.models = dict(models)
            self.device = device

    def _run(self):
        pending = []  # list of (conn, tag, arr)
        pending_total = 0
        live_conns = list(self.parent_conns)
        try:
            while not self._stop.is_set():
                if not live_conns:
                    time.sleep(0.1)
                    continue
                wait_timeout = self.max_wait if pending else 0.5
                try:
                    ready = mp_connection.wait(live_conns, timeout=wait_timeout)
                except (OSError, ValueError):
                    ready = []
                if self._stop.is_set():
                    break
                if not ready:
                    if pending:
                        self._flush(pending)
                        pending, pending_total = [], 0
                    continue
                for conn in ready:
                    try:
                        if conn.closed:
                            live_conns = [c for c in live_conns if c is not conn]
                            continue
                        msg = conn.recv()
                    except (EOFError, OSError, BrokenPipeError):
                        live_conns = [c for c in live_conns if c is not conn]
                        continue
                    if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], str):
                        tag, arr = msg
                    else:  # untagged -> generation model
                        tag, arr = TAG_GEN, msg
                    pending.append((conn, tag, arr))
                    pending_total += arr.shape[0]
                if pending_total >= self.target_batch:
                    self._flush(pending)
                    pending, pending_total = [], 0
            if pending:
                self._flush(pending)
        except Exception as e:  # noqa: BLE001
            self._error = f"coordinator loop crashed: {e!r}"
            self._wake_with_error([c for c, _, _ in pending], self._error)

    def _wake_with_error(self, conns, msg):
        for conn in conns:
            try:
                conn.send((_EVAL_ERROR_TAG, msg))
            except Exception:  # noqa: BLE001
                pass

    def _forward(self, model, device, arrays):
        batch_np = np.concatenate(arrays, axis=0) if len(arrays) > 1 else np.asarray(arrays[0])
        batch = torch.from_numpy(batch_np).float().to(device)
        batch = batch.contiguous(memory_format=preferred_memory_format(device))
        with torch.inference_mode():
            policy_logits, value_tensor = model(batch)
        probs = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        values = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
        return probs, values

    def _flush(self, pending):
        with self._lock:
            models, device = dict(self.models), self.device
        device = torch.device(device) if not isinstance(device, torch.device) else device

        by_tag = {}
        for item in pending:
            by_tag.setdefault(item[1], []).append(item)

        n_total = 0
        for tag, items in by_tag.items():
            conns = [c for c, _, _ in items]
            arrays = [a for _, _, a in items]
            model = models.get(tag)
            if model is None:
                msg = f"no model registered for eval tag {tag!r}"
                if self._error is None:
                    self._error = msg
                self._wake_with_error(conns, msg)
                continue
            try:
                probs, values = self._forward(model, device, arrays)
            except Exception as e:  # noqa: BLE001  (realistic case: CUDA OOM)
                msg = f"GPU forward pass failed ({tag}): {e!r}"
                if self._error is None:
                    self._error = msg
                self._wake_with_error(conns, msg)
                continue
            offset = 0
            for conn, arr in zip(conns, arrays):
                k = arr.shape[0]
                try:
                    conn.send((probs[offset:offset + k], values[offset:offset + k]))
                except (BrokenPipeError, OSError):
                    pass
                offset += k
            n_total += offset
            tf = self._tag_forwards.setdefault(tag, [0, 0])
            tf[0] += 1
            tf[1] += offset

        if n_total:
            self._flush_count += 1
            self._total_samples += n_total
            if n_total > self._max_flush:
                self._max_flush = n_total
            if n_total >= self.target_batch:
                self._at_target_count += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ============================================================
# Persistent pool
# ============================================================
class UnifiedGamePool:
    def __init__(self, num_workers, models, device, target_batch=DEFAULT_TARGET_BATCH,
                 max_wait=DEFAULT_MAX_WAIT, startup_timeout=180.0):
        self.ctx = mp.get_context("spawn")
        self.num_workers = num_workers
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.ready_queue = self.ctx.Queue()
        self.error_queue = self.ctx.Queue()
        # All of an epoch's tasks are enqueued up front (more tasks than
        # workers). If a call fails with tasks still queued, this process's
        # feeder thread can be left blocked writing them into a pipe nobody
        # reads, and multiprocessing would then join it -- forever -- at
        # interpreter exit, leaving a zombie trainer the watchdog never
        # restarts. Never join these feeders; shutdown() also drains.
        for q in (self.task_queue, self.result_queue, self.ready_queue, self.error_queue):
            q.cancel_join_thread()
        self.parent_conns = []
        self.workers = []
        self.evaluator = None
        self._next_call_id = 0

        for wid in range(num_workers):
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            self.parent_conns.append(parent_conn)
            p = self.ctx.Process(target=_worker_main,
                                 args=(wid, self.task_queue, self.result_queue, child_conn,
                                       self.ready_queue, self.error_queue),
                                 daemon=True)
            p.start()
            self.workers.append(p)
            child_conn.close()

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
            dead = [(i, p.exitcode) for i, p in enumerate(self.workers) if not p.is_alive()]
            if dead:
                # e.g. a child that died while bootstrapping (import-time
                # MemoryError / paging-file errors) never reaches
                # _worker_main, so it never reports on error_queue.
                errors.append(f"worker(s) died during startup (index, exitcode): {dead}")
                break
        if errors:
            self.shutdown()
            raise RuntimeError(f"MP unified worker startup failed: {errors}")
        if got < num_workers:
            self.shutdown()
            raise RuntimeError(f"only {got}/{num_workers} MP unified workers became ready within {startup_timeout}s")

        self.evaluator = TaggedPipePooledEvaluator(models, device, self.parent_conns, target_batch, max_wait)

    def run_tasks(self, tasks, stall_timeout=DEFAULT_STALL_TIMEOUT):
        """tasks: list of (task_id, kind, spec, cfg). Returns {task_id: payload}.
        Progress-based stall detection, as in selfplay_mp.MPGamePool."""
        call_id = self._next_call_id
        self._next_call_id += 1
        for (task_id, kind, spec, cfg) in tasks:
            self.task_queue.put((call_id, task_id, kind, spec, cfg))

        n = len(tasks)
        results = {}
        last_progress = time.monotonic()
        last_flushes = self.evaluator.flush_count
        last_result_count = 0
        while len(results) < n:
            try:
                err = self.error_queue.get_nowait()
                raise RuntimeError(f"MP unified worker reported a failure: {err}")
            except pyqueue.Empty:
                pass
            ev_err = self.evaluator.check_error()
            if ev_err is not None:
                raise RuntimeError(f"MP unified evaluator failed: {ev_err}")
            for p in self.workers:
                if not p.is_alive():
                    raise RuntimeError(f"MP unified worker died unexpectedly (exitcode={p.exitcode})")
            try:
                rcall_id, task_id, payload, err = self.result_queue.get(timeout=1.0)
                if rcall_id == call_id:
                    if err is not None:
                        raise RuntimeError(f"MP unified worker raised for task {task_id}: {err}")
                    results[task_id] = payload
            except pyqueue.Empty:
                pass
            flushes = self.evaluator.flush_count
            if flushes != last_flushes or len(results) != last_result_count:
                last_flushes = flushes
                last_result_count = len(results)
                last_progress = time.monotonic()
            if time.monotonic() - last_progress > stall_timeout:
                raise TimeoutError(f"MP unified pool stalled: {len(results)}/{n} tasks done, no eval flush "
                                   f"or finished game in {stall_timeout:.0f}s")
        return results

    def shutdown(self, join_budget=5.0):
        # 1. Drop any still-queued (unplayed) tasks so workers reach their
        #    sentinel right after their current game instead of playing on.
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
        # 2. Stop the GPU coordinator and close our pipe ends: a worker
        #    blocked on (or about to make) an eval request fails fast with
        #    EOF/BrokenPipe, reports it, then reads its sentinel and exits.
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
        # 3. One shared join budget for all workers, then terminate the rest.
        deadline = time.monotonic() + join_budget
        for p in self.workers:
            try:
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:  # noqa: BLE001
                pass
        for p in self.workers:
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        for q in (self.task_queue, self.result_queue, self.ready_queue, self.error_queue):
            try:
                q.close()
            except Exception:  # noqa: BLE001
                pass


_POOL = None
_STRIKES = 0
_PERMANENTLY_DISABLED = False
# Separate budget for the refresh/gate task runner (2026-09-27).
_TASK_STRIKES = 0
_TASK_DISABLED = False


def _ensure_pool(num_workers, models, device, target_batch, max_wait):
    """Returns (pool, spawned_now)."""
    global _POOL
    if _POOL is not None and _POOL.num_workers != num_workers:
        _POOL.shutdown()
        _POOL = None
    if _POOL is None:
        # Release the old self-play-only pool if a previous fallback epoch
        # spawned it, so the two never hold 2 x num_workers processes at once.
        try:
            import selfplay_mp
            selfplay_mp.shutdown_pool()
        except Exception:  # noqa: BLE001
            pass
        _POOL = UnifiedGamePool(num_workers, models, device, target_batch, max_wait)
        return _POOL, True
    _POOL.evaluator.set_models(models, device)
    return _POOL, False


def shutdown_pool():
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _POOL = None


atexit.register(shutdown_pool)


def _resolve_workers(max_workers):
    override = os.environ.get("MP_SELFPLAY_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max_workers or max(1, os.cpu_count() or 1)


# ============================================================
# Aggregation (mirrors the production generate_* functions exactly)
# ============================================================
def _aggregate_selfplay(results_in_order, augment_sample, mix, q_weight):
    raw_samples = []
    outcome_counts = {'x_win': 0, 'o_win': 0, 'draw': 0}
    opening_rows = []
    for history, outcome, ply_indices, total_plies in results_in_order:
        if outcome == 1:
            outcome_counts['x_win'] += 1
        elif outcome == -1:
            outcome_counts['o_win'] += 1
        else:
            outcome_counts['draw'] += 1
        # Only a recorded ply 0 is the game's actual first move (PCR skips
        # recording fast-search plies).
        if history and ply_indices and ply_indices[0] == 0:
            first_cell = int(np.argmax(history[0][2]))
            opening_rows.append(first_cell // BOARD_SIZE)
        for (board_3d, player_turn, target_policy_map, q), ply_idx in zip(history, ply_indices):
            final_val = mix(float(outcome if player_turn == 'X' else -outcome), q, q_weight)
            mover_won = ((outcome == 1 and player_turn == 'X') or (outcome == -1 and player_turn == 'O'))
            sample_weight = SELFPLAY_MOVER_WON_WEIGHT if mover_won else 1.0
            survival = (None if total_plies is None  # capped game: remaining length unknown
                        else min(total_plies - ply_idx, SURVIVAL_CAP) / SURVIVAL_CAP)
            for (aug_b, aug_p) in augment_sample(board_3d, target_policy_map):
                raw_samples.append((aug_b, aug_p, final_val, sample_weight, survival))
    return raw_samples, outcome_counts, opening_rows


def _aggregate_vsdet(results_in_order, depths, augment_sample, mix, q_weight):
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'det_win': 0, 'draw': 0}
    by_depth = {}
    for (history, outcome, mcts_result), depth in zip(results_in_order, depths):
        tally = by_depth.setdefault(depth, {'w': 0, 'l': 0, 'd': 0})
        if mcts_result == 'mcts':
            outcome_counts['mcts_win'] += 1
            tally['w'] += 1
        elif mcts_result == 'det':
            outcome_counts['det_win'] += 1
            tally['l'] += 1
        else:
            outcome_counts['draw'] += 1
            tally['d'] += 1
        for (board_3d, player_turn, target_policy_map, q) in history:
            final_val = mix(float(outcome if player_turn == 'X' else -outcome), q, q_weight)
            for (aug_b, aug_p) in augment_sample(board_3d, target_policy_map):
                raw_samples.append((aug_b, aug_p, final_val, VSDET_WEIGHT))
    outcome_counts['by_depth'] = by_depth
    return raw_samples, outcome_counts


def _aggregate_league(results_in_order, augment_sample, mix, q_weight):
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'league_win': 0, 'draw': 0}
    for (history, outcome, mcts_result) in results_in_order:
        if mcts_result == 'mcts':
            outcome_counts['mcts_win'] += 1
        elif mcts_result == 'league':
            outcome_counts['league_win'] += 1
        else:
            outcome_counts['draw'] += 1
        for (board_3d, player_turn, target_policy_map, q) in history:
            final_val = mix(float(outcome if player_turn == 'X' else -outcome), q, q_weight)
            for (aug_b, aug_p) in augment_sample(board_3d, target_policy_map):
                raw_samples.append((aug_b, aug_p, final_val, LEAGUE_WEIGHT))
    return raw_samples, outcome_counts


# ============================================================
# Entry point
# ============================================================
def generate_epoch_games_unified(gen_model, device, *, epoch, max_workers,
                                 selfplay_cfg, num_self_play_games, num_curriculum_games, num_opening_games,
                                 vsdet_cfg, num_vs_det_games,
                                 league_model, num_league_games,
                                 fallback,
                                 target_batch=DEFAULT_TARGET_BATCH, max_wait=DEFAULT_MAX_WAIT,
                                 stall_timeout=DEFAULT_STALL_TIMEOUT, value_q_weight=0.0):
    """Plays every generation game of this epoch in one pool.

    selfplay_cfg / vsdet_cfg: dicts with num_simulations, boosted_sims,
    boost_threshold, temperature_moves, dirichlet_alpha, noise_weight,
    c_puct, boost_when_narrow, full_search_prob, fast_sims (vsdet_cfg is
    also used for the vs-League games, as in the production loop).
    league_model None (or num_league_games 0) skips league games.

    Returns (selfplay_samples, selfplay_counts, vsdet_samples, vsdet_counts,
    league_samples, league_counts) -- vsdet/league counts are None when that
    game type was not played, exactly as in the production loop. On ANY
    failure (or when disabled) returns fallback(reason) instead, which must
    produce the same 6-tuple via the original sequential path.
    """
    global _STRIKES, _PERMANENTLY_DISABLED

    if _PERMANENTLY_DISABLED:
        return fallback("unified pool permanently disabled after repeated failures")
    if not _enabled():
        shutdown_pool()
        return fallback("unified pool kill switch is set")

    t0 = time.monotonic()
    try:
        prod = __import__(PROD_MODULE)
        from alphago_lee_make import augment_sample

        workers = _resolve_workers(max_workers)
        # 2026-10-08 (user): league_model may be a LIST of opponents; league game i
        # then plays opponent i % len(list), each under its own eval tag.
        league_list = list(league_model) if isinstance(league_model, (list, tuple)) else (
            [league_model] if league_model is not None else [])
        play_league = bool(league_list) and num_league_games > 0
        play_vsdet = num_vs_det_games > 0

        gen_model.eval()
        models = {TAG_GEN: gen_model}
        league_tags = []
        if play_league:
            for k, lm in enumerate(league_list):
                lm.eval()
                tag = TAG_LEAGUE if len(league_list) == 1 else f"{TAG_LEAGUE}{k}"
                models[tag] = lm
                league_tags.append(tag)
        pool, spawned_now = _ensure_pool(workers, models, device, target_batch, max_wait)
        if spawned_now:
            pool.evaluator.set_models(models, device)
        stats_at_start = pool.evaluator.stats_snapshot()

        # Task specs are built here, in the main process, with the same
        # module-level random state / tables the production functions use.
        tasks = []
        league_ids, vsdet_ids, self_ids = [], [], []
        if play_league:
            for i in range(num_league_games):
                tid = f"league{i}"
                opening = prod._OPENING_SQUARES[(i + epoch) % len(prod._OPENING_SQUARES)]
                tasks.append((tid, "league", (i % 2 == 0, opening, league_tags[i % len(league_tags)]), vsdet_cfg))
                league_ids.append(tid)
        vsdet_depths = []
        if play_vsdet:
            vsdet_depths = [prod._sample_vs_det_max_depth(epoch) for _ in range(num_vs_det_games)]
            for i in range(num_vs_det_games):
                tid = f"vsdet{i}"
                opening = prod._OPENING_SQUARES[(i + epoch) % len(prod._OPENING_SQUARES)]
                tasks.append((tid, "vsdet", (i % 2 == 0, opening, vsdet_depths[i]), vsdet_cfg))
                vsdet_ids.append(tid)
        for i in range(num_self_play_games):
            start_board, start_player = None, None
            if i < num_curriculum_games:
                sampled = prod._sample_tactical_start()
                if sampled is not None:
                    start_board, start_player = sampled
            elif i < num_curriculum_games + num_opening_games:
                start_board, start_player = prod._sample_forced_opening_start()
            tid = f"self{i}"
            tasks.append((tid, "self", (start_board, start_player), selfplay_cfg))
            self_ids.append(tid)

        print(f"Generating {len(tasks)} games in ONE unified pool [multi-process, workers: {workers}]: "
              f"{num_self_play_games} self-play (full_search_prob={selfplay_cfg['full_search_prob']}, "
              f"fast_sims={selfplay_cfg['fast_sims']}), {len(vsdet_ids)} vs-Det, {len(league_ids)} vs-League "
              f"(Simulations: {selfplay_cfg['num_simulations']}/move, boosted {selfplay_cfg['boosted_sims']})...",
              flush=True)

        # 2026-10-21 (user): self-play + vs-League games in C++ threads with large GPU
        # batches (selfplay_cpp); the worker pool stays as the fallback and for vs-Det.
        results = None
        if not play_vsdet:
            try:
                import selfplay_cpp
                if selfplay_cpp.enabled():
                    results = selfplay_cpp.run_tasks(tasks, models, device, prod, TAG_GEN)
                    print(f"  [C++ self-play] {len(tasks)} games: {selfplay_cpp.LAST_STATS}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [C++ self-play] FAILED ({e!r}) -- playing this epoch in the worker pool", flush=True)
                results = None
        if results is None:
            results = pool.run_tasks(tasks, stall_timeout=stall_timeout)

        mix = prod.mixed_value_target
        sp_samples, sp_counts, opening_rows = _aggregate_selfplay([results[t] for t in self_ids], augment_sample,
                                                                  mix, value_q_weight)
        vd_samples, vd_counts = (None, None)
        if play_vsdet:
            vd_samples, vd_counts = _aggregate_vsdet([results[t] for t in vsdet_ids], vsdet_depths, augment_sample,
                                                     mix, value_q_weight)
        lg_samples, lg_counts = (None, None)
        if play_league:
            lg_samples, lg_counts = _aggregate_league([results[t] for t in league_ids], augment_sample,
                                                      mix, value_q_weight)

        elapsed = time.monotonic() - t0
        print(f"Generated {len(sp_samples)} augmented MCTS training samples [unified]. "
              f"Outcomes: X {sp_counts['x_win']} / O {sp_counts['o_win']} / Draw {sp_counts['draw']} "
              f"| Opening row-coverage this epoch: {len(set(opening_rows))}/{BOARD_SIZE} "
              f"(from {len(opening_rows)} games with a recorded first move)", flush=True)
        if play_vsdet:
            print(f"Generated {len(vd_samples)} augmented vs-Deterministic training samples [unified]. "
                  f"Outcomes: MCTS {vd_counts['mcts_win']} / Det {vd_counts['det_win']} / "
                  f"Draw {vd_counts['draw']} | {prod.format_by_depth(vd_counts['by_depth'])}", flush=True)
        if play_league:
            print(f"Generated {len(lg_samples)} augmented vs-League training samples [unified]. "
                  f"Outcomes: MCTS {lg_counts['mcts_win']} / League {lg_counts['league_win']} / "
                  f"Draw {lg_counts['draw']}", flush=True)
        note = " (INCLUDES one-time pool startup)" if spawned_now else ""
        print(f"  [gen-timing] unified {elapsed:.1f}s{note} | evaluator this epoch: "
              f"{pool.evaluator.stats_since(stats_at_start)}", flush=True)
        return sp_samples, sp_counts, vd_samples, vd_counts, lg_samples, lg_counts

    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        _STRIKES += 1
        print(f"  [MP unified] FAILED after {time.monotonic() - t0:.1f}s "
              f"(strike {_STRIKES}/{MAX_STRIKES}): {e!r}", flush=True)
        shutdown_pool()
        if _STRIKES >= MAX_STRIKES:
            _PERMANENTLY_DISABLED = True
            print("  [MP unified] too many failures -- permanently reverting to the sequential "
                  "generation path for the rest of this process.", flush=True)
        return fallback(f"re-running this epoch's generation on the sequential path ({e!r})")


# ============================================================
# Refresh / gate epochs (2026-09-27)
# ============================================================
class UnifiedTaskRunner:
    """Runs a refresh epoch's mining games (+ per-game hard-example
    classification), relabel searches and promotion-gate games in the same
    persistent pool as generation. Each method raises on any failure (after
    counting a strike and tearing the pool down, like
    generate_epoch_games_unified); the production caller then falls back to
    its original threaded/sequential code for that step."""

    def __init__(self, max_workers):
        self.workers = _resolve_workers(max_workers)
        # One runner instance is created per refresh epoch. After its first
        # failure every later call on it (e.g. the gate right after a failed
        # mining step) goes straight to the caller's fallback instead of
        # respawning the pool and failing again.
        self.failed = False

    def _run(self, models, device, tasks, label):
        global _TASK_STRIKES, _TASK_DISABLED
        # Refresh/gate failures have their OWN strike budget: they may turn
        # this runner off for the process, but must never disable the
        # (separately proven) generation path's use of the pool.
        if self.failed:
            raise RuntimeError("unified task runner already failed earlier this refresh epoch")
        if _PERMANENTLY_DISABLED or _TASK_DISABLED:
            raise RuntimeError("unified pool / task runner permanently disabled after repeated failures")
        if not _enabled():
            shutdown_pool()
            raise RuntimeError("unified pool kill switch is set")
        t0 = time.monotonic()
        try:
            pool, spawned_now = _ensure_pool(self.workers, models, device, DEFAULT_TARGET_BATCH, DEFAULT_MAX_WAIT)
            pool.evaluator.set_models(models, device)
            snap = pool.evaluator.stats_snapshot()
            results = pool.run_tasks(tasks)
            note = " (INCLUDES one-time pool startup)" if spawned_now else ""
            print(f"  [{label}-timing] unified {time.monotonic() - t0:.1f}s{note} | {len(tasks)} tasks | "
                  f"evaluator: {pool.evaluator.stats_since(snap)}", flush=True)
            return results
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            self.failed = True
            _TASK_STRIKES += 1
            print(f"  [MP unified] {label} FAILED after {time.monotonic() - t0:.1f}s "
                  f"(task-runner strike {_TASK_STRIKES}/{MAX_STRIKES}): {e!r}", flush=True)
            shutdown_pool()  # generation respawns a clean pool next epoch
            if _TASK_STRIKES >= MAX_STRIKES:
                _TASK_DISABLED = True
                print("  [MP unified] refresh/gate task runner permanently disabled for this process "
                      "(generation keeps using the pool).", flush=True)
            raise

    def mine(self, model, league_model, device, sims, num_selfplay_games, vsdet_depths, num_league_games):
        """Mining games in the production order (self-play, vs-Det, vs-League).
        Returns (games, flagged): games = [(plies, outcome)] in that order;
        flagged = the concatenated per-game hard-example lists, deduped
        keeping first occurrence -- identical to running
        _classify_hard_examples once over all games in that order."""
        was_training = model.training
        model.eval()
        models = {TAG_TRAIN: model}
        if num_league_games > 0 and league_model is not None:
            league_model.eval()
            models[TAG_LEAGUE] = league_model
        else:
            num_league_games = 0
        tasks = [(f"ms{i}", "mine_self", (sims,), None) for i in range(num_selfplay_games)]
        tasks += [(f"md{g}", "mine_vsdet", (sims, g % 2 == 0, vsdet_depths[g]), None)
                  for g in range(len(vsdet_depths))]
        tasks += [(f"ml{g}", "mine_league", (sims, g % 2 == 0), None) for g in range(num_league_games)]
        try:
            results = self._run(models, device, tasks, "mining")
        finally:
            model.train(was_training)
        games, flagged, seen = [], [], set()
        for tid, _, _, _ in tasks:
            plies, outcome, game_flagged = results[tid]
            games.append((plies, outcome))
            for f in game_flagged:
                board, player = f[0], f[1]
                key = (''.join(''.join(row) for row in board), player)
                if key in seen:
                    continue
                seen.add(key)
                flagged.append(f)
        return games, flagged

    def relabel(self, model, device, items, sims):
        """Relabels `items` (already selected/ordered by
        _order_relabel_candidates) in parallel; returns results in the same
        order, dropping items whose search found no children."""
        if not items:
            return []
        was_training = model.training
        model.eval()
        tasks = [(f"rl{i}", "relabel", (item, sims), None) for i, item in enumerate(items)]
        try:
            results = self._run({TAG_TRAIN: model}, device, tasks, "relabel")
        finally:
            model.train(was_training)
        return [results[t[0]] for t in tasks if results[t[0]] is not None]

    def gate(self, model, best_model, device, cfg, openings_and_sides):
        """Promotion-gate games (model = 'a' side, best_model = 'b' side).
        Returns 'a'/'b'/'draw' per (opening, a_plays_x), in order."""
        was_training = model.training
        model.eval()
        best_model.eval()
        tasks = [(f"g{i}", "gate", (opening, a_plays_x), cfg)
                 for i, (opening, a_plays_x) in enumerate(openings_and_sides)]
        try:
            results = self._run({TAG_TRAIN: model, TAG_BEST: best_model}, device, tasks, "gate")
        finally:
            model.train(was_training)
        return [results[t[0]] for t in tasks]


def get_task_runner(max_workers):
    return UnifiedTaskRunner(max_workers)
