"""The main self-play pool: one persistent set of worker processes plays ALL of an epoch's training games --
self-play (including curriculum and opening starts), vs-Det and vs-League -- plus the refresh and gate tasks.

Why one pool: the older sequence (self-play in processes, then vs-Det and vs-League in threads) waited for the slowest
game of each phase, so most workers sat idle at every phase's end; the two-game vs-League phase left 8 of 10 workers
and most of the GPU idle (evaluator average batch 30 -> 22, "hit target" 57%). Here every game goes onto one queue
(longest first) that 10 workers drain, and the GPU coordinator keeps getting full batches.

How it differs from selfplay_pool_fallback.py:
- Workers take several task kinds and run the trainer's own game functions (naps_games) for each -- no ported game
  logic, so nothing can drift.
- Evaluation requests carry a model tag ('gen', 'league', 'train', 'best'); the coordinator batches per tag and runs
  one forward pass per model. Workers still hold no weights.
- Each worker keeps one cache per tag, reset every call (every epoch), so cached values stay valid.
- Samples are built exactly as the trainer builds them, in the same order (selfplay_pool_tasks).

Safety: any failure (a worker dying, a GPU or pipe error, a stall) tears the pool down and the epoch is re-run through
fallback(), the caller's three-phase sequence -- a bug here costs time, never the run. After MAX_STRIKES failures this
path turns itself off for the process. Turn it off mid-run by creating MP_UNIFIED_OFF in the run folder.
"""
import atexit
import multiprocessing as mp
import os
import queue as pyqueue
import sys
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root -> omok_paths
from omok_paths import run_dir  # noqa: E402
HERE = run_dir()   # the run folder of the script that imported this module
PROD_DIR = HERE
if PROD_DIR not in sys.path:
    sys.path.insert(0, PROD_DIR)

GAMES_MODULE = "naps_games"   # the game functions the workers run (naps_games.py)

from fast_board import BOARD_SIZE  # noqa: E402
from run_config import CPP_SELFPLAY  # noqa: E402
from selfplay_pool_tasks import (  # noqa: E402
    TAG_BEST, TAG_GEN, TAG_LEAGUE, TAG_TRAIN, aggregate_league,
    aggregate_selfplay, aggregate_vsdet, run_task,
)
from selfplay_pool_evaluator import (  # noqa: E402
    DEFAULT_MAX_WAIT, DEFAULT_TARGET_BATCH, TaggedPipePooledEvaluator, TaggedPipeEvalProxy,
)

DEFAULT_STALL_TIMEOUT = 180.0
MAX_STRIKES = 3

KILL_SWITCH_FILE = os.path.join(PROD_DIR, "MP_UNIFIED_OFF")


# The older selfplay_pool_fallback switch (PROD_DIR\MP_SELFPLAY_OFF / MP_SELFPLAY_DISABLE)
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


def _worker_main(worker_id, task_queue, result_queue, eval_conn, ready_queue, error_queue):
    """One persistent CPU-only worker process. Never touches CUDA: every
    leaf evaluation goes through a tagged pipe proxy, so the model/device
    arguments of the production functions are never read (None)."""
    try:
        naps_games = __import__(GAMES_MODULE)
    except Exception as e:  # noqa: BLE001
        error_queue.put(f"worker {worker_id} failed to import {GAMES_MODULE}: {e!r}")
        ready_queue.put(worker_id)
        return

    proxies = {}
    caches = {}

    def proxy(tag):
        p = proxies.get(tag)
        if p is None:
            p = proxies[tag] = TaggedPipeEvalProxy(eval_conn, tag)
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
                payload = run_task(naps_games, kind, spec, cfg, proxy, cache)
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
        Progress-based stall detection, as in selfplay_pool_fallback.MPGamePool."""
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
# Separate budget for the refresh/gate task runner.
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
            import selfplay_pool_fallback
            selfplay_pool_fallback.shutdown_pool()
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
        naps_games = __import__(GAMES_MODULE)
        from board_encoding import augment_sample

        workers = _resolve_workers(max_workers)
        # league_model may be a LIST of opponents; league game i
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
                opening = naps_games.OPENING_SQUARES[(i + epoch) % len(naps_games.OPENING_SQUARES)]
                tasks.append((tid, "league", (i % 2 == 0, opening, league_tags[i % len(league_tags)]), vsdet_cfg))
                league_ids.append(tid)
        vsdet_depths = []
        if play_vsdet:
            vsdet_depths = [naps_games.sample_vs_det_max_depth(epoch) for _ in range(num_vs_det_games)]
            for i in range(num_vs_det_games):
                tid = f"vsdet{i}"
                opening = naps_games.OPENING_SQUARES[(i + epoch) % len(naps_games.OPENING_SQUARES)]
                tasks.append((tid, "vsdet", (i % 2 == 0, opening, vsdet_depths[i]), vsdet_cfg))
                vsdet_ids.append(tid)
        for i in range(num_self_play_games):
            start_board, start_player = None, None
            if i < num_curriculum_games:
                sampled = naps_games.sample_tactical_start()
                if sampled is not None:
                    start_board, start_player = sampled
            elif i < num_curriculum_games + num_opening_games:
                start_board, start_player = naps_games.sample_forced_opening_start()
            tid = f"self{i}"
            tasks.append((tid, "self", (start_board, start_player), selfplay_cfg))
            self_ids.append(tid)

        print(f"Generating {len(tasks)} games in ONE unified pool [multi-process, workers: {workers}]: "
              f"{num_self_play_games} self-play (full_search_prob={selfplay_cfg['full_search_prob']}, "
              f"fast_sims={selfplay_cfg['fast_sims']}), {len(vsdet_ids)} vs-Det, {len(league_ids)} vs-League "
              f"(Simulations: {selfplay_cfg['num_simulations']}/move, boosted {selfplay_cfg['boosted_sims']})...",
              flush=True)

        # self-play + vs-League games in C++ threads with large GPU
        # batches (selfplay_cpp); the worker pool stays as the fallback and for vs-Det.
        results = None
        if CPP_SELFPLAY and not play_vsdet:
            try:
                import selfplay_cpp
                if selfplay_cpp.enabled():
                    results = selfplay_cpp.run_tasks(tasks, models, device, naps_games, TAG_GEN)
                    print(f"  [C++ self-play] {len(tasks)} games: {selfplay_cpp.LAST_STATS}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [C++ self-play] FAILED ({e!r}) -- playing this epoch in the worker pool", flush=True)
                results = None
        if results is None:
            results = pool.run_tasks(tasks, stall_timeout=stall_timeout)

        mix = naps_games.mixed_value_target
        sp_samples, sp_counts, opening_rows = aggregate_selfplay([results[t] for t in self_ids], augment_sample,
                                                                  mix, value_q_weight)
        vd_samples, vd_counts = (None, None)
        if play_vsdet:
            vd_samples, vd_counts = aggregate_vsdet([results[t] for t in vsdet_ids], vsdet_depths, augment_sample,
                                                     mix, value_q_weight)
        lg_samples, lg_counts = (None, None)
        if play_league:
            lg_samples, lg_counts = aggregate_league([results[t] for t in league_ids], augment_sample,
                                                      mix, value_q_weight)

        elapsed = time.monotonic() - t0
        print(f"Generated {len(sp_samples)} augmented MCTS training samples [unified]. "
              f"Outcomes: X {sp_counts['x_win']} / O {sp_counts['o_win']} / Draw {sp_counts['draw']} "
              f"| Opening row-coverage this epoch: {len(set(opening_rows))}/{BOARD_SIZE} "
              f"(from {len(opening_rows)} games with a recorded first move)", flush=True)
        if play_vsdet:
            print(f"Generated {len(vd_samples)} augmented vs-Deterministic training samples [unified]. "
                  f"Outcomes: MCTS {vd_counts['mcts_win']} / Det {vd_counts['det_win']} / "
                  f"Draw {vd_counts['draw']} | {naps_games.format_by_depth(vd_counts['by_depth'])}", flush=True)
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
# Refresh / gate epochs
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
        classify_hard_examples once over all games in that order."""
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
        order_relabel_candidates) in parallel; returns results in the same
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
