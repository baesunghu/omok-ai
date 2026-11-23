"""
Multiprocessing-based formal evaluation (evaluate_vs_deterministic) for the
11x11 hybrid MCTS pipeline.

Why this exists: evaluate_vs_deterministic's existing ThreadPoolExecutor +
PooledEvaluator implementation caps effective CPU usage at roughly 1 of 8
logical cores in production, because determinisic_play (the pure-Python
minimax opponent every game plays against) holds the GIL for the entirety of
its own call and therefore fully serializes across worker threads -- the C++
MCTS engine itself already releases the GIL correctly during its own tree
search (verified directly against the deployed mcts_core.pyd source), so that
part was never the bottleneck. Giving each concurrent game a real OS process
instead of a thread routes around the GIL entirely for the part that was
actually serializing. Measured on this hardware (GTX 1050, 8 logical cores):
~2.5x wall-clock speedup on a realistic vs-Deterministic game-generation
workload at the live sim budget, with bit-for-bit-identical search outputs
(deterministic/no-noise/greedy mode is a pure function of model weights +
opening + settings, so crossing a process boundary changes no numeric value).

Architecture: a worker-pool of up to os.cpu_count() processes each running
worker_loop, pulling (game_idx, mcts_plays_x, opening) jobs from a shared
queue until exhausted -- this avoids spawning more OS processes than there
are cores when num_games (e.g. eval_games=20) exceeds the core count. Every
worker's own C++ tree search (mcts_core.mcts_search_batched) and
board-to-plane conversion happen locally (CPU-bound, parallel across
processes, no CUDA). NN evaluation is routed to ONE centralized GPU-owning
evaluator process over a persistent shared-memory channel per worker
(see WorkerChannel/ChannelHandle), mirroring PooledEvaluator's own
accumulate-until-target_batch-or-max_wait batching loop but across a process
boundary instead of a thread boundary.

Safety: evaluate_vs_deterministic_mp raises on ANY setup failure or a worker
dying without completing its assigned games (rather than hanging or silently
returning wrong stats), so its caller (evaluate_vs_deterministic in
alphazero_mcts_make.py) can fall back to the existing, long-proven threaded
implementation. On an unattended 20000-epoch job, a periodic eval call
failing over to the slower-but-solid path is fine; a hang is not.

Not yet ported: generate_mcts_self_play_data, generate_mcts_vs_deterministic_data,
generate_mcts_vs_league_data, and run_gating_check still use the threaded
implementation -- their game-generation loops need per-move policy targets
(training data) or a model-vs-model matchup (no GIL-heavy minimax opponent in
the loop, so the same speedup would not obviously apply), which this module
does not attempt. See the investigation report this was built from for the
full reasoning on why those were left out of this pass.
"""
import multiprocessing as mp
import os
import queue as pyqueue
import time

import numpy as np

from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win, fast_check_draw

NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_INPUT_CHANNELS = 5
MAX_BATCH = 20  # eval_batch_size=16 in production + headroom


# ============================================================
# Shared-memory IPC (one persistent channel per worker, reused every round trip)
# ============================================================
class WorkerChannel:
    """Created ONCE per worker by the main process before spawning it."""
    def __init__(self, worker_id):
        req_bytes = MAX_BATCH * NUM_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE * 4  # float32
        probs_bytes = MAX_BATCH * NUM_CELLS * 8  # float64
        values_bytes = MAX_BATCH * 8  # float64
        from multiprocessing import shared_memory
        self._req_shm = shared_memory.SharedMemory(create=True, size=req_bytes)
        self._probs_shm = shared_memory.SharedMemory(create=True, size=probs_bytes)
        self._values_shm = shared_memory.SharedMemory(create=True, size=values_bytes)
        self.req_shm_name = self._req_shm.name
        self.probs_shm_name = self._probs_shm.name
        self.values_shm_name = self._values_shm.name
        self.count_val = mp.Value('i', 0)

    def close_owner(self):
        """Call from the process that created this channel (main), after
        all workers/evaluator using it have exited. Safe to call even if a
        worker crashed and never touched its channel."""
        for shm in (self._req_shm, self._probs_shm, self._values_shm):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


class ChannelHandle:
    """Attached view of a WorkerChannel's shared memory, usable from any
    process via the block names + a fresh Event reference."""
    def __init__(self, worker_id, req_shm_name, probs_shm_name, values_shm_name, response_ready_event, count_val):
        from multiprocessing import shared_memory
        self.worker_id = worker_id
        self.response_ready = response_ready_event
        self.count_val = count_val
        self._req_shm = shared_memory.SharedMemory(name=req_shm_name)
        self._probs_shm = shared_memory.SharedMemory(name=probs_shm_name)
        self._values_shm = shared_memory.SharedMemory(name=values_shm_name)
        self.req_arr = np.ndarray((MAX_BATCH, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                                   dtype=np.float32, buffer=self._req_shm.buf)
        self.probs_arr = np.ndarray((MAX_BATCH, NUM_CELLS), dtype=np.float64, buffer=self._probs_shm.buf)
        self.values_arr = np.ndarray((MAX_BATCH,), dtype=np.float64, buffer=self._values_shm.buf)

    def close(self):
        for shm in (self._req_shm, self._probs_shm, self._values_shm):
            shm.close()


# ============================================================
# Centralized GPU evaluator process
# ============================================================
def evaluator_main(ckpt_path, request_queue, channel_specs, ready_event, stop_event,
                    target_batch=48, max_wait=0.004, error_queue=None):
    try:
        import torch
        from alphago_lee import AlphaZeroTicTacToe, preferred_memory_format

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlphaZeroTicTacToe()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device, memory_format=preferred_memory_format(device))
        model.eval()

        handles = {}
        for wid, (req_name, probs_name, values_name, resp_event, count_val) in channel_specs.items():
            handles[wid] = ChannelHandle(wid, req_name, probs_name, values_name, resp_event, count_val)

        if device.type == "cuda":
            torch.cuda.synchronize()
        ready_event.set()
    except Exception as e:  # noqa: BLE001
        if error_queue is not None:
            error_queue.put(f"evaluator_main setup failed: {e!r}")
        ready_event.set()  # unblock the waiter so it sees the error queue promptly
        return

    pending_ids = []
    first_pending_time = None

    try:
        while True:
            if stop_event.is_set() and not pending_ids:
                try:
                    while True:
                        wid = request_queue.get_nowait()
                        if wid != -1:
                            pending_ids.append(wid)
                except pyqueue.Empty:
                    pass
                if not pending_ids:
                    break

            timeout = max_wait if pending_ids else 0.25
            wid = None
            try:
                wid = request_queue.get(timeout=timeout)
            except pyqueue.Empty:
                pass

            if wid is not None and wid != -1:
                pending_ids.append(wid)
                if first_pending_time is None:
                    first_pending_time = time.perf_counter()

            elapsed = (time.perf_counter() - first_pending_time) if first_pending_time else 0.0
            total_leaves = sum(handles[w].count_val.value for w in pending_ids)
            should_flush = pending_ids and (
                total_leaves >= target_batch or
                elapsed >= max_wait or
                (wid is None and stop_event.is_set())
            )

            if should_flush:
                all_planes = []
                offsets = []
                for w in pending_ids:
                    h = handles[w]
                    n = h.count_val.value
                    offsets.append((len(all_planes), n))
                    all_planes.append(h.req_arr[:n])
                batch_np = np.concatenate(all_planes, axis=0) if len(all_planes) > 1 else np.array(all_planes[0])
                batch_tensor = torch.from_numpy(batch_np).float().to(device)
                batch_tensor = batch_tensor.contiguous(memory_format=torch.contiguous_format)
                with torch.inference_mode():
                    policy_logits, value_tensor = model(batch_tensor)
                probs_batch = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
                if probs_batch.ndim == 1:
                    probs_batch = probs_batch.reshape(1, -1)
                values_batch = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)

                for (start, n), w in zip(offsets, pending_ids):
                    h = handles[w]
                    h.probs_arr[:n] = probs_batch[start:start + n]
                    h.values_arr[:n] = values_batch[start:start + n]
                    h.response_ready.set()

                pending_ids = []
                first_pending_time = None
    except Exception as e:  # noqa: BLE001
        if error_queue is not None:
            error_queue.put(f"evaluator_main loop failed: {e!r}")
    finally:
        for h in handles.values():
            h.close()


def make_remote_eval_batch_fn(handle, request_queue, board_to_input_array_from_numeric):
    def eval_batch_fn(boards, players):
        n = len(boards)
        for i in range(n):
            handle.req_arr[i] = board_to_input_array_from_numeric(boards[i], players[i])
        handle.count_val.value = n
        handle.response_ready.clear()
        request_queue.put(handle.worker_id)
        if not handle.response_ready.wait(timeout=120):
            raise TimeoutError(f"worker {handle.worker_id}: no response from evaluator within 120s")
        probs = handle.probs_arr[:n].copy()
        values = handle.values_arr[:n].copy()
        return probs, values
    return eval_batch_fn


# ============================================================
# sims_for_state -- duplicated on purpose to keep worker processes torch-free.
# MUST stay identical to alphazero_mcts_make.sims_for_state; that function is
# the source of truth if the two ever need to be reconciled.
# ============================================================
def sims_for_state(num_empty, base_sims, boosted_sims=None, boost_threshold=0, boost_when_narrow=False):
    if not boosted_sims:
        return base_sims
    if boost_when_narrow:
        sims = boosted_sims if num_empty <= boost_threshold else base_sims
    else:
        sims = boosted_sims if num_empty > boost_threshold else base_sims
    if num_empty > NUM_CELLS - 10:
        sims = max(sims, boosted_sims)
    return sims


# ============================================================
# Worker process: pulls jobs from a shared queue until exhausted
# ============================================================
def worker_loop(worker_id, job_queue, result_queue, req_shm_name, probs_shm_name, values_shm_name,
                 response_ready_event, count_val, request_queue,
                 num_simulations, boosted_sims, boost_threshold, boost_when_narrow,
                 c_puct, det_max_depth, eval_batch_size, error_queue):
    try:
        import mcts_core
        from alphago_lee import board_to_input_array_from_numeric
        import determinisic_play

        handle = ChannelHandle(worker_id, req_shm_name, probs_shm_name, values_shm_name,
                                response_ready_event, count_val)
        eval_batch_fn = make_remote_eval_batch_fn(handle, request_queue, board_to_input_array_from_numeric)

        while True:
            job = job_queue.get()
            if job is None:
                break
            game_idx, mcts_plays_x, opening = job

            mcts_player = 'X' if mcts_plays_x else 'O'
            board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            current_player = 'X'
            if opening is not None:
                r0, c0 = opening
                board[r0][c0] = 'X'
                current_player = 'O'

            result = None
            while True:
                if fast_check_win(board, mcts_player):
                    result = 'mcts'
                    break
                if fast_check_win(board, 'O' if mcts_player == 'X' else 'X'):
                    result = 'det'
                    break
                if fast_check_draw(board):
                    result = 'draw'
                    break

                if current_player == mcts_player:
                    num_empty = sum(1 for row in board for cell in row if cell == ' ')
                    sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims,
                                                     boost_threshold, boost_when_narrow)
                    _, _, candidates = mcts_core.mcts_search_batched(
                        board, current_player, eval_batch_fn, sims_this_move, c_puct,
                        False, 0.3, 0.25, BOARD_SIZE, WIN_LENGTH, eval_batch_size
                    )
                    if not candidates:
                        result = 'draw'
                        break
                    best = max(candidates, key=lambda c: c['visit_count'])
                    r, c = divmod(best['action_idx'], BOARD_SIZE)
                else:
                    r, c = determinisic_play.play(board, current_player, max_depth=det_max_depth)
                    if r == -1:
                        result = 'draw'
                        break

                board[r][c] = current_player
                current_player = 'O' if current_player == 'X' else 'X'

            result_queue.put({'game_idx': game_idx, 'result': result})

        handle.close()
    except Exception as e:  # noqa: BLE001
        if error_queue is not None:
            error_queue.put(f"worker {worker_id} failed: {e!r}")


# ============================================================
# Orchestration
# ============================================================
def evaluate_vs_deterministic_mp(model, device, num_games, num_simulations, boosted_sims=None,
                                  boost_threshold=0, c_puct=1.414, boost_when_narrow=False,
                                  det_max_depth=2, all_cells=None, max_workers=None,
                                  target_batch=48, max_wait=0.004, tmp_ckpt_dir="model"):
    """
    Drop-in alternative to alphazero_mcts_make.evaluate_vs_deterministic's
    internals: same (win_rate, draw_rate, loss_rate) return, same opening
    assignment (game i -> all_cells[i % len(all_cells)], mcts_plays_x = i%2==0)
    so per-game outcomes are directly comparable to the threaded
    implementation. Raises on any setup/runtime failure (including a worker
    dying mid-game) instead of hanging or returning partial stats -- callers
    should catch and fall back to the threaded implementation.
    """
    import torch

    if all_cells is None:
        all_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]

    os.makedirs(tmp_ckpt_dir, exist_ok=True)
    ckpt_path = os.path.abspath(os.path.join(tmp_ckpt_dir, "_mp_eval_tmp_ckpt.pth"))
    torch.save(model.state_dict(), ckpt_path)

    num_workers = max(1, min(num_games, os.cpu_count() or 1))
    ctx = mp.get_context("spawn")

    request_queue = ctx.Queue()
    result_queue = ctx.Queue()
    job_queue = ctx.Queue()
    error_queue = ctx.Queue()

    channels = {}
    response_events = {}
    channel_specs = {}
    for wid in range(num_workers):
        ch = WorkerChannel(wid)
        channels[wid] = ch
        resp_event = ctx.Event()
        response_events[wid] = resp_event
        channel_specs[wid] = (ch.req_shm_name, ch.probs_shm_name, ch.values_shm_name, resp_event, ch.count_val)

    ready_event = ctx.Event()
    stop_event = ctx.Event()

    evaluator_proc = ctx.Process(
        target=evaluator_main,
        args=(ckpt_path, request_queue, channel_specs, ready_event, stop_event),
        kwargs={'target_batch': target_batch, 'max_wait': max_wait, 'error_queue': error_queue},
        daemon=True,
    )
    worker_procs = []

    def _cleanup():
        stop_event.set()
        for p in worker_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        evaluator_proc.join(timeout=10)
        if evaluator_proc.is_alive():
            evaluator_proc.terminate()
        for ch in channels.values():
            ch.close_owner()

    try:
        evaluator_proc.start()
        if not ready_event.wait(timeout=60):
            raise RuntimeError("evaluator process did not become ready within 60s")
        try:
            err = error_queue.get_nowait()
            raise RuntimeError(f"evaluator process failed during setup: {err}")
        except pyqueue.Empty:
            pass

        for i in range(num_games):
            mcts_plays_x = (i % 2 == 0)
            opening = all_cells[i % len(all_cells)]
            job_queue.put((i, mcts_plays_x, opening))
        for _ in range(num_workers):
            job_queue.put(None)

        for wid in range(num_workers):
            ch = channels[wid]
            p = ctx.Process(
                target=worker_loop,
                args=(wid, job_queue, result_queue, ch.req_shm_name, ch.probs_shm_name, ch.values_shm_name,
                      response_events[wid], ch.count_val, request_queue,
                      num_simulations, boosted_sims, boost_threshold, boost_when_narrow,
                      c_puct, det_max_depth, 16, error_queue),
            )
            p.start()
            worker_procs.append(p)

        results = {}
        deadline = time.monotonic() + max(60.0, num_games * 90.0)
        while len(results) < num_games:
            try:
                err = error_queue.get_nowait()
                raise RuntimeError(f"worker/evaluator process reported a failure: {err}")
            except pyqueue.Empty:
                pass

            if time.monotonic() > deadline:
                raise TimeoutError(f"only {len(results)}/{num_games} games finished before the deadline")

            try:
                item = result_queue.get(timeout=1.0)
                results[item['game_idx']] = item['result']
            except pyqueue.Empty:
                pass

            if not evaluator_proc.is_alive():
                raise RuntimeError("evaluator process died unexpectedly")
            for p in worker_procs:
                if not p.is_alive() and p.exitcode not in (0, None):
                    raise RuntimeError(f"a worker process died unexpectedly (exitcode={p.exitcode})")

    finally:
        _cleanup()
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    wins = draws = losses = 0
    for i in range(num_games):
        r = results.get(i)
        if r == 'mcts':
            wins += 1
        elif r == 'draw':
            draws += 1
        else:
            losses += 1
    n = max(1, num_games)
    return wins / n, draws / n, losses / n
