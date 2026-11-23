"""The GPU side of the self-play pool: one thread in the training process collects evaluation requests from
every worker (over pipes), batches them per model tag and answers each worker. Workers hold no weights.
"""
import multiprocessing.connection as mp_connection
import numpy as np
import threading
import time
import torch

from fast_board import NUM_CELLS
from run_config import POOLED_TARGET_BATCH
from board_encoding import preferred_memory_format
from selfplay_pool_tasks import TAG_GEN




DEFAULT_TARGET_BATCH = POOLED_TARGET_BATCH   # 11x11_v2: 64 with 32-leaf requests; other runs 24
DEFAULT_MAX_WAIT = 0.015


_EVAL_ERROR_TAG = "__MP_UNIFIED_EVAL_ERROR__"


# ============================================================
# Worker side
# ============================================================
class TaggedPipeEvalProxy:
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


# ============================================================
# Coordinator: background thread in the main process, owns the GPU
# ============================================================
class TaggedPipePooledEvaluator:
    """PipePooledEvaluator (selfplay_pool_fallback.py) with per-request model tags.
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
