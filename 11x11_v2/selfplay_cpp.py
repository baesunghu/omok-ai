"""C++ self-play for the unified generation phase (11x11_v2, 2026-10-21, user).

Plays an epoch's self-play and vs-League games with mcts_core.SelfPlayRunner:
C++ threads run the games and searches inside this (trainer) process, and
every game's leaf evaluations are gathered into large batches that go to the
GPU through one PyTorch call per batch -- instead of 8 worker processes each
sending small requests over pipes (measured average GPU batch 5.9 rows).

Game rules, PCR, sims boost, temperature, forced playouts, MEGA and
max_plies mirror _play_one_mcts_game / _play_one_mcts_vs_league_game; the
returned results have exactly the formats the pool workers return, so the
aggregation code downstream is unchanged. vs-Det games are not supported
here (the caller keeps using the worker pool when an epoch has any).

Switch off: create the file CPP_SELFPLAY_OFF next to this module (or set
CPP_SELFPLAY_DISABLE=1); generation then goes back to the worker pool.
"""
import os
import time

import numpy as np
import torch

import mcts_core
import alphazero_mcts_lrfix as L
from alphago_lee import board_to_input_array, preferred_memory_format
from fast_board import BOARD_SIZE, WIN_LENGTH

HERE = os.path.dirname(os.path.abspath(__file__))
KILL_SWITCH_FILE = os.path.join(HERE, "CPP_SELFPLAY_OFF")

N_THREADS = 48        # concurrent games (C++ threads; most of them wait on the GPU at any moment)
TARGET_ROWS = 512     # flush a GPU batch once this many leaf rows are queued...
MAX_WAIT_MS = 4.0     # ...or after this long, or when every running game is waiting

LAST_STATS = ""


def enabled():
    if not hasattr(mcts_core, "SelfPlayRunner"):
        return False
    if os.path.exists(KILL_SWITCH_FILE):
        return False
    return os.environ.get("CPP_SELFPLAY_DISABLE", "").strip() not in ("1", "true", "True", "yes")


def _to_str_board(arr):
    return [['X' if v == 1 else ('O' if v == 2 else ' ') for v in row] for row in arr.tolist()]


def run_tasks(tasks, models, device, prod, gen_tag):
    """tasks: unified-pool task tuples (tid, kind, spec, cfg), kind 'self' or 'league'.
    models: {tag: model}. Returns {tid: result} in the pool workers' formats."""
    global LAST_STATS
    tags = [gen_tag] + [t for t in models if t != gen_tag]
    tag_idx = {t: i for i, t in enumerate(tags)}
    mlist = [models[t] for t in tags]
    for m in mlist:
        m.eval()

    R = mcts_core.SelfPlayRunner()
    R.board_size, R.win_length = BOARD_SIZE, WIN_LENGTH
    R.eval_batch = prod.EVAL_BATCH
    R.forced_k = L.FORCED_PLAYOUT_K
    R.mega_d1, R.mega_d2 = L.MEGA_VISITS
    R.mega_min_sims = L.MEGA_MIN_SIMS
    R.record_league_side = bool(prod.RECORD_LEAGUE_SIDE)
    for tid, kind, spec, cfg in tasks:
        if kind == "self":
            start_board, start_player = spec
            R.add_task(0, start_board, start_player or "", True, -1, 0, cfg)
        elif kind == "league":
            mcts_plays_x, opening = spec[0], spec[1]
            ltag = spec[2] if len(spec) > 2 else [t for t in tags if t != gen_tag][0]
            op = -1 if opening is None else opening[0] * BOARD_SIZE + opening[1]
            R.add_task(1, None, "", bool(mcts_plays_x), op, tag_idx[ltag], cfg)
        else:
            raise ValueError(f"C++ self-play does not play {kind!r} tasks")

    mf = preferred_memory_format(device)

    def eval_fn(tag, planes):
        batch = torch.from_numpy(planes).to(device).contiguous(memory_format=mf)
        with torch.inference_mode():
            policy_logits, value_tensor = mlist[tag](batch)
        probs = torch.exp(policy_logits).cpu().numpy().astype(np.float64).reshape(planes.shape[0], -1)
        values = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
        return probs, values

    t0 = time.monotonic()
    raw = R.run(eval_fn, min(N_THREADS, len(tasks)), TARGET_ROWS, MAX_WAIT_MS)
    LAST_STATS = (f"{time.monotonic() - t0:.1f}s, GPU batches {R.n_batches}, "
                  f"avg {R.n_rows / max(1, R.n_batches):.1f} rows, max {R.max_rows}, "
                  f"evaluated {R.n_rows}, cache hits {R.cache_hits}")

    # per-opponent vs-League tally (main net's view), appended to LAST_STATS
    tally = {}
    for (tid, kind, spec, cfg), r in zip(tasks, raw):
        if kind == "league":
            t = tally.setdefault(spec[2] if len(spec) > 2 else "league", [0, 0, 0])
            t[{1: 0, -1: 1}.get(r[4], 2)] += 1
    if tally:
        LAST_STATS += " | vs-League W/L/D: " + ", ".join(f"{k} {w}/{l}/{d}" for k, (w, l, d) in tally.items())

    results = {}
    for (tid, kind, spec, cfg), (hist, outcome, plies, total_plies, mcts_result) in zip(tasks, raw):
        history = [(board_to_input_array(_to_str_board(b), p), p, pol, q) for b, p, pol, q in hist]
        if kind == "self":
            results[tid] = (history, outcome, list(plies), total_plies)
        else:
            results[tid] = (history, outcome, {1: "mcts", -1: "league"}.get(mcts_result, "draw"))
    return results
