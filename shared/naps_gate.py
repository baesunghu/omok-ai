"""The gate: a candidate plays the current best; score = (wins + 0.5 draws - GATE_LOSS_PENALTY x losses) / games.
Per-run settings (loss penalty, opening radius) come from run_config.py.
"""
import os
import random
import time
import torch
from concurrent.futures import ThreadPoolExecutor

from fast_board import sims_for_state
from mcts_cpp import BOARD_SIZE, MCTS, PooledEvaluator, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH, GATE_LOSS_PENALTY, GATE_OPENING_RADIUS, POOLED_TARGET_BATCH
from nets import GomokuNetV3 as GomokuNet
from board_encoding import preferred_memory_format
from naps_common import FILE_SUFFIX
from naps_openings import OPENING_SQUARES

# ============================================================
# Promotion Gate (AlphaGo Zero style)
# ============================================================
# A policy trained by gradient descent does not improve monotonically (checkpoints here were non-transitive: one beat
# a checkpoint 500 epochs older and lost to one 100 epochs older). As in AlphaGo Zero, raw weights never become the
# baseline: a candidate must beat the current best in fair play (no noise, full search). Training is untouched --
# data generation keeps using the continuously updated model; only BEST_CHECKPOINT_PATH is tracked.
BEST_CHECKPOINT_PATH = f"model/best_mcts{FILE_SUFFIX}.pth"
# score = (wins + 0.5 draws - GATE_LOSS_PENALTY x losses) / games must exceed this ("a model that does not lose").
# History: 0.55 (AlphaGo Zero, draws 0.5) -> 0.50 -> 0.25 -> 0.35 (an even 6-8-6 scored 0.275 and promoted ep1005).
# The loss penalty is per run (run_config.py: 11x11 0.75, 15x15 0.4), as are the gate opening radius and batch sizes.
GATE_PROMOTION_THRESHOLD = 0.35
# Two-stage promotion: the gate (against the current best only) can prefer a model the Rapfi ladder rates weaker
# (ep1050 beat ep960 12-5-3 but measured 2,404 vs 4,432 nodes). So a pass only makes a CANDIDATE (weights in
# model/gate_candidates/, model/gate_candidate.json); the ladder watcher measures it and promotes it
# (model/ladder_promote.json) only if it is not weaker than the best.
GATE_CANDIDATE_MODE = True
RECORD_LEAGUE_SIDE = False   # See play_one_mcts_vs_league_game
LAST_GATE_CANDIDATE = {"path": None}


def _save_gate_candidate(model, score):
    import json as _json
    os.makedirs("model/gate_candidates", exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    path = f"model/gate_candidates/cand_{tag}{FILE_SUFFIX}.pth"
    torch.save(model.state_dict(), path)
    with open("model/gate_candidate.json", "w", encoding="utf-8") as f:
        _json.dump({"ckpt": path, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gate_score": round(score, 4),
                    "status": "pending"}, f, indent=1)
    LAST_GATE_CANDIDATE["path"] = path
    return path


def play_gate_match_game(model_a, model_b, device, num_simulations, boosted_sims=None, boost_threshold=0,
                           boost_when_narrow=False, c_puct=1.414, a_plays_x=True, opening=None,
                           eval_cache_a=None, eval_cache_b=None,
                           pooled_evaluator_a=None, pooled_evaluator_b=None):
    """
    One fair (no-noise, greedy, full-search) game between two frozen models,
    with a forced opening (see _canonical_symmetry_openings -- without one,
    no-noise MCTS vs no-noise MCTS is fully deterministic and every game
    with the same side assignment would just replay the same line). Returns
    'a', 'b', or 'draw'.
    """
    a_player = 'X' if a_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    mcts_a = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache_a, pooled_evaluator=pooled_evaluator_a)
    mcts_b = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache_b, pooled_evaluator=pooled_evaluator_b)

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        if fast_check_win(board, a_player):
            return 'a'
        if fast_check_win(board, 'O' if a_player == 'X' else 'X'):
            return 'b'
        if fast_check_draw(board):
            return 'draw'

        num_empty = sum(1 for row in board for cell in row if cell == ' ')
        sims = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
        active_model, active_mcts = (model_a, mcts_a) if current_player == a_player else (model_b, mcts_b)
        root = active_mcts.search(board, current_player, active_model, device,
                                   num_simulations=sims, add_noise=False)
        if not root.children:
            return 'draw'
        best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        r, c = divmod(best_action, BOARD_SIZE)
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


# Gate-check speed optimization: the periodic tactical-refresh
# epoch (every tactical_refresh_interval) was measured taking 15-25x a
# normal epoch's wall-clock -- ~42% of total training time in a 232-epoch
# stretch -- and the 42-game (21 openings x 2 sides) gate check was the
# dominant cost within it. Cut to 10 stratified openings (20 games): a
# fixed-seed shuffle of OPENING_SQUARES so the subset stays spatially
# spread rather than clustering in whatever order
# _canonical_symmetry_openings happened to emit them (same principle as the
# vs-Det opening fix). Trades some promotion-decision precision (10%-point
# granularity instead of ~2.4%-point) for roughly half the gate-check cost.
# 15x15: canonical openings include far corner squares; there the gate keeps squares within
# Chebyshev distance GATE_OPENING_RADIUS of the centre. The 11x11 runs used every canonical square (radius None).
GATE_OPENING_SQUARES = [sq for sq in OPENING_SQUARES if GATE_OPENING_RADIUS is None
                         or max(abs(sq[0] - BOARD_SIZE // 2), abs(sq[1] - BOARD_SIZE // 2)) <= GATE_OPENING_RADIUS]
random.Random(910).shuffle(GATE_OPENING_SQUARES)
GATE_OPENING_SQUARES = GATE_OPENING_SQUARES[:10]


def run_gating_check(model, gen_model, device, num_simulations, boosted_sims=None, boost_threshold=0,
                      boost_when_narrow=False, c_puct=1.414, max_workers=None, runner=None):
    """
    The gate: `model` plays the tracked best checkpoint from every symmetry-distinct opening, both sides.
    score = (wins + 0.5 draws - GATE_LOSS_PENALTY x losses) / games; a score above GATE_PROMOTION_THRESHOLD passes.
    With GATE_CANDIDATE_MODE (on) a passing model is only saved as a candidate (_save_gate_candidate) for the ladder
    to judge; otherwise it becomes the new best and gen_model -- the model data generation uses -- takes its weights.
    With no best checkpoint yet, model becomes the best.

    The games run in the selfplay_pool task runner when one is given (the two models under separate tags), else in
    threads, each model with its own PooledEvaluator and cache (a cached value is only valid for the weights that
    produced it). Either way only speed differs. Returns (win_rate, draw_rate, loss_rate, promoted).
    """
    if not os.path.exists(BEST_CHECKPOINT_PATH):
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
        gen_model.load_state_dict(model.state_dict())
        return 1.0, 0.0, 0.0, True

    best_model = GomokuNet()
    try:
        best_model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=device))
    except Exception as e:
        print(f"Gate check notice ({e}), skipping this cycle.")
        return 0.0, 0.0, 0.0, False
    best_model.to(device, memory_format=preferred_memory_format(device))
    best_model.eval()

    # model arrives here straight out of tactical_refresh(), which leaves it
    # in train() mode -- with eval_batch_size=1 every MCTS leaf eval is a
    # single-sample forward pass, so BatchNorm would normalize against that
    # one sample's own stats instead of the accumulated running mean/var,
    # systematically distorting every policy/value output for this model
    # only (the same bug was confirmed in the 5x5 sibling's gate check).
    model.eval()

    results = None
    if runner is not None:
        try:
            results = runner.gate(model, best_model, device, dict(
                num_simulations=num_simulations, boosted_sims=boosted_sims, boost_threshold=boost_threshold,
                boost_when_narrow=boost_when_narrow, c_puct=c_puct),
                [(opening, latest_plays_x) for opening in GATE_OPENING_SQUARES for latest_plays_x in (True, False)])
        except Exception as e:  # noqa: BLE001
            print(f"  [gate] unified-pool gate failed ({e!r}) -- threaded fallback", flush=True)
            results = None
    if results is None:
        max_workers = max_workers or max(1, min(len(GATE_OPENING_SQUARES) * 2, os.cpu_count() or 1))
        eval_cache_model = {}
        eval_cache_best = {}
        owns_pooled = device.type == "cuda"
        pooled_model = PooledEvaluator(model, device, target_batch=POOLED_TARGET_BATCH, max_wait=0.05) if owns_pooled else None
        pooled_best = PooledEvaluator(best_model, device, target_batch=POOLED_TARGET_BATCH, max_wait=0.05) if owns_pooled else None
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for opening in GATE_OPENING_SQUARES:
                    for latest_plays_x in (True, False):
                        futures.append(pool.submit(play_gate_match_game, model, best_model, device, num_simulations,
                                                    boosted_sims, boost_threshold, boost_when_narrow, c_puct,
                                                    latest_plays_x, opening,
                                                    eval_cache_model, eval_cache_best, pooled_model, pooled_best))
                results = [future.result() for future in futures]
        finally:
            if pooled_model is not None:
                pooled_model.stop()
            if pooled_best is not None:
                pooled_best.stop()

    wins = sum(1 for r in results if r == 'a')
    draws = sum(1 for r in results if r == 'draw')
    losses = len(results) - wins - draws
    n = max(1, wins + draws + losses)
    score = (wins + 0.5 * draws - GATE_LOSS_PENALTY * losses) / n
    promoted = score > GATE_PROMOTION_THRESHOLD
    if promoted and GATE_CANDIDATE_MODE:
        _save_gate_candidate(model, score)
        promoted = False
    elif promoted:
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
        gen_model.load_state_dict(model.state_dict())
    return wins / n, draws / n, losses / n, promoted
