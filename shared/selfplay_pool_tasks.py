"""What a pool worker does with one task (a self-play, vs-Det or league game, a mining game, a relabel, a gate
game) and how finished games become training samples (value = outcome for the mover, per-source weights).
"""
import numpy as np

from fast_board import BOARD_SIZE
from nets import SURVIVAL_CAP


TAG_GEN = "gen"
TAG_LEAGUE = "league"
# refresh/gate epochs -- the training model being mined with /
# gated, and the tracked best checkpoint it is gated against.
TAG_TRAIN = "train"
TAG_BEST = "best"

# Sample weights -- must match the production aggregation in
# naps_trainer (generate_mcts_self_play_data /
# generate_mcts_vs_deterministic_data / generate_mcts_vs_league_data).
SELFPLAY_MOVER_WON_WEIGHT = 1.15
VSDET_WEIGHT = 1.5
LEAGUE_WEIGHT = 1.25


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
        ltag = spec[2] if len(spec) > 2 else TAG_LEAGUE  # One tag per league opponent
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
