"""League play: past checkpoints (and anchors such as the best model) as opponents for training games.
"""
import glob
import numpy as np
import os
import random
import re
import torch
from concurrent.futures import ThreadPoolExecutor

from mcts_cpp import BOARD_SIZE, MCTS, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH
from nets import GomokuNetV3 as GomokuNet
from board_encoding import augment_sample, board_to_input_array, preferred_memory_format
from naps_common import FILE_SUFFIX, mixed_value_target, sims_for_state
from naps_openings import OPENING_SQUARES
from naps_gate import BEST_CHECKPOINT_PATH, RECORD_LEAGUE_SIDE


# ============================================================
# League Training (self-play vs past checkpoints)
# ============================================================
# Pure self-play (vs current-self) + vs-Deterministic (one fixed opponent)
# risks overfitting to that single opponent's style -- a weakness never
# gets tested against unless that one opponent happens to exploit it. League
# training (AlphaStar/OpenAI Five) mixes in games against a pool of the
# model's OWN past checkpoints (already saved every 10 epochs as
# model_MCTS_ep{N}{FILE_SUFFIX}.pth snapshots -- no new storage needed), so
# self-play sees a diversity of strategies instead of one fixed target.


LEAGUE_ANCHOR_PROB = 0.0   # random anchor draw (first try) -- superseded by the rotation below
LEAGUE_ANCHOR_ROTATION = True  # epoch%3 -> Dawn / ep645 / normal draw
_LEAGUE_SNAPSHOT_RE = re.compile(r"model_MCTS_ep(\d+)" + re.escape(FILE_SUFFIX) + r"\.pth$")


def _load_league_model(path, device):
    """Shared checkpoint-load helper for pick_league_opponent's two
    branches (tracked-best / snapshot-pool). Returns the loaded, eval()'d
    model, or None on any load failure (caller treats None as "skip league
    games this epoch", same as the pre-existing failure behavior)."""
    league_model = GomokuNet()
    try:
        league_model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        print(f"League opponent load notice ({e}), skipping league games this epoch.")
        return None
    league_model.to(device, memory_format=preferred_memory_format(device))
    league_model.eval()
    return league_model


def pick_league_opponent(device, pool_size=20, best_checkpoint_prob=0.28, epoch=0,
                           deep_history_prob=0.10, deep_history_min_epoch=100):
    """Picks a frozen league opponent for this epoch's vs-League games.

    With probability best_checkpoint_prob (checked first), loads the
    tracked BEST_CHECKPOINT_PATH -- the strongest validated checkpoint,
    always kept up to date by run_gating_check's promotion logic -- instead
    of sampling from the snapshot pool at all. This guarantees a real,
    strong sparring partner shows up in league play regardless of how the
    plain recency-weighted draw below happens to fall.

    Otherwise (or if that file doesn't exist), with probability
    deep_history_prob (once epoch >= deep_history_min_epoch) samples
    UNIFORMLY from the FULL snapshot history instead of just the most
    recent pool_size -- the recency-weighted pool below permanently
    excludes a snapshot once pool_size*10 epochs have passed it, so without
    this the model would only ever face styles from its own recent past,
    never its much-earlier (structurally weaker/more exploratory) self.
    Kept low-probability and floored at deep_history_min_epoch so it stays
    a small, occasional addition, not a replacement for the recency-weighted
    pool that already does most of the work.

    Otherwise, falls back to the existing snapshot pool: a random past
    checkpoint snapshot from the most recent pool_size snapshots BY EPOCH
    NUMBER, not filesystem mtime -- mtime can get scrambled by anything
    that touches the files without touching training progress, e.g. a
    backup pass or several snapshots written in quick succession right
    around a restart. Sampling within that pool is recency-weighted
    (linearly favoring the newest snapshot), not uniform, since the newest
    snapshots are the most representative of the model's current style --
    the whole point of league play is testing against strategies close to
    what self-play might otherwise overfit to.

    Returns (league_model, path), or None if the best-checkpoint load
    failed, or if fewer than 2 snapshots exist yet -- too early in training
    for league play to add anything over the fixed deterministic opponent
    already in use."""
    # Deterministic rotation over epochs --
    # epoch % 3 == 0 -> first anchor (Dawn ep625), == 1 -> second anchor (ep645),
    # == 2 -> the normal draw below (best / deep history / recency pool).
    anchors = sorted(glob.glob("model/league_anchors/*.pth")) if LEAGUE_ANCHOR_ROTATION else []
    if anchors and epoch % 3 < len(anchors) and epoch % 3 < 2:
        chosen = anchors[epoch % 3]
        league_model = _load_league_model(chosen, device)
        if league_model is not None:
            return league_model, chosen

    if best_checkpoint_prob > 0 and random.random() < best_checkpoint_prob and os.path.exists(BEST_CHECKPOINT_PATH):
        league_model = _load_league_model(BEST_CHECKPOINT_PATH, device)
        if league_model is None:
            return None
        return league_model, BEST_CHECKPOINT_PATH

    # Fixed "anchor" opponents -- the graduated Dawn (ep625) and the
    # long-standing best ep645 -- so league play keeps facing the strongest known models
    # while the gate is stuck. Checked right after the best-checkpoint draw.
    anchors = [p for p in sorted(glob.glob("model/league_anchors/*.pth"))] if LEAGUE_ANCHOR_PROB > 0 else []
    if anchors and random.random() < LEAGUE_ANCHOR_PROB:
        chosen = random.choice(anchors)
        league_model = _load_league_model(chosen, device)
        if league_model is None:
            return None
        return league_model, chosen

    pattern = f"model/model_MCTS_ep*{FILE_SUFFIX}.pth"
    numbered = []
    for path in glob.glob(pattern):
        m = _LEAGUE_SNAPSHOT_RE.search(os.path.basename(path))
        if m:  # excludes the "_final" end-of-run checkpoint, which doesn't match this pattern
            numbered.append((int(m.group(1)), path))
    numbered.sort(key=lambda t: t[0])
    snapshots = [path for _, path in numbered]
    if len(snapshots) < 2:
        return None

    if epoch >= deep_history_min_epoch and random.random() < deep_history_prob:
        chosen = random.choice(snapshots)
        league_model = _load_league_model(chosen, device)
        if league_model is None:
            return None
        return league_model, chosen

    candidates = snapshots[-pool_size:]
    # candidates is oldest-to-newest; a linear ramp (1, 2, ..., len) makes
    # the newest snapshot len(candidates) times as likely as the oldest,
    # while still leaving every snapshot in the pool a nonzero chance.
    weights = list(range(1, len(candidates) + 1))
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    league_model = _load_league_model(chosen, device)
    if league_model is None:
        return None
    return league_model, chosen


LEAGUE_NUM_OPPONENTS = 12  # Many opponents per epoch instead of one


def pick_league_opponents_multi(device, k, pool_size=40, epoch=0):
    """2026-10-08 (user, option A): k distinct league opponents for ONE epoch --
    the fixed anchors (Dawn ep625, ep645), the current best, one uniform pick from
    the whole snapshot history, and the rest recency-weighted from the newest
    pool_size snapshots without replacement. Returns [(model, path), ...]."""
    chosen = []
    seen = set()

    def add(path):
        if path in seen or len(chosen) >= k:
            return
        m = _load_league_model(path, device)
        if m is not None:
            chosen.append((m, path))
            seen.add(path)

    for a in sorted(glob.glob("model/league_anchors/*.pth")):
        add(a)
    if os.path.exists(BEST_CHECKPOINT_PATH):
        add(BEST_CHECKPOINT_PATH)
    numbered = []
    for path in glob.glob(f"model/model_MCTS_ep*{FILE_SUFFIX}.pth"):
        mm = _LEAGUE_SNAPSHOT_RE.search(os.path.basename(path))
        if mm:
            numbered.append((int(mm.group(1)), path))
    numbered.sort(key=lambda t: t[0])
    snapshots = [p for _, p in numbered]
    if snapshots:
        add(random.choice(snapshots))
    cands = snapshots[-pool_size:]
    weights = list(range(1, len(cands) + 1))
    while len(chosen) < k and any(c not in seen for c in cands):
        pool = [(c, w) for c, w in zip(cands, weights) if c not in seen]
        add(random.choices([c for c, _ in pool], weights=[w for _, w in pool], k=1)[0])
    return chosen


def play_one_mcts_vs_league_game(model, league_model, device, num_simulations, mcts_plays_x,
                                   boosted_sims=None, boost_threshold=0,
                                   temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                   c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                   pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                   opening=None, league_pooled_evaluator=None, league_eval_cache=None,
                                   max_plies=None):
    """
    Like play_one_mcts_vs_deterministic_game, but the opponent is a frozen
    past checkpoint of this same model instead of deterministic_play. Both
    sides' moves are recorded (the league opponent runs a real, always-full
    search -- see league_sims below -- so its visit distribution is just as
    valid a policy target as the live model's own): the live model was
    previously only ever training on its own moves against the league pool,
    never on the moves of the checkpoints that actually beat it, which
    starved league-derived data of the one thing it could uniquely teach.

    opening (see play_one_mcts_vs_deterministic_game): pre-places X's first
    move there before normal play resumes. The league side's own moves use
    add_noise=False (greedy), so when it moves first, every game with the
    same side assignment would otherwise replay the identical opening move.

    league_pooled_evaluator/league_eval_cache (2026-09-26, unified MP
    game pool): route the LEAGUE side's leaf evaluations through a separate
    evaluator/cache -- a worker process holds no weights, so it needs a
    proxy that the GPU coordinator maps to the league model. Both default
    to None, which keeps the original behavior (league side calls
    league_model directly with a private per-search cache).

    Returns (history, outcome, mcts_result): mcts_result is
    'mcts'/'league'/'draw' from the live model's perspective.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    league_mcts = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=league_eval_cache,
                       pooled_evaluator=league_pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    history = []
    # Decisions that record (or, for a PCR fast search, would have recorded)
    # a history entry -- drives temperature; equals len(history) at
    # full_search_prob=1.0 (PCR fix).
    decisions = 0
    outcome = 0

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        # game-length cap (stones on the board); capped = draw.
        if max_plies and sum(1 for row in board for cell in row if cell != ' ') >= max_plies:
            outcome = 0
            break
        if fast_check_win(board, 'X'):
            outcome = 1
            break
        if fast_check_win(board, 'O'):
            outcome = -1
            break
        if fast_check_draw(board):
            outcome = 0
            break

        num_empty = sum(1 for row in board for cell in row if cell == ' ')

        if current_player == mcts_player:
            is_full_search = full_search_prob >= 1.0 or random.random() < full_search_prob
            if is_full_search:
                sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
            else:
                sims_this_move = fast_sims or max(1, num_simulations // 6)
            root = mcts.search(board, current_player, model, device,
                                num_simulations=sims_this_move, add_noise=is_full_search,
                                dirichlet_alpha=dirichlet_alpha, noise_weight=noise_weight)
            if not root.children:
                outcome = 0
                break

            total_visits = sum(child.n_visits for child in root.children.values())
            target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for act, child in root.children.items():
                r, c = divmod(act, BOARD_SIZE)
                target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0

            move_count = decisions
            actions = list(root.children.keys())
            visits = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)
            if move_count < temperature_moves:
                probs = visits / visits.sum()
                chosen_act = np.random.choice(actions, p=probs)
            else:
                chosen_act = actions[np.argmax(visits)]
            r, c = divmod(chosen_act, BOARD_SIZE)
            decisions += 1

            if is_full_search:
                board_3d = board_to_input_array(board, current_player)
                history.append((board_3d, current_player, target_policy_map, float(root.value())))
        else:
            # PCR on the league side too (same full_search_prob
            # as the MCTS side): a fast search only picks the move and is not
            # recorded. At full_search_prob=1.0 this is the old behavior.
            league_full = full_search_prob >= 1.0 or random.random() < full_search_prob
            if league_full:
                league_sims = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
            else:
                league_sims = fast_sims or max(1, num_simulations // 6)
            root = league_mcts.search(board, current_player, league_model, device,
                                       num_simulations=league_sims, add_noise=False)
            if not root.children:
                outcome = 0
                break
            total_visits = sum(child.n_visits for child in root.children.values())
            target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for act, child in root.children.items():
                r, c = divmod(act, BOARD_SIZE)
                target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0

            best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
            r, c = divmod(best_action, BOARD_SIZE)

            # league-side plies are no longer training targets -- the opponents are older,
            # weaker checkpoints (some ~700-1,000 equal nodes), so their visit counts taught weaker moves.
            if league_full and RECORD_LEAGUE_SIDE:
                board_3d = board_to_input_array(board, current_player)
                # League-side ply: its search value comes from a different
                # (older) network, so this ply keeps a pure-z value target.
                history.append((board_3d, current_player, target_policy_map, None))
            decisions += 1

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'

    if outcome == 1:
        mcts_result = 'mcts' if mcts_player == 'X' else 'league'
    elif outcome == -1:
        mcts_result = 'mcts' if mcts_player == 'O' else 'league'
    else:
        mcts_result = 'draw'

    return history, outcome, mcts_result


def generate_mcts_vs_league_data(model, league_model, device, num_games=6, num_simulations=40, max_workers=None,
                                  boosted_sims=None, boost_threshold=0,
                                  temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                  c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                  pooled_evaluator=None, full_search_prob=1.0, fast_sims=None, epoch=0,
                                  value_q_weight=0.0, max_plies=None):
    """
    Same shape as generate_mcts_vs_deterministic_data, but the opponent is a
    frozen past checkpoint of this model (see play_one_mcts_vs_league_game).
    """
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    print(f"Generating {num_games} MCTS-vs-League training games "
          f"(Simulations: {num_simulations}/move, workers: {max_workers})...")

    model.eval()
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'league_win': 0, 'draw': 0}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(play_one_mcts_vs_league_game, model, league_model, device, num_simulations,
                                i % 2 == 0, boosted_sims, boost_threshold,
                                temperature_moves, dirichlet_alpha, noise_weight, c_puct, eval_cache,
                                boost_when_narrow, pooled_evaluator, full_search_prob, fast_sims,
                                OPENING_SQUARES[(i + epoch) % len(OPENING_SQUARES)],
                                None, None, max_plies)
                   for i in range(num_games)]
        for future in futures:
            history, outcome, mcts_result = future.result()
            if mcts_result == 'mcts':
                outcome_counts['mcts_win'] += 1
            elif mcts_result == 'league':
                outcome_counts['league_win'] += 1
            else:
                outcome_counts['draw'] += 1
            for (board_3d, player_turn, target_policy_map, q) in history:
                # Full +-1.0 scale (was 0.5). The halved scale
                # shrank Q's dynamic range against an unchanged c_puct, which
                # doubles the effective exploration constant and quadruples the
                # simulations needed for visit counts to overcome the prior. It
                # also fought tactical_refresh's own 0.97 value target.
                final_val = mixed_value_target(float(outcome if player_turn == 'X' else -outcome), q, value_q_weight)
                sample_weight = 1.25
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val, sample_weight))

    print(f"Generated {len(raw_samples)} augmented vs-League training samples. "
          f"Outcomes: MCTS {outcome_counts['mcts_win']} / League {outcome_counts['league_win']} / "
          f"Draw {outcome_counts['draw']}")
    return raw_samples, outcome_counts
