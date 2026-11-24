"""Hard examples: positions mined from finished games where the net was wrong or unsure, relabelled with a
deeper search, kept in a pool (model/hard_mined_examples*.pkl) and drawn into every epoch's training data.
"""
import deterministic_play
import numpy as np
import os
import pickle
import random

from mcts_cpp import BOARD_SIZE, MCTS, NUM_CELLS, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH
from naps_common import FILE_SUFFIX
from board_encoding import augment_sample, board_to_input_array
from naps_threats import fork_hits, fork_safe_cells, opponent_can_fork_next, threat_gaps


def _root_entropy_info(root):
    """(entropy, n_candidates) of a search root's visit distribution, for the
    hard-example flat-distribution detector (Hard-6, see
    scratch/multiproc_investigation/entropy_complexity_probe.py -- 2026-09-11
    finding: ~19% of live buffer entries are visit distributions EXACTLY
    uniform over their candidate set, a PUCT round-robin degeneracy when
    prior+value are both flat, most common at 17-31 stones)."""
    n = len(root.children)
    if n == 0:
        return 0.0, 0
    tot = sum(ch.n_visits for ch in root.children.values())
    if tot == 0:
        return float(np.log(n)), n
    probs = np.array([ch.n_visits / tot for ch in root.children.values()])
    probs = np.clip(probs, 1e-12, None)
    return float(-(probs * np.log(probs)).sum()), n


def _mine_game(current_player, pick):
    """The game loop the three mining games share. pick(board, player) -> ((r, c), is_model_ply, ent_info), or
    None to end the game; every ply is recorded as (board before the move, player, (r, c), is_model_ply, ent_info).
    Returns (plies, outcome) with outcome +1 X won, -1 O won, 0 otherwise (draw, no move, or the 130-move cap)."""
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    plies = []
    moves = 0
    outcome = 0
    while True:
        if fast_check_win(board, 'X'):
            outcome = 1
            break
        if fast_check_win(board, 'O'):
            outcome = -1
            break
        if fast_check_draw(board):
            outcome = 0
            break
        board_snapshot = [row[:] for row in board]
        move = pick(board, current_player)
        if move is None:
            break
        (r, c), is_model_ply, ent_info = move
        plies.append((board_snapshot, current_player, (r, c), is_model_ply, ent_info))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


def _model_move(mcts, board, player, net, device, sims, is_model_ply=True, **search_kw):
    """The most-visited move of one search, as a _mine_game move (entropy info only for the model's own plies)."""
    root = mcts.search(board, player, net, device, num_simulations=sims, **search_kw)
    if not root.children:
        return None
    act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
    return divmod(act, BOARD_SIZE), is_model_ply, (_root_entropy_info(root) if is_model_ply else None)



def mine_one_selfplay_game(model, device, sims, pooled_evaluator=None, eval_cache=None):
    """Plays one self-play game (noise on, no training-data recording) to
    collect (board, current_player) snapshots for refresh_mined_tactical_pool
    to scan -- mirrors mine_tactical_parallel.py's _play_one_selfplay.

    2026-09-11: also returns per-ply (act, entropy_info) and the game outcome
    so refresh_mined_tactical_pool can additionally run outcome-based hard-
    example detection (Hard-1/2/3/5/6) on top of the original pattern scan --
    every ply here is a model ply (self-play is the model against itself).

    pooled_evaluator/eval_cache (2026-09-27, unified MP pool): optional
    routing for the leaf evaluations; None keeps the original behavior."""
    mcts = MCTS(c_puct=3.2, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    return _mine_game(random.choice(['X', 'O']), lambda board, player: _model_move(
        mcts, board, player, model, device, sims, add_noise=True, dirichlet_alpha=1.0, noise_weight=0.85))


def mine_one_vsdet_game(model, device, sims, mcts_plays_x, det_max_depth=deterministic_play.MAX_DEPTH,
                         pooled_evaluator=None, eval_cache=None):
    """vs-Deterministic counterpart to mine_one_selfplay_game -- mirrors
    mine_tactical_parallel.py's _play_one_vsdet.

    2026-09-11 bugfix: this used to call deterministic_play.play(board,
    current_player) with no depth argument, silently mining exclusively
    against the fixed depth-2 opponent -- the same fixed opponent
    evaluate_vs_deterministic/run_gating_check hold as an untouched
    promotion metric. Callers now pass sample_vs_det_max_depth(epoch) so
    mining draws from the same 0-3 depth ladder training data generation
    already uses, keeping the fixed-depth-2 metric uncontaminated.

    pooled_evaluator/eval_cache: see mine_one_selfplay_game."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)

    def pick(board, player):
        if player == mcts_player:
            return _model_move(mcts, board, player, model, device, sims, add_noise=True, dirichlet_alpha=0.3,
                               noise_weight=0.25)
        r, c = deterministic_play.play(board, player, max_depth=det_max_depth)
        return None if r == -1 else ((r, c), False, None)
    return _mine_game('X', pick)


def mine_one_vsleague_game(model, league_model, device, sims, mcts_plays_x,
                            pooled_evaluator=None, eval_cache=None,
                            league_pooled_evaluator=None, league_eval_cache=None):
    """vs-League counterpart to mine_one_vsdet_game -- the opponent is a
    frozen past checkpoint's own greedy (no-noise) search instead of
    deterministic_play, mirroring play_one_mcts_vs_league_game's
    opponent-side logic. Mining from vs-League games closes a gap the
    self-play/vs-Det mining above never reaches: whatever tactical patterns
    specifically arise when the model plays against its own past selves.

    pooled_evaluator/eval_cache route the model side, league_* the league
    side (see mine_one_selfplay_game); all None = original behavior."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    league_mcts = MCTS(c_puct=1.6, eval_batch_size=EVAL_BATCH, shared_cache=league_eval_cache,
                       pooled_evaluator=league_pooled_evaluator)

    def pick(board, player):
        if player == mcts_player:
            return _model_move(mcts, board, player, model, device, sims, add_noise=True, dirichlet_alpha=0.3,
                               noise_weight=0.25)
        return _model_move(league_mcts, board, player, league_model, device, sims, is_model_ply=False, add_noise=False)
    return _mine_game('X', pick)


# ============================================================
# Outcome-based hard-example mining
# ============================================================
# A second mining criterion, based on outcome and disagreement, next to the pattern scan (threat_gaps is that
# scan's "4-of-5 + gap" test). A position is flagged when:
#   Hard-1: a winning move was available (own threat_gaps) and not played.
#   Hard-2: the opponent had an immediate winning threat and it was not blocked.
#   Hard-3: a fork (fork_geometries) was available or threatened and not taken or denied.
#   Hard-5: it is within the mover's last HARD5_LOOKBACK plies of a game the mover lost.
#   Hard-6: the search's visit distribution is suspiciously flat for its candidate count (~19% of the buffer was
#           exactly uniform over its candidates -- PUCT round-robin when prior and value are both flat).
# Relabelling does NOT copy the opponent's move (a fixed depth-2 minimax is most likely wrong exactly where the
# model's search disagrees with it). Each flagged position is searched again with the model's own MCTS at a boosted
# sim count, and that visit distribution becomes the policy target -- more search, aimed where the normal budget
# is not enough.
HARD_EXAMPLE_POOL_PATH = f"model/hard_mined_examples{FILE_SUFFIX}.pkl"
RELABEL_CAP = 256
# 800 -> 1800: production's own sim budget auto-ramped past this
# (currently 1100 base) since RELABEL_SIMS was first set, so the "bigger dose
# of search" this relabeling is supposed to provide had quietly fallen BELOW
# production depth. A direct measurement (live_model_tactical_check.py)
# found BLOCK-class positions (must defend a threat) still regress relative
# to the raw prior at production's own 1100 sims (85%->80%), while H2 (missed
# block) is the single largest hard-mining category by volume (~20-35/cycle)
# -- meaning the relabeling pass meant to fix exactly this was itself running
# at a shallower depth (800) than the level already shown to be insufficient
# (1100). Raised to 1800 (matching the current boosted_sims ceiling) so
# relabeled targets are reliably deeper than whatever production is doing.
RELABEL_SIMS = 1800
HARD_EXAMPLE_CAP = 8000
HARD5_LOOKBACK = 6
FLAT_ENTROPY_MARGIN = 0.05  # within this of ln(n_candidates) counts as degenerate-flat


def classify_hard_examples(games, windows, forks):
    """games: list of (plies, outcome) from _mine_one_*_game -- each ply is
    (board, player, act_rc, is_model_ply, entropy_info_or_None). Scans only
    the mover's OWN plies (is_model_ply) and returns a deduped list of
    (board, player, hard_class, outcome_for_player, g_p, g_o) -- g_p/g_o are
    kept for the relabel pass's one-hot safety net (see below)."""
    seen = set()
    flagged = []
    for plies, outcome in games:
        model_ply_positions = [i for i, p in enumerate(plies) if p[3]]
        for rank, i in enumerate(model_ply_positions):
            board, player, act, _, ent_info = plies[i]
            other = 'O' if player == 'X' else 'X'
            outcome_for_player = outcome if player == 'X' else -outcome

            g_p = threat_gaps(board, player, windows)
            g_o = threat_gaps(board, other, windows)
            forked = fork_hits(board, player, forks) + fork_hits(board, other, forks)

            # Hard-7: general, template-free fork check -- does
            # THIS move (act) leave `other` with SOME reply that manufactures
            # an unstoppable fork (or outright win)? Must look one ply past
            # the move itself (opponent_can_fork_next), not just inspect
            # the board as-is -- see that function's docstring for why a
            # same-ply check (an earlier version of this) misses the real
            # case. Checked separately from g_p/g_o/forked above since none
            # of those catch irregular real-game fork shapes (see
            # _immediate_win_cells docstring).
            after = [row[:] for row in board]
            ar, ac = act
            after[ar][ac] = player
            allowed_fork = (not fast_check_win(after, player)
                             and opponent_can_fork_next(after, other))

            hard_class = None
            if g_p and act not in g_p:
                hard_class = 1
            elif g_o and not g_p and act not in g_o:
                hard_class = 2
            elif allowed_fork:
                hard_class = 7
            elif forked and not g_p and not g_o and act not in forked:
                hard_class = 3
            elif outcome_for_player == -1 and rank >= len(model_ply_positions) - HARD5_LOOKBACK:
                hard_class = 5
            elif ent_info is not None:
                ent, n_cand = ent_info
                if n_cand >= 8 and ent > float(np.log(n_cand)) - FLAT_ENTROPY_MARGIN:
                    hard_class = 6

            if hard_class is None:
                continue
            key = (''.join(''.join(row) for row in board), player)
            if key in seen:
                continue
            seen.add(key)
            flagged.append((board, player, hard_class, outcome_for_player, g_p, g_o))
    return flagged


def relabel_hard_examples(flagged, model, device, cap=RELABEL_CAP, sims=RELABEL_SIMS):
    """Re-searches each flagged position with the CURRENT model at a boosted
    sim count (no noise) and turns the resulting visit distribution into the
    policy target. Value target: Hard-1 gets a provable +1 (the flagged gap
    IS a winning completion for `player`, independent of how the actual
    recorded game later turned out); Hard-5 gets the real recorded outcome
    (that's its whole definition); Hard-2/3/6 stay policy-only (no value
    supervision), matching generate_tactical_examples' own block/fork
    convention -- none of those three license a value claim.
    Returns a list of (board, player, policy[NUM_CELLS] float32, value_or_None, hard_class)."""
    ordered = order_relabel_candidates(flagged, cap)
    was_training = model.training
    model.eval()
    relabeled = []
    try:
        for item in ordered:
            r = relabel_one(item, model, device, sims)
            if r is not None:
                relabeled.append(r)
    finally:
        model.train(was_training)
    return relabeled


def order_relabel_candidates(flagged, cap=RELABEL_CAP):
    """Class-priority order + per-class shuffle + cap -- the selection step of
    relabel_hard_examples, factored out (2026-09-27) so the unified MP pool
    can relabel the SAME selection in parallel."""
    by_class = {}
    for f in flagged:
        by_class.setdefault(f[2], []).append(f)
    ordered = []
    for cls in (1, 2, 7, 3, 5, 6):
        items = by_class.get(cls, [])
        random.shuffle(items)
        ordered.extend(items)
    ordered = ordered[:cap]
    return ordered


def relabel_one(item, model, device, sims=RELABEL_SIMS, pooled_evaluator=None, eval_cache=None):
    """Relabels ONE flagged item (see relabel_hard_examples); returns the
    (board, player, policy, value, hard_class) tuple, or None if the search
    found no children. Factored out 2026-09-27 so the unified MP pool can
    run relabels in parallel worker processes (pooled_evaluator/eval_cache
    route the leaf evaluations there); None/None = original behavior."""
    board, player, hard_class, outcome_for_player, g_p, g_o = item
    mcts = MCTS(c_puct=1.6, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    root = mcts.search(board, player, model, device, num_simulations=sims, add_noise=False)
    if not root.children:
        return None
    total_visits = sum(ch.n_visits for ch in root.children.values())
    policy = np.zeros(NUM_CELLS, dtype=np.float32)
    for act, ch in root.children.items():
        policy[act] = (ch.n_visits / total_visits) if total_visits > 0 else 0.0

    if hard_class in (1, 2):
        gaps = g_p if hard_class == 1 else g_o
        argmax_rc = divmod(int(np.argmax(policy)), BOARD_SIZE)
        if gaps and argmax_rc not in gaps:
            forced = random.choice(gaps)
            policy[:] = 0.0
            policy[forced[0] * BOARD_SIZE + forced[1]] = 1.0
    elif hard_class == 7:
        # Same forcing as Hard-1/2, but the "known
        # correct cells" have to be computed here (not stored at
        # mining time) via fork_safe_cells -- see that function's
        # docstring for why this was needed (the re-searched policy
        # target could otherwise inherit the same weak-prior
        # blind spot it's supposed to correct).
        other = 'O' if player == 'X' else 'X'
        safe_cells = fork_safe_cells(board, player, other)
        argmax_rc = divmod(int(np.argmax(policy)), BOARD_SIZE)
        if safe_cells and argmax_rc not in safe_cells:
            forced = random.choice(safe_cells)
            policy[:] = 0.0
            policy[forced[0] * BOARD_SIZE + forced[1]] = 1.0

    if hard_class == 1:
        # 0.95, not 1.0 (label smoothing): the value head was badly overconfident on fresh games (held-out MSE 1.36,
        # worse than predicting 0; 61% of predictions past |0.9|; its "very confident" bucket won only ~67%), and
        # confident artificial labels like this one are the likeliest amplifier.
        value = 0.95
    elif hard_class == 7:
        # Policy-only (value=None). A value of -0.95 was used before, but the stored board is the position BEFORE the
        # defender moves, where fork-safe replies exist; "near-certain loss" was the value after a bad reply. Measured on 14
        # Hard-7 positions: raw value -0.78 against an actual outcome of -0.04 -- the label taught the value head that any
        # fork threat loses.
        value = None
    elif hard_class == 5:
        value = float(outcome_for_player)
    else:
        value = None

    return ([row[:] for row in board], player, policy, value, hard_class)


def load_hard_example_pool():
    if os.path.exists(HARD_EXAMPLE_POOL_PATH):
        try:
            with open(HARD_EXAMPLE_POOL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Hard-example pool load notice ({e}). Starting with an empty pool.")
    return []


def save_hard_example_pool(pool, cap=HARD_EXAMPLE_CAP):
    if len(pool) > cap:
        pool = pool[-cap:]
    os.makedirs(os.path.dirname(HARD_EXAMPLE_POOL_PATH), exist_ok=True)
    with open(HARD_EXAMPLE_POOL_PATH, "wb") as f:
        pickle.dump(pool, f)
    return pool


def get_hard_example_samples(device, n_positions=46, sample_weight=1.5):
    """Per-epoch draw from the hard-mined pool for the regular replay buffer
    (see train_mcts_alpha_zero's call site, right after the league samples).
    n_positions=46 x 8-way augmentation ~= 368 samples/epoch, matching the
    spec's target self/league/hard ~= 0.70/0.15/0.15 buffer mix. value=None
    entries (Hard-2/3/6, policy-only) get 0.0 so they never receive value
    gradient signal (value_criterion is computed the same as every other
    source; 0.0 is a placeholder, not a claim -- see how tactical_refresh
    keeps a boolean mask instead if this ever needs stricter separation)."""
    pool = load_hard_example_pool()
    if not pool:
        return []
    chosen = random.sample(pool, min(n_positions, len(pool)))
    samples = []
    for board, player, policy, value, hard_class in chosen:
        board_3d = board_to_input_array(board, player)
        target_policy_map = policy.reshape(BOARD_SIZE, BOARD_SIZE)
        final_val = float('nan') if value is None else float(value)  # NaN = no value label, masked in training
        for aug_b, aug_p in augment_sample(board_3d, target_policy_map):
            samples.append((aug_b, aug_p, final_val, sample_weight))
    return samples


def get_fork_mistake_samples(n_positions=60, sample_weight=3.0):
    """Per-epoch draw from the Hard-7 (general fork) part of the hard-example pool, separate from
    get_hard_example_samples' draw over the whole pool. Hard-7 was a small share of the 8,000-position pool (261 when
    first measured), so the shared draw of 46 picked ~1-2 per epoch, and one epoch after mining the model still
    walked into 18 of 20 fresh Hard-7 positions. This draw gives Hard-7 a fixed volume however the rest of the pool
    grows, with a higher sample_weight than the shared draw.
    """
    pool = load_hard_example_pool()
    h7 = [item for item in pool if item[4] == 7]
    if not h7:
        return []
    chosen = random.sample(h7, min(n_positions, len(h7)))
    samples = []
    for board, player, policy, value, hard_class in chosen:
        board_3d = board_to_input_array(board, player)
        target_policy_map = policy.reshape(BOARD_SIZE, BOARD_SIZE)
        final_val = float('nan') if value is None else float(value)  # NaN = no value label, masked in training
        for aug_b, aug_p in augment_sample(board_3d, target_policy_map):
            samples.append((aug_b, aug_p, final_val, sample_weight))
    return samples
