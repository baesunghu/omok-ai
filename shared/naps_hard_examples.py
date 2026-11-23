"""Hard examples: positions mined from finished games where the net was wrong or unsure, relabelled with a
deeper search, kept in a pool and mixed into training (plus the tactical tensors built from them).
"""
import deterministic_play
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from mcts_cpp import BOARD_SIZE, MCTS, NUM_CELLS, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH
from fast_board import WIN_LENGTH
from board_encoding import augment_sample, board_to_input_array, board_to_input_tensor
from naps_common import _SUFFIX
from naps_openings import _tactical_windows
from naps_vsdet import _sample_vs_det_max_depth
from naps_league import _pick_league_opponent
from naps_tactics import (
    _fork_geometries, generate_closed_three_examples, generate_diagnosed_positions_examples, generate_double_four_examples, generate_fork_examples, generate_open_three_examples,
    generate_tactical_examples,
)


_synthetic_tactical_tensors_cache = None
_mined_tactical_tensors_cache = None
_open_three_injection_cache = None

# Real positions mined from actual self-play/vs-Deterministic games (150
# sims/move, live checkpoint), scanned for the same "4-of-5+gap" pattern
# family as generate_tactical_examples() and labeled with the identical
# mover_completes semantics -- see scratch/multiproc_investigation/
# mine_tactical_examples.py, mine_tactical_parallel.py, and
# verify_mined_examples.py for provenance and independent label
# verification. Copied here (rather than read from scratch/) so the live
# training code doesn't depend on a scratch directory.
MINED_TACTICAL_EXAMPLES_PATH = f"model/mined_tactical_examples{_SUFFIX}.pkl"


def _multi_label_targets_from_rows(rows_by_key):
    """2026-09-14 multi-label fix. rows_by_key: dict mapping a board(+player)
    key -> list of ((gap_r, gap_c), mover_completes) tuples all scanned from
    that EXACT SAME input position (it commonly matches more than one
    tactical window at once, e.g. two simultaneous open threats). The old
    scheme emitted one single-hot row per matching window, so the identical
    network input got DIFFERENT "correct" one-hot targets across rows --
    an argmax target that no single-output model can ever satisfy for all
    of them. Measured on the live mined pool (2026-09-14,
    scratch/multiproc_investigation/mine_tactical_conflict_check.py): 27.1%
    of unique boards carried >=2 conflicting single-cell labels, capping the
    theoretical MAX achievable argmax accuracy on the raw pool at 64.6% --
    this is very likely the dominant reason tactical_refresh's own reported
    accuracy has been flat at ~80% for weeks (98% synthetic + ~65% real,
    averaged in the deployed 50/50 mix, ~= 80%).

    Fix: one row per unique board(+player). If ANY matching window for that
    board is a provable mover_completes=True win, the target concentrates
    ONLY on those winning cells (playing a non-winning move when a win is
    available is never correct, so diluting the target with defensive
    cells would be actively wrong). Otherwise the target spreads uniformly
    over every distinct labeled (defensive) gap cell -- all of them are
    legitimately urgent, and this removes the artificial ceiling without
    claiming false certainty about which single one is "the" answer.
    Yields (key, target_vec[NUM_CELLS] summing to 1, has_winning_cell)."""
    for key, entries in rows_by_key.items():
        winning = {gap for gap, mc in entries if mc}
        chosen = winning if winning else {gap for gap, mc in entries}
        target = np.zeros(NUM_CELLS, dtype=np.float32)
        w = 1.0 / len(chosen)
        for (r, c) in chosen:
            target[r * BOARD_SIZE + c] = w
        yield key, target, bool(winning)


def _get_synthetic_tactical_tensors(device):
    global _synthetic_tactical_tensors_cache
    if _synthetic_tactical_tensors_cache is None:
        # Five curricula combined here: the original 4-of-5-gap family (only
        # one with a provable value label, see mover_completes below), the
        # closed-three/double-three-fork/double-four/open-three families
        # (policy-only by construction, see their own docstrings). All five
        # use 'X' as mover -- normalized to (board, player, (r,c),
        # mover_completes) 4-tuples here so this flows through the same
        # grouping/tensorize path as the mined examples below. The hard-mined
        # diagnosed positions (generate_diagnosed_positions_examples) carry
        # their own current_player per position, so they're added separately
        # after this normalization instead of through fixed_x_examples.
        fixed_x_examples = (generate_tactical_examples()
                             + generate_closed_three_examples()
                             + generate_fork_examples()
                             + generate_double_four_examples()
                             + generate_open_three_examples())
        examples = [(board, 'X', gap, is_mc) for (board, gap, is_mc) in fixed_x_examples]
        examples += generate_diagnosed_positions_examples()

        # Group by (board, player) first (see _multi_label_targets_from_rows)
        # -- a board can be produced by more than one generator above, or
        # matched by more than one window/gap_idx within a single generator
        # once clutter is involved.
        board_by_key, rows_by_key = {}, {}
        for board, player, (gr, gc), is_mc in examples:
            key = (tuple(tuple(row) for row in board), player)
            board_by_key.setdefault(key, (board, player))
            rows_by_key.setdefault(key, []).append(((gr, gc), is_mc))

        inputs, targets, value_targets = [], [], []
        for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
            board, player = board_by_key[key]
            t = board_to_input_tensor(board, player, device).squeeze(0).cpu().numpy()
            inputs.append(t)
            targets.append(target)
            value_targets.append(0.97 if has_win else float('nan'))

        _synthetic_tactical_tensors_cache = (
            torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
            torch.tensor(np.array(value_targets), dtype=torch.float32),
        )
        print(f"Tactical curriculum: {len(examples)} synthetic rows -> {len(inputs)} rows "
              f"(4-of-5-gap + closed-three + fork + double-four + open-three + diagnosed-positions, "
              f"multi-label, cached for reuse)")
    return _synthetic_tactical_tensors_cache


def _get_mined_tactical_tensors(device):
    """Loads and tensorizes the real mined examples (board, current_player,
    (gap_r, gap_c), mover_completes) tuples once and caches. Unlike the
    synthetic boards (always encoded from 'X' mover's perspective), each
    mined example carries its own real current_player -- passed through
    as-is to board_to_input_tensor, whose perspective-relative encoding
    (curr_plane = current_player's stones) needs no color canonicalization.

    2026-09-09: an 8-way dihedral-augmented version of this (same
    augment_sample every other data source gets) was tried and reverted --
    it reproducibly stalled the live process for 2+ hours at the third
    tactical_refresh cycle (ep63; the first two, ep21/ep42, ran fine), most
    likely GPU-memory fragmentation from repeatedly building/discarding an
    8x-larger (150K+) tensor cache every refresh compounding across cycles.
    Not re-attempted here; a safer version would need to build the
    per-example encoding via board_to_input_array (pure CPU numpy, no
    per-item .to(device) round-trip) instead of board_to_input_tensor, and
    ideally batch rather than rebuild-from-scratch every refresh.

    Returns empty tensors (falls back to synthetic-only) if the pickle is
    missing, so a fresh checkout without the mined-examples file still runs."""
    global _mined_tactical_tensors_cache
    if _mined_tactical_tensors_cache is None:
        if os.path.exists(MINED_TACTICAL_EXAMPLES_PATH):
            with open(MINED_TACTICAL_EXAMPLES_PATH, "rb") as f:
                data = pickle.load(f)

            # Group by (board, player) first -- see _multi_label_targets_from_rows.
            # Real positions match multiple tactical windows at once far more
            # often than synthetic ones (measured: 27.1% of unique mined
            # boards, 65.2% of all rows), so this matters even more here.
            board_by_key, rows_by_key = {}, {}
            for board, current_player, (gr, gc), is_mover_completion in data:
                key = (tuple(tuple(row) for row in board), current_player)
                board_by_key.setdefault(key, (board, current_player))
                rows_by_key.setdefault(key, []).append(((gr, gc), is_mover_completion))

            inputs, targets, value_targets = [], [], []
            for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
                board, current_player = board_by_key[key]
                t = board_to_input_tensor(board, current_player, device).squeeze(0).cpu().numpy()
                inputs.append(t)
                targets.append(target)
                value_targets.append(0.97 if has_win else float('nan'))

            _mined_tactical_tensors_cache = (
                torch.tensor(np.array(inputs), dtype=torch.float32),
                torch.tensor(np.array(targets), dtype=torch.float32),
                torch.tensor(np.array(value_targets), dtype=torch.float32),
            )
            print(f"Tactical curriculum: {len(data)} real mined rows -> {len(inputs)} unique (board,player) "
                  f"loaded from {MINED_TACTICAL_EXAMPLES_PATH} (multi-label, cached for reuse)")
        else:
            print(f"Tactical curriculum: no mined examples found at {MINED_TACTICAL_EXAMPLES_PATH}, "
                  f"falling back to synthetic-only")
            _mined_tactical_tensors_cache = (
                torch.empty((0, 5, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32),
                torch.empty((0, NUM_CELLS), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
    return _mined_tactical_tensors_cache


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


def _mine_one_selfplay_game(model, device, sims, pooled_evaluator=None, eval_cache=None):
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
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = random.choice(['X', 'O'])
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
        root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                            dirichlet_alpha=1.0, noise_weight=0.85)
        if not root.children:
            break
        act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        ent_info = _root_entropy_info(root)
        r, c = divmod(act, BOARD_SIZE)
        plies.append((board_snapshot, current_player, (r, c), True, ent_info))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


def _mine_one_vsdet_game(model, device, sims, mcts_plays_x, det_max_depth=deterministic_play.MAX_DEPTH,
                         pooled_evaluator=None, eval_cache=None):
    """vs-Deterministic counterpart to _mine_one_selfplay_game -- mirrors
    mine_tactical_parallel.py's _play_one_vsdet.

    2026-09-11 bugfix: this used to call deterministic_play.play(board,
    current_player) with no depth argument, silently mining exclusively
    against the fixed depth-2 opponent -- the same fixed opponent
    evaluate_vs_deterministic/run_gating_check hold as an untouched
    promotion metric. Callers now pass _sample_vs_det_max_depth(epoch) so
    mining draws from the same 0-3 depth ladder training data generation
    already uses, keeping the fixed-depth-2 metric uncontaminated.

    pooled_evaluator/eval_cache: see _mine_one_selfplay_game."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
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
        if current_player == mcts_player:
            root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                                dirichlet_alpha=0.3, noise_weight=0.25)
            if not root.children:
                break
            act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
            ent_info = _root_entropy_info(root)
            r, c = divmod(act, BOARD_SIZE)
            plies.append((board_snapshot, current_player, (r, c), True, ent_info))
        else:
            r, c = deterministic_play.play(board, current_player, max_depth=det_max_depth)
            if r == -1:
                break
            plies.append((board_snapshot, current_player, (r, c), False, None))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


def _mine_one_vsleague_game(model, league_model, device, sims, mcts_plays_x,
                            pooled_evaluator=None, eval_cache=None,
                            league_pooled_evaluator=None, league_eval_cache=None):
    """vs-League counterpart to _mine_one_vsdet_game -- the opponent is a
    frozen past checkpoint's own greedy (no-noise) search instead of
    deterministic_play, mirroring _play_one_mcts_vs_league_game's
    opponent-side logic. Mining from vs-League games closes a gap the
    self-play/vs-Det mining above never reaches: whatever tactical patterns
    specifically arise when the model plays against its own past selves.

    pooled_evaluator/eval_cache route the model side, league_* the league
    side (see _mine_one_selfplay_game); all None = original behavior."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    league_mcts = MCTS(c_puct=1.6, eval_batch_size=EVAL_BATCH, shared_cache=league_eval_cache,
                       pooled_evaluator=league_pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
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
        if current_player == mcts_player:
            root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                                dirichlet_alpha=0.3, noise_weight=0.25)
            is_model_ply = True
        else:
            root = league_mcts.search(board, current_player, league_model, device, num_simulations=sims,
                                       add_noise=False)
            is_model_ply = False
        if not root.children:
            break
        act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        ent_info = _root_entropy_info(root) if is_model_ply else None
        r, c = divmod(act, BOARD_SIZE)
        plies.append((board_snapshot, current_player, (r, c), is_model_ply, ent_info))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


# ============================================================
# Outcome-based hard-example mining
# ============================================================
# Adds a second, outcome/disagreement-based mining criterion on top of the
# pattern-only scan below (_threat_gaps mirrors that scan's own "4-of-5+gap"
# test exactly, factored out so both share one implementation). Approved
# design (see this session's diagnosis workflow, agent "diag:hard-mining-spec"):
#   Hard-1: a winning move was available (own threat_gaps) and not played.
#   Hard-2: opponent had an immediate winning threat and it was not blocked.
#   Hard-3: a fork (_fork_geometries) was available/threatened and not taken/denied.
#   Hard-5: position within the last HARD5_LOOKBACK of the mover's own plies
#           in a game that mover actually lost.
#   Hard-6 (an addition, not in the original spec): the search's own
#           visit distribution came out suspicion-of-degenerate flat for its
#           candidate count -- see entropy_complexity_probe.py's finding that
#           ~19% of the live buffer is EXACTLY uniform over its candidate set,
#           a PUCT round-robin degeneracy when prior+value are both flat.
# Relabeling deliberately does NOT imitate the opponent's move (that review
# rejected minimax-as-teacher: a shallow, fixed depth-2 opponent is most
# likely to be WRONG exactly where the model's own search disagrees with it
# most). Instead every flagged position is re-searched with the model's OWN
# MCTS at a boosted sim count and the resulting visit distribution becomes
# the policy target -- purely a bigger dose of the exact fix already applied
# this morning (more search), targeted at the positions where today's smaller
# production budget still isn't enough.
HARD_EXAMPLE_POOL_PATH = f"model/hard_mined_examples{_SUFFIX}.pkl"
RELABEL_CAP = 256
# 800 -> 1800: production's own sim budget auto-ramped past this
# (currently 1100 base) since RELABEL_SIMS was first set, so the "bigger dose
# of search" this relabeling is supposed to provide had quietly fallen BELOW
# production depth. Direct measurement today (live_model_tactical_check.py)
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


def _threat_gaps(board, color, windows):
    """Every (r, c) where `color` already has 4-of-5 filled and exactly one
    empty cell in some tactical window -- the same test
    refresh_mined_tactical_pool's own pattern scan uses, factored out so the
    hard-example detectors below (Hard-1/Hard-2) reuse it exactly rather than
    re-deriving it."""
    gaps = []
    for cells in windows:
        vals = [board[r][c] for (r, c) in cells]
        if sum(v == color for v in vals) == 4 and vals.count(' ') == 1:
            gaps.append(cells[vals.index(' ')])
    return gaps


def _fork_hits(board, color, forks):
    """Every fork center (see _fork_geometries) where `color` already holds
    all 4 stone cells with both open ends still empty and the center itself
    still empty -- i.e. a live, not-yet-taken/denied fork."""
    hits = []
    for center, stone_cells, open_cells in forks:
        cr, cc = center
        if board[cr][cc] != ' ':
            continue
        if (all(board[r][c] == color for (r, c) in stone_cells)
                and all(board[r][c] == ' ' for (r, c) in open_cells)):
            hits.append(center)
    return hits


def _opponent_can_fork_next(board, opponent):
    """True if `opponent` has SOME reply (placed right now) that either wins
    outright or creates 2+ simultaneous immediate-winning cells for
    `opponent` -- i.e. the position already contains a next-move fork
    `opponent` can walk into unopposed. This is the check that actually
    matters for Hard-7 (see its call site): a fork isn't visible by
    inspecting the board as-is (that only shows CURRENTLY-live winning
    cells, which is what _immediate_win_cells checks) -- it's visible one
    ply further, in whether the opponent HAS a move that manufactures a
    fork out of stones already on the board. O(BOARD_SIZE^2) candidate
    replies x O(BOARD_SIZE^2) fast_check_win each -- only run on already-
    mined game plies at refresh_mined_tactical_pool's cadence, not in the
    self-play hot loop."""
    # 15x15, now every run: same answer, window-based. The brute force was O(N^6) fast_check_win
    # calls per _fork_safe_cells and stalled the relabel step for 30+ minutes on 15x15.
    if _immediate_win_cells(board, opponent):
        return True
    # With no win cell on the board, a reply can only create one inside a window that already holds
    # 3 opponent stones and 2 empties, so only those empties need trying.
    cand = set()
    for cells_w in _WIN_WINDOWS:
        vals = [board[r][c] for r, c in cells_w]
        if vals.count(opponent) == WIN_LENGTH - 2 and vals.count(' ') == 2:
            cand.update(cell for cell, v in zip(cells_w, vals) if v == ' ')
    for r, c in cand:
        board[r][c] = opponent
        n = len(_immediate_win_cells(board, opponent))
        board[r][c] = ' '
        if n >= 2:
            return True
    return False


def _immediate_win_cells(board, color):
    """Every empty cell where placing `color` right now completes 5-in-a-row.
    Template-free (unlike _fork_hits/_fork_geometries, which only match the
    clean symmetric double-open-three shape) -- brute-forces every empty
    cell through fast_check_win, so it also catches irregular, real-game
    multi-threat shapes _fork_geometries structurally can't represent.
    len(...) >= 2 means an UNSTOPPABLE fork for whoever must move next (no
    single move blocks two simultaneous winning replies).

    2026-09-23: added after 4 straight HLT (human) losses this session, all
    ending in this exact kind of fork, NONE of which were caught by the
    existing (narrow, template-based) Hard-3 detector below -- see the
    session's find_fork_points.py/test_fork_prevention.py scratch analysis.
    O(BOARD_SIZE^2) fast_check_win calls per invocation; only run on
    already-mined game plies (a few hundred per refresh cycle), not inside
    the hot self-play loop itself.
    """
    # 15x15, now every run: window scan instead of one fast_check_win per empty cell. A cell completes
    # 5+ in a row iff some 5-cell window through it holds 4 `color` stones and that one empty cell.
    cells = set()
    for cells_w in _WIN_WINDOWS:
        vals = [board[r][c] for r, c in cells_w]
        if vals.count(color) == WIN_LENGTH - 1 and vals.count(' ') == 1:
            cells.add(cells_w[vals.index(' ')])
    return sorted(cells)


_WIN_WINDOWS = [[(r + k * dr, c + k * dc) for k in range(WIN_LENGTH)]
                for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                if 0 <= r + (WIN_LENGTH - 1) * dr < BOARD_SIZE and 0 <= c + (WIN_LENGTH - 1) * dc < BOARD_SIZE]


def _fork_safe_cells(board, player, other):
    """Every empty cell where `player` moving there does NOT hand `other` an
    unstoppable next-move fork (winning outright always counts as safe too)
    -- the Hard-7 analogue of _threat_gaps' g_p/g_o: a verified, brute-force
    ground truth for "which replies are actually correct here", not just
    "is a fork possible". Used by _relabel_hard_examples to force the
    Hard-7 policy target onto a real safe cell when the re-searched MCTS
    visit distribution doesn't already favor one -- mirrors how hard_class
    1/2 force their policy onto g_p/g_o. Without this, Hard-7's policy
    target was just whatever a RELABEL_SIMS-budget search happened to
    converge to, which an artificial verified-answer puzzle test
    (2026-09-24 session) showed can itself fail to concentrate visits on
    the best-Q move -- i.e. the relabel step could silently inherit and
    reinforce the same policy-prior blind spot it's meant to correct.
    O(BOARD_SIZE^2) candidate moves x _opponent_can_fork_next's own
    O(BOARD_SIZE^2)-worst-case scan; only run inside
    _relabel_hard_examples's already-capped (RELABEL_CAP) pass, not the
    self-play hot loop."""
    safe = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = player
            if fast_check_win(board, player):
                safe.append((r, c))
            elif not _opponent_can_fork_next(board, other):
                safe.append((r, c))
            board[r][c] = ' '
    return safe


def _classify_hard_examples(games, windows, forks):
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

            g_p = _threat_gaps(board, player, windows)
            g_o = _threat_gaps(board, other, windows)
            forked = _fork_hits(board, player, forks) + _fork_hits(board, other, forks)

            # Hard-7: general, template-free fork check -- does
            # THIS move (act) leave `other` with SOME reply that manufactures
            # an unstoppable fork (or outright win)? Must look one ply past
            # the move itself (_opponent_can_fork_next), not just inspect
            # the board as-is -- see that function's docstring for why a
            # same-ply check (an earlier version of this) misses the real
            # case. Checked separately from g_p/g_o/forked above since none
            # of those catch irregular real-game fork shapes (see
            # _immediate_win_cells docstring).
            after = [row[:] for row in board]
            ar, ac = act
            after[ar][ac] = player
            allowed_fork = (not fast_check_win(after, player)
                             and _opponent_can_fork_next(after, other))

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


def _relabel_hard_examples(flagged, model, device, cap=RELABEL_CAP, sims=RELABEL_SIMS):
    """Re-searches each flagged position with the CURRENT model at a boosted
    sim count (no noise) and turns the resulting visit distribution into the
    policy target. Value target: Hard-1 gets a provable +1 (the flagged gap
    IS a winning completion for `player`, independent of how the actual
    recorded game later turned out); Hard-5 gets the real recorded outcome
    (that's its whole definition); Hard-2/3/6 stay policy-only (no value
    supervision), matching generate_tactical_examples' own block/fork
    convention -- none of those three license a value claim.
    Returns a list of (board, player, policy[NUM_CELLS] float32, value_or_None, hard_class)."""
    ordered = _order_relabel_candidates(flagged, cap)
    was_training = model.training
    model.eval()
    relabeled = []
    try:
        for item in ordered:
            r = _relabel_one(item, model, device, sims)
            if r is not None:
                relabeled.append(r)
    finally:
        model.train(was_training)
    return relabeled


def _order_relabel_candidates(flagged, cap=RELABEL_CAP):
    """Class-priority order + per-class shuffle + cap -- the selection step of
    _relabel_hard_examples, factored out (2026-09-27) so the unified MP pool
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


def _relabel_one(item, model, device, sims=RELABEL_SIMS, pooled_evaluator=None, eval_cache=None):
    """Relabels ONE flagged item (see _relabel_hard_examples); returns the
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
        # mining time) via _fork_safe_cells -- see that function's
        # docstring for why this was needed (the re-searched policy
        # target could otherwise inherit the same weak-prior
        # blind spot it's supposed to correct).
        other = 'O' if player == 'X' else 'X'
        safe_cells = _fork_safe_cells(board, player, other)
        argmax_rc = divmod(int(np.argmax(policy)), BOARD_SIZE)
        if safe_cells and argmax_rc not in safe_cells:
            forced = random.choice(safe_cells)
            policy[:] = 0.0
            policy[forced[0] * BOARD_SIZE + forced[1]] = 1.0

    if hard_class == 1:
        # 1.0 -> 0.95 (label smoothing). value_overfit_check.py
        # found the live value head badly overconfident on FRESH,
        # never-trained-on self-play games (held-out MSE 1.36 --
        # worse than a constant-0 predictor; 61% of predictions
        # saturated past |0.9|; the model's own "very confident"
        # bucket only actually won ~67% of the time, not ~100%).
        # Confident artificial labels (this class and Hard-7) are
        # the most likely amplifier -- softening them off the
        # tanh's saturated tail should reduce that overconfidence
        # without giving up the corrective signal's direction.
        value = 0.95
    elif hard_class == 7:
        # Unlike Hard-1/5, this is a confident NEGATIVE
        # label, not a policy-only class -- letting `other` reach an
        # unstoppable fork is a near-certain loss for `player`
        # regardless of what happens afterward (session evidence:
        # test_fork_prevention.py found 0/3 real HLT forks
        # preventable even with a fresh 10000-sim search, so this
        # needs a direct value correction, not just more search at
        # relabel time). -1.0 -> -0.95: same label-smoothing
        # rationale as Hard-1 above.
        # REMOVED -- policy-only now (value=None). The
        # board stored here is the position BEFORE the defender
        # moves, where fork-safe replies exist, so "near-certain
        # loss" was the value of the position AFTER a bad reply, not
        # this one. pessimism_probe (14 Hard-7 positions, safe reply
        # then best-vs-best playouts) measured raw value -0.78 vs an
        # empirical outcome of -0.04: this label was teaching the
        # value head that any fork threat means the game is lost.
        value = None
    elif hard_class == 5:
        value = float(outcome_for_player)
    else:
        value = None

    return ([row[:] for row in board], player, policy, value, hard_class)


def _load_hard_example_pool():
    if os.path.exists(HARD_EXAMPLE_POOL_PATH):
        try:
            with open(HARD_EXAMPLE_POOL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Hard-example pool load notice ({e}). Starting with an empty pool.")
    return []


def _save_hard_example_pool(pool, cap=HARD_EXAMPLE_CAP):
    if len(pool) > cap:
        pool = pool[-cap:]
    os.makedirs(os.path.dirname(HARD_EXAMPLE_POOL_PATH), exist_ok=True)
    with open(HARD_EXAMPLE_POOL_PATH, "wb") as f:
        pickle.dump(pool, f)
    return pool


def _get_hard_example_samples(device, n_positions=46, sample_weight=1.5):
    """Per-epoch draw from the hard-mined pool for the regular replay buffer
    (see train_mcts_alpha_zero's call site, right after the league samples).
    n_positions=46 x 8-way augmentation ~= 368 samples/epoch, matching the
    spec's target self/league/hard ~= 0.70/0.15/0.15 buffer mix. value=None
    entries (Hard-2/3/6, policy-only) get 0.0 so they never receive value
    gradient signal (value_criterion is computed the same as every other
    source; 0.0 is a placeholder, not a claim -- see how tactical_refresh
    keeps a boolean mask instead if this ever needs stricter separation)."""
    pool = _load_hard_example_pool()
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


def _get_fork_mistake_samples(n_positions=60, sample_weight=3.0):
    """Per-epoch DEDICATED draw from the Hard-7 (general, template-free
    fork) slice of the hard-mined pool, separate from
    _get_hard_example_samples' whole-pool draw.

    2026-09-23: Hard-7 is currently a small fraction of the shared 8000-cap
    pool (261/8000 the first time it was measured), so
    _get_hard_example_samples' n_positions=46 draw only pulls ~1-2 Hard-7
    examples per epoch by chance -- measured directly
    (hard7_adaptation_test.py): the live model still walked into 18/20
    freshly-mined Hard-7 positions one epoch after mining, essentially no
    real training exposure yet. This dedicated draw guarantees Hard-7 gets
    real per-epoch volume regardless of how large the rest of the pool
    (mostly Hard-5, outcome-based) grows around it. sample_weight=2.0
    (higher than the shared draw's 1.5) since Hard-7 carries a confident
    value=-1.0 label (see _relabel_hard_examples), not a policy-only one.
    """
    pool = _load_hard_example_pool()
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


def _get_open_three_injection_samples(n_examples=120, sample_weight=1.0):
    """Per-epoch draw of open-three + open-3-3-fork (+ diagnosed-position)
    examples for the regular replay buffer, same pattern as
    _get_hard_example_samples right below this.

    2026-09-22: replaces tactical_refresh()'s periodic burst supervised
    fine-tune as this curriculum's delivery mechanism. Isolated before/after
    testing (scratch/.../tactical_refresh_destroy_test.py) found ONE
    tactical_refresh() call immediately dropped vs-Det win rate 65%->25%
    (a single 20-game/600-sim sample -- noisy, but the direction matches the
    loss-spike pattern already documented in tactical_refresh's own
    docstring at ep22/ep43/ep64). A separate retention check
    (scratch/.../tactical_retention_check.py) found the ~100% tactical
    accuracy right after a refresh had already decayed to ~70% just 2
    epochs later, before the next refresh even fired -- so each burst was
    paying a real, measured cost for a gain that mostly didn't survive to
    the next cycle anyway. Blending a smaller amount of this curriculum into
    EVERY epoch's ordinary training batch (like the hard-mined pool
    already does) avoids both problems: no isolated-batch overfit spike,
    and no burst to decay away between refreshes since there's no longer a
    gap between applications.

    n_examples=120 positions x up to 8-way dihedral augmentation ~= up to
    ~960 samples/epoch -- deliberately a larger share of the epoch's buffer
    than the hard-mined pool's ~368/epoch (n_positions=46), since this is
    now the sole delivery path for the open-three curriculum rather than a
    supplement to periodic bursts.
    """
    global _open_three_injection_cache
    if _open_three_injection_cache is None:
        # generate_fork_examples() (the open-3-3 / double-open-
        # three geometry -- see _fork_geometries) added to this SAME
        # continuous pool. It was already implemented and already used to
        # get delivered via tactical_refresh()'s burst fine-tune, but that
        # call was removed from the main loop this session (see the
        # tactical_refresh_interval guard's comment) and nothing replaced
        # it for this specific curriculum the way open-three's own got
        # migrated here -- so it silently went from
        # "burst-delivered" to "never delivered". A held-out T-shape probe
        # this session (test_t_shape_33fork_realistic.py) found this
        # measurably regressing over the ~200 epochs since: raw policy
        # rank for the correct denial cell 3rd/121 at the Gen12 reset
        # point -> 47th/121 at the current live checkpoint, MCTS solving
        # it correctly at epoch 0 and failing every checkpoint since --
        # i.e. real forgetting of knowledge the supervised-pretrain
        # checkpoint already had, not just slow acquisition. No new
        # examples are being authored here -- this is reconnecting an
        # existing generator to an existing delivery mechanism.
        examples = [(board, 'X', gap, is_mc) for (board, gap, is_mc) in generate_open_three_examples()]
        examples += [(board, 'X', gap, is_mc) for (board, gap, is_mc) in generate_fork_examples()]
        examples += generate_diagnosed_positions_examples()
        board_by_key, rows_by_key = {}, {}
        for board, player, (gr, gc), is_mc in examples:
            key = (tuple(tuple(row) for row in board), player)
            board_by_key.setdefault(key, (board, player))
            rows_by_key.setdefault(key, []).append(((gr, gc), is_mc))
        pool = []
        for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
            board, player = board_by_key[key]
            board_arr = board_to_input_array(board, player)
            target_policy_map = target.reshape(BOARD_SIZE, BOARD_SIZE)
            value = 0.97 if has_win else 0.0
            pool.append((board_arr, target_policy_map, value))
        _open_three_injection_cache = pool
        print(f"Open-three(+3-3-fork) injection pool built: {len(pool)} unique positions "
              f"(continuous per-epoch draw, replaces tactical_refresh burst)")
    pool = _open_three_injection_cache
    if not pool:
        return []
    chosen = random.sample(pool, min(n_examples, len(pool)))
    samples = []
    for board_arr, target_policy_map, value in chosen:
        for aug_b, aug_p in augment_sample(board_arr, target_policy_map):
            samples.append((aug_b, aug_p, value, sample_weight))
    return samples


def refresh_mined_tactical_pool(model, device, num_selfplay_games=16, num_vsdet_games=16,
                                 sims=150, max_workers=6, cap=20000, num_vsleague_games=16,
                                 epoch=0, league_pool_size=20, league_best_checkpoint_prob=0.28,
                                 runner=None):
    """
    Periodic top-up for the real-game-mined tactical pool (see
    _get_mined_tactical_tensors), called once per tactical_refresh cycle
    (right before tactical_refresh itself) so the pool keeps tracking the
    model's CURRENT play style instead of staying frozen at whatever epoch
    it was first mined at.

    Why this matters: the mined pool's board positions come from games the
    model itself plays, so a one-time-only pool reflects only that
    snapshot's blind spots (a self-referential limitation -- see project
    discussion). As self-play's own exploration (Dirichlet noise + MCTS's
    PUCT exploration bonus, which visits under-prior-weighted moves and
    feeds back into training when they turn out good) gradually corrects
    those blind spots over many epochs, re-mining periodically lets the
    tactical curriculum's data source ride along with that improvement
    instead of being stuck at its original snapshot forever.

    Plays num_selfplay_games + num_vsdet_games real games with the CURRENT
    model (threaded, mirroring generate_mcts_self_play_data's own
    ThreadPoolExecutor pattern -- torch releases the GIL during the forward
    pass so this is real concurrency), scans every ply of every game for
    the same "4-of-5+gap" pattern _tactical_windows() defines, appends any
    matches to the on-disk pool at MINED_TACTICAL_EXAMPLES_PATH (capped at
    `cap` total examples -- oldest examples dropped first once exceeded, so
    the pool doesn't grow unboundedly over a 20000-epoch run), and
    invalidates the in-memory tensor cache so the very next
    _get_tactical_tensors() call (from the tactical_refresh() call right
    after this one) picks up the fresh pool instead of stale cached tensors.

    Uses a modest sims count (150, not the live training sim-ramp budget)
    since realistic stone clustering matters more here than search
    strength -- matches the original one-time mining's own choice (see
    scratch/multiproc_investigation/mine_tactical_examples.py).

    2026-09-11: the same games also feed a SECOND, outcome-based mining pass
    (_classify_hard_examples / _relabel_hard_examples) writing to the
    separate HARD_EXAMPLE_POOL_PATH -- see the block of comments/functions
    just above this one for the full design. Kept in the same function
    (rather than a second refresh call) so the expensive game-generation
    itself is only paid once per tactical_refresh cycle.

    runner (2026-09-27): optional selfplay_mp_unified task runner. When
    given, the mining games (+ their per-game hard-example classification)
    and the relabel searches run in the persistent multi-process game pool
    instead of threads in this process; any runner failure falls back to
    the original threaded/sequential code below for that step.
    """
    global _mined_tactical_tensors_cache

    was_training = model.training
    model.eval()
    windows = _tactical_windows()
    forks = _fork_geometries()
    games = []  # (plies, outcome) per game, for the hard-example pass
    games_played = 0
    flagged = None
    # Opponent picks / depth draws happen here, once, for either path.
    mine_depths = [_sample_vs_det_max_depth(epoch) for _ in range(num_vsdet_games)]
    mine_league_model = None
    if num_vsleague_games > 0:
        # deep_history_prob=0.0: mining wants the model's CURRENT blind
        # spots (see docstring above), so this pick stays biased toward
        # recent/strong opponents only.
        _mine_pick = _pick_league_opponent(device, league_pool_size, league_best_checkpoint_prob,
                                           epoch=epoch, deep_history_prob=0.0)
        if _mine_pick is not None:
            mine_league_model = _mine_pick[0]
    if runner is not None:
        try:
            games, flagged = runner.mine(model, mine_league_model, device, sims,
                                         num_selfplay_games, mine_depths,
                                         num_vsleague_games if mine_league_model is not None else 0)
            games_played = len(games)
        except Exception as e:  # noqa: BLE001
            print(f"  [refresh] unified-pool mining failed ({e!r}) -- threaded fallback", flush=True)
            games, flagged, games_played = [], None, 0
            runner = None
    try:
        if runner is None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_mine_one_selfplay_game, model, device, sims)
                           for _ in range(num_selfplay_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_selfplay_games
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_mine_one_vsdet_game, model, device, sims, (g % 2 == 0), mine_depths[g])
                           for g in range(num_vsdet_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_vsdet_games
            if mine_league_model is not None:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_mine_one_vsleague_game, model, mine_league_model, device, sims, (g % 2 == 0))
                               for g in range(num_vsleague_games)]
                    for fut in futures:
                        games.append(fut.result())
                games_played += num_vsleague_games
    finally:
        model.train(was_training)

    all_positions = [(board, player) for plies, _ in games for (board, player, _, _, _) in plies]

    new_examples = []
    for board, current_player in all_positions:
        for cells in windows:
            vals = [board[r][c] for (r, c) in cells]
            for color in ('X', 'O'):
                if sum(v == color for v in vals) == 4 and vals.count(' ') == 1:
                    gap_cell = cells[vals.index(' ')]
                    new_examples.append(([row[:] for row in board], current_player,
                                          gap_cell, color == current_player))

    existing = []
    if os.path.exists(MINED_TACTICAL_EXAMPLES_PATH):
        with open(MINED_TACTICAL_EXAMPLES_PATH, "rb") as f:
            existing = pickle.load(f)

    combined = existing + new_examples
    if len(combined) > cap:
        combined = combined[-cap:]  # drop oldest first, keep the pool's most recent snapshot bounded

    os.makedirs(os.path.dirname(MINED_TACTICAL_EXAMPLES_PATH), exist_ok=True)
    with open(MINED_TACTICAL_EXAMPLES_PATH, "wb") as f:
        pickle.dump(combined, f)

    _mined_tactical_tensors_cache = None  # force _get_mined_tactical_tensors to reload from disk next call

    if flagged is None:
        flagged = _classify_hard_examples(games, windows, forks)
    relabeled = None
    if runner is not None:
        try:
            relabeled = runner.relabel(model, device, _order_relabel_candidates(flagged), RELABEL_SIMS)
        except Exception as e:  # noqa: BLE001
            print(f"  [refresh] unified-pool relabel failed ({e!r}) -- sequential fallback", flush=True)
            relabeled = None
    if relabeled is None:
        relabeled = _relabel_hard_examples(flagged, model, device)
    hard_pool = _load_hard_example_pool()
    hard_pool = _save_hard_example_pool(hard_pool + relabeled)
    class_counts = Counter(r[4] for r in relabeled)
    class_note = " ".join(f"H{c}={class_counts.get(c,0)}" for c in (1, 2, 7, 3, 5, 6))

    msg = (f"Tactical pool re-mined: +{len(new_examples)} new (from {len(all_positions)} positions "
           f"across {games_played} games) -> {len(combined)} total"
           f"{' (capped, oldest dropped)' if len(existing) + len(new_examples) > cap else ''}"
           f" | Hard examples: {len(flagged)} flagged -> {len(relabeled)} relabeled ({class_note}) "
           f"-> {len(hard_pool)} total")
    print(msg)
    return len(new_examples), len(combined)


def _get_tactical_tensors(device):
    """
    Builds this refresh's tactical training pool as an EQUAL-WEIGHTED mix:
    ALL currently-available real mined examples (naturally-clustered
    positions mined from actual played games) combined with a FRESH random
    subsample of the synthetic pool of equal size (drawn without
    replacement, independently on every call -- not cached/fixed). This is
    "candidate d" from the held-out generalization study (Phase 2:
    scratch/multiproc_investigation/tactical_mix_experiment.py; Phase 3
    independent re-verification: scratch/multiproc_investigation/
    phase3_independent_reverify.py) -- the only mix among 5 tested that
    materially improved held-out-REAL argmax accuracy (Phase 2: 0.5079 ->
    0.6947; independent re-verification with a different seed/split:
    0.3135 -> 0.6239) with no measurable cost to held-out-SYNTHETIC
    accuracy (both runs: within ~0.2pp of the synthetic-only baseline).
    Naive full-concatenation and real-upsampled-to-synthetic-volume mixes
    were both measurably WORSE than synthetic-only on held-out real
    accuracy in Phase 2 -- do not "simplify" this back to a plain
    concatenation.

    Re-drawing the synthetic subsample fresh every call (rather than fixing
    it once) means the full synthetic pool's window/direction/color
    coverage guarantee is still achieved in expectation across many refresh
    cycles (every tactical_refresh_interval epochs), while any single
    refresh's pool stays balanced 50/50 instead of being diluted the way a
    naive full-concatenation was measured to be.
    """
    syn_inputs, syn_targets, syn_mc = _get_synthetic_tactical_tensors(device)
    real_inputs, real_targets, real_mc = _get_mined_tactical_tensors(device)

    n_real = real_inputs.shape[0]
    if n_real == 0:
        return syn_inputs, syn_targets, syn_mc

    n_syn = syn_inputs.shape[0]
    n_down = min(n_real, n_syn)
    syn_idx = torch.from_numpy(np.random.choice(n_syn, size=n_down, replace=False))

    inputs = torch.cat([syn_inputs[syn_idx], real_inputs], dim=0)
    targets = torch.cat([syn_targets[syn_idx], real_targets], dim=0)
    mover_completes = torch.cat([syn_mc[syn_idx], real_mc], dim=0)
    return inputs, targets, mover_completes


def tactical_refresh(model, device, epochs=3, batch_size=32):
    """
    Runs a few epochs of supervised fine-tuning on the tactical curriculum.

    Uses its OWN local Adam (2026-09-10) rather than the main RL loop's
    optimizer. Sharing it meant 8 supervised epochs on a different objective
    wrote into the shared Adam moment estimates, and every refresh was
    followed by a 2-3 epoch loss spike as the main loop recovered (ep22:
    P 0.98->2.04, V 0.06->0.26; ep43 and ep64 the same shape) -- roughly 14%
    of all epochs spent undoing it. Meant to be called
    periodically (every N epochs) during train_mcts_alpha_zero so this
    signal doesn't get diluted back out by the much larger volume of
    ordinary self-play samples between refreshes.

    Policy loss is computed over the whole batch, as before. Value loss is
    added ONLY over the subset of the batch carrying a real value label --
    a per-row float value_target (NaN = no label, e.g. the pure-block-only
    case, which gets none: "the defender played correctly" alone doesn't
    prove a value by itself). 0.97 (not literal 1.0) stays off the tanh
    output's saturated-gradient tail.
    """
    inputs, targets, value_targets = _get_tactical_tensors(device)
    inputs, targets, value_targets = inputs.to(device), targets.to(device), value_targets.to(device)
    n = inputs.shape[0]

    # lr-fix restart: 5e-4 -> 5e-5. This was tuned
    # against the OLD main-loop lr=1e-3 (a 0.5x ratio); the main loop is now
    # 1e-4 (this restart's lr-fix), and tactical_refresh_interval dropped
    # 21->5 for denser gate-check signal, so this refresh now fires ~4.2x
    # more often. Left at 5e-4 it would repeatedly overpower the just-
    # lowered main LR with the same-ratio-violating nudge, ~4x more often --
    # likely why gate checks are still stuck at ~0% 25+ epochs into this
    # restart. Scaling down to preserve the original 0.5x-of-main-lr ratio.
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    model.train()
    last_acc = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_in, b_tgt, b_val = inputs[idx], targets[idx], value_targets[idx]
            b_mask = ~torch.isnan(b_val)
            optimizer.zero_grad()
            logits, value_pred = model(b_in)
            policy_loss = -(b_tgt * logits).sum(dim=1).mean()
            if b_mask.any():
                # value_pred is (batch,1); b_val[b_mask] is (n,) -- must match
                # shape or mse_loss silently broadcasts to (n,n) and computes
                # the wrong (inflated) loss instead of an elementwise one.
                value_loss = nn.functional.mse_loss(value_pred[b_mask], b_val[b_mask].unsqueeze(-1))
                loss = policy_loss + 1.0 * value_loss
            else:
                loss = policy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                pred = torch.exp(logits)
                # Multi-label-correct: the predicted cell counts as right if
                # it's ANY of the valid cells for this board (b_tgt > 0), not
                # only if it matches b_tgt's own argmax -- b_tgt can now carry
                # several equally-valid cells (see _multi_label_targets_from_rows),
                # and tie-breaking on argmax would wrongly mark the others wrong.
                pred_idx = pred.argmax(dim=1, keepdim=True)
                correct += (b_tgt.gather(1, pred_idx).squeeze(1) > 0).sum().item()
        last_acc = correct / n
    return last_acc
