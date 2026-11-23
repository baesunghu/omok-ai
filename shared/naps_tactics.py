"""Synthetic tactical training positions: closed threes, forks, double fours, open threes, diagnosed positions.
"""
import random

from mcts_cpp import BOARD_SIZE, fast_check_win
from fast_board import WIN_LENGTH
from naps_openings import _TACTICAL_DIRECTIONS, _tactical_clutter, _tactical_windows


def generate_tactical_examples():
    """
    Every length-WIN_LENGTH window on the board, in all 4 directions, with
    exactly 3 cells filled by one player and 1 empty -- covers open-3s and
    internal gaps uniformly, for both the deterministic AI's typical
    attacking shapes ('O' filled -> X must block) and the mirror offensive
    case ('X' filled -> X should complete its own win). Each base pattern
    also gets a couple of "clutter" variants with a few extra random stones
    elsewhere, so the model sees the shape embedded in messier positions
    instead of only clean isolated ones.
    """
    examples = []
    for cells in _tactical_windows():
        for gap_idx in range(WIN_LENGTH):
            filled_cells = [cells[k] for k in range(WIN_LENGTH) if k != gap_idx]
            gap_r, gap_c = cells[gap_idx]
            # clutter_variants was 2; measured held-out generalization on the
            # exact same abstract pattern was only ~17.5% argmax-correct (avg
            # rank 2.4/9) despite ~95%+ *train* accuracy in tactical_refresh --
            # the model was memorizing the few clutter variants, not learning
            # a clutter-robust "block/complete the gap" feature. Raised to 12,
            # then to 16 for even broader clutter coverage of the same generic
            # patterns going into the next full reset.
            for stone, clutter_variants in (('O', 16), ('X', 16)):
                for variant in range(clutter_variants):
                    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                    for (r, c) in filled_cells:
                        board[r][c] = stone
                    if variant > 0:
                        _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                    if board[gap_r][gap_c] != ' ':
                        continue  # clutter collided with the gap cell, skip
                    # The mover is always 'X' in these synthetic boards (see
                    # board_to_input_tensor(board, 'X', ...) below), so
                    # stone == 'X' is the "mover completes their own line
                    # here" case -- a provable +1-for-the-mover outcome.
                    # stone == 'O' is the "must block" case: correct play,
                    # but not itself a provable value label (blocking one
                    # threat says nothing about who ultimately wins), so it
                    # must NOT get a value target.
                    mover_completes = (stone == 'X')
                    examples.append((board, (gap_r, gap_c), mover_completes))
    return examples


# ------------------------------------------------------------
# Closed-three and double-three-fork curricula
# ------------------------------------------------------------
# Two more tactical blind spots, diagnosed separately (see
# scratch/multiproc_investigation/closed_three_diagnostic_newrun.py and
# double_three_fork_diagnostic.py) as weak even on the latest checkpoint,
# and NOT covered by the 4-of-5-gap family above (that family is always 4
# stones + 1 gap; both shapes here are one filled square short of it):
#   - closed-three: 3-in-a-row blocked on one end, open (with room to
#     extend) on the other. Offense = extend into the open end. Defense =
#     block the open end first.
#   - double-three-fork: a single point that completes an open-three in
#     two directions at once (unstoppable -- the defender can only block
#     one line). Offense = play the shared point to create the fork.
#     Defense = play it first to deny the opponent's fork.
# Unlike the 4-of-5-gap family, NEITHER pattern's correct move is ever a
# provable win on its own (extending/blocking a closed-three, or
# creating/denying a fork, doesn't complete a 5-in-a-row) -- so both
# offense and defense are policy-only here, mover_completes always False.
# This keeps the two new curricula symmetric by construction, unlike the
# existing family's offense/defense value-supervision asymmetry.
_FORK_DIRECTION_PAIRS = [(_TACTICAL_DIRECTIONS[i], _TACTICAL_DIRECTIONS[j])
                         for i in range(len(_TACTICAL_DIRECTIONS))
                         for j in range(i + 1, len(_TACTICAL_DIRECTIONS))]


def _closed_three_spans():
    """
    Every length-5 (blocker, 3-in-a-row, target) span on the board, in all 4
    directions -- same exhaustive per-direction/per-cell sweep as
    _tactical_windows(), just anchored one cell later along the line (cells
    run k=-1..3 relative to the loop cell instead of k=0..4), so index 0 is
    the blocker, indices 1-3 are the three-in-a-row, and index 4 is the
    target (the empty cell past the open end, with room to extend into).
    Translation-invariant along each line, so this covers the same total
    count of length-5 windows as _tactical_windows() despite the different
    anchor.
    """
    spans = []
    for dr, dc in _TACTICAL_DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = [(r + dr * k, c + dc * k) for k in range(-1, 4)]
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    spans.append(cells)
    return spans


def generate_closed_three_examples():
    """
    OFFENSE: the mover ('X', same "mover is always 'X'" convention as
    generate_tactical_examples) owns the three-stone line; 'O' holds the
    blocking stone. Correct move = the target cell -- extends the three
    into a four (still capped on one end by 'O', so NOT a 5-in-a-row).
    DEFENSE: 'O' owns the three-stone line and its blocked end; the mover
    ('X') owns the blocking stone (played earlier). Correct move = the SAME
    target cell -- denies the extension before 'O' can play it.

    Each span gets a clean variant plus a couple of clutter variants (reusing
    _tactical_clutter, same pattern as generate_tactical_examples).
    """
    examples = []
    for cells in _closed_three_spans():
        blocker_cell = cells[0]
        three_cells = cells[1:4]
        target_cell = cells[4]
        for offense in (True, False):
            three_color = 'X' if offense else 'O'
            blocker_color = 'O' if offense else 'X'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                board[blocker_cell[0]][blocker_cell[1]] = blocker_color
                for (r, c) in three_cells:
                    board[r][c] = three_color
                if variant > 0:
                    _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                tr, tc = target_cell
                if board[tr][tc] != ' ':
                    continue  # clutter collided with the target cell, skip
                examples.append((board, target_cell, False))
    return examples


def _fork_geometries():
    """
    Every (direction-pair, center) combo where a single move at the shared
    center cell would complete an open-three ("_XXX_") in BOTH directions at
    once -- the double-three fork. For each of the 2 directions, needs
    stones at +-1 step from center and empty cells at +-2 step (the open
    ends); the two directions' cells never overlap since they only meet at
    the center itself, which stays empty until the move is made. Exhaustive
    over all 6 direction pairs and every center cell whose full 8-cell
    footprint (4 per direction) stays in bounds -- same spirit as
    _tactical_windows()'s exhaustive sweep, over direction PAIRS instead of
    single directions.
    """
    combos = []
    for d1, d2 in _FORK_DIRECTION_PAIRS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone_cells, open_cells, ok = [], [], True
                for dr, dc in (d1, d2):
                    s1, s2 = (r - dr, c - dc), (r + dr, c + dc)
                    o1, o2 = (r - 2 * dr, c - 2 * dc), (r + 2 * dr, c + 2 * dc)
                    if not all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in (s1, s2, o1, o2)):
                        ok = False
                        break
                    stone_cells += [s1, s2]
                    open_cells += [o1, o2]
                if ok:
                    combos.append(((r, c), stone_cells, open_cells))
    return combos


def generate_fork_examples():
    """
    OFFENSE: the mover ('X') owns all 4 stones (2 per direction); playing
    the shared center cell creates the fork. DEFENSE: 'O' owns all 4 stones
    (would create the fork if 'O' got to move next); the mover ('X') must
    occupy the shared center FIRST to deny it -- same target cell either
    way.

    Each geometry gets a clean variant plus a couple of clutter variants
    (reusing _tactical_clutter, same pattern as generate_tactical_examples),
    excluding both the stone cells and the required-open end cells so
    clutter can't collide with either.
    """
    examples = []
    for center, stone_cells, open_cells in _fork_geometries():
        for offense in (True, False):
            stone_color = 'X' if offense else 'O'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in stone_cells:
                    board[r][c] = stone_color
                if variant > 0:
                    exclude = set(stone_cells) | set(open_cells) | {center}
                    _tactical_clutter(board, exclude, n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                cr, cc = center
                if board[cr][cc] != ' ' or any(board[r][c] != ' ' for (r, c) in open_cells):
                    continue  # clutter collided with the target or a required-open end, skip
                examples.append((board, center, False))
    return examples


def _double_four_geometries():
    """
    Every (direction-pair, center) combo where a single move at the shared
    center cell would complete a FOUR-in-a-row (one cell short of the
    winning 5) in BOTH directions at once -- a double-four. For each of the
    2 directions, needs 3 consecutive stones immediately adjacent to center
    (center+d, center+2d, center+3d) and an empty cell one step past them
    (center+4d) for the resulting four to still be completable next move.
    The two directions' cells never overlap since they only meet at the
    center itself, which stays empty until the move is made. Exhaustive
    over all 6 direction pairs and every center cell whose full footprint
    (4 per direction) stays in bounds -- same spirit as _fork_geometries().
    """
    combos = []
    for d1, d2 in _FORK_DIRECTION_PAIRS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone_cells, open_cells, ok = [], [], True
                for dr, dc in (d1, d2):
                    line_stones = [(r + dr * k, c + dc * k) for k in (1, 2, 3)]
                    line_open = (r + dr * 4, c + dc * 4)
                    if not all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                               for rr, cc in line_stones + [line_open]):
                        ok = False
                        break
                    stone_cells += line_stones
                    open_cells.append(line_open)
                if ok:
                    combos.append(((r, c), stone_cells, open_cells))
    return combos


def generate_double_four_examples():
    """
    OFFENSE: the mover ('X') owns all 6 stones (3 per direction); playing
    the shared center cell completes a four in both directions at once --
    each threatens a win next move via its own open end, and the defender
    can only block one, a strictly more decisive threat than the open-three
    fork (generate_fork_examples) even though, like that fork, it is one
    move short of an outright win itself. DEFENSE: 'O' owns all 6 stones
    (would create the double-four if 'O' got to move next); the mover ('X')
    must occupy the shared center FIRST to deny it -- same target cell
    either way. mover_completes is always False for the same reason
    generate_fork_examples' is: completing/denying this shape isn't itself
    a provable 5-in-a-row.

    Each geometry gets a clean variant plus a couple of clutter variants
    (reusing _tactical_clutter, same pattern as generate_fork_examples),
    excluding both the stone cells and the required-open end cells so
    clutter can't collide with either.
    """
    examples = []
    for center, stone_cells, open_cells in _double_four_geometries():
        for offense in (True, False):
            stone_color = 'X' if offense else 'O'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in stone_cells:
                    board[r][c] = stone_color
                if variant > 0:
                    exclude = set(stone_cells) | set(open_cells) | {center}
                    _tactical_clutter(board, exclude, n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                cr, cc = center
                if board[cr][cc] != ' ' or any(board[r][c] != ' ' for (r, c) in open_cells):
                    continue  # clutter collided with the target or a required-open end, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # geometry/clutter combo already completes a win elsewhere, skip
                examples.append((board, center, False))
    return examples


# ------------------------------------------------------------
# Open-three curriculum
# ------------------------------------------------------------
# 2026-09-15 (added after re-diagnosing the HLT-loss value/MCTS blind spot --
# see scratch/multiproc_investigation/rediag_value_blindspot.py, run against
# the live epoch443 checkpoint): a genuine OPEN three (3-in-a-row with BOTH
# immediately-flanking cells empty) is a shape this curriculum never actually
# covered. generate_tactical_examples' "4-of-5-gap" family is always one move
# from a proven outcome (mover_completes or a single forced block);
# generate_closed_three_examples' shape has only ONE open end. An open three
# has TWO equally-correct response cells (either flank creates/denies an open
# four), and re-testing 6 real/synthetic open-three positions against the
# live checkpoint found the model still gets roughly half of them wrong --
# MCTS converging on an unrelated move despite the raw policy prior usually
# still ranking at least one true response cell near the top -- the same
# "good-ish prior, MCTS/value diverges" pattern diagnosed earlier, just never
# targeted by a dedicated curriculum shape until now.
def generate_open_three_examples():
    """
    Reuses _closed_three_spans()'s exact 5-cell (k=-1..3) windows, but
    treats BOTH end cells (index 0 and index 4) as empty and equally
    correct, instead of one being an occupied blocker and the other the
    sole target -- so each variant yields TWO rows for the SAME board (one
    per valid end), which _multi_label_targets_from_rows groups back into a
    single 50/50 target over both cells, exactly like the two symmetric
    block cells in generate_tactical_examples' gap family.

    OFFENSE: mover ('X') owns the three-stone line; either end extends it
    into an open four. DEFENSE: 'O' owns the three-stone line; the mover
    ('X') must occupy either end to prevent the open four. Neither is a
    provable 5-in-a-row on its own, so mover_completes is always False --
    same policy-only convention as generate_closed_three_examples/
    generate_fork_examples/generate_double_four_examples.
    """
    examples = []
    for cells in _closed_three_spans():
        end1, three_cells, end2 = cells[0], cells[1:4], cells[4]
        # 2026-09-23: edge/corner oversampling (added earlier today, REMOVED
        # here) -- the organic self-play open-three test
        # (open3_organic_test.py) found the model missing threats located
        # centrally while repeatedly playing near row 0 instead, matching
        # exactly the kind of "edge habit" this oversampling could produce
        # if it overshot: skewing the network toward edge/corner moves as a
        # general prior rather than genuinely improving edge-tactic
        # recognition. Reverting to a flat, position-independent variant
        # count (matches the window-count distribution the board's own
        # geometry produces, not artificially reweighted) to isolate
        # whether the organic-test miss rate improves without it.
        n_variants = 14
        for offense in (True, False):
            three_color = 'X' if offense else 'O'
            # 2026-09-21: 3->8 variants (1 clean + more clutter diversity) --
            # open-three defense is the pattern the passive open3 tracker
            # has shown persistently flat/noisy (0.2-0.7 miss rate) despite
            # training, so this deliberately over-represents it relative to
            # the other four tactical families in the combined curriculum.
            # 2026-09-22: 8->14 -- tactical_refresh() (the periodic burst
            # supervised re-injection this curriculum used to feed) was
            # removed the same day: isolated testing
            # (scratch/.../tactical_refresh_destroy_test.py) measured ONE
            # refresh call immediately dropping vs-Det win rate 65%->25%,
            # and a separate retention check
            # (scratch/.../tactical_retention_check.py) found that gain
            # decayed from ~100% back to ~70% tactical accuracy within just
            # 2 epochs anyway -- net negative. Replaced by continuous
            # per-epoch injection into the ordinary replay buffer (see
            # _get_open_three_injection_samples), so this pool now needs
            # more inherent variant diversity since it's no longer refreshed
            # by a dedicated fine-tuning pass, just sampled from directly.
            for variant in range(n_variants):  # was: flat range(14); now edge/corner-weighted
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in three_cells:
                    board[r][c] = three_color
                if variant > 0:
                    _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))
                if board[end1[0]][end1[1]] != ' ' or board[end2[0]][end2[1]] != ' ':
                    continue  # clutter collided with one of the two open ends, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # clutter combo already completes a win elsewhere, skip
                examples.append((board, end1, False))
                examples.append((board, end2, False))
    return examples


# ------------------------------------------------------------
# Diagnosed-position hard-mining
# ------------------------------------------------------------
# 2026-09-15: Position A (epoch291 HLT loss) is a real open-three-defense
# instance whose raw POLICY prior (not just the value head/MCTS search) has
# drifted onto the historically-wrong move after repeated self-play/HLT
# reinforcement -- confirmed by direct re-check against the live checkpoint
# (rank 1/121, 12.6% prior on the wrong cell, vs the two correct blocks at
# 3-5% each). A diffuse generic curriculum (generate_open_three_examples
# above) teaches the general shape but has no special weight on this EXACT,
# already-entrenched association -- unlearning it needs direct, repeated
# counter-supervision on the specific pattern, not just broader coverage.
# 2026-09-16: Position B (live test game, Claude vs the epoch462-promoted
# checkpoint) -- a DIAGONAL open-three missed in a denser, real-game-like
# position (5 X + 4 O stones already on the board, unlike Position A's
# emptier board) despite the model correctly blocking a near-identical
# horizontal open-three earlier in the SAME game. Confirms the open-three
# fix from generate_open_three_examples works on the shape in isolation but
# does not yet fully generalize to busier boards -- clutter density was
# widened 3-36 (was 3-6, see _tactical_clutter calls throughout this file)
# specifically to close this gap generically, and this exact position is
# also hard-mined directly, same rationale as Position A.
_DIAGNOSED_POSITIONS = [
    # (moves, current_player, correct_response_cells)
    (
        [('X', 5, 5), ('O', 5, 4), ('X', 4, 5), ('O', 6, 5), ('X', 4, 3), ('O', 7, 6), ('X', 4, 4)],
        'O', [(4, 6), (4, 2)],
    ),
    (
        [('X', 5, 5), ('O', 4, 6), ('X', 5, 6), ('O', 2, 4), ('X', 5, 7), ('O', 5, 4),
         ('X', 7, 5), ('O', 6, 5), ('X', 6, 6)],
        'O', [(8, 4), (4, 8)],
    ),
    # 2026-09-16: Positions C/D/E found by the autonomous tactical_watch.py
    # loop's very first run -- randomly-generated open-three test boards
    # with dense (heavy-clutter) surrounding stones, all missed by the
    # live checkpoint (block_visit_share 2.0-6.5%) despite the widened
    # 3-36 clutter curriculum deployed the SAME epoch (expected: that
    # curriculum hasn't been trained on yet, only code-deployed -- these
    # are exactly the kind of busy-board misses it should start fixing
    # once the next tactical_refresh cycle actually trains on it).
    (
        [('X', 1, 0), ('X', 1, 2), ('X', 1, 3), ('O', 2, 4), ('X', 3, 6), ('X', 4, 0), ('X', 4, 5), ('O', 4, 6), ('O', 5, 2), ('O', 5, 4), ('X', 5, 5), ('X', 5, 6), ('O', 5, 8), ('O', 6, 3), ('X', 7, 0), ('X', 7, 1), ('X', 7, 2), ('O', 7, 4), ('X', 7, 6), ('X', 7, 8), ('O', 8, 3), ('X', 8, 7), ('X', 8, 8), ('X', 9, 0), ('X', 9, 3), ('X', 9, 6), ('X', 10, 2), ('O', 10, 3), ('O', 10, 4), ('X', 10, 8)],
        'X', [(4, 1), (8, 5)],
    ),
    (
        [('X', 0, 2), ('O', 0, 5), ('O', 1, 2), ('O', 1, 4), ('X', 1, 5), ('O', 1, 6), ('X', 1, 7), ('X', 1, 9), ('O', 2, 3), ('O', 2, 6), ('O', 2, 7), ('O', 2, 8), ('O', 3, 2), ('O', 3, 4), ('X', 3, 6), ('O', 3, 7), ('O', 3, 9), ('O', 4, 2), ('O', 4, 3), ('O', 4, 5), ('O', 4, 9), ('X', 5, 4), ('O', 5, 8), ('O', 5, 9)],
        'X', [(2, 5), (2, 9)],
    ),
    (
        [('O', 2, 2), ('X', 2, 3), ('X', 3, 1), ('X', 3, 3), ('X', 4, 4), ('O', 4, 6), ('X', 5, 0), ('X', 5, 1), ('O', 5, 5), ('O', 5, 6), ('O', 5, 7), ('X', 6, 0), ('O', 6, 3), ('O', 6, 4), ('X', 6, 5), ('X', 7, 3), ('O', 7, 4), ('X', 7, 9), ('O', 8, 2), ('O', 8, 5), ('O', 8, 9), ('O', 9, 0), ('O', 9, 1), ('O', 9, 4), ('O', 10, 1), ('O', 10, 4), ('X', 10, 5)],
        'X', [(5, 2), (9, 6)],
    ),
    # 2026-09-16: 2nd tactical_watch.py manual run (~7h after the 1st,
    # loop wakeup chain appears to have stopped firing autonomously --
    # see chat) -- 3 more dense-board misses, block_visit_share
    # 0.0-1.2%. 2/5 tested passed this time (vs 0/3 the first run),
    # consistent with gradual, partial improvement rather than solved.
    (
        [('X', 0, 0), ('X', 0, 1), ('O', 0, 2), ('X', 0, 3), ('X', 1, 4), ('O', 2, 2), ('X', 2, 3), ('X', 2, 5), ('O', 3, 3), ('O', 4, 3), ('O', 4, 4), ('O', 4, 5), ('X', 4, 7), ('O', 5, 1), ('O', 5, 2), ('O', 5, 4), ('O', 5, 6), ('X', 5, 7), ('O', 6, 5), ('O', 6, 7), ('X', 6, 9), ('O', 7, 0), ('X', 7, 3), ('X', 7, 4), ('O', 8, 1), ('O', 8, 3), ('X', 8, 6), ('X', 8, 7), ('X', 8, 9), ('X', 9, 5), ('X', 10, 6), ('X', 10, 7), ('O', 10, 9)],
        'X', [(3, 2), (7, 6)],
    ),
    (
        [('O', 0, 1), ('X', 0, 5), ('X', 0, 6), ('X', 0, 7), ('O', 1, 2), ('O', 1, 4), ('O', 1, 7), ('X', 2, 1), ('X', 2, 2), ('O', 2, 3), ('O', 2, 4), ('O', 2, 6), ('O', 3, 1), ('X', 3, 3), ('O', 3, 4), ('O', 3, 5), ('X', 4, 2), ('O', 4, 3), ('X', 4, 5), ('X', 4, 6), ('O', 5, 1), ('X', 5, 2), ('O', 5, 3), ('X', 5, 4), ('O', 5, 5), ('O', 5, 7), ('O', 6, 1), ('X', 6, 2), ('O', 6, 3), ('X', 6, 6), ('O', 6, 7), ('O', 7, 1), ('O', 7, 2), ('X', 7, 3), ('O', 7, 6)],
        'X', [(0, 4), (4, 4)],
    ),
    (
        [('X', 0, 6), ('X', 0, 10), ('X', 1, 8), ('O', 1, 10), ('X', 3, 4), ('O', 3, 6), ('O', 3, 7), ('X', 3, 8), ('X', 4, 3), ('X', 4, 4), ('O', 4, 6), ('O', 4, 9), ('X', 5, 2), ('O', 5, 5), ('O', 6, 1), ('X', 6, 3), ('X', 6, 6), ('X', 7, 2), ('X', 7, 4), ('O', 7, 5), ('O', 7, 6), ('O', 7, 8), ('O', 8, 2), ('O', 8, 3), ('X', 8, 5), ('O', 8, 6), ('O', 8, 8), ('X', 9, 1), ('O', 9, 3), ('O', 9, 6)],
        'X', [(2, 8), (6, 4)],
    ),
    # 2026-09-23 Position F: HLT (human) loss for Gen13/NAPS, epoch129,
    # cumulative_epoch 129 checkpoint. Root value swung to +0.977 (AI very
    # confident of winning) at move 24, then collapsed to -0.921 by move 36
    # as the human found the refutation -- see the session's HLT-analysis
    # discussion. Root cause traced by replaying row 8's stone history:
    # after X's move at (8,3) (ply 35), row 8 reads "...X.XXXOX." -- X at
    # cols 3,5,6,7 with a SINGLE empty gap at col 4 and O blocking col 8.
    # This is a plain 4-of-5-gap shape (the OLDEST tactical curriculum
    # family in this file, generate_tactical_examples) with exactly one
    # winning cell (8,4) -- yet the live model's actual move-36 search
    # never surfaced (8,4) in its top-5 candidates at any simulation
    # checkpoint (only (3,7)/(8,4-adjacent) cells the network's prior
    # never weighted), meaning this wasn't a search-depth miss but a
    # policy-prior blind spot on a pattern the model has trained on
    # extensively. Distinct from Positions A-E (all open-three shapes) --
    # this is a gap-four shape missed on a busy, asymmetric real board.
    (
        [('X', 5, 5), ('O', 4, 6), ('X', 5, 4), ('O', 5, 6), ('X', 4, 4), ('O', 6, 6), ('X', 3, 6), ('O', 6, 5),
         ('X', 6, 3), ('O', 4, 5), ('X', 3, 4), ('O', 6, 4), ('X', 3, 5), ('O', 6, 7), ('X', 6, 8), ('O', 7, 8),
         ('X', 8, 9), ('O', 7, 6), ('X', 8, 6), ('O', 3, 3), ('X', 7, 7), ('O', 5, 9), ('X', 8, 5), ('O', 4, 7),
         ('X', 7, 4), ('O', 3, 8), ('X', 2, 9), ('O', 4, 8), ('X', 4, 9), ('O', 5, 2), ('X', 5, 3), ('O', 2, 6),
         ('X', 8, 7), ('O', 8, 8), ('X', 8, 3)],
        'O', [(8, 4)],
    ),
]


def _dihedral_transform(r, c, k, size):
    """Applies the k-th (0-7) of the 8 square-board symmetries (4 rotations,
    optionally preceded by a horizontal reflection) to cell (r, c). k=0 is
    the identity. Verified self-consistent: applying k=1..3 four times in a
    row returns the original cell, and k=4..7 apply the same rotations after
    an initial column-flip -- together the standard dihedral group of order 8."""
    if k >= 4:
        c = size - 1 - c
        k -= 4
    for _ in range(k):
        r, c = c, size - 1 - r
    return r, c


def generate_diagnosed_positions_examples():
    """
    Hard-mines the exact real diagnosed position(s) in _DIAGNOSED_POSITIONS,
    each replicated across all 8 dihedral symmetries and a batch of clutter
    variants (reusing _tactical_clutter), so this specific pattern gets
    meaningfully more representation than a single clean board would --
    comparable in order of magnitude to how other targeted families in this
    file are weighted, while staying a small fraction of the full curriculum.
    Policy-only (mover_completes always False), same convention as
    generate_open_three_examples, whose shape this position is an instance
    of.

    Returns (board, current_player, (r, c), mover_completes) 4-tuples --
    current_player varies per position, not always 'X' (same shape
    generate_ignored_threat_examples used to return).
    """
    DIAGNOSED_CLUTTER_VARIANTS = 20
    examples = []
    for moves, current_player, correct_cells in _DIAGNOSED_POSITIONS:
        for k in range(8):
            base_board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            placed_cells = set()
            for p, r, c in moves:
                tr, tc = _dihedral_transform(r, c, k, BOARD_SIZE)
                base_board[tr][tc] = p
                placed_cells.add((tr, tc))
            correct_transformed = [_dihedral_transform(r, c, k, BOARD_SIZE) for (r, c) in correct_cells]
            for variant in range(DIAGNOSED_CLUTTER_VARIANTS):
                board = [row[:] for row in base_board]
                if variant > 0:
                    _tactical_clutter(board, placed_cells | set(correct_transformed), n_stones=random.randint(3, 36))  # 2026-09-16: widened 2-5 -> 3-36 for consistency with the other families (see generate_open_three_examples' header comment)
                if any(board[r][c] != ' ' for (r, c) in correct_transformed):
                    continue  # clutter collided with a correct response cell, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # clutter combo already completes a win elsewhere, skip
                for (r, c) in correct_transformed:
                    examples.append((board, current_player, (r, c), False))
    return examples
