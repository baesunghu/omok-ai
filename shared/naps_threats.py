"""Threat checks on a board position: four-in-a-row gaps, fork points, immediate wins, and whether the opponent
could make a fork next. Used to classify mined positions and to pick fork-safe moves.
"""

from mcts_cpp import BOARD_SIZE, fast_check_win
from fast_board import WIN_LENGTH


def threat_gaps(board, color, windows):
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


def fork_hits(board, color, forks):
    """Every fork center (see fork_geometries) where `color` already holds
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


def opponent_can_fork_next(board, opponent):
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
    # calls per fork_safe_cells and stalled the relabel step for 30+ minutes on 15x15.
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
    Template-free (unlike fork_hits/fork_geometries, which only match the
    clean symmetric double-open-three shape) -- brute-forces every empty
    cell through fast_check_win, so it also catches irregular, real-game
    multi-threat shapes fork_geometries structurally can't represent.
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


def fork_safe_cells(board, player, other):
    """Every empty cell where `player` moving there does NOT hand `other` an
    unstoppable next-move fork (winning outright always counts as safe too)
    -- the Hard-7 analogue of threat_gaps' g_p/g_o: a verified, brute-force
    ground truth for "which replies are actually correct here", not just
    "is a fork possible". Used by relabel_hard_examples to force the
    Hard-7 policy target onto a real safe cell when the re-searched MCTS
    visit distribution doesn't already favor one -- mirrors how hard_class
    1/2 force their policy onto g_p/g_o. Without this, Hard-7's policy
    target was just whatever a RELABEL_SIMS-budget search happened to
    converge to, which an artificial verified-answer puzzle test
    (2026-09-24 session) showed can itself fail to concentrate visits on
    the best-Q move -- i.e. the relabel step could silently inherit and
    reinforce the same policy-prior blind spot it's meant to correct.
    O(BOARD_SIZE^2) candidate moves x opponent_can_fork_next's own
    O(BOARD_SIZE^2)-worst-case scan; only run inside
    relabel_hard_examples's already-capped (RELABEL_CAP) pass, not the
    self-play hot loop."""
    safe = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = player
            if fast_check_win(board, player):
                safe.append((r, c))
            elif not opponent_can_fork_next(board, other):
                safe.append((r, c))
            board[r][c] = ' '
    return safe
