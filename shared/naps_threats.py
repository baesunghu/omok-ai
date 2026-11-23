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
    """Every empty cell where placing `color` now completes five in a row, found by trying every empty cell with
    fast_check_win -- unlike fork_hits / fork_geometries, which only match the clean double-open-three template, it
    also sees the irregular multi-threat shapes of real games. Two or more cells mean an unstoppable fork for the
    side to move next. Added after four straight losses to human players, all ending in such forks and none caught
    by the template-based Hard-3. O(BOARD_SIZE^2) fast_check_win calls; used on mined plies only, not in self-play.
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
    """Every empty cell where `player` moving there does NOT give `other` an unstoppable fork on the next move
    (winning outright counts as safe) -- a brute-force answer to "which replies are correct here". relabel uses it to
    move the Hard-7 policy target onto a safe cell when the re-search does not already favour one: a puzzle test
    showed a RELABEL_SIMS search can itself fail to concentrate on the best move, so the relabel could reinforce the
    very blind spot it is meant to fix. O(BOARD_SIZE^2) x opponent_can_fork_next per call; used only in the capped
    relabel pass, not in self-play."""
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
