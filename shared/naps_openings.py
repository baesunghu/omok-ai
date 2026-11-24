"""Where training games start: canonical (symmetry-reduced) opening squares and the tactical / forced-opening
start positions some self-play games begin from.
"""
import random

from mcts_cpp import BOARD_SIZE, fast_check_win
from fast_board import WIN_LENGTH


def _canonical_symmetry_openings(board_size):
    """One representative (row, col) per dihedral-symmetry equivalence class
    of first moves on an empty board_size x board_size board. Used to fan a
    fixed no-noise/greedy matchup out into genuinely different games instead
    of one deterministic line replayed under different game-index labels."""
    def symmetries(r, c, n):
        yield (r, c)
        yield (c, n - 1 - r)
        yield (n - 1 - r, n - 1 - c)
        yield (n - 1 - c, r)
        yield (r, n - 1 - c)
        yield (n - 1 - r, c)
        yield (c, r)
        yield (n - 1 - c, n - 1 - r)

    seen = set()
    reps = []
    for r in range(board_size):
        for c in range(board_size):
            if (r, c) in seen:
                continue
            orbit = set(symmetries(r, c, board_size))
            seen |= orbit
            reps.append((r, c))
    return reps


OPENING_SQUARES = _canonical_symmetry_openings(BOARD_SIZE)
# All board cells, not just one representative per symmetry class -- used by
# evaluate_vs_deterministic (see its docstring): with only len(OPENING_SQUARES)
# distinct openings, requesting a game count not coprime with that count made
# many "games" exact repeats of an earlier one under a different index label.
# Cycling through every cell instead gives eval_games far more room before
# that collision recurs.
ALL_CELLS = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
# Gen3 eval-diversification: row-major order meant every vs-Det
# check (sliced to the first num_games) only ever sampled the top-left
# edge/corner region of the board, every check, for the whole project's
# history. Shuffle once with a fixed seed so a slice is a representative
# spread across the whole board, while keeping openings reproducible
# run-to-run (same index -> same cell every time, preserving the
# pure-function-of-weights property callers/caching rely on).
random.Random(1109).shuffle(ALL_CELLS)


# ============================================================
# Tactical Curriculum (periodic refresh)
# ============================================================
# Diagnosis: pure self-play plateaued at 0% win rate
# against the deterministic opponent because it keeps missing obvious
# "3-filled-of-4 + 1 empty" line threats (open-3s and internal gaps alike).
# A supervised sanity check showed the network CAN learn this pattern
# instantly (94% val accuracy in 15 epochs on 448 synthetic examples) --
# capacity isn't the bottleneck, self-play's own search just rarely
# generates strong enough targets for it. A one-time pretrain on this
# curriculum gave a real but temporary boost (loss broke through its old
# plateau, entropy kept climbing) that faded over ~30 epochs as the much
# larger stream of ordinary self-play/vs-det samples diluted it back out.
# So instead of a single injection, refresh periodically throughout training.
TACTICAL_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def tactical_windows():
    windows = []
    for dr, dc in TACTICAL_DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = [(r + dr * k, c + dc * k) for k in range(WIN_LENGTH)]
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    windows.append(cells)
    return windows


def tactical_clutter(board, exclude_cells, n_stones=3, cluster_radius=3):
    """2026-09-14: clutter now clusters NEAR the tactical window (within
    cluster_radius of any exclude_cells cell) instead of scattering
    uniformly across the whole 11x11 board. Real games don't produce
    stones scattered board-wide around an isolated pattern -- play
    naturally clusters around the area of active contest, which is
    exactly the shape a real mined position has. Uniform-random clutter
    was a distribution the model could learn to recognize (and exploit)
    that never matches real games, plausibly a contributor to synthetic-
    trained candidates underperforming even the real-data label-conflict
    ceiling on held-out real positions (see
    scratch/multiproc_investigation/mine_tactical_conflict_check.py).
    Falls back to anywhere-empty if the local neighborhood is too crowded
    to fit n_stones (rare)."""
    near = set()
    for (r0, c0) in exclude_cells:
        for dr in range(-cluster_radius, cluster_radius + 1):
            for dc in range(-cluster_radius, cluster_radius + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == ' ' and (r, c) not in exclude_cells:
                    near.add((r, c))
    candidates = list(near)
    random.shuffle(candidates)
    if len(candidates) < n_stones:
        rest = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                if board[r][c] == ' ' and (r, c) not in exclude_cells and (r, c) not in near]
        random.shuffle(rest)
        candidates += rest
    chosen = candidates[:n_stones]
    for (r, c) in chosen:
        board[r][c] = random.choice(['X', 'O'])
    # Never keep a clutter placement that accidentally creates/completes a
    # 4-in-a-row elsewhere -- that would make the intended target wrong.
    if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
        for (r, c) in chosen:
            board[r][c] = ' '


def sample_tactical_start(max_attempts=10):
    """
    A reachable board with a random tactical pattern ("3 in a row + 1 gap", as in generate_tactical_examples) for
    self-play to start from now and then, so self-play has to resolve this family of positions with its own search
    more often than full games produce it. The clutter is exactly 2 stones of the other colour, so X and O have equal
    counts with X to move -- the parity of a real game. X is always the mover (as in _get_tactical_tensors), whether
    it must block O's line or complete its own.

    Only the starting position changes; the move played is still whatever this epoch's search decides.

    Returns (board, 'X'), or None if no placement is found within max_attempts (then start from an empty board).
    """
    windows = tactical_windows()
    for _ in range(max_attempts):
        cells = random.choice(windows)
        gap_idx = random.randrange(WIN_LENGTH)
        filled_cells = [cells[k] for k in range(WIN_LENGTH) if k != gap_idx]
        gap_r, gap_c = cells[gap_idx]
        stone = random.choice(['O', 'X'])
        clutter_stone = 'O' if stone == 'X' else 'X'

        board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for (r, c) in filled_cells:
            board[r][c] = stone

        empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                   if board[r][c] == ' ' and (r, c) != (gap_r, gap_c)]
        if len(empties) < 2:
            continue
        random.shuffle(empties)
        for (r, c) in empties[:2]:
            board[r][c] = clutter_stone

        if board[gap_r][gap_c] != ' ':
            continue
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        return board, 'X'
    return None


def sample_forced_opening_start():
    """
    Places a single stone at a stratified board-region sample (reusing
    OPENING_SQUARES -- the dihedral-symmetry-class representatives already
    used for eval/gate openings, so this picks up their spread across the
    whole board for free) as self-play's starting position, for a small
    forced fraction of games -- see generate_mcts_self_play_data's
    num_opening_games. Unlike sample_tactical_start (a mid-game tactical
    shape), this only forces the FIRST move, guaranteeing literal
    board-region coverage independent of whatever the network's own root
    prior plus Dirichlet noise would have picked on an empty board.

    Returns (board, next_player_to_move).
    """
    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    r, c = random.choice(OPENING_SQUARES)
    first_player = random.choice(['X', 'O'])
    board[r][c] = first_player
    next_player = 'O' if first_player == 'X' else 'X'
    return board, next_player
