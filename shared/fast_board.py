import numpy as np

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from run_config import BOARD_SIZE  # noqa: E402  (each run folder's run_config.py; 11 or 15)
WIN_LENGTH = 5
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
# every model / report / state file of a run carries this suffix (best_mcts_11x11.pth, training_report_11x11.txt)
FILE_SUFFIX = "" if BOARD_SIZE == 3 else f"_{BOARD_SIZE}x{BOARD_SIZE}"

if HAS_NUMBA:
    @jit(nopython=True)
    def check_win_numba(board_int, player_val, board_size=5, win_len=3):
        """
        Numba C-compiled fast win check.
        board_int: 2D int32 numpy array (1 for X, -1 for O, 0 for empty)
        """
        for r in range(board_size):
            for c in range(board_size):
                if board_int[r, c] != player_val:
                    continue
                # Horizontal
                if c <= board_size - win_len:
                    match = True
                    for k in range(1, win_len):
                        if board_int[r, c + k] != player_val:
                            match = False
                            break
                    if match:
                        return True
                # Vertical
                if r <= board_size - win_len:
                    match = True
                    for k in range(1, win_len):
                        if board_int[r + k, c] != player_val:
                            match = False
                            break
                    if match:
                        return True
                # Diagonal Down-Right
                if r <= board_size - win_len and c <= board_size - win_len:
                    match = True
                    for k in range(1, win_len):
                        if board_int[r + k, c + k] != player_val:
                            match = False
                            break
                    if match:
                        return True
                # Diagonal Down-Left
                if r <= board_size - win_len and c >= win_len - 1:
                    match = True
                    for k in range(1, win_len):
                        if board_int[r + k, c - k] != player_val:
                            match = False
                            break
                    if match:
                        return True
        return False

    @jit(nopython=True)
    def count_patterns_numba(board_int, player_val, opponent_val, board_size=5, win_len=3):
        score = 0
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for r in range(board_size):
            for c in range(board_size):
                if board_int[r, c] != player_val:
                    continue
                for d_idx in range(4):
                    dr, dc = directions[d_idx]
                    p_count = 1
                    blocked = False
                    for k in range(1, win_len):
                        nr, nc = r + dr * k, c + dc * k
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            blocked = True
                            break
                        cell = board_int[nr, nc]
                        if cell == player_val:
                            p_count += 1
                        elif cell == opponent_val:
                            blocked = True
                            break
                    if not blocked:
                        if p_count == 2:
                            score += 10
                        elif p_count >= 3:
                            score += 1000
        return score

def board_to_int_array(board):
    """Converts 5x5 board list of strings to int32 numpy array."""
    arr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 'X':
                arr[r, c] = 1
            elif board[r][c] == 'O':
                arr[r, c] = -1
    return arr

def fast_check_win(board, player):
    val = 1 if player == 'X' else -1
    if HAS_NUMBA:
        arr = board_to_int_array(board)
        return check_win_numba(arr, val, BOARD_SIZE, WIN_LENGTH)
    else:
        # Fallback fast Python
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != player:
                    continue
                if c <= BOARD_SIZE - WIN_LENGTH:
                    if all(board[r][c + k] == player for k in range(1, WIN_LENGTH)):
                        return True
                if r <= BOARD_SIZE - WIN_LENGTH:
                    if all(board[r + k][c] == player for k in range(1, WIN_LENGTH)):
                        return True
                if r <= BOARD_SIZE - WIN_LENGTH and c <= BOARD_SIZE - WIN_LENGTH:
                    if all(board[r + k][c + k] == player for k in range(1, WIN_LENGTH)):
                        return True
                if r <= BOARD_SIZE - WIN_LENGTH and c >= WIN_LENGTH - 1:
                    if all(board[r + k][c - k] == player for k in range(1, WIN_LENGTH)):
                        return True
        return False

def fast_check_draw(board):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == ' ':
                return False
    return True


# Search budget per move (here, not in naps_common, so torch-free worker processes such as mp_eval's can use it)
def sims_for_state(num_empty, base_sims, boosted_sims=None, boost_threshold=0, boost_when_narrow=False):
    """
    Simulations for one move. Extra search pays only where it changes the answer: on 5x5 a fixed budget had
    not converged on a wide-open board (top move's visit share 49% -> 79% from 100 to 1,600 sims at 25 empty cells)
    but had on narrower ones (flat ~38% at 14 empty). So boosted_sims applies while num_empty > boost_threshold.

    boost_when_narrow=True inverts that, boosting the late game instead: on 3x3 a critical defensive move stayed at
    ~1/100 visits in the mid/late game. Because that switch dropped the wide-open boost, the opening is boosted in
    any case (num_empty > NUM_CELLS - 10, before the 10th stone): at 23 empty cells a 100-sim search picked a move the
    search itself rejects with 600+ sims (24% -> 92% visit share), and that move led to a loss.
    """
    if not boosted_sims:
        return base_sims
    if boost_when_narrow:
        sims = boosted_sims if num_empty <= boost_threshold else base_sims
    else:
        sims = boosted_sims if num_empty > boost_threshold else base_sims
    if num_empty > NUM_CELLS - 10:
        sims = max(sims, boosted_sims)
    return sims
