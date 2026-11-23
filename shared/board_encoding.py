"""Board -> network input planes (5 planes: own, opponent, empty, own fork cells, opponent fork cells),
win/draw checks, and device / memory-format helpers. Every network in nets.py reads these planes.
(Until 2026-11 this lived in alphago_lee.py together with the V1 network, now nets.GomokuNetV1.)
"""
import torch
import numpy as np
import os

# ============================================================
# GAME SETTINGS
# ============================================================

from fast_board import BOARD_SIZE, WIN_LENGTH  # noqa: E402  one source for the board size: fast_board.py

NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_INPUT_CHANNELS = 5
INPUT_SIZE = NUM_CELLS * NUM_INPUT_CHANNELS

# ============================================================
# Fork-Detection Plane Helpers (analytical, no search)
# ============================================================
# _LINES enumerates every WIN_LENGTH-length contiguous window of board
# coordinates (horizontal, vertical, and both diagonal directions) -- for
# BOARD_SIZE=10, WIN_LENGTH=5 this is 192 windows total (60 horizontal +
# 60 vertical + 36 diagonal "\" + 36 diagonal "/").

# 2026-09-16: optional native C++/pybind11 acceleration for _fork_mask
# (cpp_mcts/fork_mask_cpp.cpp, same build pattern as mcts_core/
# deterministic_play_core -- see cpp_mcts/build_fork_mask_cpp.bat). Explored
# via a workflow that tried 4 independent optimization strategies (numba
# JIT, fully-vectorized numpy, this C++ port, a bitboard/SWAR approach) and
# benchmarked them against each other; this one and numba tied for fastest
# (~9-11x over the pure-numpy version below), C++ was picked to match this
# project's existing convention of a compiled engine + pure-Python fallback
# (mcts_core, deterministic_play_core) rather than adding a new numba
# runtime dependency to this hot path. Verified byte-for-byte identical to
# the pure-Python version below on 374 random/edge-case boards (including a
# duplicate-threat-pair dedup stress case) both by the workflow agent and
# independently re-verified in this session. Falls back to the pure-Python
# implementation if the compiled extension isn't present (e.g. a different
# machine/Python ABI) -- same try/except ImportError pattern as
# mcts_python.py's _mcts_core.
try:
    import fork_mask_cpp as _fork_mask_native
except ImportError:
    _fork_mask_native = None

def _generate_lines():
    lines = []
    max_start = BOARD_SIZE - WIN_LENGTH

    # Horizontal windows
    for r in range(BOARD_SIZE):
        for c in range(max_start + 1):
            lines.append([(r, c + k) for k in range(WIN_LENGTH)])

    # Vertical windows
    for c in range(BOARD_SIZE):
        for r in range(max_start + 1):
            lines.append([(r + k, c) for k in range(WIN_LENGTH)])

    # Diagonal windows (down-right, "\")
    for r in range(max_start + 1):
        for c in range(max_start + 1):
            lines.append([(r + k, c + k) for k in range(WIN_LENGTH)])

    # Diagonal windows (down-left, "/")
    for r in range(max_start + 1):
        for c in range(WIN_LENGTH - 1, BOARD_SIZE):
            lines.append([(r + k, c - k) for k in range(WIN_LENGTH)])

    return lines


_LINES = _generate_lines()
# Flattened board-index form of _LINES (num_lines, WIN_LENGTH), precomputed once,
# for a fully vectorized (no Python-level line loop) _immediate_win_mask -- the
# naive per-line Python loop is fine on a 5x5/28-line board but far too slow on
# a 10x10/192-line board once _fork_mask calls it once per empty candidate cell
# (up to ~100 times per call, called for every MCTS simulation).
_LINES_IDX = np.array([[r * BOARD_SIZE + c for (r, c) in line] for line in _LINES], dtype=np.int64)


def _immediate_win_mask(curr_plane, empty_plane):
    """Returns a (BOARD_SIZE, BOARD_SIZE) float32 array with 1.0 at every
    empty cell that would complete a WIN_LENGTH-in-a-row for the owner of
    curr_plane if a stone were placed there right now. Numerically identical
    to a per-line Python-loop implementation (verified against it on 300+
    random boards plus hand-built edge cases), just vectorized via a
    precomputed flat index array instead of iterating _LINES in Python."""
    curr_flat = curr_plane.ravel()
    empty_flat = empty_plane.ravel()
    line_curr = curr_flat[_LINES_IDX]
    line_empty = empty_flat[_LINES_IDX]
    curr_counts = line_curr.sum(axis=1)
    empty_counts = line_empty.sum(axis=1)
    hit = (curr_counts == WIN_LENGTH - 1) & (empty_counts == 1)
    mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    if hit.any():
        hit_lines = _LINES_IDX[hit]
        hit_empty = line_empty[hit]
        slot = hit_empty.argmax(axis=1)
        target = hit_lines[np.arange(len(slot)), slot]
        mask[target] = 1.0
    return mask.reshape(BOARD_SIZE, BOARD_SIZE)


def _fork_mask(curr_plane, empty_plane):
    """Dispatches to the native C++ extension (fork_mask_cpp, ~9-10x faster,
    see cpp_mcts/fork_mask_cpp.cpp) when available, else falls back to the
    pure-Python/numpy implementation below (_fork_mask_python) -- both are
    verified byte-for-byte identical. See _fork_mask_python's docstring for
    the full algorithm description."""
    if _fork_mask_native is not None:
        return _fork_mask_native.fork_mask(curr_plane, empty_plane)
    return _fork_mask_python(curr_plane, empty_plane)


def _fork_mask_python(curr_plane, empty_plane):
    """Returns a (BOARD_SIZE, BOARD_SIZE) float32 array with 1.0 at every
    empty cell where placing a curr_plane-owner stone there would create 2
    or more distinct empty cells that would each independently complete a
    WIN_LENGTH-in-a-row on the following move (a "double threat" / fork).

    O(num_lines) instead of the naive O(num_empty_cells * num_lines)
    (simulate-every-candidate-then-rescan-everything) approach -- the naive
    version was the dominant cost of MCTS search on the 10x10/11x11 boards
    (measured: fork detection alone was ~60% of total simulation time).
    Numerically verified identical to the naive simulate-per-candidate
    version on 600+ random boards plus hand-built edge cases (including a
    board with pre-existing separate threats before any hypothetical
    placement, which the naive version's ">=2 total lit cells" definition
    also counts).

    Key fact this relies on: if a line already has exactly one empty cell
    and candidate cell A (currently empty) belongs to that line, A MUST BE
    that one empty cell -- so any line through A that is already an
    immediate-win line has A itself as its target. That means placing at A
    only removes A from the pre-existing lit set (if A itself was already
    a winning move) and otherwise leaves every other pre-existing threat
    untouched; the only NEW threats placing at A can create come from
    lines that currently have exactly WIN_LENGTH-2 of curr's stones and
    exactly 2 empty cells (one of which is A) -- filling A then completes
    that line to WIN_LENGTH-1 with the OTHER empty cell as the new target.

    2026-09-16: the per-candidate combination used to be a Python loop over
    every empty cell (up to ~120 on a sparse board) that rebuilt a new set
    via `(baseline_lit - {a}) | threats.get(a, set())` on EVERY iteration --
    profiled as the single largest cost in the whole dataset-generation
    pipeline (~74s of a 158s run, ~47%; board_to_input_array as a whole was
    77%), because set construction/union has real per-call overhead even
    when the sets involved are tiny. Rewritten to be fully vectorized: the
    identity len((baseline_lit - {a}) | T) = baseline_count - [a in
    baseline_lit] + len(T) - len(T & baseline_lit) (valid because `a` is
    never a member of its own threat set T, by construction) is computed as
    plain numpy array arithmetic over all cells at once -- the only
    remaining Python loop is over `threats.items()`, bounded by ~2x the
    count of "two-empty-cell" lines, not by empty-cell count. Verified
    numerically identical to the original on 966+ boards (random densities
    0-120 stones + hand-built edge cases including multiple simultaneous
    baseline threats); measured ~1.25x faster in isolation, and this
    function's overall dominance in board encoding means that carries
    through to end-to-end dataset generation and self-play.
    """
    L = WIN_LENGTH
    n_cells = BOARD_SIZE * BOARD_SIZE
    curr_flat = curr_plane.ravel()
    empty_flat = empty_plane.ravel()
    line_curr = curr_flat[_LINES_IDX]
    line_empty = empty_flat[_LINES_IDX]
    curr_counts = line_curr.sum(axis=1)
    empty_counts = line_empty.sum(axis=1)

    baseline_arr = np.zeros(n_cells, dtype=np.int32)
    baseline_hit = (curr_counts == L - 1) & (empty_counts == 1)
    if baseline_hit.any():
        hit_lines = _LINES_IDX[baseline_hit]
        hit_empty = line_empty[baseline_hit]
        slot = hit_empty.argmax(axis=1)
        targets = hit_lines[np.arange(len(slot)), slot]
        baseline_arr[targets] = 1  # set-like dedup: plain assignment, not +=
    baseline_count = int(baseline_arr.sum())

    threat_count = np.zeros(n_cells, dtype=np.int32)
    overlap = np.zeros(n_cells, dtype=np.int32)
    two_empty_lines = np.where((curr_counts == L - 2) & (empty_counts == 2))[0]
    threats = {}
    for li in two_empty_lines:
        positions = np.where(line_empty[li] > 0)[0]
        a_idx = int(_LINES_IDX[li, positions[0]])
        b_idx = int(_LINES_IDX[li, positions[1]])
        threats.setdefault(a_idx, set()).add(b_idx)
        threats.setdefault(b_idx, set()).add(a_idx)
    for a, s in threats.items():
        threat_count[a] = len(s)
        if baseline_count:
            overlap[a] = sum(1 for b in s if baseline_arr[b])

    total = baseline_count - baseline_arr + threat_count - overlap
    mask = ((total >= 2) & (empty_flat > 0)).astype(np.float32)
    return mask.reshape(BOARD_SIZE, BOARD_SIZE)


# ============================================================
# Model Directory
# ============================================================

if not os.path.exists('model'):
    os.makedirs('model')


# ============================================================
# Optimized Vectorized Board Utilities
# ============================================================

def board_to_input_array(board_state, current_player):
    """Canonical board -> network-input-planes builder (CPU-only numpy,
    no device transfer). Builds the curr/opp/empty occupancy planes, then
    appends two analytically-computed fork-detection planes (fork_self,
    fork_opp), returning all NUM_INPUT_CHANNELS planes stacked in this
    fixed order: curr, opp, empty, fork_self, fork_opp.

    This is the single source of truth for the board -> input-tensor
    representation -- every other function that needs the planes (including
    board_to_input_tensor below) calls this one instead of re-deriving them.
    Lets a caller collect many boards into one combined RAM-side array
    before doing a single .to(device) call for the whole batch, instead of
    one small host->device copy per board -- see boards_to_input_tensor
    below."""
    arr = np.array(board_state)
    opponent_player = 'O' if current_player == 'X' else 'X'
    curr_plane = (arr == current_player).astype(np.float32)
    opp_plane = (arr == opponent_player).astype(np.float32)
    empty_plane = (arr == ' ').astype(np.float32)
    fork_self = _fork_mask(curr_plane, empty_plane)
    fork_opp = _fork_mask(opp_plane, empty_plane)
    return np.stack([curr_plane, opp_plane, empty_plane, fork_self, fork_opp])


def board_to_input_array_from_numeric(raw_board_int8_array, current_player):
    """Fast numeric counterpart to board_to_input_array, for the C++ MCTS
    engine's leaf-eval callbacks. mcts_core.pyd (built from the
    round-trip-elimination rewrite in cpp_mcts\\mcts_core_v2.cpp) hands the
    leaf board to Python as a (BOARD_SIZE, BOARD_SIZE) numpy int8 array
    (0=empty, 1=X, 2=O) instead of a nested list of "X"/"O"/" " Python
    strings, so this avoids the per-leaf Python string/list object churn
    that board_to_input_array's numpy string-equality path pays.

    Returns the same (NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE) float32
    stack as board_to_input_array -- planes in order
    [curr, opp, empty, fork_self, fork_opp] -- computed via direct integer
    comparisons instead of numpy string equality, reusing _fork_mask
    verbatim (not re-derived) for the fork planes so that logic stays in
    exactly one place."""
    arr = np.asarray(raw_board_int8_array)
    curr_val = 1 if current_player == 'X' else 2
    opp_val = 2 if current_player == 'X' else 1
    curr_plane = (arr == curr_val).astype(np.float32)
    opp_plane = (arr == opp_val).astype(np.float32)
    empty_plane = (arr == 0).astype(np.float32)
    fork_self = _fork_mask(curr_plane, empty_plane)
    fork_opp = _fork_mask(opp_plane, empty_plane)
    return np.stack([curr_plane, opp_plane, empty_plane, fork_self, fork_opp])


def preferred_memory_format(device):
    """channels_last packs num_channels contiguous floats per pixel, which
    vectorizes much better on CPU for this small board (benchmarked: ~20%
    faster forward pass paired with a channels_last model -- see
    board_to_input_tensor below). On CUDA it's the opposite on this GPU:
    GTX 1050 (Pascal, no Tensor Cores) measured channels_last 1.4x-3.4x
    SLOWER than plain contiguous_format across batch sizes 8-96 for this
    exact network shape (scratch/perf_fixes/bench_memory_format.py) --
    channels_last's usual CUDA advantage comes from Tensor Core NHWC
    kernels, which this GPU doesn't have. So contiguous_format on CUDA,
    channels_last everywhere else."""
    return torch.contiguous_format if device.type == "cuda" else torch.channels_last


def boards_to_input_tensor(arrays, device):
    """Combines a list of board_to_input_array() outputs into one batch
    tensor with a single host->device transfer (np.stack concatenates in
    host RAM first), instead of transferring each one individually and
    torch.cat-ing already-on-device tensors."""
    batch_np = np.stack(arrays)
    batch = torch.from_numpy(batch_np).float().to(device)
    return batch.contiguous(memory_format=preferred_memory_format(device))


def board_to_input_tensor(board_state, current_player, device):
    stacked = board_to_input_array(board_state, current_player)
    input_tensor = torch.from_numpy(stacked).float().to(device)
    return input_tensor.unsqueeze(0).contiguous(memory_format=preferred_memory_format(device))


def check_win(board, player):
    """Fast win check for BOARD_SIZE x BOARD_SIZE with WIN_LENGTH."""
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


def check_draw(board):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == ' ':
                return False
    return True
