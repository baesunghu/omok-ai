import torch
import torch.nn as nn
import numpy as np
import os

# ============================================================
# GAME SETTINGS
# ============================================================

BOARD_SIZE = 15  # 2026-10-13: 15x15 run (must match fast_board.BOARD_SIZE)
WIN_LENGTH = 5

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
# alphazero_mcts.py's _mcts_core.
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
# ResNet Block & Dual-Head Policy-Value Network
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class AlphaZeroTicTacToe(nn.Module):
    """
    High-performance 2D Convolutional Policy-Value Network for 10x10 Board.
    Supports both 4D tensors (B, NUM_INPUT_CHANNELS, 10, 10) and 2D tensors
    (B, NUM_INPUT_CHANNELS * 100) -- 5 input planes: curr, opp, empty,
    fork_self, fork_opp (see board_to_input_array).
    """

    def __init__(self, board_size=BOARD_SIZE, num_channels=None, num_res_blocks=None, value_hidden=None):
        super(AlphaZeroTicTacToe, self).__init__()
        # Scale capacity with board size so every call site (training and
        # inference alike) gets a consistent architecture automatically --
        # 3x3 already converges well at 64/2 (loss ~0.04), but on 5x5 that
        # size plateaued (loss flat ~0.45 for dozens of epochs even after
        # the replay buffer was mostly refreshed with better search targets)
        # which points at capacity, not data quality, as the bottleneck.
        # 10x10 (4x the cells of 5x5, WIN_LENGTH=5 needing a longer-range
        # receptive field) reused the 5x5-sized 128/3 config with no board
        # size big enough to fall outside board_size<=5's tier -- adding a
        # third tier here before the same capacity plateau shows up again.
        if board_size <= 3:
            num_channels = num_channels or 64
            num_res_blocks = num_res_blocks or 2
            value_hidden = value_hidden or 64
        elif board_size <= 5:
            num_channels = num_channels or 128
            num_res_blocks = num_res_blocks or 3
            value_hidden = value_hidden or 128
        else:
            # Gen3 capacity bump (2026-09-08): 192/5 plateaued (Policy Loss
            # never below 0.548 across 400+ epochs, same plateau-shape as the
            # 5x5 tier's precedent above) -- 256/7 is ~2.5x params.
            num_channels = num_channels or 256
            num_res_blocks = num_res_blocks or 7
            value_hidden = value_hidden or 256

        self.board_size = board_size
        self.num_cells = board_size * board_size
        # Stored as an instance attribute (not read as the NUM_INPUT_CHANNELS
        # global directly) because forward() references it under
        # torch.jit.script -- TorchScript can close over self.* attributes
        # but rejects closing over a plain module-level Python int global.
        self.num_input_channels = NUM_INPUT_CHANNELS

        # Common Conv Feature Extractor
        self.conv_in = nn.Sequential(
            nn.Conv2d(NUM_INPUT_CHANNELS, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy Head (Outputs log-probabilities for BOARD_SIZE*BOARD_SIZE actions)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.num_cells, self.num_cells),
            nn.LogSoftmax(dim=-1)
        )

        # Value Head (Outputs state value in range [-1, 1])
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * self.num_cells, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, self.num_input_channels, self.board_size, self.board_size)
        out = self.conv_in(x)
        for block in self.res_blocks:
            out = block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


# ============================================================
# Global Model / Device State Caching
# ============================================================

_current_model = None
_current_device = None
_current_model_path = None


def _get_model_and_device_internal():
    global _current_model
    global _current_device

    if _current_model is None or _current_device is None:
        _current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _current_model = AlphaZeroTicTacToe()
        _current_model.to(_current_device)

    return _current_model, _current_device


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


# ============================================================
# AI Move Selection
# ============================================================

def _ai_make_move_with_model_internal(model, device, board_state, ai_player):
    model_input = board_to_input_tensor(board_state, ai_player, device)
    model.eval()

    with torch.no_grad():
        policy_logits, _ = model(model_input)

    move_probabilities = torch.exp(policy_logits).squeeze().cpu().numpy()

    valid_moves = []
    for i in range(NUM_CELLS):
        r, c = divmod(i, BOARD_SIZE)
        if board_state[r][c] == ' ':
            valid_moves.append((move_probabilities[i], r, c))

    if not valid_moves:
        return -1, -1

    valid_moves.sort(key=lambda x: x[0], reverse=True)
    return valid_moves[0][1], valid_moves[0][2]


def _load_model_internal(model, path, device):
    global _current_model_path
    if _current_model_path == path:
        return  # Model already loaded in memory!

    try:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            _current_model_path = path
    except Exception as e:
        print(f"AlphaGoLee Notice: Model load fallback ({e}).")


# ============================================================
# Public API
# ============================================================

def play(board, turn, model_name):
    global _current_model
    global _current_device

    if _current_model is None or _current_device is None:
        _current_model, _current_device = _get_model_and_device_internal()

    model_path = os.path.join('model', model_name)
    _load_model_internal(_current_model, model_path, _current_device)

    row, col = _ai_make_move_with_model_internal(_current_model, _current_device, board, turn)

    if row == -1 and col == -1:
        return None

    return (row, col)