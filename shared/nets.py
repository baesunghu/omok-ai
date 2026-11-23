"""The networks of the NAPS lineage (until 2026-11: alphago_lee.py, alphago_lee_v2_seblock.py, alphago_lee_v3_survival.py).

    GomokuNetV1   first 11x11 network (residual CNN, BatchNorm)
    GomokuNetV2   SE blocks + GroupNorm (Gen12 onward)
    GomokuNetV3   V2 + survival head (NAPS from ep285, 11x11_v2, 15x15)

The v3 run's network (11x11_v2/v3/net_v3.py, NetV3: 128 channels x 8 blocks, auxiliary heads) is a different,
smaller net built from GomokuNetV3. Checkpoints are state_dicts, so they load under the new class names.
"""
import os

import torch
import torch.nn as nn

from board_encoding import (  # noqa: F401 -- re-exported for convenience
    BOARD_SIZE, WIN_LENGTH, NUM_CELLS, NUM_INPUT_CHANNELS, INPUT_SIZE,
    board_to_input_array, board_to_input_array_from_numeric,
    board_to_input_tensor, boards_to_input_tensor, preferred_memory_format,
    _fork_mask,
)


# ======================================================================
# V1 (until 2026-11 in alphago_lee.py)
# ======================================================================


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


class GomokuNetV1(nn.Module):
    """
    High-performance 2D Convolutional Policy-Value Network for 10x10 Board.
    Supports both 4D tensors (B, NUM_INPUT_CHANNELS, 10, 10) and 2D tensors
    (B, NUM_INPUT_CHANNELS * 100) -- 5 input planes: curr, opp, empty,
    fork_self, fork_opp (see board_to_input_array).
    """

    def __init__(self, board_size=BOARD_SIZE, num_channels=None, num_res_blocks=None, value_hidden=None):
        super(GomokuNetV1, self).__init__()
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
            # Gen3 capacity bump: 192/5 plateaued (Policy Loss
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
        _current_model = GomokuNetV1()
        _current_model.to(_current_device)

    return _current_model, _current_device


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


# ======================================================================
# V2 (until 2026-11 alphago_lee_v2_seblock.py)
# ======================================================================
# Experimental fused-block architecture, built alongside (not
# replacing) alphago_lee.py's V1 network (GomokuNetV1). Imports everything except
# the model classes from board_encoding -- fork-mask helpers, board_to_input_array,
# board_to_input_tensor, BOARD_SIZE/NUM_CELLS/NUM_INPUT_CHANNELS -- so there is
# exactly one source of truth for the board/tensor representation; only the
# network internals differ here.
#
# Combines three changes discussed in chat, each independently motivated:
# 1. Depthwise-separable convolutions (MobileNetV3/EfficientNet-style) instead
#    of full 3x3 convs -- ~8-9x fewer multiply-adds per conv layer at this
#    channel count (256), directly targeting this project's recurring GPU
#    bottleneck.
# 2. GroupNorm instead of BatchNorm -- GroupNorm normalizes per-sample (no
#    batch-wide running statistics), so it behaves identically regardless of
#    batch size or train()/eval() mode. This eliminates an entire bug class
#    this project has hit more than once (the 5x5 gate-check BatchNorm
#    train-mode bug and others referenced in this session) at
#    the source, rather than continuing to rely on callers remembering to
#    call .eval() before every MCTS leaf batch.
# 3. Squeeze-and-Excitation channel attention (KataGo uses this in its own
#    trunk) -- lets the network dynamically emphasize/suppress channels based
#    on GLOBAL board context, which suits whole-board pattern reasoning
#    (open-three / fork detection) better than treating every channel with
#    fixed importance regardless of position.
#
# This combination (depthwise-separable conv + SE, "MBConv"-style) is not
# novel -- it is the standard building block of MobileNetV3 and EfficientNet,
# chosen here because it is well-validated in the literature, not because it
# is untested.
#
# IMPORTANT: this is architecturally incompatible with existing checkpoints
# (nets.GomokuNetV1's ResBlock uses full Conv2d + BatchNorm2d
# with different parameter shapes) -- a model built from this file must be
# trained from scratch, it cannot load model/best_mcts_11x11.pth or any
# existing checkpoint. Nothing under 11x11_naps/model/ is touched by this file
# merely existing; it only takes effect wherever something explicitly
# imports GomokuNetV2 instead of the original.


def _group_norm(channels, max_groups=32):
    """GroupNorm needs channels % groups == 0 -- pick the largest divisor
    of `channels` that is <= max_groups instead of assuming 32 always
    divides evenly (true for 256, not guaranteed for arbitrary channel
    counts a caller might pass)."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate. Squeeze: global-average-pool
    each channel to one scalar (how active is this channel across the
    WHOLE board). Excitation: a small bottleneck MLP turns those C scalars
    into C gate values in [0,1]; the input feature map is rescaled
    channel-wise by those gates."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s


class DSConv(nn.Module):
    """2026-09-21: depthwise/grouped separable convs were tried (full
    depthwise, then groups=32) and both measured far SLOWER wall-clock on
    this GPU than a standard conv despite fewer FLOPs -- grouped conv
    benchmarked at 18.75s/train-step (batch 256) vs the original
    architecture's ~0.78s/step, a ~24x regression. Low arithmetic intensity
    (few multiply-adds per byte moved) makes these patterns memory-
    bandwidth-bound, which this older Pascal-era GTX 1050 (no Tensor Cores,
    older cuDNN grouped-conv kernel support) handles poorly regardless of
    group size. Reverted to a plain full 3x3 Conv2d -- this class now just
    wraps conv + GroupNorm (see module docstring point 2); the class name
    is kept (not renamed to avoid churn) even though it is no longer
    depthwise-separable at all. Only the SE gate (point 3) and GroupNorm
    (point 2) remain from the original three-part proposal."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _group_norm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class FusedSEResBlock(nn.Module):
    """Residual block built from two DSConv layers + an SE gate placed
    just before the skip connection is added back -- the standard
    MobileNetV3/EfficientNet MBConv placement."""
    def __init__(self, channels, se_reduction=16):
        super().__init__()
        self.conv1 = DSConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DSConv(channels, channels)
        self.se = SEBlock(channels, reduction=se_reduction)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        out = out + residual
        return self.relu(out)


class GomokuNetV2(nn.Module):
    """Same external interface as nets.GomokuNetV1 (same
    constructor signature and defaults, same forward(x) -> (policy, value)
    contract, same 4D/2D input handling) -- only the internal block type
    and normalization differ. NOT weight-compatible with the original;
    always trained from scratch."""

    def __init__(self, board_size=BOARD_SIZE, num_channels=None, num_res_blocks=None, value_hidden=None):
        super().__init__()
        if board_size <= 3:
            num_channels = num_channels or 64
            num_res_blocks = num_res_blocks or 2
            value_hidden = value_hidden or 64
        elif board_size <= 5:
            num_channels = num_channels or 128
            num_res_blocks = num_res_blocks or 3
            value_hidden = value_hidden or 128
        else:
            # Back to 256/7 (matching the original ResBlock
            # architecture's sizing) -- the 320/16 bump was sized assuming
            # depthwise-separable's compute savings, which were dropped
            # (see DSConv's docstring: measured far slower wall-clock on
            # this GPU). Without that savings, 320/16 would have been a
            # 3.5x-bigger network (29,993,062 params) than the original,
            # not a lighter one -- reverting to the original size keeps
            # this an apples-to-apples GroupNorm+SE comparison.
            num_channels = num_channels or 256
            num_res_blocks = num_res_blocks or 7
            value_hidden = value_hidden or 256

        self.board_size = board_size
        self.num_cells = board_size * board_size
        self.num_input_channels = NUM_INPUT_CHANNELS

        self.conv_in = nn.Sequential(
            nn.Conv2d(NUM_INPUT_CHANNELS, num_channels, kernel_size=3, padding=1),
            _group_norm(num_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([FusedSEResBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1),
            _group_norm(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.num_cells, self.num_cells),
            nn.LogSoftmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            _group_norm(4),
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


# ======================================================================
# V3 (until 2026-11 alphago_lee_v3_survival.py)
# ======================================================================
# V3 = V2 (GroupNorm + SE, nets.py) + an auxiliary
# "survival" head that predicts how many plies remain until the game ends.
#
# Why: in 11x11 free-style gomoku the first player wins almost every game
# (elo_roundrobin.py: 68/72), so nearly every defender-side training
# position carries the same -1 value label no matter how well it was
# defended -- the value head has no reason to tell a good defence from a
# bad one, and defence quality measurably fell during self-play
# (elo_defense.py: the supervised start model survived 21.5 plies as the
# defender, the self-play models 11-14). Predicting remaining plies gives
# the shared trunk a dense per-position signal that does distinguish
# "losing slowly" from "losing fast" (KataGo-style auxiliary target).
#
# Compatibility:
# - forward(x) still returns (policy, value) exactly like V2, so every
#   existing call site (MCTS eval callbacks, gating, league, self-play pool,
#   test scripts) works unchanged. forward_all(x) additionally returns the
#   survival prediction and is used only by the training step.
# - load_state_dict() accepts V2 checkpoints: the survival_head.* keys are
#   allowed to be missing (the head keeps its fresh init); any OTHER
#   missing or unexpected key still raises, so a genuinely wrong checkpoint
#   is not silently accepted.
# - V3 checkpoints carry extra survival_head.* keys, so they do NOT load
#   into the V2 class with strict=True. The last V2 best is kept at
#   model/v2_final_best_ep284.pth for old-vs-new matches.
#
# Survival target: min(remaining_plies, SURVIVAL_CAP) / SURVIVAL_CAP, in
# [0, 1]; the head ends in a sigmoid.

SURVIVAL_CAP = 40


class GomokuNetV3(GomokuNetV2):
    def __init__(self, *args, survival_hidden=64, **kwargs):
        super().__init__(*args, **kwargs)
        num_channels = self.conv_in[0].out_channels
        self.survival_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            _group_norm(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.num_cells, survival_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(survival_hidden, 1),
            nn.Sigmoid(),
        )

    def _trunk(self, x):
        if x.dim() == 2:
            x = x.view(-1, self.num_input_channels, self.board_size, self.board_size)
        out = self.conv_in(x)
        for block in self.res_blocks:
            out = block(out)
        return out

    def forward(self, x):
        out = self._trunk(x)
        return self.policy_head(out), self.value_head(out)

    def forward_all(self, x):
        out = self._trunk(x)
        return self.policy_head(out), self.value_head(out), self.survival_head(out)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=False, **kwargs)
        if strict:
            bad_missing = [k for k in result.missing_keys if not k.startswith("survival_head.")]
            if bad_missing or result.unexpected_keys:
                raise RuntimeError(
                    f"GomokuNetV3.load_state_dict: missing={bad_missing} "
                    f"unexpected={result.unexpected_keys}")
        return result
