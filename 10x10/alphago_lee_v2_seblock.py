"""
2026-09-21: experimental fused-block architecture, built alongside (not
replacing) alphago_lee.py's AlphaZeroTicTacToe. Imports everything except
the model classes from alphago_lee -- fork-mask helpers, board_to_input_array,
board_to_input_tensor, BOARD_SIZE/NUM_CELLS/NUM_INPUT_CHANNELS -- so there is
exactly one source of truth for the board/tensor representation; only the
network internals differ here.

Combines three changes discussed in chat, each independently motivated:
1. Depthwise-separable convolutions (MobileNetV3/EfficientNet-style) instead
   of full 3x3 convs -- ~8-9x fewer multiply-adds per conv layer at this
   channel count (256), directly targeting this project's recurring GPU
   bottleneck.
2. GroupNorm instead of BatchNorm -- GroupNorm normalizes per-sample (no
   batch-wide running statistics), so it behaves identically regardless of
   batch size or train()/eval() mode. This eliminates an entire bug class
   this project has hit more than once (the 5x5 gate-check BatchNorm
   train-mode bug, 2026-08-29, and others referenced in this session) at
   the source, rather than continuing to rely on callers remembering to
   call .eval() before every MCTS leaf batch.
3. Squeeze-and-Excitation channel attention (KataGo uses this in its own
   trunk) -- lets the network dynamically emphasize/suppress channels based
   on GLOBAL board context, which suits whole-board pattern reasoning
   (open-three / fork detection) better than treating every channel with
   fixed importance regardless of position.

This combination (depthwise-separable conv + SE, "MBConv"-style) is not
novel -- it is the standard building block of MobileNetV3 and EfficientNet,
chosen here because it is well-validated in the literature, not because it
is untested.

IMPORTANT: this is architecturally incompatible with existing checkpoints
(alphago_lee.AlphaZeroTicTacToe's ResBlock uses full Conv2d + BatchNorm2d
with different parameter shapes) -- a model built from this file must be
trained from scratch, it cannot load model/best_mcts_11x11.pth or any
existing checkpoint. Nothing under 10x10/model/ is touched by this file
merely existing; it only takes effect wherever something explicitly
imports AlphaZeroTicTacToeV2 instead of the original.
"""
import torch
import torch.nn as nn

from alphago_lee import (  # noqa: F401 -- re-exported for convenience
    BOARD_SIZE, WIN_LENGTH, NUM_CELLS, NUM_INPUT_CHANNELS, INPUT_SIZE,
    board_to_input_array, board_to_input_array_from_numeric,
    board_to_input_tensor, boards_to_input_tensor, preferred_memory_format,
    _fork_mask,
)


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


class AlphaZeroTicTacToeV2(nn.Module):
    """Same external interface as alphago_lee.AlphaZeroTicTacToe (same
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
            # 2026-09-21: back to 256/7 (matching the original ResBlock
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
