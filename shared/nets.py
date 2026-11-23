"""The networks of the NAPS lineage (until 2026-11: alphago_lee.py, alphago_lee_v2_seblock.py, alphago_lee_v3_survival.py).

    GomokuNetV1   first 11x11 network (residual CNN, BatchNorm)
    GomokuNetV2   SE blocks + GroupNorm (Gen12 onward)
    GomokuNetV3   V2 + survival head (NAPS from ep285, 11x11_v2, 15x15)

The v3 run's network (11x11_v2/v3/net_v3.py, NetV3: 128 channels x 8 blocks, auxiliary heads) is a different,
smaller net built from GomokuNetV3. Checkpoints are state_dicts, so they load under the new class names.
"""
import torch.nn as nn

from board_encoding import BOARD_SIZE, NUM_INPUT_CHANNELS


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


# ======================================================================
# V2 (until 2026-11 alphago_lee_v2_seblock.py)
# ======================================================================
# SE blocks + GroupNorm. Shares the board encoding with V1 (board_encoding); only the network differs. Three changes:
# 1. Depthwise-separable convolutions instead of full 3x3 -- about 8-9x fewer multiply-adds per layer at 256 channels,
#    aimed at the GPU bottleneck.
# 2. GroupNorm instead of BatchNorm -- normalizes per sample, so it behaves the same at any batch size and in
#    train() or eval() mode. This removes the BatchNorm-mode bug class (e.g. a gate check that ran in train mode) at
#    the source instead of relying on every caller to call .eval().
# 3. Squeeze-and-excitation channel attention (as in KataGo's trunk): channels weighted by global board context,
#    which suits whole-board patterns such as open threes and forks.
# Depthwise-separable conv + SE is the standard MobileNetV3 / EfficientNet block. V2 cannot load V1 checkpoints
# (different parameter shapes); it was trained from scratch.


def group_norm(channels, max_groups=32):
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
        self.norm = group_norm(out_channels)

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
            group_norm(num_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([FusedSEResBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1),
            group_norm(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.num_cells, self.num_cells),
            nn.LogSoftmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            group_norm(4),
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
# V3 = V2 + an auxiliary "survival" head that predicts how many plies remain.
#
# Why: in 11x11 free-style the first player wins almost every game (elo_roundrobin.py: 68/72), so nearly every
# defender position carries the value label -1 however well it was defended; the value head cannot tell good
# defence from bad, and defence measurably got worse in self-play (elo_defense.py: the supervised start model lasted
# 21.5 plies as defender, the self-play models 11-14). Remaining plies is a dense per-position signal that separates
# losing slowly from losing fast (a KataGo-style auxiliary target).
#
# Compatibility: forward(x) still returns (policy, value), so every caller works unchanged; forward_all(x) adds the
# survival prediction for the training step. load_state_dict accepts V2 checkpoints (survival_head.* may be missing;
# any other missing or unexpected key still raises). V3 checkpoints do not load into V2 with strict=True; the last V2
# best is model/v2_final_best_ep284.pth.
#
# Target: min(remaining plies, SURVIVAL_CAP) / SURVIVAL_CAP in [0, 1]; the head ends in a sigmoid.

SURVIVAL_CAP = 40


class GomokuNetV3(GomokuNetV2):
    def __init__(self, *args, survival_hidden=64, **kwargs):
        super().__init__(*args, **kwargs)
        num_channels = self.conv_in[0].out_channels
        self.survival_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            group_norm(2),
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
