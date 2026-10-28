"""11x11_v3 network (2026-10-22, user): 128 ch x 8 blocks, same trunk/heads as the v2 net
(GroupNorm + SE res blocks, policy / value / survival heads) plus two KataGo-style auxiliary heads:

  opp_policy_head -- the opponent's actual reply to this position (log-probs over cells)
  owner_head      -- who occupies each cell when the game ends: empty / mover / opponent
                     (log-probs, shape (B, 3, N, N))

forward(x) still returns (policy, value), so every search / match script works unchanged;
forward_all(x) returns all five outputs and is used only by train_v3.py.
"""
import os
import sys

import torch.nn as nn

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 11x11_v2 modules
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from alphago_lee_v2_seblock import _group_norm

CHANNELS, BLOCKS = 128, 8


class NetV3(AlphaZeroTicTacToeV3):
    def __init__(self, num_channels=CHANNELS, num_res_blocks=BLOCKS, **kwargs):
        super().__init__(num_channels=num_channels, num_res_blocks=num_res_blocks,
                         value_hidden=max(64, num_channels), **kwargs)
        n = self.num_cells
        self.opp_policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1), _group_norm(8), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(8 * n, n), nn.LogSoftmax(dim=-1))
        self.owner_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1), _group_norm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1), nn.LogSoftmax(dim=1))

    def forward_all(self, x):
        out = self._trunk(x)
        return (self.policy_head(out), self.value_head(out), self.survival_head(out),
                self.opp_policy_head(out), self.owner_head(out))

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return nn.Module.load_state_dict(self, state_dict, strict=strict, **kwargs)
