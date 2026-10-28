"""
V3 = V2 (GroupNorm + SE, alphago_lee_v2_seblock.py) + an auxiliary
"survival" head that predicts how many plies remain until the game ends.

Why: in 11x11 free-style gomoku the first player wins almost every game
(elo_roundrobin.py: 68/72), so nearly every defender-side training
position carries the same -1 value label no matter how well it was
defended -- the value head has no reason to tell a good defence from a
bad one, and defence quality measurably fell during self-play
(elo_defense.py: the supervised start model survived 21.5 plies as the
defender, the self-play models 11-14). Predicting remaining plies gives
the shared trunk a dense per-position signal that does distinguish
"losing slowly" from "losing fast" (KataGo-style auxiliary target).

Compatibility:
- forward(x) still returns (policy, value) exactly like V2, so every
  existing call site (MCTS eval callbacks, gating, league, self-play pool,
  test scripts) works unchanged. forward_all(x) additionally returns the
  survival prediction and is used only by the training step.
- load_state_dict() accepts V2 checkpoints: the survival_head.* keys are
  allowed to be missing (the head keeps its fresh init); any OTHER
  missing or unexpected key still raises, so a genuinely wrong checkpoint
  is not silently accepted.
- V3 checkpoints carry extra survival_head.* keys, so they do NOT load
  into the V2 class with strict=True. The last V2 best is kept at
  model/v2_final_best_ep284.pth for old-vs-new matches.

Survival target: min(remaining_plies, SURVIVAL_CAP) / SURVIVAL_CAP, in
[0, 1]; the head ends in a sigmoid.
"""
import torch.nn as nn

from alphago_lee_v2_seblock import AlphaZeroTicTacToeV2, _group_norm

SURVIVAL_CAP = 40


class AlphaZeroTicTacToeV3(AlphaZeroTicTacToeV2):
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
                    f"AlphaZeroTicTacToeV3.load_state_dict: missing={bad_missing} "
                    f"unexpected={result.unexpected_keys}")
        return result
