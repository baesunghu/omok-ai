"""메기 리그 / CATFISH league (2026-10-18, user) -- CATFISH = Crippled Anchors To Fight Inbred Self-play Habits.
Catfish ("메기") league opponents: two deliberately broken copies of the current best that
stir up the league so self-play stops sharing one set of blind spots. No training-code change: the trainer
already loads every file in model/league_anchors/ as a fixed league opponent each epoch.
  catfish_value0   -- value head output forced to 0 (last Linear zeroed -> tanh(0)); search judges only real
                      wins/losses, so it does not share the main net's overconfidence (cf. ep960 value ablation).
  catfish_value_half -- value head softened (last Linear x0.5 -> tanh(0.5 z)), 2026-10-21.
  catfish_policy0  -- policy head forced to uniform (last Linear zeroed -> equal logits); search spreads over
                      moves the main net's policy never looks at.
League-side moves are not recorded as training targets (RECORD_LEAGUE_SIDE = False); only the main net's own
moves and the game results from these games are learned.
Usage: python make_catfish.py   (re-run after a promotion to rebase them on the new best)
"""
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))   # shared/: code common to every run
from fast_board import FILE_SUFFIX  # noqa: E402  ('_11x11' / '_15x15', from the board size)

SRC = f"model/best_mcts{FILE_SUFFIX}.pth"
OUT = "model/league_anchors"
os.makedirs(OUT, exist_ok=True)
sd = torch.load(SRC, map_location="cpu")
v = {k: t.clone() for k, t in sd.items()}
v["value_head.6.weight"].zero_(); v["value_head.6.bias"].zero_()
torch.save(v, os.path.join(OUT, "catfish_value0.pth"))
# A "half" value catfish -- last value Linear x0.5, so tanh(0.5 z): the same judgement,
# just less sure of it (a softened, not a dead, value head).
h = {k: t.clone() for k, t in sd.items()}
h["value_head.6.weight"].mul_(0.5); h["value_head.6.bias"].mul_(0.5)
torch.save(h, os.path.join(OUT, "catfish_value_half.pth"))
p = {k: t.clone() for k, t in sd.items()}
p["policy_head.4.weight"].zero_(); p["policy_head.4.bias"].zero_()
torch.save(p, os.path.join(OUT, "catfish_policy0.pth"))
print("catfish opponents written from", SRC)
# Dawn (ep625, graduated 9/29) as an intact league anchor -- the cleanest value head in the
# lineage (before the Hard-7 contamination) and it passed the graduation problems 3/3.
DAWN = "../11x11_naps/model/graduated_11x11/ep625_graduated_20260929.pth"
if os.path.exists(DAWN):
    torch.save(torch.load(DAWN, map_location="cpu"), os.path.join(OUT, "catfish_dawn.pth"))
    print("catfish_dawn written from", DAWN)
