"""Do the v2 engine (11x11_v2/mcts_core, used by v3/head2head.py) and the v3 Gumbel build (v3/cpp/mcts_core, the only
engine source in the repo) play identical head2head games? If yes, Kaggle can run head2head with the v3 build.
Usage: python kaggle/engine_parity.py <engine dir> <A.pth> <B.pth>   (prints the move list of 2 short games)
"""
import importlib.util
import os
import sys

import torch

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eng_dir = os.path.abspath(sys.argv[1])
sys.path.append(os.path.dirname(V2))                 # repository root -> omok_paths
from omok_paths import find_extension  # noqa: E402
spec = importlib.util.spec_from_file_location("mcts_core", find_extension(eng_dir, "mcts_core"))
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
sys.modules["mcts_core"] = mc
sys.path.insert(0, V2); sys.path.insert(1, os.path.join(V2, "v3"))
os.chdir(V2)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))   # shared/: code common to every run
from board_encoding import make_ev
from fast_board import fast_check_win
from net_v3 import NetV3

mc.set_vcf(10, 3000)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(p):
    m = NetV3(); m.load_state_dict(torch.load(p, map_location=DEV)); return m.to(DEV).eval()





a, b = make_ev(load(sys.argv[2]), DEV), make_ev(load(sys.argv[3]), DEV)
for opening in (((5, 5), (4, 4)), ((5, 5), (4, 5))):
    board = [[' '] * 11 for _ in range(11)]; turn = 'X'; moves = []
    for r, c in opening:
        board[r][c] = turn; turn = 'O' if turn == 'X' else 'X'
    for _ in range(30):
        ev = a if turn == 'X' else b
        _, _, cands = mc.MCTSEngine().search_batched(board, turn, ev, 1100, 1.414, False, 0.3, 0.25, 11, 5, 16)
        r, c = max(cands, key=lambda x: x["visit_count"])["move"]
        board[r][c] = turn; moves.append(r * 11 + c)
        if fast_check_win(board, turn):
            break
        turn = 'O' if turn == 'X' else 'X'
    print(moves)
