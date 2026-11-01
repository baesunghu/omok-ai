"""Do the v2 engine (11x11_v2/mcts_core, used by v3/head2head.py) and the v3 Gumbel build (v3/cpp/mcts_core, the only
engine source in the repo) play identical head2head games? If yes, Kaggle can run head2head with the v3 build.
Usage: python kaggle/engine_parity.py <engine dir> <A.pth> <B.pth>   (prints the move list of 2 short games)
"""
import importlib.util
import os
import sys

import numpy as np
import torch

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eng_dir = os.path.abspath(sys.argv[1])
spec = importlib.util.spec_from_file_location("mcts_core", os.path.join(eng_dir, "mcts_core.pyd"))
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
sys.modules["mcts_core"] = mc
sys.path.insert(0, V2); sys.path.insert(1, os.path.join(V2, "v3"))
os.chdir(V2)
from alphago_lee import board_to_input_array_from_numeric
from fast_board import fast_check_win
from net_v3 import NetV3

mc.set_vcf(10, 3000)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(p):
    m = NetV3(); m.load_state_dict(torch.load(p, map_location=DEV)); return m.to(DEV).eval()


def ev_of(m):
    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(DEV)
        with torch.inference_mode():
            pl, v = m(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
    return ev


a, b = ev_of(load(sys.argv[2])), ev_of(load(sys.argv[3]))
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
