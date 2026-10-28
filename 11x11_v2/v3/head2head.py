"""Head-to-head between two v3 checkpoints (2026-10-23, user: did ep120 drift by over-fitting the catfish?).

Each opening (first two stones fixed) is played twice with colours swapped, so 10 openings = 20 games.
Same noiseless PUCT search for both sides (1,100 sims, VCF on), like the Rapfi ladder.
Usage: python head2head.py A.pth B.pth [sims]   (run from 11x11_v2/v3)
"""
import os
import sys

import numpy as np
import torch

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)
os.chdir(V2)
import mcts_core
from alphago_lee import board_to_input_array_from_numeric
from fast_board import fast_check_win
sys.path.insert(1, os.path.join(V2, "v3"))
from net_v3 import NetV3

N = 11
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OPENINGS = [((5, 5), (4, 4)), ((5, 5), (4, 5)), ((5, 5), (6, 4)), ((5, 5), (3, 3)), ((5, 5), (5, 3)),
            ((4, 5), (5, 5)), ((5, 4), (4, 6)), ((4, 4), (6, 6)), ((5, 6), (4, 4)), ((6, 5), (5, 3))]
if os.environ.get("H2H_MORE"):   # 2026-10-26 (user: evaluate ep400 more): 36 openings = 72 games, the 10 above first
    OPENINGS = OPENINGS + [((r, c), (r + dr, c + dc)) for r in (4, 5, 6) for c in (4, 5, 6) for dr, dc in ((-1, -1), (-1, 1), (0, 2))
                           if ((r, c), (r + dr, c + dc)) not in OPENINGS][:26]
mcts_core.set_vcf(10, 3000)


def load(p):
    m = NetV3(); m.load_state_dict(torch.load(p, map_location=DEV)); return m.to(DEV).eval()


def make_ev(m):
    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(DEV)
        with torch.inference_mode():
            pl, v = m(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
    return ev


def play(evx, evo, opening, sims):
    board = [[' '] * N for _ in range(N)]
    turn = 'X'
    for r, c in opening:
        board[r][c] = turn; turn = 'O' if turn == 'X' else 'X'
    for _ in range(N * N - len(opening)):
        ev = evx if turn == 'X' else evo
        _, _, cands = mcts_core.MCTSEngine().search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, 5, 16)
        r, c = max(cands, key=lambda x: x["visit_count"])["move"]
        board[r][c] = turn
        if fast_check_win(board, turn):
            return turn
        turn = 'O' if turn == 'X' else 'X'
    return None


def main():
    a, b = sys.argv[1], sys.argv[2]
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
    eva, evb = make_ev(load(a)), make_ev(load(b))
    w = l = d = 0
    for i, op in enumerate(OPENINGS):
        for a_is_x in (True, False):
            win = play(eva, evb, op, sims) if a_is_x else play(evb, eva, op, sims)
            res = "draw" if win is None else ("A" if (win == 'X') == a_is_x else "B")
            w += res == "A"; l += res == "B"; d += res == "draw"
            print(f"game {2 * i + (0 if a_is_x else 1) + 1}: opening {op} A={'X' if a_is_x else 'O'} -> {res}", flush=True)
    n = w + l + d
    print(f"\nA={os.path.basename(a)} vs B={os.path.basename(b)}: A {w} / B {l} / draw {d}, A score {(w + 0.5 * d) / n:.1%}")


if __name__ == "__main__":
    main()
