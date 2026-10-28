"""Does a root-only VCT check fix a defence problem? (2026-10-23, user)

Same trial as tactical_problem_check.py (our model vs Rapfi 1M nodes, 40 plies, 900/1100/1300 sims, VCF on), but at
each of our moves the candidates are tried in visit order and a candidate after which the opponent has a VCT is
skipped (prototype vct.py, node budget per check). Usage (from 11x11_v2):
  python v3/vct_root_trial.py CKPT PROBLEM_ID [top_k] [budget]
"""
import json
import os
import sys
import time

import numpy as np
import torch

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)
sys.path.insert(1, os.path.join(V2, "v3"))
import mcts_core
from alphago_lee import board_to_input_array_from_numeric
from fast_board import fast_check_win
from net_v3 import NetV3
from rapfi_match import Rapfi
import vct

mcts_core.set_vcf(10, 3000)
N, MAX_PLIES = 11, 40
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ckpt, pid = sys.argv[1], sys.argv[2]
TOP_K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

sd = torch.load(ckpt, map_location=DEV)
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
model = NetV3(); model.load_state_dict(sd); model.to(DEV).eval()


def ev(bs, ps):
    t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(DEV)
    with torch.inference_mode():
        pl, v = model(t)
    return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)


pr = next(p for p in json.load(open(os.path.join(V2, "tactical_problems", "web_game58_problems.json"), encoding="utf-8"))["problems"] if p["id"] == pid)
ai = pr["side_to_move"]
opp = 'O' if ai == 'X' else 'X'
rapfi = Rapfi(30000, 1000000)
passed = 0
for sims in (900, 1100, 1300):
    moves = [tuple(m) for m in pr["moves"]]
    board = [[' '] * N for _ in range(N)]
    for p, r, c in moves:
        board[r][c] = p
    turn, winner, first, swaps, t0 = ai, None, None, 0, time.time()
    for _ in range(MAX_PLIES):
        if turn == ai:
            _, rv, cands = mcts_core.MCTSEngine().search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, 5, 16)
            order = [tuple(x["move"]) for x in sorted(cands, key=lambda x: -x["visit_count"])]
            r, c = order[0]
            for i, (rr, cc) in enumerate(order[:TOP_K]):
                board[rr][cc] = ai
                lost = vct.has_vct(board, opp, budget=vct.Budget(BUDGET))
                board[rr][cc] = ' '
                if not lost:
                    r, c = rr, cc
                    swaps += i > 0
                    if i > 0:
                        print(f"  ply {len(moves) - len(pr['moves']) + 1}: MCTS {order[0]} gives a VCT -> {order[i]} (rank {i + 1})", flush=True)
                    break
            first = first or (r, c)
        else:
            r, c = rapfi.move(list(moves), turn)
        board[r][c] = turn
        moves.append((turn, r, c))
        if fast_check_win(board, turn):
            winner = turn
            break
        turn = opp if turn == ai else ai
    ok = first != tuple(pr.get("bad_move") or ()) and winner != opp
    passed += ok
    print(f"{sims}: {'PASS' if ok else 'FAIL'} first {first} winner {winner} plies {len(moves) - len(pr['moves'])} "
          f"VCT swaps {swaps} ({time.time() - t0:.0f}s)", flush=True)
rapfi.close()
print(f"{pid}: {passed}/3 with root VCT (top {TOP_K}, budget {BUDGET})")
