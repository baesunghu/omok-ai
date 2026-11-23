"""Print the moves and our root values of one graded trial (2026-10-24: why did ep400 lose the corner-diagonal problem?).
Usage (from 11x11_v2): python v3/trace_problem.py CKPT PROBLEM_ID SIMS"""
import os as _os, sys as _sys
_sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))   # repository root
from omok_paths import PROBLEMS_FILE  # noqa: E402  (11x11_naps/tactical_problems: the 11x11 problem set, one copy)
import json, os, sys
import torch
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2); sys.path.insert(1, os.path.join(V2, "v3"))
import mcts_core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))   # shared/: code common to every run
from board_encoding import make_ev
from fast_board import fast_check_win
from net_v3 import NetV3
from rapfi_match import Rapfi
mcts_core.set_vcf(10, 3000)
ckpt, pid, sims = sys.argv[1], sys.argv[2], int(sys.argv[3])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
sd = torch.load(ckpt, map_location=DEV); sd = sd["model"] if "model" in sd else sd
m = NetV3(); m.load_state_dict(sd); m.to(DEV).eval()
ev = make_ev(m, DEV)
pr = next(p for p in json.load(open(PROBLEMS_FILE, encoding="utf-8"))["problems"] if p["id"] == pid)
ai = pr["side_to_move"]; rapfi = Rapfi(30000, 1000000)
moves = [tuple(x) for x in pr["moves"]]; b = [[' '] * 11 for _ in range(11)]
for p, r, c in moves: b[r][c] = p
turn = ai
for k in range(40):
    if turn == ai:
        _, rv, cands = mcts_core.MCTSEngine().search_batched(b, turn, ev, sims, 1.414, False, 0.3, 0.25, 11, 5, 16)
        top = sorted(cands, key=lambda x: -x["visit_count"])[:3]
        r, c = top[0]["move"]
        print(f"{k+1:2d} us  {turn} ({r},{c})  root {rv:+.2f}  top3 " + ", ".join(f"{tuple(x['move'])}:{x['visit_count']}" for x in top), flush=True)
    else:
        r, c = rapfi.move(list(moves), turn); print(f"{k+1:2d} Rapfi {turn} ({r},{c})", flush=True)
    b[r][c] = turn; moves.append((turn, r, c))
    if fast_check_win(b, turn): print("winner", turn); break
    turn = 'O' if turn == 'X' else 'X'
rapfi.close()
