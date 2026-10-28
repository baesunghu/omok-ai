"""Is a defence problem fair? Play it Rapfi vs Rapfi (2026-10-23, user: p18 is really "survive 40 plies vs 1M-node Rapfi").
The defender is also Rapfi; if even Rapfi loses as the defender, the position is lost and the problem cannot be passed.
Usage: python v3/rapfi_selfcheck.py PROBLEM_ID [defender_nodes,...]   (run from 11x11_v2)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rapfi_match import Rapfi
from fast_board import fast_check_win

N, MAX_PLIES, ATTACK_NODES = 11, 40, 1000000
pid = sys.argv[1]
defs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1000000,30000").split(",")]
pr = next(p for p in json.load(open("tactical_problems/web_game58_problems.json", encoding="utf-8"))["problems"] if p["id"] == pid)
ai = pr["side_to_move"]
FIRST = tuple(int(x) for x in os.environ["FIRST"].split(",")) if os.environ.get("FIRST") else None
for dn in defs:
    attacker, defender = Rapfi(30000, ATTACK_NODES), Rapfi(30000, dn)
    moves = [tuple(m) for m in pr["moves"]]
    board = [[' '] * N for _ in range(N)]
    for p, r, c in moves:
        board[r][c] = p
    turn, winner, first = ai, None, None
    for _ in range(MAX_PLIES):
        if first is None and FIRST:          # 2026-10-24: force the defender's first move (snap-through check)
            r, c = FIRST
        else:
            r, c = (defender if turn == ai else attacker).move(list(moves), turn)
        first = first or (r, c)
        board[r][c] = turn
        moves.append((turn, r, c))
        if fast_check_win(board, turn):
            winner = turn
            break
        turn = 'O' if turn == 'X' else 'X'
    ok = first != tuple(pr.get("bad_move") or ()) and winner != ('X' if ai == 'O' else 'O')
    print(f"{pid} defender Rapfi {dn:,} nodes vs attacker 1,000,000: first {first}, "
          f"{'PASS' if ok else 'FAIL'} (winner {winner}, {len(moves) - len(pr['moves'])} plies)", flush=True)
    for e in (attacker, defender):
        try:
            e.close()
        except Exception:
            pass
