"""Is 900's (3,7) a real answer to the 13-ply problem? Force the first move, then Rapfi defends at N nodes
against Rapfi attacking at 200k nodes for 40 plies. Same for Rapfi's own answer (8,5) and the bad move (8,9)."""
import json, os, sys
sys.path.insert(0, r"C:\Users\My PC\Desktop\final\11x11_v2"); os.chdir(r"C:\Users\My PC\Desktop\final\11x11_v2")
from rapfi_match import Rapfi
from fast_board import fast_check_win
N = 11
p = [x for x in json.load(open("tactical_problems/web_game58_problems.json", encoding="utf-8"))["problems"] if x["id"] == "web58_d13_defense"][0]
me = p["side_to_move"]; opp = 'O' if me == 'X' else 'X'
att = Rapfi(30000, 200000)
try:
    for first in ((3, 7), (8, 5), (8, 9)):
        row = []
        for n in (1000, 2000, 5000, 10000, 50000, 200000):
            dfn = Rapfi(30000, n)
            try:
                moves = [tuple(m) for m in p["moves"]] + [(me,) + first]
                b = [[' '] * N for _ in range(N)]
                for pl, r, c in moves: b[r][c] = pl
                turn, winner, k = opp, None, 0
                for k in range(1, 40):
                    r, c = (att if turn == opp else dfn).move(list(moves), turn)
                    b[r][c] = turn; moves.append((turn, r, c))
                    if fast_check_win(b, turn): winner = turn; break
                    turn = me if turn == opp else opp
            finally:
                dfn.close()
            row.append(f"{n}:{'hold' if winner != opp else 'lost@' + str(k)}")
        print(first, "  ".join(row), flush=True)
finally:
    att.close()
