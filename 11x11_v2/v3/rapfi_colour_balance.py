"""How often does black win when Rapfi plays Rapfi on 11x11? (2026-10-25, user: is the 53:47 black:white value weight realistic?)
Same openings as the ladder (one X stone, then O to move; RAPFI_MORE_OPENINGS set = 25 cells within 2 of the centre),
both sides Rapfi with the same node limit, game to five in a row or a full board (draw).
Usage (from 11x11_v2): python v3/rapfi_colour_balance.py [nodes,...]
"""
import os
import sys

os.environ["RAPFI_MORE_OPENINGS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rapfi_match import Rapfi, OPENINGS
from fast_board import fast_check_win

N = 11
for nodes in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "10000,30000").split(",")]:
    bx = by = d = 0
    for op in OPENINGS:
        eng = {'X': Rapfi(30000, nodes), 'O': Rapfi(30000, nodes)}   # one process per colour, as in rapfi_selfcheck
        moves = [("X", op[0], op[1])]
        b = [[' '] * N for _ in range(N)]
        b[op[0]][op[1]] = 'X'
        turn, win = 'O', None
        while len(moves) < N * N:
            r, c = eng[turn].move(list(moves), turn)
            b[r][c] = turn
            moves.append((turn, r, c))
            if fast_check_win(b, turn):
                win = turn
                break
            turn = 'X' if turn == 'O' else 'O'
        for e in eng.values():
            e.close()
        bx += win == 'X'; by += win == 'O'; d += win is None
        print(f"{nodes} opening {op}: {win or 'draw'} in {len(moves)} plies", flush=True)
    dec = bx + by
    print(f"== Rapfi {nodes:,} vs itself, {len(OPENINGS)} openings: black {bx} / white {by} / draw {d}"
          f"  -> decisive black share {100 * bx / dec:.0f}%" if dec else "  (no decisive games)", flush=True)
