"""Rapfi Elo scale on 15x15 (2026-10-14, user): Rapfi(n_lo) vs Rapfi(n_hi), colors alternating, 2 random
opening stones within 2 of the centre. Elo(hi) - Elo(lo) = 400 log10(s / (1 - s)), s = hi's score.
Usage: python rapfi_calib.py <n_lo> <n_hi> <games>   -> appends a line to model/rapfi_calib.jsonl
"""
import json
import math
import os
import random
import sys
import time

import sl_rapfi_data as S

N = 15


class R(S.Rapfi):
    def __init__(self, nodes):
        S.NODES = nodes
        super().__init__()


def main():
    lo, hi, games = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rnd = random.Random(lo * 7 + hi)
    eng = {lo: R(lo), hi: R(hi)}
    w = d = l = 0
    g = 0
    while g < games:
        b = [[' '] * N for _ in range(N)]; moves = []
        for i in range(2):
            p = 'X' if i == 0 else 'O'
            while True:
                r, c = 7 + rnd.randint(-2, 2), 7 + rnd.randint(-2, 2)
                if b[r][c] == ' ':
                    break
            b[r][c] = p; moves.append((p, r, c))
        hi_color = 'X' if g % 2 == 0 else 'O'
        turn, winner, ok = 'X', None, True
        while len(moves) < N * N:
            n = hi if turn == hi_color else lo
            try:
                r, c, _ = eng[n].move(list(moves), turn)
            except Exception:
                eng[n].close(); eng[n] = R(n); ok = False
                break
            b[r][c] = turn; moves.append((turn, r, c))
            if S.five(b, r, c, turn):
                winner = turn
                break
            turn = 'O' if turn == 'X' else 'X'
        if not ok:
            continue
        g += 1
        if winner is None:
            d += 1
        elif winner == hi_color:
            w += 1
        else:
            l += 1
    for e in eng.values():
        e.close()
    s = (w + 0.5 * d) / games
    sc = min(max(s, 0.02), 0.98)
    elo = 400 * math.log10(sc / (1 - sc))
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), lo=lo, hi=hi, games=games, hi_win=w, draw=d, hi_loss=l,
               score=round(s, 3), elo_diff=round(elo, 1))
    with open(os.path.join("model", "rapfi_calib.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


if __name__ == "__main__":
    main()
