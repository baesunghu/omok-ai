"""
Double-three (3-3 fork) test set, hard-coded shapes -- MEASUREMENT ONLY
(2026-09-27). Nothing here feeds training.

Eight canonical 3-3 shapes (two lines of two stones meeting at an empty
fork point P; contiguous, broken, orthogonal and diagonal variants), each
placed near the centre under 4 of the 8 board symmetries, with a few
far-away stones for the other side so stone counts are legal.

ATTACK : the shape's owner is to move. Correct = any move that creates two
         or more threats at once (open three / four), or wins.
DEFENSE: the other side is to move. Correct = any move after which the
         shape's owner has no double-threat move left.

A threat is judged by brute force from the rules, not by shape templates:
  * four        = the mover has a cell that completes five on that line;
  * open three  = the mover has a cell that makes a STRAIGHT four (four in a
                  row with both ends empty) on that line.
Run with the production engine (currently the MCTS-Solver build).

Usage:  python double_three_test.py [checkpoint]   (default: best)
"""
import sys

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\10x10")
import numpy as np
import torch

import mcts_core
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH

N = BOARD_SIZE
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
SIMS = 1100

# (dr, dc) offsets of the owner's four stones relative to the fork point P.
SHAPES = {
    "cross +":            [(0, -1), (0, 1), (-1, 0), (1, 0)],
    "T (O_O over O/O)":   [(0, -1), (0, 1), (1, 0), (2, 0)],
    "L":                  [(0, 1), (0, 2), (1, 0), (2, 0)],
    "diagonal V":         [(1, 1), (2, 2), (1, -1), (2, -2)],
    "diagonal X":         [(-1, -1), (1, 1), (-1, 1), (1, -1)],
    "row + diagonal":     [(0, 1), (0, 2), (1, 1), (2, 2)],
    "broken + column":    [(0, 1), (0, 3), (1, 0), (2, 0)],
    "broken diag + row":  [(1, 1), (3, 3), (0, -1), (0, -2)],
}
OTHER_STONES = [(-4, -4), (-4, 4), (4, -4)]  # far from every shape (defender / spare stones)


def sym(dr, dc, k):
    for _ in range(k % 4):
        dr, dc = dc, -dr
    return (dr, -dc) if k >= 4 else (dr, dc)


def in_b(r, c):
    return 0 <= r < N and 0 <= c < N


def five_at(b, r, c, p):
    for dr, dc in DIRS:
        n = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while in_b(rr, cc) and b[rr][cc] == p:
                n += 1
                rr += s * dr
                cc += s * dc
        if n >= WIN_LENGTH:
            return True
    return False


def straight_four_through(b, r, c, dr, dc, p):
    """Is there 4-in-a-row of p through (r,c) along (dr,dc) with both ends empty?"""
    run = [(r, c)]
    rr, cc = r + dr, c + dc
    while in_b(rr, cc) and b[rr][cc] == p:
        run.append((rr, cc)); rr += dr; cc += dc
    end1 = (rr, cc)
    rr, cc = r - dr, c - dc
    while in_b(rr, cc) and b[rr][cc] == p:
        run.append((rr, cc)); rr -= dr; cc -= dc
    end2 = (rr, cc)
    return len(run) == 4 and all(in_b(*e) and b[e[0]][e[1]] == ' ' for e in (end1, end2))


def threats_after(b, r, c, p):
    """Number of directions through (r,c) where p (having just played there)
    has a four (a five-completing cell on that line) or an open three (a cell
    making a straight four on that line)."""
    count = 0
    for dr, dc in DIRS:
        line_cells = [(r + k * dr, c + k * dc) for k in range(-4, 5) if k != 0]
        four = open3 = False
        for (er, ec) in line_cells:
            if not in_b(er, ec) or b[er][ec] != ' ':
                continue
            b[er][ec] = p
            if five_at(b, er, ec, p):
                four = True
            elif straight_four_through(b, er, ec, dr, dc, p):
                # the straight four must include (r,c): check (r,c) is on that run
                open3 = True
            b[er][ec] = ' '
        if four or open3:
            count += 1
    return count


def is_fork_move(b, r, c, p):
    if b[r][c] != ' ':
        return False
    b[r][c] = p
    try:
        if five_at(b, r, c, p):
            return True
        return threats_after(b, r, c, p) >= 2
    finally:
        b[r][c] = ' '


def fork_moves(b, p):
    return [(r, c) for r in range(N) for c in range(N) if b[r][c] == ' ' and is_fork_move(b, r, c, p)]


def build(shape, k, owner):
    other = 'O' if owner == 'X' else 'X'
    b = [[' '] * N for _ in range(N)]
    P = (5, 5)
    for dr, dc in SHAPES[shape]:
        sr, sc = sym(dr, dc, k)
        b[P[0] + sr][P[1] + sc] = owner
    for dr, dc in OTHER_STONES:
        sr, sc = sym(dr, dc, k)
        b[P[0] + sr][P[1] + sc] = other
    return b, P


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\My PC\Desktop\final\10x10\model\best_mcts_11x11.pth"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroTicTacToeV3()
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.to(dev).eval()

    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(bb, pp) for bb, pp in zip(bs, ps)])).float().to(dev)
        with torch.inference_mode():
            pl, v = model(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)

    def num(b):
        return np.array([[1 if x == 'X' else 2 if x == 'O' else 0 for x in row] for row in b], dtype=np.int8)

    rows = []
    for shape in SHAPES:
        for k in (0, 1, 5, 6):
            for mode in ("attack", "defense"):
                owner = 'X'
                b, P = build(shape, k, owner)
                mover = owner if mode == "attack" else 'O'
                if mode == "defense":
                    # defender to move: give the owner one extra far stone so counts stay legal
                    pass
                if mode == "attack":
                    correct = set(fork_moves(b, owner))
                else:
                    correct = set()
                    for r in range(N):
                        for c in range(N):
                            if b[r][c] != ' ':
                                continue
                            b[r][c] = mover
                            if not fork_moves(b, owner):
                                correct.add((r, c))
                            b[r][c] = ' '
                if not correct or (mode == "attack" and P not in correct):
                    rows.append((shape, k, mode, None, None, None, None))
                    continue
                probs, val = ev([num(b)], [mover])
                probs = probs[0]
                order = np.argsort(-probs)
                best_rank = min(int(np.where(order == r * N + c)[0][0]) + 1 for (r, c) in correct)
                eng = mcts_core.MCTSEngine()
                _, rv, cands = eng.search_batched(b, mover, ev, SIMS, 1.414, False, 0.3, 0.25, N, WIN_LENGTH, 16)
                top = tuple(max(cands, key=lambda x: x['visit_count'])['move'])
                rows.append((shape, k, mode, top in correct, best_rank, float(val[0]), float(rv)))

    for mode in ("attack", "defense"):
        print(f"\n=== {mode.upper()} ===")
        tot = ok = 0
        for shape in SHAPES:
            rs = [r for r in rows if r[0] == shape and r[2] == mode and r[3] is not None]
            if not rs:
                print(f"  {shape:<20s}: (no valid positions)")
                continue
            n_ok = sum(r[3] for r in rs)
            tot += len(rs); ok += n_ok
            ranks = [r[4] for r in rs]
            vals = [r[5] for r in rs]
            print(f"  {shape:<20s}: MCTS {n_ok}/{len(rs)} | best correct-move prior rank {ranks} | raw value (mover) {np.mean(vals):+.2f}")
        print(f"  TOTAL {mode}: {ok}/{tot} = {ok / max(tot, 1):.0%}")


if __name__ == "__main__":
    main()
