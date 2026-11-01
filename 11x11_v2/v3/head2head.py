"""Head-to-head between two v3 checkpoints (2026-10-23, user: did ep120 drift by over-fitting the catfish?).

Each opening (first two stones fixed) is played twice with colours swapped, so 10 openings = 20 games.
Same noiseless PUCT search for both sides (1,100 sims, VCF on), like the Rapfi ladder.
Usage: python head2head.py A.pth B.pth [sims]   (run from 11x11_v2/v3)
A or B may also be a 15x15 V4 checkpoint (board-size-independent heads), played here on 11x11.
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
if os.environ.get("H2H_ALL"):    # 2026-10-31 (user: catfish A/B at ep30/60): 100 distinct openings = 200 games, the 10 above first
    OPENINGS = OPENINGS + [((r, c), (r + dr, c + dc)) for r in (4, 5, 6) for c in (4, 5, 6)
                           for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)
                           and ((r, c), (r + dr, c + dc)) not in OPENINGS][:90]
if os.environ.get("H2H_SHARD"):  # "k/n": play only openings k, k+n, k+2n, ... (parallel processes)
    _k, _n = map(int, os.environ["H2H_SHARD"].split("/"))
    OPENINGS = OPENINGS[_k::_n]
mcts_core.set_vcf(10, 3000)


def load(p):
    sd = torch.load(p, map_location=DEV)
    if "model" in sd:                       # a *_latest.pth training checkpoint
        sd = sd["model"]
    if "policy_head.out.weight" in sd:      # 2026-10-30 (user): 15x15 V4 net played on the 11x11 board
        sys.path.insert(1, os.path.join(os.path.dirname(V2), "15x15_v3"))
        from net_v4 import NetV4, upgrade_adj_head
        m = NetV4(board_size=N); upgrade_adj_head(sd, m)
    else:
        m = NetV3()
    m.load_state_dict(sd); return m.to(DEV).eval()


def make_ev(m):
    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(DEV)
        with torch.inference_mode():
            pl, v = m(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
    return ev


LIVE = None   # 2026-10-31 (user: 관전): sharded runs write the game in progress for ab_scratch/ab_live.py
if os.environ.get("H2H_SHARD"):
    LIVE = os.path.join(V2, "v3", "ab_scratch", "live", f"shard{os.environ['H2H_SHARD'].split('/')[0]}.json")
    os.makedirs(os.path.dirname(LIVE), exist_ok=True)


def live(info, moves):
    if LIVE:   # best effort: a viewer reading the file can make os.replace fail on Windows (11-01: crashed 2 shards)
        import json
        try:
            with open(LIVE + ".tmp", "w", encoding="utf-8") as f:
                json.dump(dict(info, moves=moves, time=__import__("time").time()), f)
            os.replace(LIVE + ".tmp", LIVE)
        except OSError:
            pass


def play(evx, evo, opening, sims, info=None):
    board = [[' '] * N for _ in range(N)]
    turn = 'X'
    moves = []
    for r, c in opening:
        board[r][c] = turn; moves.append([r, c]); turn = 'O' if turn == 'X' else 'X'
    live(info or {}, moves)
    for _ in range(N * N - len(opening)):
        ev = evx if turn == 'X' else evo
        _, _, cands = mcts_core.MCTSEngine().search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, 5, 16)
        r, c = max(cands, key=lambda x: x["visit_count"])["move"]
        board[r][c] = turn
        moves.append([r, c]); live(info or {}, moves)
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
            info = {"a": os.path.basename(os.path.dirname(os.path.dirname(a))) + "/" + os.path.basename(a),
                    "b": os.path.basename(os.path.dirname(os.path.dirname(b))) + "/" + os.path.basename(b),
                    "a_is_x": a_is_x, "game": 2 * i + (0 if a_is_x else 1) + 1, "of": 2 * len(OPENINGS)}
            win = play(eva, evb, op, sims, info) if a_is_x else play(evb, eva, op, sims, info)
            res = "draw" if win is None else ("A" if (win == 'X') == a_is_x else "B")
            w += res == "A"; l += res == "B"; d += res == "draw"
            print(f"game {2 * i + (0 if a_is_x else 1) + 1}: opening {op} A={'X' if a_is_x else 'O'} -> {res}", flush=True)
    n = w + l + d
    print(f"\nA={os.path.basename(a)} vs B={os.path.basename(b)}: A {w} / B {l} / draw {d}, A score {(w + 0.5 * d) / n:.1%}")


if __name__ == "__main__":
    main()
