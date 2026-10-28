"""Distillation data for small-net training (11x11_v2, 2026-10-21, user: "more and better data").

Plays many games at once with the teacher's raw policy (no search) and records every position with the
teacher's outputs as labels. Diversity: 0-4 random opening stones near the centre, policy sampling at
temperature 1, a small share of uniformly random legal moves, and exact-duplicate positions dropped.
Each record keeps its game id, so the test split can hold out whole games.
Output: model/distill_data.npz with
  boards  int8   (N, 121)   0 empty / 1 X / 2 O
  player  int8   (N,)       side to move (1 X / 2 O)
  logp    float16 (N, 121)  teacher log-policy
  value   float16 (N,)      teacher value (side to move)
  game    int32  (N,)
Usage: python distill_data_gen.py [n_positions] [teacher.pth]
"""
import os
import sys
import time

import numpy as np
import torch

import mcts_core
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3 as Net
from fast_board import BOARD_SIZE as N, WIN_LENGTH

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
TEACHER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "model", "best_mcts_11x11.pth")
PARALLEL = 768          # games advanced together (one teacher batch per ply)
RANDOM_MOVE = 0.03      # share of uniformly random legal moves
NC = N * N
dev = torch.device("cuda")
rng = np.random.default_rng(int(time.time()))

net = Net()
net.load_state_dict(torch.load(TEACHER, map_location=dev))
net.to(dev).eval()


def five_through(b, i, p):
    r0, c0 = divmod(i, N)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            r, c = r0 + s * dr, c0 + s * dc
            while 0 <= r < N and 0 <= c < N and b[r * N + c] == p:
                cnt += 1
                r += s * dr
                c += s * dc
        if cnt >= WIN_LENGTH:
            return True
    return False


def new_game():
    b = np.zeros(NC, np.int8)
    p = 1
    for _ in range(rng.integers(0, 5)):
        while True:
            r, c = rng.integers(2, N - 2, size=2)
            if b[r * N + c] == 0:
                break
        b[r * N + c] = p
        p = 3 - p
    return b, p


games = [new_game() for _ in range(PARALLEL)]
gid = list(range(PARALLEL))
next_gid = PARALLEL
seen = set()
out_b, out_p, out_lp, out_v, out_g = [], [], [], [], []
t0 = time.time()
while len(out_b) < TARGET:
    planes = np.stack([mcts_core.encode_planes(b.reshape(N, N), "X" if p == 1 else "O", N, WIN_LENGTH)
                       for b, p in games])
    with torch.inference_mode():
        lp, v = net(torch.from_numpy(planes).to(dev))
    lp = lp.float().cpu().numpy()
    v = v.float().view(-1).cpu().numpy()
    for k, (b, p) in enumerate(games):
        key = b.tobytes() + bytes([p])
        if key not in seen:
            seen.add(key)
            out_b.append(b.copy()); out_p.append(p); out_lp.append(lp[k].astype(np.float16))
            out_v.append(np.float16(v[k])); out_g.append(gid[k])
        empty = np.flatnonzero(b == 0)
        if rng.random() < RANDOM_MOVE:
            a = int(rng.choice(empty))
        else:
            pr = np.exp(lp[k][empty].astype(np.float64))
            pr /= pr.sum()
            a = int(rng.choice(empty, p=pr))
        b[a] = p
        if five_through(b, a, p) or len(empty) == 1:
            games[k] = new_game()
            gid[k] = next_gid
            next_gid += 1
        else:
            games[k] = (b, 3 - p)
    if len(out_b) % 50000 < PARALLEL:
        print(f"{len(out_b):,} positions, {next_gid:,} games, {time.time() - t0:.0f}s", flush=True)

np.savez(os.path.join(HERE, "model", "distill_data.npz"), boards=np.stack(out_b), player=np.array(out_p, np.int8),
         logp=np.stack(out_lp), value=np.array(out_v, np.float16), game=np.array(out_g, np.int32))
print(f"saved {len(out_b):,} positions from {next_gid:,} games in {time.time() - t0:.0f}s")
