"""
Round-robin between checkpoints to check whether the chained gate-Elo
estimate (elo_trend.py) matches real head-to-head results.

Players: supervised Gen12 start (ep0), ep50, ep150, current best.
Each pair plays N_OPENINGS random openings x 2 colors. Openings are a
random X first move near center plus a random adjacent O reply, so
deterministic (no-noise) search doesn't replay one identical game.
Ratings are fit by Bradley-Terry maximum likelihood, anchored ep0 = 1500.

Read-only on checkpoints. Writes model/elo_roundrobin_results.json.

Usage:  python elo_roundrobin.py
"""
import sys
import json
import math
import random
import itertools

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\11x11_v2")
sys.path.append(r"C:\Users\My PC\Desktop\final\cpp_mcts")

import torch
import numpy as np

import mcts_core_verbose
from alphago_lee_v2_seblock import AlphaZeroTicTacToeV2 as AlphaZeroTicTacToe
from alphago_lee import preferred_memory_format, board_to_input_array_from_numeric
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

MODEL_DIR = r"C:\Users\My PC\Desktop\final\11x11_v2\model"
PLAYERS = {
    "sup_ep0": MODEL_DIR + r"\model_MCTS_ep0_11x11.pth",
    "ep50": MODEL_DIR + r"\model_MCTS_ep50_11x11.pth",
    "ep150": MODEL_DIR + r"\model_MCTS_ep150_11x11.pth",
    "best": MODEL_DIR + r"\best_mcts_11x11.pth",
}
N_OPENINGS = 6
SIMS = 400
C_PUCT = 1.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    m = AlphaZeroTicTacToe()
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device, memory_format=preferred_memory_format(device))
    m.eval()
    return m


models = {name: load(p) for name, p in PLAYERS.items()}


def make_eval(model):
    def eval_batch_fn(boards, players):
        inputs = [board_to_input_array_from_numeric(b, p) for b, p in zip(boards, players)]
        t = torch.from_numpy(np.stack(inputs)).float().to(device).contiguous(
            memory_format=preferred_memory_format(device))
        with torch.inference_mode():
            logits, v = model(t)
        probs = torch.exp(logits).cpu().numpy().astype(np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        return probs, v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
    return eval_batch_fn


evals = {name: make_eval(m) for name, m in models.items()}

rng = random.Random(20260925)
openings = []
while len(openings) < N_OPENINGS:
    xr, xc = rng.randint(4, 6), rng.randint(4, 6)
    dr, dc = rng.choice([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    op = ((xr, xc), (xr + dr, xc + dc))
    if op not in openings:
        openings.append(op)


def play(x_name, o_name, opening):
    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    (xr, xc), (orr, oc) = opening
    board[xr][xc] = 'X'
    board[orr][oc] = 'O'
    turn = 'X'
    engines = {'X': mcts_core_verbose.MCTSEngine(), 'O': mcts_core_verbose.MCTSEngine()}
    names = {'X': x_name, 'O': o_name}
    empty = BOARD_SIZE * BOARD_SIZE - 2
    while empty > 0:
        _, _, cands = engines[turn].search_batched(
            board, turn, evals[names[turn]], SIMS, C_PUCT, False, 0.3, 0.25,
            BOARD_SIZE, WIN_LENGTH, 16, 0)
        r, c = max(cands, key=lambda d: d['visit_count'])['move']
        board[r][c] = turn
        empty -= 1
        if fast_check_win(board, turn):
            return turn
        turn = 'O' if turn == 'X' else 'X'
    return 'D'


# score[a][b] = points a earned vs b
score = {a: {b: 0.0 for b in PLAYERS} for a in PLAYERS}
games = {a: {b: 0 for b in PLAYERS} for a in PLAYERS}
log = []
pairs = list(itertools.combinations(PLAYERS, 2))
total = len(pairs) * N_OPENINGS * 2
gno = 0
for a, b in pairs:
    for op in openings:
        for x_name, o_name in ((a, b), (b, a)):
            gno += 1
            res = play(x_name, o_name, op)
            if res == 'D':
                score[x_name][o_name] += 0.5
                score[o_name][x_name] += 0.5
                winner = "draw"
            else:
                w = x_name if res == 'X' else o_name
                l = o_name if res == 'X' else x_name
                score[w][l] += 1.0
                winner = w
            games[x_name][o_name] += 1
            games[o_name][x_name] += 1
            log.append({"X": x_name, "O": o_name, "opening": op, "result": res})
            print(f"[{gno}/{total}] X={x_name:8s} O={o_name:8s} opening={op} -> {winner}", flush=True)

print("\nHead-to-head (row's score vs column, out of games played):")
names = list(PLAYERS)
print(" " * 10 + "".join(f"{n:>10s}" for n in names))
for a in names:
    row = "".join(f"{'-':>10s}" if a == b else f"{score[a][b]:>5.1f}/{games[a][b]:<4d}" for b in names)
    print(f"{a:>10s}{row}")

# Bradley-Terry MLE (MM algorithm) with a tiny prior to keep 0%/100% finite
strength = {n: 1.0 for n in names}
for _ in range(2000):
    new = {}
    for i in names:
        wins_i = sum(score[i][j] for j in names if j != i) + 0.5
        denom = sum((games[i][j] + 1) / (strength[i] + strength[j]) for j in names if j != i)
        new[i] = wins_i / denom
    g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
    strength = {k: v / g for k, v in new.items()}
elo = {n: 400 * math.log10(strength[n]) for n in names}
base = elo["sup_ep0"]
elo = {n: 1500 + e - base for n, e in elo.items()}

print("\nRound-robin Elo (Bradley-Terry, sup_ep0 anchored at 1500):")
for n in sorted(names, key=lambda k: elo[k]):
    print(f"  {n:>8s}: {elo[n]:7.1f}")

with open(MODEL_DIR + r"\elo_roundrobin_results.json", "w", encoding="utf-8") as f:
    json.dump({"elo": elo, "score": score, "games": games, "log": log,
               "sims": SIMS, "openings": openings}, f, indent=1)
print("\nSaved model/elo_roundrobin_results.json\nDONE")
