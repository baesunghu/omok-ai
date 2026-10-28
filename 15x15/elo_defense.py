"""
Strength comparison that factors out the first-player advantage.

elo_roundrobin.py showed X (first player) won 68/72 games regardless of
which checkpoint played it, so win/loss alone can't separate models.
Instead, every ordered pair (attacker as X, defender as O) plays the same
set of openings, and we record:

  - defender result (O win / draw / loss) -- rare but decisive signal
  - game length in plies -- a stronger defender survives longer before
    losing; a stronger attacker wins faster

Per model we report: as defender, average plies survived and O
wins/draws; as attacker, average plies to win.

Read-only on checkpoints. Writes model/elo_defense_results.json.

Usage:  python elo_defense.py
"""
import sys
import json
import random
import itertools

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\15x15")
sys.path.append(r"C:\Users\My PC\Desktop\final\cpp_mcts")

import torch
import numpy as np

import mcts_core_verbose
from alphago_lee_v2_seblock import AlphaZeroTicTacToeV2 as AlphaZeroTicTacToe
from alphago_lee import preferred_memory_format, board_to_input_array_from_numeric
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

MODEL_DIR = r"C:\Users\My PC\Desktop\final\15x15\model"
PLAYERS = {
    "sup_ep0": MODEL_DIR + r"\model_MCTS_ep0_11x11.pth",
    "ep50": MODEL_DIR + r"\model_MCTS_ep50_11x11.pth",
    "ep150": MODEL_DIR + r"\model_MCTS_ep150_11x11.pth",
    "best": MODEL_DIR + r"\best_mcts_11x11.pth",
}
N_OPENINGS = 8
SIMS = 400
C_PUCT = 1.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    m = AlphaZeroTicTacToe()
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device, memory_format=preferred_memory_format(device))
    m.eval()
    return m


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


evals = {name: make_eval(load(p)) for name, p in PLAYERS.items()}

rng = random.Random(20260926)
openings = []
while len(openings) < N_OPENINGS:
    xr, xc = rng.randint(3, 7), rng.randint(3, 7)
    dr, dc = rng.choice([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    op = ((xr, xc), (xr + dr, xc + dc))
    if op not in openings:
        openings.append(op)


def play(x_name, o_name, opening):
    """Returns (result 'X'/'O'/'D', plies played after the opening)."""
    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    (xr, xc), (orr, oc) = opening
    board[xr][xc] = 'X'
    board[orr][oc] = 'O'
    turn = 'X'
    engines = {'X': mcts_core_verbose.MCTSEngine(), 'O': mcts_core_verbose.MCTSEngine()}
    names = {'X': x_name, 'O': o_name}
    empty = BOARD_SIZE * BOARD_SIZE - 2
    plies = 0
    while empty > 0:
        _, _, cands = engines[turn].search_batched(
            board, turn, evals[names[turn]], SIMS, C_PUCT, False, 0.3, 0.25,
            BOARD_SIZE, WIN_LENGTH, 16, 0)
        r, c = max(cands, key=lambda d: d['visit_count'])['move']
        board[r][c] = turn
        empty -= 1
        plies += 1
        if fast_check_win(board, turn):
            return turn, plies
        turn = 'O' if turn == 'X' else 'X'
    return 'D', plies


log = []
pairs = list(itertools.permutations(PLAYERS, 2))
total = len(pairs) * N_OPENINGS
gno = 0
for x_name, o_name in pairs:
    for op in openings:
        gno += 1
        res, plies = play(x_name, o_name, op)
        log.append({"X": x_name, "O": o_name, "opening": op, "result": res, "plies": plies})
        print(f"[{gno}/{total}] X={x_name:8s} O={o_name:8s} -> {res} in {plies} plies", flush=True)

names = list(PLAYERS)
print("\n=== As DEFENDER (O) ===")
print(f"{'model':>8s} {'O wins':>7s} {'draws':>6s} {'avg plies survived (losses)':>28s}")
for n in names:
    g = [e for e in log if e["O"] == n]
    ow = sum(e["result"] == "O" for e in g)
    dr = sum(e["result"] == "D" for e in g)
    lost = [e["plies"] for e in g if e["result"] == "X"]
    avg = sum(lost) / len(lost) if lost else float("nan")
    print(f"{n:>8s} {ow:>4d}/{len(g):<2d} {dr:>6d} {avg:>28.1f}")

print("\n=== As ATTACKER (X) ===")
print(f"{'model':>8s} {'X wins':>7s} {'avg plies to win':>17s}")
for n in names:
    g = [e for e in log if e["X"] == n]
    xw = [e["plies"] for e in g if e["result"] == "X"]
    avg = sum(xw) / len(xw) if xw else float("nan")
    print(f"{n:>8s} {len(xw):>4d}/{len(g):<2d} {avg:>17.1f}")

print("\n=== Defender survival matrix (avg plies before loss; row=defender, col=attacker) ===")
print(" " * 9 + "".join(f"{n:>9s}" for n in names))
for d in names:
    cells = []
    for a in names:
        if a == d:
            cells.append(f"{'-':>9s}")
            continue
        lost = [e["plies"] for e in log if e["O"] == d and e["X"] == a and e["result"] == "X"]
        cells.append(f"{(sum(lost) / len(lost) if lost else float('nan')):>9.1f}")
    print(f"{d:>9s}" + "".join(cells))

with open(MODEL_DIR + r"\elo_defense_results.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "sims": SIMS, "openings": openings}, f, indent=1)
print("\nSaved model/elo_defense_results.json\nDONE")
