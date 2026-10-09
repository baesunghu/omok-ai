"""
One-shot combined status check: general training status (status_check.py)
+ Hard-7 (double-four fork) adaptation rate + open-three organic-miss
trend (from track_open3_progress.py's own log) + the open-3-3 (T-shape)
fork probe -- all in a single run, so it can be launched standalone
without a live Claude session.

Usage (from a PowerShell prompt):

    cd "C:\\Users\\My PC\\Desktop\\final\\10x10"
    python fork_status_check.py

Read-only everywhere: reads the live checkpoint, the hard-mined pool, and
the open-three progress log; writes nothing.
"""
import sys
import os
import random
import pickle
import subprocess

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\10x10")
sys.path.append(r"C:\Users\My PC\Desktop\final\cpp_mcts")

import torch
import numpy as np

print("=" * 70)
print("1) GENERAL TRAINING STATUS (status_check.py)")
print("=" * 70)
try:
    result = subprocess.run(
        [sys.executable, "status_check.py", "10x10/model/training_report_11x11.txt"],
        cwd=r"C:\Users\My PC\Desktop\final",
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)
except Exception as e:
    print(f"status_check.py failed to run: {e}")

print("\n" + "=" * 70)
print("Loading live checkpoint (shared by the fork checks below)...")
print("=" * 70)

import mcts_core as mcts_core_verbose  # production engine (virtual-loss fix + mate distance)
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3 as AlphaZeroTicTacToe
from alphago_lee import preferred_memory_format, board_to_input_array_from_numeric
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win
import alphazero_mcts_make_lrfix as amk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"C:\Users\My PC\Desktop\final\10x10\model\best_mcts_11x11.pth"
model = AlphaZeroTicTacToe()
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device, memory_format=preferred_memory_format(device))
model.eval()
print("loaded.\n")


def eval_batch_fn(boards, players):
    inputs = [board_to_input_array_from_numeric(b, p) for b, p in zip(boards, players)]
    batch = np.stack(inputs)
    tensor = torch.from_numpy(batch).float().to(device).contiguous(memory_format=preferred_memory_format(device))
    with torch.inference_mode():
        policy_logits, value_tensor = model(tensor)
    probs = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    values = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
    return probs, values


SUMMARY = {}

print("=" * 70)
print("1b) LEVEL-1: block an immediate four (1-ply defense)")
print("=" * 70)
try:
    rng = random.Random(7)
    dirs4 = [(0, 1), (1, 0), (1, 1), (1, -1)]
    tested = ok = 0
    tries = 0
    while tested < 20 and tries < 2000:
        tries += 1
        dr, dc = rng.choice(dirs4)
        r0, c0 = rng.randrange(BOARD_SIZE), rng.randrange(BOARD_SIZE)
        line = [(r0 + k * dr, c0 + k * dc) for k in range(5)]
        if not all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE for r, c in line):
            continue
        gap = rng.randrange(5)
        board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for k, (r, c) in enumerate(line):
            if k != gap:
                board[r][c] = 'O'
        # a few scattered stones for both sides (equal-ish counts)
        empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if (r, c) not in line]
        rng.shuffle(empties)
        for (r, c) in empties[:5]:
            board[r][c] = 'X'
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        # X must not have its own immediate win; collect O's winning cells
        x_wins = o_wins = []
        o_wins = []
        x_can_win = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != ' ':
                    continue
                board[r][c] = 'X'
                if fast_check_win(board, 'X'):
                    x_can_win = True
                board[r][c] = 'O'
                if fast_check_win(board, 'O'):
                    o_wins.append((r, c))
                board[r][c] = ' '
        if x_can_win or len(o_wins) != 1:
            continue
        engine = mcts_core_verbose.MCTSEngine()
        _, rv, cands = engine.search_batched(
            board, 'X', eval_batch_fn, 1100, 1.414, False, 0.3, 0.25,
            BOARD_SIZE, WIN_LENGTH, 16,
        )
        top = tuple(max(cands, key=lambda c: c['visit_count'])['move'])
        tested += 1
        ok += int(top in o_wins)
    print(f"LEVEL-1 BLOCK RATE: {ok}/{tested} = {ok/max(tested,1):.1%}")
    SUMMARY['L1'] = f"{ok}/{tested} = {ok/max(tested,1):.0%}"
except Exception as e:
    print(f"Level-1 check failed: {e}")

print("=" * 70)
print("2) HARD-7 (double-four fork) ADAPTATION RATE -- real mined pool")
print("=" * 70)
try:
    with open(r"C:\Users\My PC\Desktop\final\10x10\model\hard_mined_examples_11x11.pkl", "rb") as f:
        pool = pickle.load(f)
    h7 = [item for item in pool if item[4] == 7]
    print(f"Hard-7 pool size: {len(h7)}")
    random.seed(42)
    SAMPLE_N = 20
    sample = random.sample(h7, min(SAMPLE_N, len(h7)))
    SIMS = 4000
    avoided = 0
    for i, (board, player, policy, value, hard_class) in enumerate(sample):
        other = 'O' if player == 'X' else 'X'
        engine = mcts_core_verbose.MCTSEngine()
        _, root_value, candidates = engine.search_batched(
            board, player, eval_batch_fn, SIMS, 1.414, False, 0.3, 0.25,
            BOARD_SIZE, WIN_LENGTH, 16,
        )
        top = max(candidates, key=lambda c: c['visit_count'])
        r, c = top['move']
        after = [row[:] for row in board]
        after[r][c] = player
        if fast_check_win(after, player):
            result = "AVOIDED (won outright)"
            avoided += 1
        elif amk._opponent_can_fork_next(after, other):
            result = "STILL FORKS"
        else:
            result = "AVOIDED"
            avoided += 1
        print(f"  [{i+1}/{len(sample)}] move=({r},{c}) root_value={root_value:.4f} -> {result}")
    print(f"\nADAPTATION RATE: {avoided}/{len(sample)} = {avoided/len(sample):.1%}")
    SUMMARY['L3_hard7'] = f"{avoided}/{len(sample)} = {avoided/len(sample):.0%}"
except Exception as e:
    print(f"Hard-7 check failed: {e}")

print("\n" + "=" * 70)
print("3) OPEN-THREE organic-miss trend (from track_open3_progress.py's own log)")
print("=" * 70)
try:
    log_path = r"C:\Users\My PC\Desktop\final\scratch\multiproc_investigation_multilabel_v1\open3_progress_log.jsonl"
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()[-8:]
    import json as _json
    _recs = [_json.loads(l) for l in lines]
    _mr = [r["miss_rate"] for r in _recs if r.get("miss_rate") is not None]
    if _mr:
        SUMMARY['L2'] = f"block {1-_mr[-1]:.0%} latest (ep{_recs[-1]['epoch']}), avg of last {len(_mr)} checks {1-sum(_mr)/len(_mr):.0%}"
    for l in lines:
        print("  " + l.strip())
except Exception as e:
    print(f"open-three log read failed: {e}")

print("\n" + "=" * 70)
print("4) OPEN-3-3 (T-shape) FORK PROBE")
print("=" * 70)
print("""    O _ O
      O
      O
""")
try:
    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    GAP = (5, 5)
    board[5][4] = 'O'
    board[5][6] = 'O'
    board[6][5] = 'O'
    board[7][5] = 'O'
    board[3][3] = 'X'
    board[8][8] = 'X'
    board[3][7] = 'X'

    def is_open_three(board, r0, c0, dr, dc, color):
        def count_dir(dr, dc):
            n = 0
            r, c = r0 + dr, c0 + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
                n += 1
                r, c = r + dr, c + dc
            end_open = 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == ' '
            return n, end_open
        n1, open1 = count_dir(dr, dc)
        n2, open2 = count_dir(-dr, -dc)
        return (1 + n1 + n2) == 3 and open1 and open2

    def o_can_make_33_fork(board):
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != ' ':
                    continue
                board[r][c] = 'O'
                if fast_check_win(board, 'O'):
                    board[r][c] = ' '
                    return True
                open3_dirs = sum(1 for (dr, dc) in dirs if is_open_three(board, r, c, dr, dc, 'O'))
                board[r][c] = ' '
                if open3_dirs >= 2:
                    return True
        return False

    safe_cells = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = 'X'
            still = o_can_make_33_fork(board)
            board[r][c] = ' '
            if not still:
                safe_cells.append((r, c))

    SIMS = 1100  # production budget
    engine = mcts_core_verbose.MCTSEngine()
    _, root_value, candidates = engine.search_batched(
        board, 'X', eval_batch_fn, SIMS, 1.6, False, 0.3, 0.25,
        BOARD_SIZE, WIN_LENGTH, 16,
    )
    top_move = tuple(max(candidates, key=lambda c: c['visit_count'])['move'])
    if top_move == GAP:
        verdict = "BEST"
    elif top_move in safe_cells:
        verdict = "OK"
    else:
        verdict = "WRONG"

    inp_num = np.array([[1 if v == 'X' else 2 if v == 'O' else 0 for v in row] for row in board], dtype=np.int8)
    inp = board_to_input_array_from_numeric(inp_num, 'X')
    tensor = torch.from_numpy(inp).float().unsqueeze(0).to(device, memory_format=preferred_memory_format(device))
    with torch.inference_mode():
        policy_logits, value = model(tensor)
    probs = torch.exp(policy_logits).squeeze(0).cpu().numpy()
    gap_idx = GAP[0] * BOARD_SIZE + GAP[1]
    gap_prob = float(probs[gap_idx])
    rank = int(np.where(np.argsort(-probs) == gap_idx)[0][0]) + 1

    print(f"MCTS (SIMS={SIMS}): root_value={root_value:.4f} top_move={top_move} -> {verdict}")
    print(f"Raw policy prior: P(gap)={gap_prob:.4f}, rank={rank}/121, raw_value={float(value.item()):.4f}")
    SUMMARY['L3_tshape'] = f"{verdict} (prior rank of gap {rank}/121)"
except Exception as e:
    print(f"T-shape probe failed: {e}")

print("\n" + "=" * 70)
print("5) ATTACK LADDER (attack_ladder.py)")
print("=" * 70)
ATTACK = {}
try:
    from attack_ladder import run_attack_ladder

    def prior_rank(board, cell):
        num = np.array([[1 if v == 'X' else 2 if v == 'O' else 0 for v in row] for row in board], dtype=np.int8)
        t = torch.from_numpy(board_to_input_array_from_numeric(num, 'X')).float().unsqueeze(0).to(
            device, memory_format=preferred_memory_format(device))
        with torch.inference_mode():
            pl, _ = model(t)
        pr = torch.exp(pl).squeeze(0).cpu().numpy()
        idx = cell[0] * BOARD_SIZE + cell[1]
        return int(np.where(np.argsort(-pr) == idx)[0][0]) + 1

    ATTACK = run_attack_ladder(mcts_core_verbose, eval_batch_fn, prior_rank)
except Exception as e:
    print(f"Attack ladder failed: {e}")

print("\n" + "=" * 70)
print("ATTACK LADDER SUMMARY")
print("=" * 70)
print(f"  A1 complete own four (1 ply)         : {ATTACK.get('A1', 'n/a')}")
print(f"  A2 open three -> open four (2-3 ply) : {ATTACK.get('A2', 'n/a')}")
print(f"  A3 make double-four fork             : {ATTACK.get('A3', 'n/a')}")
print(f"  A3 T-shape double open three         : {ATTACK.get('A3_tshape', 'n/a')}")

print("\n" + "=" * 70)
print("DEFENSE LADDER SUMMARY (low -> high reading depth)")
print("=" * 70)
print(f"  L1 block immediate four (1 ply)      : {SUMMARY.get('L1', 'n/a')}")
print(f"  L2 block open three (2-3 ply)        : {SUMMARY.get('L2', 'n/a')}")
print(f"  L3 avoid double-four fork (Hard-7)   : {SUMMARY.get('L3_hard7', 'n/a')}")
print(f"  L3 open-3-3 T-shape fork             : {SUMMARY.get('L3_tshape', 'n/a')}")
print("  (read bottom-up: a higher level only means something once the lower ones are stable)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
