"""
Same fair (no-noise, greedy, full-search) MCTS-vs-past-checkpoint eval as
5x5/eval_league_greedy.py, adapted for 10x10. Reduced games/opponent vs the
5x5 version (2 instead of 4) since 10x10's sims are 3x bigger (900/1800 vs
300/600) on a 4x larger board -- each game costs much more wall-clock here.

Run from within this directory:
    python eval_league_greedy.py

Appends one summary line per run to model/greedy_league_eval_10x10.txt, which
status_check.py reads to show the latest fair-eval result alongside the
regular (handicapped) training-time league win rate.
"""
import glob
import os
import re
from datetime import datetime

import torch

from alphago_lee import AlphaZeroTicTacToe
from alphazero_mcts import MCTS, BOARD_SIZE, fast_check_win, fast_check_draw
from alphazero_mcts_make import sims_for_state, _SUFFIX, REPORT_LOG_PATH

DEVICE = "cpu"  # avoid contending with the live GPU trainers
NUM_SIMULATIONS = 900
BOOSTED_SIMS = 1800
BOOST_THRESHOLD = 5
BOOST_WHEN_NARROW = True
C_PUCT = 3.2
POOL_SIZE = 20
GAMES_PER_OPPONENT = 2  # 1 as X, 1 as O -- reduced for 10x10's much higher per-move cost
NUM_OPPONENTS = 5

GREEDY_LOG_PATH = f"model/greedy_league_eval{_SUFFIX}.txt"
LAST_EPOCH_RE = re.compile(r"Epoch (\d+)/\d+ \|")


def latest_epoch():
    if not os.path.exists(REPORT_LOG_PATH):
        return None
    last = None
    with open(REPORT_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            m = LAST_EPOCH_RE.search(line)
            if m:
                last = int(m.group(1))
    return last


def load_model(path):
    m = AlphaZeroTicTacToe()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


def play_one(live_model, opp_model, live_plays_x):
    live_player = 'X' if live_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    live_mcts = MCTS(c_puct=C_PUCT)
    opp_mcts = MCTS(c_puct=C_PUCT)

    while True:
        if fast_check_win(board, live_player):
            return 'live'
        if fast_check_win(board, 'O' if live_player == 'X' else 'X'):
            return 'opp'
        if fast_check_draw(board):
            return 'draw'

        num_empty = sum(1 for row in board for cell in row if cell == ' ')
        sims = sims_for_state(num_empty, NUM_SIMULATIONS, BOOSTED_SIMS, BOOST_THRESHOLD, BOOST_WHEN_NARROW)

        if current_player == live_player:
            root = live_mcts.search(board, current_player, live_model, DEVICE,
                                     num_simulations=sims, add_noise=False)
        else:
            root = opp_mcts.search(board, current_player, opp_model, DEVICE,
                                    num_simulations=sims, add_noise=False)

        if not root.children:
            return 'draw'
        best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        r, c = divmod(best_action, BOARD_SIZE)
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


def main():
    live_model = load_model(f"model/alphazero_mcts{_SUFFIX}.pth")

    pattern = f"model/model_MCTS_ep*{_SUFFIX}.pth"
    snapshots = sorted(glob.glob(pattern), key=os.path.getmtime)
    pool = snapshots[-POOL_SIZE:]
    if len(pool) <= NUM_OPPONENTS:
        chosen = pool
    else:
        idxs = [round(i * (len(pool) - 1) / (NUM_OPPONENTS - 1)) for i in range(NUM_OPPONENTS)]
        chosen = [pool[i] for i in sorted(set(idxs))]

    print(f"Live model: model/alphazero_mcts{_SUFFIX}.pth")
    print(f"Pool size on disk (last {POOL_SIZE}): {len(pool)}, testing against {len(chosen)} spread-out opponents")
    print(f"Settings: sims={NUM_SIMULATIONS} (boosted {BOOSTED_SIMS} at <={BOOST_THRESHOLD} empty), "
          f"c_puct={C_PUCT}, add_noise=False both sides, greedy argmax\n")

    total_w = total_d = total_l = 0
    for opp_path in chosen:
        opp_model = load_model(opp_path)
        w = d = l = 0
        for i in range(GAMES_PER_OPPONENT):
            live_plays_x = (i % 2 == 0)
            result = play_one(live_model, opp_model, live_plays_x)
            if result == 'live':
                w += 1
            elif result == 'draw':
                d += 1
            else:
                l += 1
        total_w += w
        total_d += d
        total_l += l
        print(f"  vs {os.path.basename(opp_path):32s} Win {w}/{GAMES_PER_OPPONENT}  Draw {d}/{GAMES_PER_OPPONENT}  Loss {l}/{GAMES_PER_OPPONENT}")

    n = total_w + total_d + total_l
    print(f"\nTotal: Win {total_w}/{n} ({total_w/n:.0%})  Draw {total_d}/{n} ({total_d/n:.0%})  Loss {total_l}/{n} ({total_l/n:.0%})")

    epoch = latest_epoch()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(GREEDY_LOG_PATH), exist_ok=True)
    with open(GREEDY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] epoch={epoch if epoch is not None else '?'} opponents={len(chosen)} games={n} "
                f"sims={NUM_SIMULATIONS}/{BOOSTED_SIMS} Win={total_w} Draw={total_d} Loss={total_l} "
                f"(Win {total_w/n:.1%} Draw {total_d/n:.1%} Loss {total_l/n:.1%})\n")
    print(f"\nAppended to {GREEDY_LOG_PATH}")


if __name__ == "__main__":
    main()
