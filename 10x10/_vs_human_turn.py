"""
Turn-by-turn human-vs-AI driver for chat-based play (no interactive stdin).
State is persisted to _vs_human_state.json between invocations. Adapted
from the 5x5 project's _vs_human_turn.py for the 11x11 board.

Usage:
    python _vs_human_turn.py new [X|O] [best|live]  # start a new game; human plays the given side (default X),
                                                      # vs the gated best checkpoint or the live training checkpoint (default best)
    python _vs_human_turn.py move <r> <c>            # human plays at 1-indexed row,col; AI replies if game continues

Runs on GPU (falls back to CPU if unavailable) for fast interactive replies;
this does share the GPU with the live trainer, so moves may be a bit slower
during heavy training phases. The live checkpoint is being overwritten
continuously by the trainer, so it is copied to a snapshot file first to
avoid reading a half-written file.
"""
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from alphago_lee import AlphaZeroTicTacToe
from alphazero_mcts import MCTS, BOARD_SIZE, fast_check_win, fast_check_draw
from alphazero_mcts_make import sims_for_state, BEST_CHECKPOINT_PATH, _SUFFIX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SIMULATIONS = 600
BOOSTED_SIMS = 1200
BOOST_THRESHOLD = 5
BOOST_WHEN_NARROW = True
C_PUCT = 1.6
STATE_PATH = "_vs_human_state.json"
LIVE_CHECKPOINT_PATH = f"model/alphazero_mcts{_SUFFIX}.pth"

_model = None


def resolve_model_path(which):
    if which == "live":
        if not os.path.exists(LIVE_CHECKPOINT_PATH):
            print(f"Model checkpoint not found at {LIVE_CHECKPOINT_PATH}.")
            sys.exit(1)
        snapshot_path = "_vs_human_live_snapshot.pth"
        shutil.copy2(LIVE_CHECKPOINT_PATH, snapshot_path)
        return snapshot_path
    if not os.path.exists(BEST_CHECKPOINT_PATH):
        print(f"Model checkpoint not found at {BEST_CHECKPOINT_PATH}.")
        sys.exit(1)
    snapshot_path = "_vs_human_best_snapshot.pth"
    shutil.copy2(BEST_CHECKPOINT_PATH, snapshot_path)
    return snapshot_path


def get_model(model_path):
    global _model
    if _model is None:
        m = AlphaZeroTicTacToe()
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        _model = m
    return _model


def render_board(board):
    lines = ["     " + " ".join(f"{c + 1:2d}" for c in range(BOARD_SIZE))]
    for r in range(BOARD_SIZE):
        row_str = " ".join(f" {board[r][c]}" if board[r][c] != ' ' else " ." for c in range(BOARD_SIZE))
        lines.append(f"{r + 1:3d}: {row_str}")
    return "\n".join(lines)


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def load_state():
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_status(board):
    if fast_check_win(board, 'X'):
        return 'X'
    if fast_check_win(board, 'O'):
        return 'O'
    if fast_check_draw(board):
        return 'draw'
    return None


def ai_move(board, ai_player, model_path):
    mcts = MCTS(c_puct=C_PUCT)
    num_empty = sum(1 for row in board for cell in row if cell == ' ')
    sims = sims_for_state(num_empty, NUM_SIMULATIONS, BOOSTED_SIMS, BOOST_THRESHOLD, BOOST_WHEN_NARROW)
    root = mcts.search(board, ai_player, get_model(model_path), DEVICE, num_simulations=sims, add_noise=False)
    if not root.children:
        return None, sims
    best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
    return divmod(best_action, BOARD_SIZE), sims


def cmd_new(human_player, which):
    ai_player = 'O' if human_player == 'X' else 'X'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    model_path = resolve_model_path(which)
    state = {"board": board, "human_player": human_player, "ai_player": ai_player,
              "current_player": current_player, "over": False, "result": None,
              "model_path": model_path, "which": which}

    print(f"새 게임 시작. 당신: {human_player} / AI: {ai_player} (모델: {which}, {model_path})")

    if current_player == ai_player:
        move, sims = ai_move(board, ai_player, model_path)
        r, c = move
        board[r][c] = current_player
        print(f"AI 생각 중... ({sims} simulations)")
        print(f"AI가 ({r + 1},{c + 1})에 두었습니다.")
        current_player = 'O' if current_player == 'X' else 'X'
        state["board"] = board
        state["current_player"] = current_player

    result = check_status(state["board"])
    if result:
        state["over"] = True
        state["result"] = result

    save_state(state)
    print(render_board(state["board"]))
    if state["over"]:
        print(status_message(state))


def status_message(state):
    result = state["result"]
    if result == "draw":
        return "\n무승부입니다."
    winner_is_human = (result == state["human_player"])
    return f"\n{result} 승리! " + ("축하합니다!" if winner_is_human else "AI가 이겼습니다.")


def cmd_move(r1, c1):
    state = load_state()
    if state["over"]:
        print("게임이 이미 종료되었습니다. 'new'로 새 게임을 시작하세요.")
        print(render_board(state["board"]))
        print(status_message(state))
        return

    board = state["board"]
    r, c = r1 - 1, c1 - 1
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        print(f"1부터 {BOARD_SIZE} 사이의 값으로 입력하세요.")
        return
    if board[r][c] != ' ':
        print("이미 돌이 놓인 칸입니다.")
        return
    if state["current_player"] != state["human_player"]:
        print("지금은 당신 차례가 아닙니다.")
        return

    board[r][c] = state["current_player"]
    state["current_player"] = 'O' if state["current_player"] == 'X' else 'X'

    result = check_status(board)
    if result:
        state["over"] = True
        state["result"] = result
        save_state(state)
        print(render_board(board))
        print(status_message(state))
        return

    move, sims = ai_move(board, state["ai_player"], state["model_path"])
    if move is None:
        state["over"] = True
        state["result"] = "draw"
        save_state(state)
        print(render_board(board))
        print(status_message(state))
        return
    print(f"AI 생각 중... ({sims} simulations)")
    ar, ac = move
    board[ar][ac] = state["current_player"]
    print(f"AI가 ({ar + 1},{ac + 1})에 두었습니다.")
    state["current_player"] = 'O' if state["current_player"] == 'X' else 'X'

    result = check_status(board)
    if result:
        state["over"] = True
        state["result"] = result

    state["board"] = board
    save_state(state)
    print(render_board(board))
    if state["over"]:
        print(status_message(state))


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python _vs_human_turn.py new [X|O]  |  python _vs_human_turn.py move <r> <c>")
        sys.exit(1)
    if args[0] == "new":
        human_player = args[1].upper() if len(args) > 1 else "X"
        which = args[2].lower() if len(args) > 2 else "best"
        cmd_new(human_player, which)
    elif args[0] == "move":
        r1, c1 = int(args[1]), int(args[2])
        cmd_move(r1, c1)
    else:
        print("Unknown command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
