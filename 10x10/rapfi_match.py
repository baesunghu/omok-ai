"""
Our model vs Rapfi (Gomocup 2024 champion engine), 11x11 freestyle.
MEASUREMENT ONLY (2026-09-28) -- nothing here feeds training.

Rapfi is driven through the Gomocup (pbrain) protocol and is sent the full
position with BOARD before each of its moves, so it is stateless between
moves. Our side: the tracked best checkpoint + the production engine
(mcts_core: solver build), no noise, greedy.

Each game starts from a fixed opening stone for X (cycling through a
spread-out list) so games differ; colors alternate game by game.

Usage: python rapfi_match.py [games] [our_sims] [rapfi_limit] [checkpoint]
  rapfi_limit: "2000" = 2000 ms per move, or "n10000" = 10,000 search nodes
  per move (machine-independent; the time cap is then 30 s as a safety net).
"""
import os
import subprocess
import sys
import time
import json

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\10x10")
os.chdir(r"C:\Users\My PC\Desktop\final\10x10")
import numpy as np
import torch

import mcts_core
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

N = BOARD_SIZE
RAPFI_DIR = r"C:\Users\My PC\Desktop\final\engines\rapfi"
RAPFI_EXE = os.path.join(RAPFI_DIR, "pbrain-rapfi-windows-avx2.exe")
OPENINGS = [(5, 5), (4, 5), (5, 4), (4, 4), (6, 4), (3, 5), (5, 3), (4, 6)]
LOG = r"C:\Users\My PC\Desktop\final\10x10\model\rapfi_match_log.jsonl"


class Rapfi:
    def __init__(self, ms, nodes=None):
        self.p = subprocess.Popen([RAPFI_EXE], cwd=RAPFI_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send(f"START {N}")
        self._wait_ok()
        for cmd in ("INFO rule 0", f"INFO timeout_turn {ms}", "INFO timeout_match 100000000", "INFO max_memory 0"):
            self.send(cmd)
        if nodes:
            self.send(f"INFO max_node {nodes}")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _readline(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("rapfi exited")
        return line.strip()

    def _wait_ok(self):
        while True:
            l = self._readline()
            if l == "OK":
                return
            if l.startswith("ERROR"):
                raise RuntimeError(l)

    def move(self, moves, me):
        """moves: list of (player, r, c) played so far; me: Rapfi's color."""
        self.send("BOARD")
        for pl, r, c in moves:
            self.send(f"{c},{r},{1 if pl == me else 2}")  # Gomocup: x=column, y=row
        self.send("DONE")
        while True:
            l = self._readline()
            if l.startswith(("MESSAGE", "DEBUG", "OK")) or not l:
                continue
            if l.startswith("ERROR"):
                raise RuntimeError(l)
            x, y = l.split(",")
            return int(y), int(x)

    def close(self):
        try:
            self.send("END")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
    lim = sys.argv[3] if len(sys.argv) > 3 else "2000"
    nodes = int(lim[1:]) if lim.startswith("n") else None
    ms = 30000 if nodes else int(lim)
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "model/best_mcts_11x11.pth"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroTicTacToeV3()
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.to(dev).eval()

    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
        with torch.inference_mode():
            pl, v = model(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)

    rapfi = Rapfi(ms, nodes)
    tally = {"ai_win": 0, "rapfi_win": 0, "draw": 0}
    by_color = {"X": [0, 0, 0], "O": [0, 0, 0]}  # ai as color: W/L/D
    try:
        for g in range(games):
            ai = 'X' if g % 2 == 0 else 'O'
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            board = [[' '] * N for _ in range(N)]
            moves = [('X', opening[0], opening[1])]
            board[opening[0]][opening[1]] = 'X'
            turn = 'O'
            engine = mcts_core.MCTSEngine()
            result = None
            t0 = time.time()
            while True:
                if len(moves) >= N * N:
                    result = "draw"
                    break
                if turn == ai:
                    _, rv, cands = engine.search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, WIN_LENGTH, 16)
                    r, c = max(cands, key=lambda x: x['visit_count'])['move']
                else:
                    r, c = rapfi.move(moves, turn)
                if board[r][c] != ' ':
                    raise RuntimeError(f"illegal move {(r, c)} by {'ai' if turn == ai else 'rapfi'}")
                board[r][c] = turn
                moves.append((turn, r, c))
                if fast_check_win(board, turn):
                    result = "ai_win" if turn == ai else "rapfi_win"
                    break
                turn = 'O' if turn == 'X' else 'X'
            tally[result] += 1
            idx = {"ai_win": 0, "rapfi_win": 1, "draw": 2}[result]
            by_color[ai][idx] += 1
            print(f"game {g + 1}/{games}: AI={ai} opening={opening} -> {result} in {len(moves)} plies "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ai": ai, "opening": opening,
                                    "result": result, "plies": len(moves), "sims": sims, "rapfi_ms": None if nodes else ms, "rapfi_nodes": nodes,
                                    "checkpoint": os.path.basename(ckpt),
                                    "moves": moves}) + "\n")
    finally:
        rapfi.close()
    print(f"\nTOTAL: AI {tally['ai_win']} / Rapfi {tally['rapfi_win']} / draw {tally['draw']}")
    print(f"AI as X (first): W{by_color['X'][0]} L{by_color['X'][1]} D{by_color['X'][2]} | "
          f"AI as O (second): W{by_color['O'][0]} L{by_color['O'][1]} D{by_color['O'][2]}")


if __name__ == "__main__":
    main()
