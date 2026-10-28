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
_MEGA_EVAL = tuple(int(x) for x in __import__('os').environ.get('MEGA_EVAL', '0,0').split(','))

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\11x11_v2")
os.chdir(r"C:\Users\My PC\Desktop\final\11x11_v2")
import numpy as np
import torch

import mcts_core
# 2026-10-23 (user): VCF on everywhere (was off: engine default depth 0); VCF=0 turns it off
if os.environ.get("VCF", "1") != "0":
    mcts_core.set_vcf(10, 3000)
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

N = BOARD_SIZE
RAPFI_DIR = r"C:\Users\My PC\Desktop\final\engines\rapfi"
RAPFI_EXE = os.path.join(RAPFI_DIR, "pbrain-rapfi-windows-avx2.exe")
OPENINGS = [(5, 5), (4, 5), (5, 4), (4, 4), (6, 4), (3, 5), (5, 3), (4, 6)]
# 2026-10-11: RAPFI_MORE_OPENINGS=1 -> all 25 cells within 2 of the centre (validation runs with >16 games,
# so games do not repeat; the default 8 openings keep the ladder identical to its history)
if os.environ.get("RAPFI_MORE_OPENINGS"):
    OPENINGS = OPENINGS + [(r, c) for r in range(3, 8) for c in range(3, 8) if (r, c) not in OPENINGS]
LOG = os.environ.get("RAPFI_MATCH_LOG", r"C:\Users\My PC\Desktop\final\11x11_v2\model\rapfi_match_log.jsonl")
# 2026-10-10: games are deterministic (Rapfi single-threaded + node limit, our side greedy without noise), so a
# level can be split across RAPFI_PAR worker processes (game g -> worker g % PAR) with identical per-game results.
PAR = int(os.environ.get("RAPFI_PAR", "3"))  # 3 workers: 20 games at 5000 nodes in ~12 min (2026-10-10 benchmark)


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
    if PAR > 1 and "--worker" not in sys.argv and int(sys.argv[1] if len(sys.argv) > 1 else 6) > 1:
        procs = [subprocess.Popen([sys.executable, "-u", __file__] + sys.argv[1:] + ["--worker", str(k), str(PAR)])
                 for k in range(PAR)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"worker failed: {codes}")
        return
    wk, wn = (int(sys.argv[sys.argv.index("--worker") + 1]), int(sys.argv[sys.argv.index("--worker") + 2])) \
        if "--worker" in sys.argv else (0, 1)
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
    lim = sys.argv[3] if len(sys.argv) > 3 else "2000"
    nodes = int(lim[1:]) if lim.startswith("n") else None
    ms = 30000 if nodes else int(lim)
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "model/best_mcts_11x11.pth"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(ckpt, map_location=dev)
    if any(k.startswith("opp_policy_head.") for k in sd):   # 2026-10-22: 11x11_v3 checkpoint (NetV3, 128x8)
        sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3"))
        from net_v3 import NetV3
        model = NetV3()
    else:
        model = AlphaZeroTicTacToeV3()
    model.load_state_dict(sd)
    model.to(dev).eval()

    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
        with torch.inference_mode():
            pl, v = model(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)

    # 2026-10-10: RAPFI_THREADS > 1 plays that many games at once in this process and evaluates their
    # positions in ONE shared GPU batch (cross-game batching). The search itself is unchanged (1100 sims,
    # batch 16 per game), but cuDNN picks different kernels for different batch sizes, so outputs differ by
    # ~1e-5 and a near-tied move can occasionally flip: results are statistically, not bit-for-bit, equal.
    threads = int(os.environ.get("RAPFI_THREADS", "1"))
    if threads > 1:
        import queue, threading
        req_q = queue.Queue()
        active = [0]
        alock = threading.Lock()

        def gpu_loop():
            while True:
                first = req_q.get()
                if first is None:
                    return
                batch = [first]
                deadline = time.time() + 0.004
                while True:
                    with alock:
                        want = active[0]
                    if len(batch) >= want:
                        break
                    try:
                        item = req_q.get(timeout=max(0.0, deadline - time.time()))
                    except queue.Empty:
                        break
                    if item is None:
                        req_q.put(None)
                        break
                    batch.append(item)
                bs = [x for it in batch for x in it[0]]
                ps = [x for it in batch for x in it[1]]
                pol, val = ev(bs, ps)
                k = 0
                for it in batch:
                    n = len(it[0])
                    it[2].append((pol[k:k + n], val[k:k + n]))
                    it[3].set()
                    k += n

        def shared_ev(bs, ps):
            box, done = [], threading.Event()
            req_q.put((list(bs), list(ps), box, done))
            done.wait()
            return box[0]

        gpu = threading.Thread(target=gpu_loop, daemon=True)
        gpu.start()
        evf = shared_ev
    else:
        evf = ev

    tally = {"ai_win": 0, "rapfi_win": 0, "draw": 0}
    by_color = {"X": [0, 0, 0], "O": [0, 0, 0]}  # ai as color: W/L/D
    out_lock = __import__("threading").Lock()

    def play(g, rapfi):
        ai = 'X' if g % 2 == 0 else 'O'
        opening = OPENINGS[(g // 2) % len(OPENINGS)]
        board = [[' '] * N for _ in range(N)]
        moves = [('X', opening[0], opening[1])]
        board[opening[0]][opening[1]] = 'X'
        turn = 'O'
        engine = mcts_core.MCTSEngine()
        result = None
        t0 = time.time(); t_ai = t_rapfi = 0.0
        while True:
            tm = time.time()
            if len(moves) >= N * N:
                result = "draw"
                break
            if turn == ai:
                # 2026-10-21 (user): MEGA_EVAL=30,15 also applies MEGA stage 1 to these noiseless match searches
                _, rv, cands = engine.search_batched(board, turn, evf, sims, 1.414, False, 0.3, 0.25, N, WIN_LENGTH, 16,
                                                     0.0, *_MEGA_EVAL)
                r, c = max(cands, key=lambda x: x['visit_count'])['move']
            else:
                r, c = rapfi.move(moves, turn)
            if turn == ai: t_ai += time.time() - tm
            else: t_rapfi += time.time() - tm
            if board[r][c] != ' ':
                raise RuntimeError(f"illegal move {(r, c)} by {'ai' if turn == ai else 'rapfi'}")
            board[r][c] = turn
            moves.append((turn, r, c))
            if fast_check_win(board, turn):
                result = "ai_win" if turn == ai else "rapfi_win"
                break
            turn = 'O' if turn == 'X' else 'X'
        with out_lock:
            tally[result] += 1
            by_color[ai][{"ai_win": 0, "rapfi_win": 1, "draw": 2}[result]] += 1
            print(f"game {g + 1}/{games}: AI={ai} opening={opening} -> {result} in {len(moves)} plies "
                  f"({time.time() - t0:.0f}s: ours {t_ai:.0f}s, rapfi {t_rapfi:.0f}s)", flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ai": ai, "opening": opening,
                                    "result": result, "plies": len(moves), "sims": sims, "rapfi_ms": None if nodes else ms, "rapfi_nodes": nodes,
                                    "checkpoint": os.path.basename(ckpt), "game": g,
                                    "moves": moves}) + "\n")

    mine = [g for g in range(games) if g % wn == wk]
    if threads <= 1:
        rapfi = Rapfi(ms, nodes)
        try:
            for g in mine:
                play(g, rapfi)
        finally:
            rapfi.close()
    else:
        import threading
        todo = list(mine)
        errors = []

        def runner():
            rapfi = Rapfi(ms, nodes)
            with alock:
                active[0] += 1
            try:
                while True:
                    with out_lock:
                        if not todo or errors:
                            return
                        g = todo.pop(0)
                    play(g, rapfi)
            except Exception as e:
                errors.append(e)
            finally:
                with alock:
                    active[0] -= 1
                rapfi.close()

        ts = [threading.Thread(target=runner) for _ in range(min(threads, len(mine)))]
        for t in ts: t.start()
        for t in ts: t.join()
        req_q.put(None)
        if errors:
            raise errors[0]
    print(f"\nTOTAL: AI {tally['ai_win']} / Rapfi {tally['rapfi_win']} / draw {tally['draw']}")
    print(f"AI as X (first): W{by_color['X'][0]} L{by_color['X'][1]} D{by_color['X'][2]} | "
          f"AI as O (second): W{by_color['O'][0]} L{by_color['O'][1]} D{by_color['O'][2]}")


if __name__ == "__main__":
    main()
