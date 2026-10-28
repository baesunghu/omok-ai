"""15x15 supervised-pretraining data (2026-10-13, user decision): Rapfi at 100 nodes labels positions.

Each game: 2-4 random opening stones within 3 of the centre, then Rapfi(100 nodes) plays both sides; with
probability EPS a ply is a random move near existing stones instead (diversity: Rapfi is deterministic at a
node limit, so identical openings would replay identical games). Every position where Rapfi is to move is
recorded with Rapfi's chosen move (policy target = one-hot) and its evaluation (value target, side to move).
The game's final result is stored too, so the eval -> [-1, 1] mapping can be fitted afterwards.

Usage: python sl_rapfi_data.py <games> <worker_id> <out.jsonl>
"""
import json
import os
import random
import re
import subprocess
import sys

N = 15
NODES = int(os.environ.get("SL_NODES", "100"))   # 2026-10-16 (user): 10 for a deliberately weak epoch 0
EPS = 0.08
RAPFI_DIR = r"C:\Users\My PC\Desktop\final\engines\rapfi"
RAPFI_EXE = os.path.join(RAPFI_DIR, "pbrain-rapfi-windows-avx2.exe")
EVAL_RE = re.compile(r"Eval (\S+)")


class Rapfi:
    def __init__(self):
        self.p = subprocess.Popen([RAPFI_EXE], cwd=RAPFI_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        for cmd in (f"START {N}", "INFO timeout_turn 30000", "INFO timeout_match 100000000",
                    f"INFO max_node {NODES}", "INFO thread_num 1"):
            self.send(cmd)
        while True:                       # wait for START's OK
            l = self.p.stdout.readline().strip()
            if l.startswith("OK") or not l:
                break

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def sync(self):
        """2026-10-14: wait until the engine is idle (a command sent while it is still busy is swallowed)."""
        self.send("ABOUT")
        while True:
            raw = self.p.stdout.readline()
            if raw == "":
                raise RuntimeError("rapfi closed")
            if "name=" in raw:
                return

    def move(self, moves, me):
        self.sync()
        self.send("BOARD")
        for pl, r, c in moves:
            self.send(f"{c},{r},{1 if pl == me else 2}")
        self.send("DONE")
        ev = None
        while True:
            l = self.p.stdout.readline().strip()
            if not l:
                raise RuntimeError("rapfi closed")
            if l.startswith("MESSAGE"):
                m = EVAL_RE.search(l)
                if m:
                    ev = m.group(1)
                continue
            if l.startswith(("DEBUG", "OK")):
                continue
            if l.startswith("ERROR") or "," not in l:
                raise RuntimeError(l)
            x, y = l.split(",")
            return int(y), int(x), ev

    def close(self):
        try:
            self.send("END"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def five(b, r, c, p):
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        k = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < N and 0 <= cc < N and b[rr][cc] == p:
                k += 1; rr += s * dr; cc += s * dc
        if k >= 5:
            return True
    return False


def near_empty(b, rnd):
    cells = [(r, c) for r in range(N) for c in range(N) if b[r][c] == ' ' and any(
        0 <= r + dr < N and 0 <= c + dc < N and b[r + dr][c + dc] != ' '
        for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2))]
    return rnd.choice(cells)


def main():
    games, wid, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rnd = random.Random(1000 + wid)
    eng = Rapfi()
    done = 0
    with open(out, "a", encoding="utf-8") as f:
        while done < games:
            b = [[' '] * N for _ in range(N)]; moves = []
            for i in range(rnd.randint(2, 4)):
                p = 'X' if i % 2 == 0 else 'O'
                while True:
                    r, c = 7 + rnd.randint(-3, 3), 7 + rnd.randint(-3, 3)
                    if b[r][c] == ' ':
                        break
                b[r][c] = p; moves.append((p, r, c))
            turn = 'X' if len(moves) % 2 == 0 else 'O'
            recs, winner, ok = [], None, True
            while len(moves) < N * N:
                try:
                    r, c, ev = eng.move(list(moves), turn)
                except Exception:
                    eng.close(); eng = Rapfi(); ok = False
                    break
                recs.append([len(moves), r, c, ev])          # position = moves[:ply], label = (r, c), eval
                if rnd.random() < EPS and len(moves) < 40:
                    r, c = near_empty(b, rnd)                # diversity: play a random nearby move instead
                b[r][c] = turn; moves.append((turn, r, c))
                if five(b, r, c, turn):
                    winner = turn
                    break
                turn = 'O' if turn == 'X' else 'X'
            if not ok:
                continue
            f.write(json.dumps({"moves": moves, "labels": recs, "winner": winner}) + "\n"); f.flush()
            done += 1
    eng.close()


if __name__ == "__main__":
    main()
