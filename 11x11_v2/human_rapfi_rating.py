"""Put human players on the Rapfi node ruler (method A, 2026-10-10). MEASUREMENT ONLY.

For every human move in the web games (hlt/web/games.jsonl) we measure how much the move gave away,
judged by a strong reference Rapfi, and compare that with what Rapfi limited to N nodes gives away
in the very same positions. The N whose average loss matches the humans' is the humans' strength
on the same ruler used for the models (equal-strength nodes).

  loss(move) = W(position) - W(position after move)      W = win probability for the player to move,
                                                         from the reference Rapfi's eval (logistic)
Reference: Rapfi at REF_NODES nodes. Levels: Rapfi at LEVELS nodes. All node-limited, single thread:
deterministic. Results are cached in model/human_rapfi_cache.json so the run can resume.

Usage: python human_rapfi_rating.py [max_games]
"""
import ast, json, math, os, random, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, HERE)
from fast_board import fast_check_win
RAPFI_DIR = r"C:\Users\My PC\Desktop\final\engines\rapfi"
RAPFI_EXE = os.path.join(RAPFI_DIR, "pbrain-rapfi-windows-avx2.exe")
N = 11
REF_NODES = 100000
LEVELS = [5, 10, 15, 20, 30, 100, 300, 1000]   # 2026-10-10 (user): low end. Rapfi plays identically at 1/3/5 nodes (no search: only forced blocks, else (1,1)), so 5 stands for all three; 15/20 added for resolution; 3000/10000 dropped
K = 200.0                 # eval -> win probability scale (relative comparison, same metric on both sides)
BLUNDER = 0.30            # a move that gives away >= 30% win probability
CACHE = os.path.join("model", "human_rapfi_cache.json")
EXCLUDE_TIMES = ("2026-09-28 14:35",)   # the 9/28 game played by a human WITH engine help


class Rapfi:
    def __init__(self, nodes):
        self.p = subprocess.Popen([RAPFI_EXE], cwd=RAPFI_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send(f"START {N}"); self._wait("OK")
        for c in ("INFO rule 0", "INFO timeout_turn 30000", "INFO timeout_match 100000000", "INFO max_memory 0",
                  f"INFO max_node {nodes}"):
            self.send(c)

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, tok):
        while self.p.stdout.readline().strip() != tok:
            pass

    def think(self, moves, me):
        """returns ((r, c), eval for `me`) -- eval is the last reported search eval"""
        self.send("BOARD")
        for pl, r, c in moves:
            self.send(f"{c},{r},{1 if pl == me else 2}")
        self.send("DONE")
        ev = 0.0
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("rapfi exited")
            l = l.strip()
            m = re.search(r"Eval ([+-]?)(M?)(\d+)", l)
            if m:
                sign = -1 if m.group(1) == "-" else 1
                ev = sign * (20000.0 if m.group(2) else float(m.group(3)))
                continue
            if l.startswith(("MESSAGE", "DEBUG", "OK")) or not l:
                continue
            mv = re.fullmatch(r"(\d+),(\d+)", l)
            if not mv:   # "-5,-5", "ERROR ...": the protocol state is no longer trustworthy -> restart engine
                raise RuntimeError(f"rapfi exited/desynced: {l!r}")
            return (int(mv.group(2)), int(mv.group(1))), ev

    def close(self):
        try:
            self.send("END"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class NoMove(Exception):
    pass


def win(ev):
    return 1.0 / (1.0 + math.exp(-max(-5000.0, min(5000.0, ev)) / K))


def other(p):
    return 'O' if p == 'X' else 'X'


def main():
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    games = [json.loads(l) for l in open(os.path.join("hlt", "web", "games.jsonl"), encoding="utf-8")]
    games = [g for g in games if not g["time"].startswith(EXCLUDE_TIMES)][:max_games]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    eng = {"ref": Rapfi(REF_NODES), **{n: Rapfi(n) for n in LEVELS}}
    crashes = []

    def ask(which, moves, me):
        """think(); if the Rapfi process dies on this position, restart it and skip the position"""
        try:
            return eng[which].think(moves, me)
        except RuntimeError as e:
            if "exited" not in str(e):
                raise
            crashes.append((which, len(moves), str(e)[-40:]))
            eng[which].close()
            eng[which] = Rapfi(REF_NODES if which == "ref" else which)
            raise NoMove

    def key(moves):
        return ";".join(f"{p}{r},{c}" for p, r, c in moves)

    def ref_win(moves, to_move):
        if moves:   # the last move already made five: the side to move has lost
            b = [[' '] * N for _ in range(N)]
            for p, rr, cc in moves: b[rr][cc] = p
            if fast_check_win(b, moves[-1][0]):
                return 0.0
        k = "R" + key(moves)
        if k not in cache:
            cache[k] = ask("ref", moves, to_move)[1]   # eval is valid even when no move comes back
        return win(cache[k])

    t0 = time.time(); per_game = []; skipped = [0]
    try:
        for gi, g in enumerate(games):
            mv = ast.literal_eval(g["moves"]) if isinstance(g["moves"], str) else g["moves"]
            mv = [tuple(m) for m in mv]
            human = g["human"]
            rec = {"human": [], **{str(n): [] for n in LEVELS}}
            for i, (pl, r, c) in enumerate(mv):
                if pl != human:
                    continue
                pos = mv[:i]
                if len(pos) < 2:      # skip the very first stone
                    continue
                try:
                    w0 = ref_win(pos, pl)
                except NoMove:
                    skipped[0] += 1; continue
                if w0 < 0.02 or w0 > 0.98:   # already decided: errors there say little about skill
                    continue
                try:   # all-or-nothing per position, so every column is measured on the same positions
                    row = {"human": w0 - (1.0 - ref_win(pos + [(pl, r, c)], other(pl)))}
                    for n in LEVELS:
                        k = f"L{n}" + key(pos)
                        if k not in cache:
                            m_ = ask(n, pos, pl)[0]
                            if m_ is None:
                                raise NoMove
                            cache[k] = list(m_)
                        rr, cc = cache[k]
                        row[str(n)] = w0 - (1.0 - ref_win(pos + [(pl, rr, cc)], other(pl)))
                except NoMove:
                    skipped[0] += 1; continue
                for kk, vv in row.items():
                    rec[kk].append(vv)
            per_game.append(rec)
            json.dump(cache, open(CACHE, "w"))
            h = rec["human"]
            print(f"game {gi + 1}/{len(games)} ({g['time']}, {len(h)} moves): human mean loss "
                  f"{(sum(h) / len(h) if h else float('nan')):.3f}  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        for e in eng.values():
            e.close()
        json.dump(cache, open(CACHE, "w"))

    def summary(games_):
        out = {}
        for k in ["human"] + [str(n) for n in LEVELS]:
            xs = [x for g in games_ for x in g[k]]
            out[k] = (sum(xs) / len(xs), sum(x >= BLUNDER for x in xs) / len(xs), len(xs)) if xs else (float("nan"), 0, 0)
        return out

    def match(s):
        """node count whose mean loss equals the humans' (log-linear interpolation)"""
        h = s["human"][0]
        pts = [(n, s[str(n)][0]) for n in LEVELS]
        for (n1, l1), (n2, l2) in zip(pts, pts[1:]):
            if (l1 - h) * (l2 - h) <= 0 and l1 != l2:
                t = (l1 - h) / (l1 - l2)
                return 10 ** (math.log10(n1) + t * (math.log10(n2) - math.log10(n1)))
        return LEVELS[0] if h > pts[0][1] else LEVELS[-1]

    print(f"positions skipped (Rapfi gave no move or crashed): {skipped[0]}  crashes: {crashes}")
    s = summary(per_game)
    print("\n            mean loss   blunder rate   moves")
    for k, (m, b, n) in s.items():
        print(f"{('human' if k == 'human' else 'Rapfi ' + k):>12}  {m:9.3f}   {b * 100:9.1f}%   {n:6d}")
    est = match(s)
    random.seed(0)
    bs = sorted(match(summary([random.choice(per_game) for _ in per_game])) for _ in range(1000))
    print(f"\nhumans ~ Rapfi {est:,.0f} nodes  (95% by resampling games: {bs[25]:,.0f} ~ {bs[974]:,.0f})")
    json.dump({"summary": s, "estimate": est, "ci95": [bs[25], bs[974]], "games": len(per_game),
               "ref_nodes": REF_NODES, "levels": LEVELS, "K": K},
              open(os.path.join("model", "human_rapfi_rating.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
