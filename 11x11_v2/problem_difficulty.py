"""Rapfi-node difficulty of every problem in tactical_problems/web_game58_problems.json (2026-10-10).
MEASUREMENT ONLY -- nothing here feeds training.

The side to move is played by Rapfi limited to N nodes per move, the other side by Rapfi at OPP_NODES
(strong, node-limited so every result is deterministic), for up to 40 plies -- the graduation check's
setting, with Rapfi in place of the model.
  defense : pass if the first move is not the problem's bad move and the side to move does not lose
  attack  : pass if the side to move wins
Difficulty = the smallest N from which every larger level also passes ("stable"); the first passing
level is reported too. Results: model/problem_difficulty.json (re-uses finished problems).

Usage: python problem_difficulty.py
"""
import json, os, sys, threading
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from rapfi_match import Rapfi
from fast_board import fast_check_win
N = 11
OPP_NODES = 200000
LEVELS = [100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000]
OUT = os.path.join("model", "problem_difficulty.json")
TRIAL_TIMEOUT = 300   # s; 2026-10-10: a Rapfi pipe once stalled for an hour (both engines idle)


def trial(p, n, opp, reg=None):
    me = p["side_to_move"]; other = 'O' if me == 'X' else 'X'
    dfn = Rapfi(30000, n)
    if reg is not None:
        reg["dfn"] = dfn   # so a stalled trial can be unblocked from outside
    try:
        moves = [tuple(m) for m in p["moves"]]
        b = [[' '] * N for _ in range(N)]
        for pl, r, c in moves: b[r][c] = pl
        turn, first, winner = me, None, None
        for _ in range(40):
            r, c = (dfn if turn == me else opp).move(list(moves), turn)
            if first is None: first = [r, c]
            b[r][c] = turn; moves.append((turn, r, c))
            if fast_check_win(b, turn): winner = turn; break
            turn = other if turn == me else me
    finally:
        dfn.close()
    if p.get("type") == "attack":
        ok = winner == me
    else:
        ok = first != p.get("bad_move") and winner != other
    return ok, first, winner


def trial_guarded(p, n, holder):
    """trial() with a timeout: on a stall, kill both engines (unblocks the read), report a timeout, and
    give the caller a fresh opponent engine."""
    box = {}

    def run():
        try:
            box["r"] = trial(p, n, holder["opp"], holder)
        except Exception as e:
            box["e"] = e

    th = threading.Thread(target=run, daemon=True); th.start(); th.join(TRIAL_TIMEOUT)
    if th.is_alive() or "e" in box:
        for k in ("opp", "dfn"):
            try:
                holder[k].p.kill()
            except Exception:
                pass
        th.join(10)
        holder["opp"] = Rapfi(30000, OPP_NODES)
        return None, None, "timeout" if "e" not in box else f"error: {box['e']}"
    return box["r"]


def main():
    probs = json.load(open(os.path.join("tactical_problems", "web_game58_problems.json"), encoding="utf-8"))["problems"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    holder = {"opp": Rapfi(30000, OPP_NODES)}
    try:
        for p in probs:
            if p["id"] in res:
                continue
            row = []
            for n in LEVELS:
                ok, first, winner = trial_guarded(p, n, holder)
                if ok is None:
                    print(f"  {p['id']} {n} nodes: {winner} -> counted as fail", flush=True)
                    ok = False
                row.append([n, ok, first, winner])
            stable = next((n for i, (n, *_r) in enumerate(row) if all(x[1] for x in row[i:])), None)
            firstp = next((n for n, ok, *_r in row if ok), None)
            res[p["id"]] = {"type": p.get("type"), "stage": p.get("stage"), "tier": p.get("tier"),
                            "stable": stable, "first_pass": firstp, "rows": row}
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"{p['id']:<28} stage {p.get('stage')} tier {p.get('tier')}: stable from {stable}, first pass {firstp}", flush=True)
    finally:
        holder["opp"].close()


if __name__ == "__main__":
    main()
