"""
Rapfi node-limited strength ladder (2026-09-28). MEASUREMENT ONLY.

Plays our model against Rapfi at fixed search-node budgets (a
machine-independent yardstick, unlike a time limit), alternating colours,
then converts each level's score into an Elo difference
(Elo = 400 * log10(s / (1 - s)), s = (wins + draws/2) / games, clamped to
[1/(2n), 1 - 1/(2n)] so a whitewash stays finite), and reports the node
budget at which the model would score 50% (log-linear interpolation).

Results append to model/rapfi_ladder_log.jsonl, one line per run, so the
same ladder can be re-run every N epochs to draw an Elo-over-time curve.

Usage: python rapfi_ladder.py [games_per_level] [our_sims] [checkpoint]
"""
import json
import math
import os
import subprocess
import sys
import time

TENX = r"C:\Users\My PC\Desktop\final\15x15"
LEVELS = [int(x) for x in os.environ.get("RAPFI_LEVELS", "1000,2000,3000,5000,10000,20000").split(",")]  # 2026-10-10 night (user): 20000 added (the 정오 bar sits around 20k)  # 2026-10-10 evening (user, option B): 300 dropped (85%+, no information), 10000 added (890 scored 62% at 5000)  # 2026-10-10 (user): 5000 added -- 860/880 still scored ~42% at 3000, so the top end was unmeasurable  # Rapfi search nodes per move (around the current equal-strength point)
LOG = os.path.join(TENX, "model", "rapfi_ladder_log.jsonl")
MATCH_LOG = os.path.join(TENX, "model", "rapfi_match_log.jsonl")


def elo(score, n):
    lo, hi = 1.0 / (2 * n), 1.0 - 1.0 / (2 * n)
    s = min(max(score, lo), hi)
    return 400.0 * math.log10(s / (1.0 - s))


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "model/best_mcts_11x11.pth"
    results = []
    for nodes in LEVELS:
        start = sum(1 for _ in open(MATCH_LOG, encoding="utf-8")) if os.path.exists(MATCH_LOG) else 0
        t0 = time.time()
        subprocess.run([sys.executable, "-u", "rapfi_match.py", str(games), str(sims), f"n{nodes}", ckpt],
                       cwd=TENX, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(MATCH_LOG, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f.readlines()[start:]]
        w = sum(r["result"] == "ai_win" for r in rows)
        l = sum(r["result"] == "rapfi_win" for r in rows)
        d = sum(r["result"] == "draw" for r in rows)
        n = max(1, w + l + d)
        s = (w + 0.5 * d) / n
        res = dict(nodes=nodes, games=n, win=w, loss=l, draw=d, score=s, elo_vs_level=elo(s, n),
                   minutes=round((time.time() - t0) / 60, 1))
        results.append(res)
        print(f"Rapfi {nodes:>7} nodes: W{w} L{l} D{d}  score {s:.0%}  Elo vs level {res['elo_vs_level']:+.0f}  "
              f"({res['minutes']} min)", flush=True)

    # node budget where our score would be 50% (log-linear interpolation)
    eq = None
    for a, b in zip(results, results[1:]):
        if a["score"] >= 0.5 >= b["score"] and a["score"] != b["score"]:
            t = (a["score"] - 0.5) / (a["score"] - b["score"])
            eq = 10 ** (math.log10(a["nodes"]) + t * (math.log10(b["nodes"]) - math.log10(a["nodes"])))
    if eq is None:
        eq = "above max level" if results[-1]["score"] > 0.5 else (
            "below min level" if results[0]["score"] < 0.5 else None)
    crossing = eq
    # 2026-09-29 (user decision): the official equal-strength point is a games-weighted
    # least-squares line of Elo vs log10(nodes) over ALL levels, solved for Elo = 0.
    # The single-crossing rule above swung 100..1250 on 20-game noise; kept for reference.
    xs = [math.log10(r["nodes"]) for r in results]
    ws = [r["games"] for r in results]
    ys = [400 * math.log10(min(max(r["score"], 0.03), 0.97) / (1 - min(max(r["score"], 0.03), 0.97))) for r in results]
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    eq = 10 ** (mx - my / slope) if slope < 0 else ("above max level" if my > 0 else "below min level")
    print(f"\n50% point, fitted (official): {eq if isinstance(eq, str) else f'~{eq:,.0f} nodes'}  "
          f"[slope {slope:.0f} Elo per 10x nodes; old single-crossing rule: "
          f"{crossing if isinstance(crossing, str) or crossing is None else f'~{crossing:,.0f}'}]")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), checkpoint=os.path.basename(ckpt),
                                sims=sims, games_per_level=games, levels=results, equal_nodes=eq,
                                equal_nodes_crossing=crossing, fit_slope=slope)) + "\n")


if __name__ == "__main__":
    main()
