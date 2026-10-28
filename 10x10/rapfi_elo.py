"""Rapfi node count -> Elo curve (2026-10-27, user: "make a rapfi(node) -> elo function").

Why: the node ladder is a log ruler that also saturates at high nodes, so "50x the nodes" does not translate
into a fixed Elo gap. This fits the ruler itself from every model-vs-Rapfi game we have, so node counts can
be turned into Elo (Rapfi at 100 nodes = 0, as in the 15x15 plan).

Model (Davidson with a first-move term), one game of player p against Rapfi at n nodes:
    x = (E_p - R(n) + c * (+1 if p is black else -1)) * ln10/400
    P(p wins) ~ e^(x/2),  P(Rapfi wins) ~ e^(-x/2),  P(draw) ~ e^nu
R(n) is piecewise linear in log10(n) with one slope per decade (10, 100, ..., 1e6 nodes), each >= 0, R(100) = 0.
Every player (checkpoint, per day: some file names were reused for different weights) gets its own Elo.
Data: 11x11_v2/model/rapfi_match_log.jsonl (per game, with colour) and 10x10/model/rapfi_ladder_log.jsonl
(per ladder run and level, colour unknown -> no first-move term; each run is its own player).

Usage:  python rapfi_elo.py                fit, print the table, save model/rapfi_elo_curve.json
        from rapfi_elo import rapfi_elo, rapfi_nodes     (read the saved curve)
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2_LOG = os.path.join(HERE, "..", "11x11_v2", "model", "rapfi_match_log.jsonl")
LADDER_LOG = os.path.join(HERE, "model", "rapfi_ladder_log.jsonl")
OUT = os.path.join(HERE, "model", "rapfi_elo_curve.json")
KNOTS = np.arange(1.0, 7.0)          # log10 nodes: 10 .. 1e6; outside, the end slopes are extended
ANCHOR = 2.0                         # R(100 nodes) = 0
K = math.log(10) / 400


def _r(t, slopes):
    """R at log10 nodes t for per-decade slopes (len(KNOTS) - 1 of them)."""
    t = np.asarray(t, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(slopes)])            # R at each knot, relative to KNOTS[0]
    seg = np.clip(np.floor(t - KNOTS[0]).astype(int), 0, len(slopes) - 1)
    val = cum[seg] + slopes[seg] * (t - KNOTS[seg])
    a = int(ANCHOR - KNOTS[0])
    return val - cum[a]


def load():
    """-> list of (player, log10 nodes, colour +1/-1/0, wins, losses, draws) cells."""
    cells = collections.defaultdict(lambda: [0, 0, 0])
    for line in open(V2_LOG, encoding="utf-8"):
        g = json.loads(line)
        if g["checkpoint"].startswith("_smoke"):
            continue
        key = (g["checkpoint"] + "@" + g["time"][:10], math.log10(g["rapfi_nodes"]), 1 if g["ai"] == "X" else -1)
        cells[key][{"ai_win": 0, "rapfi_win": 1, "draw": 2}[g["result"]]] += 1
    for i, run in enumerate(json.loads(l) for l in open(LADDER_LOG, encoding="utf-8")):
        for lv in run["levels"]:
            key = (f"{run['checkpoint']}#{i}", math.log10(lv["nodes"]), 0)
            c = cells[key]
            c[0] += lv["win"]; c[1] += lv["loss"]; c[2] += lv["draw"]
    return [(p, t, col, *v) for (p, t, col), v in cells.items()]


def fit(cells):
    """Maximum likelihood with exact gradients (torch, L-BFGS). Slopes are softplus (>= 0) and a weak
    smoothness penalty ties neighbouring decades together, so a decade with few games borrows from its
    neighbours instead of collapsing to 0 (2026-10-27: the gradient-free first try did exactly that)."""
    import torch
    players = sorted({c[0] for c in cells})
    pid = {p: i for i, p in enumerate(players)}
    tt = lambda a, dt=torch.float64: torch.tensor(a, dtype=dt)
    P = tt([pid[c[0]] for c in cells], torch.long); T = tt([c[1] for c in cells]); C = tt([c[2] for c in cells])
    W, L, D = (tt([c[i] for c in cells]) for i in (3, 4, 5))
    ns = len(KNOTS) - 1
    knots = tt(KNOTS)
    raw = torch.full((ns,), 2.0, dtype=torch.float64, requires_grad=True)     # softplus -> ~200 Elo/decade start
    cn = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)     # half first-move term, draw nu
    E = torch.zeros(len(players), dtype=torch.float64, requires_grad=True)

    def r_of(t, slopes):
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(slopes, 0)])
        seg = torch.clamp(torch.floor(t - knots[0]).long(), 0, ns - 1)
        val = cum[seg] + slopes[seg] * (t - knots[seg])
        return val - cum[int(ANCHOR - KNOTS[0])]

    def loss():
        slopes = torch.nn.functional.softplus(raw) * 100
        x = (E[P] - r_of(T, slopes) + cn[0] * C) * K
        lz = torch.logaddexp(torch.logaddexp(x / 2, -x / 2), cn[1])
        nll = -(W * (x / 2 - lz) + L * (-x / 2 - lz) + D * (cn[1] - lz)).sum()
        smooth = ((slopes[1:] - slopes[:-1]) / 100).pow(2).sum() * 2.0
        return nll + smooth + 1e-6 * E.pow(2).sum()

    opt = torch.optim.LBFGS([raw, cn, E], lr=1, max_iter=5000, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss()
        l.backward()
        return l
    for _ in range(5):
        opt.step(closure)
    g = torch.cat([raw.grad.abs(), cn.grad.abs(), E.grad.abs()]).max().item()
    slopes = (torch.nn.functional.softplus(raw) * 100).detach().numpy()
    games = collections.Counter()
    for c in cells:
        games[round(10 ** c[1])] += c[3] + c[4] + c[5]
    return dict(knots_log10=KNOTS.tolist(), anchor_log10=ANCHOR, slopes_per_decade=slopes.tolist(),
                first_move_elo=float(2 * cn[0].item()), draw_nu=float(cn[1].item()), converged=g < 1e-3,
                players={p: float(e) for p, e in zip(players, E.detach().numpy())},
                games=int(W.sum() + L.sum() + D.sum()), games_by_nodes=dict(sorted(games.items())))


_CURVE = None


def _curve():
    global _CURVE
    if _CURVE is None:
        _CURVE = json.load(open(OUT, encoding="utf-8"))
    return _CURVE


def rapfi_elo(nodes):
    """Elo of Rapfi at `nodes` nodes per move (Rapfi at 100 nodes = 0)."""
    return float(_r(math.log10(nodes), np.array(_curve()["slopes_per_decade"])))


def rapfi_nodes(elo):
    """Inverse: the node count at which Rapfi has this Elo."""
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if rapfi_elo(10 ** mid) < elo:
            lo = mid
        else:
            hi = mid
    return 10 ** ((lo + hi) / 2)


def main():
    cells = load()
    r = fit(cells)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    global _CURVE
    _CURVE = r
    print(f"{r['games']} games, converged {r['converged']}, first-move {r['first_move_elo']:.0f} Elo, draw nu {r['draw_nu']:.2f}")
    print("decade          Elo per 10x nodes   games")
    for k, s in zip(KNOTS[:-1], r["slopes_per_decade"]):
        g = sum(v for n, v in r["games_by_nodes"].items() if k <= math.log10(n) < k + 1)
        print(f"{10 ** k:>9,.0f}-{10 ** (k + 1):<9,.0f} {s:8.0f}            {g}")
    print("nodes -> Elo:", ", ".join(f"{n:,}: {rapfi_elo(n):.0f}" for n in (10, 100, 1000, 10000, 51696, 100000, 1000000)))
    for name in sorted(r["players"]):
        if name.startswith("v3_ep") or "ep625" in name:
            print(f"  {name:32s} {r['players'][name]:7.0f}")


if __name__ == "__main__":
    main()
