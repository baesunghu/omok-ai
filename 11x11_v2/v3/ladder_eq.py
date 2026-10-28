"""Draw- and colour-aware Rapfi equal point for v3 checkpoints (2026-10-25, user: draws now dominate the ladder).

Model (Davidson with a first-move term), per game against Rapfi n nodes:
    x = (a + b*log10(n) + c*(+1 if we are X else -1)) * ln10/400      (Elo diff -> nats)
    P(win) ~ e^(x/2),  P(loss) ~ e^(-x/2),  P(draw) ~ e^nu
Equal point = colour-neutral n where P(win) = P(loss), i.e. a + b*log10(n) = 0.
If the fitted slope is too flat (b > FLAT, the Rapfi ruler cannot separate the levels), b is fixed to the median
slope of the earlier v3 checkpoints and only a, c, nu are fitted; the result is flagged "slope fixed".
68/95% intervals: bootstrap over games (resampled within each level x colour cell).
Usage (from 11x11_v2): python v3/ladder_eq.py ep500 [ep400 ...]     (no args: every v3 checkpoint with >= 2 levels)
"""
import collections
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(V2, "model", "rapfi_match_log.jsonl")
OUT = os.path.join(V2, "v3", "ladder_eq_v3.json")
FLAT = -120          # Elo per 10x nodes; flatter than this = ruler saturated
K = np.log(10) / 400


def games():
    D = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for l in open(LOG, encoding="utf-8"):
        if '"v3_ep' not in l:
            continue
        r = json.loads(l)
        ck = r["checkpoint"][3:-4]
        if ck == "ep100" and r["time"] < "2026-10-22 23:02":      # the 10/22 pre-restart partial run
            continue
        if ck == "ep200" and r["rapfi_nodes"] == 32838:            # 10/23 spot check, repeats the same games
            continue
        D[ck][(r["rapfi_nodes"], r["ai"])][{"ai_win": 0, "rapfi_win": 1, "draw": 2}[r["result"]]] += 1
    return D


def nll(p, cells, b_fixed=None):
    if b_fixed is None:
        a, b, c, nu = p
    else:
        (a, c, nu), b = p, b_fixed
    t = 0.0
    for (n, col), (w, l, d) in cells:
        x = (a + b * np.log10(n) + (c if col == "X" else -c)) * K
        lz = np.logaddexp(np.logaddexp(x / 2, -x / 2), nu)
        t -= w * (x / 2 - lz) + l * (-x / 2 - lz) + d * (nu - lz)
    return t


def fit(cells, b_fixed=None):
    if b_fixed is None:
        r = minimize(nll, [1500, -300, 300, 0], args=(cells,), method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-3, "fatol": 1e-6})
        a, b, c, nu = r.x
    else:
        r = minimize(nll, [1500, 300, 0], args=(cells, b_fixed), method="Nelder-Mead", options={"maxiter": 6000})
        (a, c, nu), b = r.x, b_fixed
    return a, b, c, nu


def eq_of(a, b):
    return 10 ** (-a / b) if b < 0 else float("nan")


def analyse(ck, cells, prior_slopes, boots=400, seed=0):
    a, b, c, nu = fit(cells)
    fixed = None
    if b > FLAT and prior_slopes:
        fixed = float(np.median(prior_slopes))
        a, b, c, nu = fit(cells, fixed)
    eq = eq_of(a, b)
    rng, bs = np.random.default_rng(seed), []
    for _ in range(boots):
        rc = [(k, list(rng.multinomial(sum(v), np.array(v) / sum(v)))) for k, v in cells]
        aa, bb, _, _ = fit(rc, fixed)
        e = eq_of(aa, bb)
        if np.isfinite(e) and 0 < e < 1e9:
            bs.append(e)
    q = np.percentile(bs, [2.5, 16, 84, 97.5]) if bs else [float("nan")] * 4
    col = {s: [sum(v[i] for (n, cc), v in cells if cc == s) for i in range(3)] for s in ("X", "O")}
    return dict(checkpoint=ck, equal_nodes=eq, lo95=q[0], lo68=q[1], hi68=q[2], hi95=q[3], slope=b, slope_fixed=fixed is not None,
                first_move_elo=2 * c, draw_nu=nu, games=sum(sum(v) for _, v in cells),
                levels=sorted({n for (n, _), _ in cells}), black_wld=col["X"], white_wld=col["O"])


def main():
    D = games()
    order = sorted((k for k in D if len({n for n, _ in D[k]}) >= 2), key=lambda s: int(s[2:]))
    want = sys.argv[1:] or order
    out, slopes = {}, []
    for ck in order:
        cells = sorted(D[ck].items())
        if ck in want:
            r = analyse(ck, cells, list(slopes))
            out[ck] = r
            print(f"{ck}: equal {r['equal_nodes']:,.0f}  68% {r['lo68']:,.0f}-{r['hi68']:,.0f}  95% {r['lo95']:,.0f}-{r['hi95']:,.0f}  "
                  f"slope {r['slope']:.0f}{' (fixed)' if r['slope_fixed'] else ''}  first-move {r['first_move_elo']:.0f} Elo  "
                  f"levels {r['levels']}  black W/L/D {r['black_wld']}  white W/L/D {r['white_wld']}", flush=True)
            if not r["slope_fixed"]:
                slopes.append(r["slope"])
        else:
            b = fit(cells)[1]
            if b <= FLAT:
                slopes.append(b)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    old.update(out)
    json.dump(old, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
