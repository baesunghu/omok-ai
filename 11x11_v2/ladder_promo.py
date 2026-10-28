"""Rapfi-ladder promotion check (2026-10-12, user decision).

The gate only plays the current best, so a matchup ("상성") can block a stronger model for a long time.
This adds a second promotion path: after a ladder run on a checkpoint, compare it with the best's own
ladder run (model/ladder_best_ref.json). If its fitted equal-node point is higher AND the bootstrap
probability that it is stronger is >= P_MIN, write model/ladder_promote.json; the trainer applies it at
the start of its next epoch (best checkpoint + self-play generation model) and logs "PROMOTED (ladder)".

Usage:
    python ladder_promo.py <checkpoint_basename_in_ladder_log> <path_to_weights> [--dry-run]
"""
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "model", "rapfi_ladder_log.jsonl")
REF = os.path.join(HERE, "model", "ladder_best_ref.json")
FLAG = os.path.join(HERE, "model", "ladder_promote.json")
P_MIN = 0.90   # 2026-10-12 (user): 80% -> 90% (fewer lucky promotions / inflated reference)
P_CAND = 0.60  # 2026-10-13 (user): gate candidates also need P(stronger than best) >= 60% (an equal model
               # passed "eq >= best" about half the time by measurement luck)
N_BOOT = 4000


def games(level):
    return [1.0] * level["win"] + [0.5] * level["draw"] + [0.0] * level["loss"]


def fit(levels):
    """Games-weighted least squares of Elo vs log10(nodes), solved for Elo = 0 (same as rapfi_ladder.py)."""
    xs = [math.log10(n) for n in levels]
    ws = [len(s) for s in levels.values()]
    ys = []
    for s in levels.values():
        p = min(max(sum(s) / len(s), 0.03), 0.97)
        ys.append(400 * math.log10(p / (1 - p)))
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    return min(10 ** (mx - my / slope), 1e6) if slope < 0 else 1e6


def last_run(ckpt_name):
    """Merged levels of all ladder log entries for this checkpoint name (one-off runs add levels)."""
    levels = {}
    for line in open(LOG, encoding="utf-8"):
        r = json.loads(line)
        if r.get("checkpoint") == ckpt_name:
            for L in r["levels"]:
                levels[L["nodes"]] = games(L)
    return levels


TACT = os.path.join(HERE, "model", "tactical_problem_log.jsonl")


def grad_score(ckpt_name):
    """(graduation passed, total, all passed) from the newest problem check of this checkpoint, or None.
    2026-10-12 (user): guard against Rapfi bias -- a ladder promotion must not do worse on the
    graduation problems than the current best."""
    last = None
    if os.path.exists(TACT):
        for line in open(TACT, encoding="utf-8"):
            r = json.loads(line)
            if r.get("checkpoint") == ckpt_name:
                last = r
    return (last["grad_passed"], last["grad_total"], last["passed"]) if last else None


def p_stronger(a, b, seed=0):
    rnd = random.Random(seed)
    wins = 0
    for _ in range(N_BOOT):
        ea = fit({n: rnd.choices(s, k=len(s)) for n, s in a.items()})
        eb = fit({n: rnd.choices(s, k=len(s)) for n, s in b.items()})
        wins += ea > eb
    return wins / N_BOOT


def check(ckpt_name, weights_path, dry_run=False, candidate=False):
    """candidate=True (2026-10-13, gate-passed candidate): promote if its point estimate is not below the
    best's (no P threshold -- the gate already supplied evidence); graduation must not be worse."""
    ref = json.load(open(REF, encoding="utf-8"))
    a, b = last_run(ckpt_name), ref["levels"] if "levels" in ref else last_run(ref["ckpt_name"])
    b = {int(k): v for k, v in b.items()}
    if len(a) < 3:
        return f"{ckpt_name}: only {len(a)} ladder levels, no check"
    ea, eb = fit(a), fit(b)
    p = p_stronger(a, b)
    msg = f"{ckpt_name}: eq {ea:,.0f} vs best {ref['name']} {eb:,.0f}, P(stronger) {p:.0%}"
    g = grad_score(ckpt_name)
    gb = ref.get("grad")
    msg += f", graduation {g[0]}/{g[1] if g else '?'} vs best {gb if gb is not None else '?'}" if g else ", graduation: not graded yet"
    if g is None:
        return msg + " -> no promotion (grade the graduation problems first)"
    if gb is not None and g[0] < gb:
        return msg + " -> no promotion (graduation problems worse than the best)"
    if (ea >= eb and p >= P_CAND) if candidate else (ea > eb and p >= P_MIN):
        if dry_run:
            return msg + " -> WOULD PROMOTE (dry run)"
        name = ckpt_name.replace("_rapfi_latest_", "").replace(".pth", "")
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), ckpt=weights_path, name=name, eq=round(ea),
                       best_eq=round(eb), p=round(p, 3), applied=False), open(FLAG, "w", encoding="utf-8"), indent=1)
        json.dump(dict(name=name, ckpt_name=ckpt_name, ckpt=weights_path, levels=a, eq=round(ea), grad=g[0],
                       since=time.strftime("%Y-%m-%d %H:%M:%S")), open(REF, "w", encoding="utf-8"), indent=1)
        return msg + " -> PROMOTION FLAG WRITTEN"
    return msg + " -> no promotion"


if __name__ == "__main__":
    print(check(sys.argv[1], sys.argv[2], dry_run="--dry-run" in sys.argv, candidate="--candidate" in sys.argv))
