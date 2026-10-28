"""Bootstrap CI of the fitted equal-strength node count for each ladder run in ladder_hist.json,
using the real per-game results in model/rapfi_match_log.jsonl. Writes ladder_ci.json."""
import json, math, random
SP = r"C:\Users\My PC\Desktop\final\11x11_v2\model"  # ladder_hist.json / ladder_ci.json live here
rows = [json.loads(l) for l in open(r"C:\Users\My PC\Desktop\final\11x11_v2\model\rapfi_match_log.jsonl", encoding="utf-8")]
groups, prev = [], None
for x in rows:
    key = (x.get("checkpoint"), x.get("rapfi_nodes"))
    if key != prev: groups.append({"key": key, "rows": []}); prev = key
    groups[-1]["rows"].append(x)
for g in groups:
    g["rows"] = g["rows"][-20:]                      # an aborted+rerun level keeps only the rerun
    g["start"], g["end"] = g["rows"][0]["time"][:16], g["rows"][-1]["time"][:16]
    g["sc"] = [{"ai_win": 1, "draw": .5}.get(r["result"], 0) for r in g["rows"]]

def fit(levels):
    xs = [math.log10(n) for n, s in levels]; ws = [len(s) for n, s in levels]
    ys = []
    for n, s in levels:
        p = min(max(sum(s) / len(s), .03), .97); ys.append(400 * math.log10(p / (1 - p)))
    sw = sum(ws); mx = sum(w * x for w, x in zip(ws, xs)) / sw; my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs)); sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    sl = sxy / sxx if sxx else 0
    if sl < 0: return 10 ** min(mx - my / sl, 6)
    return 1e6 if my > 0 else 1.0

hist = [r for r in json.load(open(SP + r"\ladder_hist.json")) if not r["dup"] and r["eq"]]
random.seed(0); out = []
for h in hist:
    # match the run by its own fitted value (hist times are not consistently start or end times)
    best = None
    for i, k in ((i, k) for k in (6, 5, 4, 3) for i in range(len(groups) - k + 1)):   # 5 levels from 2026-10-10 (5000 added)
        w = groups[i:i + k]
        if any(g["key"][1] is None for g in w) or [g["key"][1] for g in w] != sorted(g["key"][1] for g in w) or len({g["key"][1] for g in w}) < k or len({g["key"][0] for g in w}) > 1: continue
        e = fit([(g["key"][1], g["sc"]) for g in w])
        if abs(e - h["eq"]) <= max(2, h["eq"] * .002):
            d = abs((int(w[0]["start"][8:10]) * 24 + int(w[0]["start"][11:13])) - (int(h["time"][8:10]) * 24 + int(h["time"][11:13])))
            if best is None or d < best[0]: best = (d, w)
    if best is None: print("no match", h["ep"], h["eq"]); continue
    run = best[1]
    if h.get("add"):   # a level measured separately later (e.g. 5000 nodes added 2026-10-10)
        ad = h["add"]
        run = run + [next(g for g in groups if g["key"] == (ad["ckpt"], ad["nodes"]) and g["start"] >= ad["after"])]
    lv = [(g["key"][1], g["sc"]) for g in run]
    est = fit(lv)
    bs = sorted(fit([(n, random.choices(s, k=len(s))) for n, s in lv]) for _ in range(4000))
    q = lambda p: bs[int(p * (len(bs) - 1))]
    out.append(dict(ep=h["ep"], kind=h["kind"], eq_hist=h["eq"], eq=round(est), lo95=q(.025), lo68=q(.16), hi68=q(.84), hi95=q(.975),
                    levels=[(n, len(s), sum(s)) for n, s in lv], ckpt=run[0]["key"][0], start=run[0]["start"]))
    o = out[-1]
    print(f"{o['ep']:>4} {o['kind']:<6} hist {o['eq_hist']:>5} refit {o['eq']:>5}  68% {o['lo68']:>6.0f}-{o['hi68']:<7.0f} 95% {o['lo95']:>6.0f}-{o['hi95']:<7.0f} {o['start']} {[(n,c) for n,c,_ in o['levels']]}")
json.dump(out, open(SP + r"\ladder_ci.json", "w"), indent=1)
