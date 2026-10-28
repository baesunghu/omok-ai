# ladder_all_chart.py + a third panel for 11x11_v3 (Gumbel cold start), same bootstrap as ladder_v2_chart (2026-10-22).
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util, io, contextlib
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
old = json.load(open(r"..\10x10\model\ladder_ci.json"))
spec = importlib.util.spec_from_file_location("v2", "ladder_v2_chart.py")
with contextlib.redirect_stdout(io.StringIO()):
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
# v3 levels straight from the per-game log; ep100 = the 23:02 restart (hardest-first) only
rows = [json.loads(l) for l in open("model/rapfi_match_log.jsonl", encoding="utf-8") if '"v3_ep' in l]
V3 = {}
for x in rows:
    ck = x["checkpoint"]
    if ck == "v3_ep100.pth" and x["time"] < "2026-10-22 23:02":
        continue
    if ck == "v3_ep200.pth" and x["rapfi_nodes"] == 32838:   # 10/23 spot check (repeats the same 10 games), not the ladder
        continue
    lv = V3.setdefault(ck[3:-4], {}).setdefault(x["rapfi_nodes"], [0, 0, 0])
    lv[{"ai_win": 0, "rapfi_win": 1, "draw": 2}[x["result"]]] += 1
rng = np.random.default_rng(0)
v3res = []
for k in sorted(V3, key=lambda s: int(s[2:])):
    lv = [(n, *c) for n, c in sorted(V3[k].items())]
    if len(lv) < 2:      # a single-level spot check (e.g. ep120 at 20k) cannot be fitted
        continue
    bs = []
    for _ in range(4000):
        bs.append(v2.eq([(n, *rng.multinomial(w + l + d, np.array([w, l, d]) / (w + l + d))) for n, w, l, d in lv]))
    bs = np.array(bs); bs = bs[np.isfinite(bs)]
    v3res.append((k, v2.eq(lv), *np.percentile(bs, [2.5, 16, 84, 97.5])))
    print(k, sum(sum(c) for c in V3[k].values()), "games", f"eq {v3res[-1][1]:,.0f}", "68%", np.round(v3res[-1][3:5]), "95%", np.round([v3res[-1][2], v3res[-1][5]]))
INK, MUTED, SURF, BEST, LATEST, V2C, V3C = "#1d1d1b", "#8a8780", "#fcfcfb", "#2a78d6", "#eb6834", "#2f9e6a", "#8e44ad"
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 6.5), facecolor=SURF, sharey=True, gridspec_kw={"width_ratios": [4, 1.3, 1.1]})
clip = lambda v: min(max(v, 60), 1000000)
for kind, col in (("best", BEST), ("latest", LATEST)):
    r = sorted([c for c in old if c["kind"] == kind], key=lambda c: c["ep"])
    x = [c["ep"] for c in r]
    if kind == "best":
        a1.fill_between(x, [clip(c["lo95"]) for c in r], [clip(c["hi95"]) for c in r], color=col, alpha=.08, lw=0)
        a1.fill_between(x, [clip(c["lo68"]) for c in r], [clip(c["hi68"]) for c in r], color=col, alpha=.2, lw=0)
        a1.plot(x, [c["eq"] for c in r], color=col, lw=2, label="best (승격 모델)")
    else:
        for c in r:
            a1.plot([c["ep"]] * 2, [clip(c["lo68"]), clip(c["hi68"])], color=col, lw=3, alpha=.4, solid_capstyle="butt")
        a1.plot(x, [c["eq"] for c in r], "o", color=col, ms=6, label="latest (승격 전 모델)")
a1.axvline(960, color=MUTED, ls=":", lw=1); a1.text(963, 70, "ep960 → v2 분기", color=MUTED, fontsize=9)
a1.set_title("NAPS (10x10 폴더, VCF 없는 엔진)", loc="left", color=INK)
a1.set_xlabel("에폭"); a1.legend(loc="upper left", frameon=False)
def panel(a, res, col, title, first_col=None):
    for i, (k, e, l95, l68, h68, h95) in enumerate(res):
        c = first_col if (first_col and i == 0) else col
        a.plot([i, i], [clip(l95), clip(h95)], color=c, lw=1.5, alpha=.35)
        a.plot([i, i], [clip(l68), clip(h68)], color=c, lw=7, alpha=.45, solid_capstyle="butt")
        a.plot(i, e, "o", color=c, ms=8, mec=SURF, mew=2)
        a.text(i + .15, e, f"{e:,.0f}", va="center", fontsize=9, color=INK)
    a.set_xticks(range(len(res))); a.set_xticklabels([r[0] for r in res]); a.set_xlim(-.5, len(res) - .2)
    a.set_title(title, loc="left", color=INK)
panel(a2, v2.res, V2C, "11x11_v2 (VCF 엔진)", BEST)
panel(a3, v3res, V3C, "v3 검벨 (콜드 스타트)")
best = max(c["eq"] for c in old)
for a in (a1, a2, a3):
    a.set_facecolor(SURF); a.set_yscale("log"); a.set_ylim(60, 1000000)
    a.grid(axis="y", color="#e4e2dc", lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for y in (959, 3000): a.axhline(y, color=MUTED, ls="--", lw=.8)
    a.axhline(best, color=BEST, ls=":", lw=1)
a1.text(545, 959 * 1.04, "여명 959", color=MUTED, fontsize=9); a1.text(545, 3000 * 1.04, "3,000 벽", color=MUTED, fontsize=9)
a1.text(545, best * 1.04, f"NAPS 최고 {best:,.0f}", color=BEST, fontsize=9)
a1.set_yticks([100, 300, 1000, 3000, 10000, 30000, 100000, 300000]); a1.set_yticklabels(["100", "300", "1,000", "3,000", "10,000", "30,000", "100,000", "300,000"])
a1.set_ylabel("Rapfi 대등 노드 (로그) — 막대/띠 68%, 옅은 띠·선 95%")
fig.tight_layout(); fig.savefig("model/ladder_all_v3_ci.png", dpi=120)
print("ok")
