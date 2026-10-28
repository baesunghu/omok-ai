# 11x11_v2 equal-point chart with bootstrap 68/95% bands (2026-10-22). Data: rapfi_ladder_watch.log / ladder_ep40_rest.log.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
D = {  # label: [(nodes, W, L, D)]
    "ep960\n(최고)": [(1000, 21, 10, 9), (3000, 18, 10, 12), (10000, 6, 29, 5)],
    "ep20": [(900, 19, 12, 9), (2710, 18, 11, 11), (8120, 9, 23, 8), (27080, 2, 31, 7)],
    "ep30": [(900, 19, 14, 7), (2710, 13, 17, 10), (8120, 11, 23, 6), (27080, 6, 31, 3)],
    "ep40": [(900, 25, 8, 6), (2710, 16, 18, 6), (8120, 9, 23, 8), (27080, 1, 37, 2)],
}
def eq(levels):
    x, y = [], []
    for n, w, l, d in levels:
        s = min(max((w + .5 * d) / (w + l + d), .02), .98)
        x.append(np.log10(n)); y.append(-400 * np.log10(1 / s - 1))
    a, b = np.polyfit(x, y, 1)
    return 10 ** (-b / a) if a < 0 else np.nan
rng = np.random.default_rng(0)
res = []
for k, lv in D.items():
    bs = []
    for _ in range(4000):
        r = []
        for n, w, l, d in lv:
            g = rng.multinomial(w + l + d, np.array([w, l, d]) / (w + l + d))
            r.append((n, *g))
        bs.append(eq(r))
    bs = np.array(bs); bs = bs[np.isfinite(bs)]
    res.append((k, eq(lv), *np.percentile(bs, [2.5, 16, 84, 97.5])))
    print(k.replace("\n", " "), round(res[-1][1]), [round(v) for v in res[-1][2:]])
INK, MUTED, SURF, C = "#1d1d1b", "#8a8780", "#fcfcfb", "#2a78d6"
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURF); ax.set_facecolor(SURF)
xs = np.arange(len(res))
for i, (k, e, l95, l68, h68, h95) in enumerate(res):
    col = "#eb6834" if i == 0 else C
    ax.plot([i, i], [l95, h95], color=col, lw=1.5, alpha=.35)
    ax.plot([i, i], [l68, h68], color=col, lw=7, alpha=.45, solid_capstyle="butt")
    ax.plot(i, e, "o", color=col, ms=9, mec=SURF, mew=2, zorder=3)
    ax.text(i + .12, e, f"{e:,.0f}", va="center", color=INK, fontsize=11)
ax.axhline(res[0][1], color="#eb6834", ls="--", lw=1)
ax.set_yscale("log"); ax.set_ylim(700, 12000)
ax.set_yticks([1000, 2000, 3000, 5000, 10000]); ax.set_yticklabels(["1,000", "2,000", "3,000", "5,000", "10,000"])
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in res], fontsize=11); ax.set_xlim(-.5, len(res) - .3)
ax.set_ylabel("Rapfi 대등 노드 (로그)", color=INK)
ax.set_title("11x11_v2 대등점 — 굵은 막대 68%, 가는 선 95% (부트스트랩)", color=INK, loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e4e2dc", lw=.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("model/ladder_v2_ci.png", dpi=130)
