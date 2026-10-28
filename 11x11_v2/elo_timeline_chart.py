# Equal points as Elo over wall-clock time: NAPS (ladder_ci.json), v2, v3 (2026-10-22, user).
# Elo vs Rapfi 100 nodes = K * log10(eq / 100), K = 304 Elo per 10x nodes (v3 ep100 fitted slope), same K for every run
# so the Elo axis is a relabelled log axis; bars are the 68% bootstrap intervals.
import json, datetime as dt, numpy as np, matplotlib, importlib.util, io, contextlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as md
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
K = 304
elo = lambda n: K * np.log10(max(n, 1) / 100)
old = json.load(open(r"..\10x10\model\ladder_ci.json"))
with contextlib.redirect_stdout(io.StringIO()):
    spec = importlib.util.spec_from_file_location("c", "ladder_all_v3_chart.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
T = lambda s: dt.datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
v2t = {"ep20": "2026-10-22 03:39", "ep30": "2026-10-22 08:00", "ep40": "2026-10-22 16:00"}
v3t = {"ep40": "2026-10-22 17:47", "ep70": "2026-10-22 20:25", "ep100": "2026-10-22 23:36", "ep170": "2026-10-23 09:10", "ep200": "2026-10-23 14:09", "ep300": "2026-10-24 04:16", "ep400": "2026-10-24 17:35"}
INK, MUTED, SURF, NAPS, V2C, V3C = "#1d1d1b", "#8a8780", "#fcfcfb", "#2a78d6", "#2f9e6a", "#8e44ad"
fig, (ax, az) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURF, sharey=True, gridspec_kw={"width_ratios": [3.2, 1]}); ax.set_facecolor(SURF); az.set_facecolor(SURF)
def series(pts, col, lab, line=True):
    pts.sort()
    for t, e, lo, hi in pts:
        ax.plot([t, t], [elo(lo), elo(hi)], color=col, lw=4, alpha=.35, solid_capstyle="butt")
    ax.plot([p[0] for p in pts], [elo(p[1]) for p in pts], "o-" if line else "o", color=col, ms=6, lw=1.5, mec=SURF, label=lab)
best = [c_ for c_ in old if c_["kind"] == "best"]
series([(T(x["start"]), x["eq"], x["lo68"], x["hi68"]) for x in best], NAPS, "NAPS (승격 모델)")
series([(T(v2t[k]), e, l68, h68) for k, e, l95, l68, h68, h95 in c.v2.res if k in v2t], V2C, "v2 (VCF 엔진)")
v3 = [(T(v3t[k]), e, l68, h68) for k, e, l95, l68, h68, h95 in c.v3res]
series(v3, V3C, "v3 검벨 (콜드 스타트)")
ax.annotate("v3 →", (v3[-1][0], elo(v3[-1][1])), xytext=(-60, 40), textcoords="offset points", color=V3C, fontsize=10, weight="bold")
eps = [(int(k[2:]), e, l68, h68) for k, e, l95, l68, h68, h95 in c.v3res]
for x, e, lo, hi in eps:
    az.plot([x, x], [elo(lo), elo(hi)], color=V3C, lw=6, alpha=.35, solid_capstyle="butt")
    az.annotate(f"{elo(e):+.0f} ({e:,.0f}노드)", (x, elo(e)), xytext=(8, -4), textcoords="offset points", color=V3C, fontsize=9, weight="bold")
az.plot([p[0] for p in eps], [elo(p[1]) for p in eps], "o-", color=V3C, ms=7, mec=SURF)
napsbest = max(x["eq"] for x in old)
for n, lab in ((959, "여명"), (3000, "3천 벽"), (napsbest, "NAPS 최고")):
    az.axhline(elo(n), color=NAPS if lab == "NAPS 최고" else MUTED, ls=":" if lab == "NAPS 최고" else "--", lw=.8)
az.text(2, elo(napsbest) + 8, f"NAPS 최고 {elo(napsbest):+.0f}", color=NAPS, fontsize=9)
az.set_xlim(0, 420); az.set_xlabel("v3 에폭 (10/22~24)"); az.set_title("v3 확대", loc="left", color=INK)
az.grid(axis="y", color="#e4e2dc", lw=.8); az.set_axisbelow(True)
for sp in ("top", "right"): az.spines[sp].set_visible(False)
for n, lab in ((959, "여명 959노드"), (3000, "3,000 벽")):
    ax.axhline(elo(n), color=MUTED, ls="--", lw=.8); ax.text(T("2026-10-08 12:00"), elo(n) + 8, f"{lab} = {elo(n):+.0f}", color=MUTED, fontsize=9)
ax.axvline(T("2026-09-29 14:22"), color=MUTED, ls=":", lw=1); ax.text(T("2026-09-29 18:00"), elo(70), "여명 졸업 9/29", color=MUTED, fontsize=9)
ax.set_xlim(T("2026-09-28 00:00"), T("2026-10-25 00:00"))
ax.xaxis.set_major_formatter(md.DateFormatter("%m/%d"))
ax.set_ylabel(f"Elo (Rapfi 100노드 = 0, 노드 10배 = {K} Elo)"); ax.set_ylim(elo(60), elo(400000))
ax.grid(axis="y", color="#e4e2dc", lw=.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False)
ax.set_title("대등점을 Elo로: 날짜별 (막대 = 68% 구간)", loc="left", color=INK)
fig.tight_layout(); fig.savefig("model/elo_timeline.png", dpi=120); print("ok")
