# 10x10-folder 11x11 run (NAPS, ep540~) + 11x11_v2 equal points with 68/95% bands, one log y-axis (2026-10-22).
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
old = json.load(open(r"..\10x10\model\ladder_ci.json"))
spec = importlib.util.spec_from_file_location("v2", "ladder_v2_chart.py")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
INK, MUTED, SURF, BEST, LATEST, V2 = "#1d1d1b", "#8a8780", "#fcfcfb", "#2a78d6", "#eb6834", "#2f9e6a"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURF, sharey=True, gridspec_kw={"width_ratios": [4, 1.3]})
clip = lambda v: min(max(v, 60), 30000)
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
for y, lab in ((959, "여명 959"), (2000, "2차 목표 2000")):
    a1.axhline(y, color=MUTED, ls="--", lw=.8); a1.text(a1.get_xlim()[0] if False else 545, y * 1.04, lab, color=MUTED, fontsize=9)
a1.set_title("기존 11x11 (10x10 폴더, VCF 없는 엔진)", loc="left", color=INK)
a1.set_xlabel("에폭"); a1.legend(loc="upper left", frameon=False)
for i, (k, e, l95, l68, h68, h95) in enumerate(v2.res):
    col = BEST if i == 0 else V2
    a2.plot([i, i], [l95, h95], color=col, lw=1.5, alpha=.35)
    a2.plot([i, i], [l68, h68], color=col, lw=7, alpha=.45, solid_capstyle="butt")
    a2.plot(i, e, "o", color=col, ms=8, mec=SURF, mew=2)
    a2.text(i + .15, e, f"{e:,.0f}", va="center", fontsize=9, color=INK)
a2.axhline(959, color=MUTED, ls="--", lw=.8); a2.axhline(2000, color=MUTED, ls="--", lw=.8)
a2.set_xticks(range(len(v2.res))); a2.set_xticklabels([r[0] for r in v2.res]); a2.set_xlim(-.5, len(v2.res) - .2)
a2.set_title("11x11_v2 (VCF 엔진)", loc="left", color=INK)
for a in (a1, a2):
    a.set_facecolor(SURF); a.set_yscale("log"); a.set_ylim(60, 30000)
    a.grid(axis="y", color="#e4e2dc", lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
a1.set_yticks([100, 300, 1000, 3000, 10000, 30000]); a1.set_yticklabels(["100", "300", "1,000", "3,000", "10,000", "30,000"])
a1.set_ylabel("Rapfi 대등 노드 (로그) — 막대/띠 68%, 옅은 띠·선 95%")
fig.tight_layout(); fig.savefig("model/ladder_all_ci.png", dpi=120)
print("ok")
