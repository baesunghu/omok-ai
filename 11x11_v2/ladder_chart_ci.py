import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SP = r"C:\Users\My PC\Desktop\final\11x11_v2\model"  # ladder_hist.json / ladder_ci.json live here
rows = [r for r in json.load(open(SP + r"\ladder_hist.json")) if not r["dup"] and r["eq"]]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BEST, LATEST = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(13, 7), facecolor=SURF); ax.set_facecolor(SURF)
for y, lab, ls in ((1000, "1차 졸업 기준 1000", ":"), (2000, "2차 목표 2000", ":"), (959, "여명 959 (5단계)", "--")):
    ax.axhline(y, color=MUTED if ls == ":" else INK2, ls=ls, lw=1, zorder=1)
    below = ls == "--"   # Dawn sits just under the 1000 line: put her label below her line
    ax.text(533, y * (0.97 if below else 1.03), lab, color=INK2, fontsize=9, ha="left", va="top" if below else "bottom")
b = [r for r in rows if r["kind"] == "best"]; l = [r for r in rows if r["kind"] == "latest"]
CI = {(c["ep"], c["kind"]): c for c in json.load(open(SP + r"\ladder_ci.json"))}
for r in rows:   # use the refit (5 levels where a 5000-node run exists)
    c = CI.get((r["ep"], r["kind"]))
    if c: r["eq"] = c["eq"]
YMIN, YMAX = 60, 30000
clip = lambda v: min(max(v, YMIN), YMAX)
bx = [r["ep"] for r in b]
ax.fill_between(bx, [clip(CI[(e, "best")]["lo95"]) for e in bx], [clip(CI[(e, "best")]["hi95"]) for e in bx],
                color=BEST, alpha=.08, lw=0, zorder=0)
ax.fill_between(bx, [clip(CI[(e, "best")]["lo68"]) for e in bx], [clip(CI[(e, "best")]["hi68"]) for e in bx],
                color=BEST, alpha=.20, lw=0, zorder=0)
for r in l:
    c = CI[(r["ep"], "latest")]
    ax.plot([r["ep"]] * 2, [clip(c["lo95"]), clip(c["hi95"])], color=LATEST, lw=1, alpha=.35, zorder=1)
    ax.plot([r["ep"]] * 2, [clip(c["lo68"]), clip(c["hi68"])], color=LATEST, lw=3, alpha=.45, zorder=1, solid_capstyle="butt")
ax.plot([r["ep"] for r in b], [r["eq"] for r in b], color=BEST, lw=2, zorder=2)
for r in b:
    old = r["old_levels"]
    ax.scatter(r["ep"], r["eq"], s=70, color=SURF if old else BEST, edgecolor=BEST, linewidth=2, zorder=3)
ax.scatter([r["ep"] for r in l], [r["eq"] for r in l], s=70, marker="D", color=SURF, edgecolor=LATEST, linewidth=2, zorder=3)
LBL = {625: ("625 여명 (졸업, 5단계)", 0, -22), 645: ("645 (best 최고)", 22, 8), 740: ("740", 0, 10),
       795: ("795", 0, -22), 820: ("820", 0, -22), 840: ("840", 0, 10),
       855: ("855", -14, 10), 860: ("860", 0, 10), 870: ("870", 10, -14), 890: ("890", -16, 12), 905: ("905 (현재 best)", 26, -34), 920: ("920", 0, 10), 940: ("940 (LR 충격)", 0, -18), 960: ("960 (4단계)", -10, 12), 1005: ("1005 (롤백 충격 중 승격)", 0, -18), 1020: ("1020", 0, 10), 1050: ("1050 (게이트 12-5-3)", 0, 10), 900: ("900", -2, 12), 880: ("880 (5단계)", 14, -14), 545: ("545", 0, -18), 760: ("760", 0, 10)}
for r in rows:
    if r["ep"] in LBL:
        t, dx, dy = LBL[r["ep"]]
        ax.annotate(f"{t}\n{r['eq']:,}", (r["ep"], r["eq"]), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="bottom" if dy > 0 else "top", fontsize=8.5, color=INK)
ax.set_yscale("log"); ax.set_ylim(YMIN, YMAX)
ax.set_yticks([100, 300, 1000, 2000, 4000, 10000]); ax.set_yticklabels(["100", "300", "1,000", "2,000", "4,000", "10,000"])
ax.set_xlim(530, 1065)
ax.set_xlabel("에폭", color=INK2); ax.set_ylabel("Rapfi 동등 노드 (로그 눈금)", color=INK2)
ax.grid(True, which="major", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=BEST, marker="o", lw=2, ms=8, label="best 체크포인트"),
     Line2D([], [], color=LATEST, marker="D", lw=0, mfc=SURF, mew=2, ms=7, label="최신 체크포인트 (승격 안 된 것)"),
     Line2D([], [], color=BEST, marker="o", lw=0, mfc=SURF, mew=2, ms=8, label="초기 측정 (100노드 단계 포함, 비교 약함)"),
     __import__("matplotlib.patches", fromlist=["Patch"]).Patch(color=BEST, alpha=.35, label="오차 범위 68% (진한 띠, 약 ±1σ)"),
     __import__("matplotlib.patches", fromlist=["Patch"]).Patch(color=BEST, alpha=.12, label="오차 범위 95% (옅은 띠, 약 ±2σ, 위·아래 끝은 그래프 밖일 수 있음)")]
ax.legend(handles=h, loc="lower right", fontsize=9, frameon=False, labelcolor=INK2)
ax.set_title("Rapfi 대등 노드와 오차 범위 (실제 대국 결과 부트스트랩 4,000회) — ep1050까지 (960 이후 기록은 롤백 전 측정)", color=INK, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(r"C:\Users\My PC\Desktop\final\11x11_v2\model\rapfi_ladder_history_ci.png", dpi=120, facecolor=SURF)
print("saved")
