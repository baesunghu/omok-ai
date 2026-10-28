"""Head-to-head summary (2026-10-10): spectator-board matches vs Dawn and matches vs the gomokuwithai Hard AI,
with 95% intervals (per-game score 1 / 0.5 / 0, normal approximation). Output: model/matchup_summary.png"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
os.chdir(os.path.dirname(os.path.abspath(__file__)))
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BEST, DAWN = "#2a78d6", "#eb6834"


def stats(xs):
    n = len(xs); m = sum(xs) / n
    se = math.sqrt(max(sum(x * x for x in xs) / n - m * m, 1e-9) / n)
    return m * 100, 1.96 * se * 100, n


spec = [json.loads(l) for l in open("model/spectate_log.jsonl", encoding="utf-8")]


def vs(a, b):
    out = []
    for r in spec:
        if {r["black"], r["white"]} == {a, b}:
            side = "black" if r["black"] == a else "white"
            out.append(.5 if r["result"] == "draw" else (1.0 if r["result"] == side else 0.0))
    return out


def gwa(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [1.0 if r["result"] == "dawn" else .5 if r["result"] == "draw" else 0.0 for r in rows]


bars = [("820 vs 여명", stats(vs("ep820", "dawn")), BEST, "관전 (300 시뮬)"),
        ("860 vs 여명", stats(vs("ep860", "dawn")), BEST, "관전 (300 시뮬)"),
        ("880 vs 여명", stats(vs("ep880", "dawn")), BEST, "관전 (300 시뮬)"),
        ("870 vs 860", stats(vs("ep870", "ep860")), MUTED, "관전 (300 시뮬)"),
        ("여명 vs 사이트 AI", stats(gwa("model/gomokuwithai_match_log.jsonl")), DAWN, "Gomoku With AI Hard (1100 시뮬)"),
        ("880 vs 사이트 AI", stats(gwa("model/gomokuwithai_match_model_MCTS_ep880_11x11_log.jsonl")), BEST, "Gomoku With AI Hard (1100 시뮬)")]
fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURF); ax.set_facecolor(SURF)
ys = list(range(len(bars)))[::-1]
for y, (lab, (m, e, n), col, kind) in zip(ys, bars):
    ax.barh(y, m, height=.56, color=col, alpha=.85, zorder=2)
    ax.plot([max(0, m - e), min(100, m + e)], [y, y], color=INK, lw=1.4, zorder=3)
    ax.text(min(100, m + e) + 1.2, y, f"{m:.1f}%  (±{e:.1f}, {n}판)", va="center", fontsize=9.5, color=INK)
ax.axvline(50, color=INK2, ls="--", lw=1, zorder=1)
ax.text(50.6, -0.62, "50% (대등)", color=INK2, fontsize=9)
ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=10.5, color=INK)
ax.set_xlim(0, 112); ax.set_xlabel("앞 모델의 점수 (승 1, 무 0.5) · 막대 끝 선 = 95% 범위", color=INK2)
ax.grid(True, axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.set_title("직접 대결 요약 — 관전석과 Gomoku With AI (2026-10-10)", color=INK, fontsize=12, loc="left")
fig.tight_layout(); fig.savefig("model/matchup_summary.png", dpi=120, facecolor=SURF)
print("saved")
