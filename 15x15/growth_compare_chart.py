"""11x11 NAPS vs 15x15: Rapfi equal-strength nodes against training hours since the supervised start (2026-10-15).

Both runs start from a supervised-learning model at epoch 0 (NAPS: Gen12 SL, 2026-09-23 restart; 15x15: Rapfi-100 SL).
x = training hours (epoch-to-epoch intervals summed; gaps > 2 h such as the exam pause or the BSOD count as one median
epoch). y = fitted equal-node point with bootstrap 68/95% bands. Writes model/growth_compare_11v15.png.
"""
import json
import math
import os
import random
import re
import statistics
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FINAL = r"C:\Users\My PC\Desktop\final"
OUT = os.path.join(FINAL, "15x15", "model", "growth_compare_11v15.png")
for f in ("Malgun Gothic", "NanumGothic"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False


def T(s):
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def epoch_hours(report, since):
    """epoch -> training hours since the run start (first occurrence of each epoch after `since`)."""
    xs = []
    for line in open(report, encoding="utf-8", errors="ignore"):
        m = re.match(r"\[(\S+ \S+)\] Epoch (\d+)/", line)
        if m and m.group(1) >= since:
            ep = int(m.group(2))
            if not xs or ep > xs[-1][0]:
                xs.append((ep, T(m.group(1))))
    gaps = [(b[1] - a[1]).total_seconds() / 3600 for a, b in zip(xs, xs[1:])]
    med = statistics.median(gaps)
    h, out = med, {xs[0][0]: med, xs[0][0] - 1: 0.0}      # the first logged epoch took ~one median epoch
    for (a, b), g in zip(zip(xs, xs[1:]), gaps):
        h += g if g <= 2.0 else med
        out[b[0]] = h
    return out


def fit(levels):
    xs = [math.log10(n) for n, s in levels]
    ws = [len(s) for n, s in levels]
    ys = []
    for n, s in levels:
        p = min(max(sum(s) / len(s), .03), .97)
        ys.append(400 * math.log10(p / (1 - p)))
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    sl = sxy / sxx if sxx else 0
    return 10 ** min(mx - my / sl, 6) if sl < 0 else 1e6


def boot(levels, n=2000, seed=0):
    rnd = random.Random(seed)
    v = sorted(fit([(k, rnd.choices(s, k=len(s))) for k, s in levels]) for _ in range(n))
    q = lambda p: v[int(p * (n - 1))]
    return q(.025), q(.16), q(.84), q(.975)


# ---- 11x11 NAPS (2nd run, from Gen12 SL; ladder from ep540) ----
h11 = epoch_hours(os.path.join(FINAL, "10x10", "model", "training_report_11x11.txt"), "2026-09-23 16:00")
naps = []
for r in json.load(open(os.path.join(FINAL, "10x10", "model", "ladder_ci.json"), encoding="utf-8")):
    if r["ep"] in h11 and r["ep"] <= 960:
        naps.append((h11[r["ep"]], r["ep"], r["eq"], r["lo95"], r["lo68"], r["hi68"], r["hi95"], r["kind"]))
naps.sort()

# ---- 15x15 ----
h15 = epoch_hours(os.path.join(FINAL, "15x15", "model", "training_report_15x15.txt"), "2026-10-14")
h15[0] = 0.0
games = {}
for line in open(os.path.join(FINAL, "15x15", "model", "rapfi_match_log.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    games.setdefault(r["checkpoint"], {}).setdefault(r["rapfi_nodes"], []).append(
        {"ai_win": 1.0, "draw": .5}.get(r["result"], 0.0))
b15 = []
for line in open(os.path.join(FINAL, "15x15", "model", "rapfi_ladder_log.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    ck = r["checkpoint"]
    m = re.search(r"ep(\d+)", ck)
    if m:
        ep = int(m.group(1))
    elif "cand_" in ck:                                    # gate candidate = the epoch logged just before its timestamp
        ep = 25
    else:
        continue
    lv = [(L["nodes"], games[ck][L["nodes"]][-L["games"]:]) for L in r["levels"]]
    lo95, lo68, hi68, hi95 = boot(lv)
    b15.append((h15.get(ep, 0.0), ep, fit(lv), lo95, lo68, hi68, hi95, "cand" if "cand_" in ck else "latest"))
b15.sort()

# ---- plot ----
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)
for rows, col, name in ((naps, "#2f6db5", "11×11 NAPS"), (b15, "#d0632a", "15×15")):
    x = [r[0] for r in rows]
    ax.fill_between(x, [r[3] for r in rows], [r[6] for r in rows], color=col, alpha=.10, lw=0)
    ax.fill_between(x, [r[4] for r in rows], [r[5] for r in rows], color=col, alpha=.20, lw=0)
    ax.plot(x, [r[2] for r in rows], "-o", color=col, ms=4.5, lw=1.8, label=name)
    for r in rows:
        if r[1] in (0, 20, 40, 60, 80, 100, 540, 625, 645, 740, 840, 860, 960) or r[7] == "cand":
            ax.annotate(f"ep{r[1]}", (r[0], r[2]), textcoords="offset points", xytext=(4, 6), fontsize=7.5, color=col)
# 960 re-measured with the 25-opening protocol (the 8-opening run above gave 8,563)
x960 = h11.get(960)
if x960 is not None:
    ax.plot([x960], [3454], marker="*", ms=13, color="#2f6db5", mec="white", zorder=5)
    ax.annotate("ep960 재측정 3,454\n(25개 시작 국면)", (x960, 3454), textcoords="offset points", xytext=(10, -30),
                fontsize=8, color="#2f6db5")
ax.axvline(h15[46], color="#d0632a", ls=":", lw=1)
ax.text(h15[46], 160, " 학습률 1e-4\n (ep47~)", fontsize=8, color="#d0632a", va="bottom")
ax.set_yscale("log")
ax.set_ylim(80, 30000)
ax.set_xlim(-8, 300)
ax.set_xlabel("지도학습 출발점(에포크 0)부터 학습 시간 (시간, 정지 구간 제외)")
ax.set_ylabel("Rapfi 대등국면 (노드, 로그 눈금)")
ax.set_title("11×11 NAPS vs 15×15: 학습 시간에 따른 Rapfi 대등국면 (띠: 부트스트랩 68% / 95%)")
ax.grid(True, which="both", alpha=.25)
ax.legend(loc="upper left")
ax.text(0.35, 0.98, "11×11: 시작 국면 8개, 단계당 20판(ep960 별표만 25개) · 15×15: 시작 국면 25개, 단계당 40판\n"
        "판 크기가 달라 Rapfi 노드당 강도도 다름 → 절대값보다 기울기(성장 속도)를 비교",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#555")
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
for name, rows in (("11x11", naps), ("15x15", b15)):
    for r in rows:
        print(name, f"{r[0]:7.1f}h ep{r[1]:<4} eq {r[2]:8.0f}  95% {r[3]:8.0f}-{r[6]:8.0f}")
