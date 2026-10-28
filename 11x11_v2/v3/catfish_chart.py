# v3 CATFISH figure for the report (2026-10-23): value loss before/after catfish, and per-epoch losses to each catfish.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
E, V, C = [], [], {}
for l in open("training_report_v3.txt", encoding="utf-8"):
    m = re.search(r"Epoch (\d+) \| loss \S+ policy \S+ value (\S+)", l)
    if not m: continue
    E.append(int(m.group(1))); V.append(float(m.group(2)))
    c = re.search(r"value0 (\d+)/(\d+)/(\d+), value_half (\d+)/(\d+)/(\d+), policy(?:0|_half) (\d+)/(\d+)/(\d+)", l)   # 10/24: policy0 -> policy_half at ep129
    if c: C[int(m.group(1))] = [int(x) for x in c.groups()]
INK, MUTED, SURF = "#1d1d1b", "#8a8780", "#fcfcfb"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), facecolor=SURF, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
a1.plot(E, V, color="#2a78d6", lw=1.2)
a1.axvline(79, color="#B05A17", ls="--", lw=1); a1.text(80, 0.285, "메기 투입 (ep79)", color="#B05A17", fontsize=10)
a1.set_ylim(0.1, 0.3); a1.set_ylabel("가치 손실"); a1.set_title("v3 가치 손실과 메기 대국 결과", loc="left", color=INK)
ep = sorted(C)
import numpy as np
for k, lab, col in ((0, "가치망 0 메기", "#B05A17"), (3, "가치망 절반 메기", "#8e44ad"), (6, "정책 메기 (ep129까지 0, 이후 절반)", "#2f9e6a")):
    # 10/24: games per catfish dropped 16 -> 8 at ep139, so plot the loss RATE (10-epoch rolling mean)
    r = np.array([100 * C[e][k + 1] / max(1, sum(C[e][k:k + 3])) for e in ep])
    sm = np.convolve(r, np.ones(10) / 10, mode="valid")
    a2.plot(ep, r, color=col, lw=.6, alpha=.25)
    a2.plot(ep[9:], sm, color=col, lw=1.8, label=lab)
for x, t in ((129, "정책 0 → 정책 절반"), (139, "메기당 16 → 8판")):
    a2.axvline(x, color=MUTED, ls=":", lw=1); a2.text(x + 1, 95, t, color=MUTED, fontsize=9)
a2.set_ylim(0, 100)
a2.set_ylabel("본 모델이 진 비율 % (굵은 선 = 10에폭 평균)"); a2.set_xlabel("에폭"); a2.legend(frameon=False, loc="upper left")
for a in (a1, a2):
    a.set_facecolor(SURF); a.grid(axis="y", color="#e4e2dc", lw=.8)
    for s in ("top", "right"): a.spines[s].set_visible(False)
a2.set_xlim(20, E[-1] + 1)
fig.tight_layout(); fig.savefig("model/catfish_v3.png", dpi=120); print("ok")
