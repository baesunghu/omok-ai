# v3 policy / value loss with key events (2026-10-23, user).
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
E, P, V = [], [], []
for l in open("training_report_v3.txt", encoding="utf-8"):
    m = re.search(r"Epoch (\d+) \| loss \S+ policy (\S+) value (\S+)", l)
    if m: E.append(int(m.group(1))); P.append(float(m.group(2))); V.append(float(m.group(3)))
EV = [(40, "래더 1,558"), (70, "래더 4,620"), (79, "메기 투입"), (100, "래더 5,214 · 문제 9/10"),
      (120, "p18 첫 해결 · ep100에 40%"), (130, "정책 절반 메기")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), facecolor="#fcfcfb", sharex=True)
a1.plot(E, P, color="#2a78d6", lw=1.4); a1.set_ylabel("정책 손실"); a1.set_ylim(1.35, 1.75)
a2.plot(E, V, color="#B05A17", lw=1.4); a2.set_ylabel("가치 손실"); a2.set_ylim(0.1, 0.3); a2.set_xlabel("에폭")
for a in (a1, a2):
    a.set_facecolor("#fcfcfb"); a.grid(axis="y", color="#e4e2dc", lw=.8)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for e, t in EV:
        a.axvline(e, color="#8a8780", ls=":", lw=1)
for i, (e, t) in enumerate(EV):
    a1.text(e + 0.8, 1.72 - (i % 3) * 0.035, t, fontsize=9, color="#4a5058")
a1.set_title("v3 정책·가치 손실과 주요 사건", loc="left")
a1.set_xlim(20, E[-1] + 2)
fig.tight_layout(); fig.savefig("model/v3_loss_events.png", dpi=120); print("ok", E[-1])
