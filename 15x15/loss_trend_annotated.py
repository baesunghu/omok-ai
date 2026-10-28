"""Loss curves of the current (NAPS) run with the 2026-09-26..28 pipeline
changes marked, so loss drops caused by target changes are not read as
strength gains. Writes model/loss_trend_annotated.png."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = r"C:\Users\My PC\Desktop\final\15x15\model\training_report_15x15.txt"
PAT = re.compile(r"Epoch (\d+)/\d+ \| Policy Loss: ([\d.]+) \| Value Loss: ([\d.]+)")
GATE = re.compile(r"Gate vs best \(\d+ games\): Win (\d+)% / Draw (\d+)% / Loss (\d+)% -> (\w+)")
rows = []
for line in open(REPORT, encoding="utf-8", errors="ignore"):
    m = PAT.search(line)
    if m:
        g = GATE.search(line)
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                     (int(g.group(1)), int(g.group(2)), int(g.group(3)), g.group(4)) if g else None))
# keep the last contiguous run (a reset shows up as an epoch number going down)
start = 0
for i in range(1, len(rows)):
    if rows[i][0] < rows[i - 1][0]:
        start = i
run = {}
gates = {}
for e, p, v, g in rows[start:]:
    run[e] = (p, v)   # later duplicates (restarts) overwrite
    if g:
        gates[e] = g
FROM = 1  # 2026-10-08 (user): show the whole NAPS run, not just ep300+
ep = sorted(e for e in run if e >= FROM)
pol = [run[e][0] for e in ep]
val = [run[e][1] for e in ep]


def smooth(x, k=5):
    return [sum(x[max(0, i - k + 1):i + 1]) / len(x[max(0, i - k + 1):i + 1]) for i in range(len(x))]


EVENTS = [
    (285, "vlfix + mate-dist + survival head"),
    (370, "Hard-7 value label removed"),
    (371, "PCR + unified pool"),
    (408, "value head reset"),
    (452, "z/q value mix"),
    (492, "MCTS-Solver"),
    (494, "forced playouts, league 6, 90-ply cap"),
    (521, "league PCR"),
    (625, "GRADUATED: NAPS-625 Dawn"),
    (721, "LR halved + league anchors"),
    (740, "anchor rotation"),
    (805, "12 league opponents, buffer 196k, no vs-Det eval"),
    (905, "q weight 0.5 -> 0.2, half-life 20"),
    (911, "gate: loss -0.75, promote if > 50%"),
]
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
for ax, y, name, color in ((axes[0], pol, "Policy loss", "#2a6fdb"), (axes[1], val, "Value loss", "#d9480f")):
    ax.plot(ep, y, color=color, alpha=0.25, lw=1)
    ax.plot(ep, smooth(y), color=color, lw=2, label=f"{name} (5-epoch mean)")
    ax.set_ylabel(name)
    ax.grid(alpha=0.2)
    top = ax.get_ylim()[1]
    for i, (e, label) in enumerate(EVENTS):
        if ep and ep[0] <= e <= ep[-1]:
            ax.axvline(e, color="#555", ls="--", lw=0.8)
            if ax is axes[0]:
                ax.text(e + 1, top * (0.98 - 0.07 * (i % 4)), label, fontsize=7, color="#333", va="top")
    # 2026-10-08 (user): reference line at Dawn's (graduated ep625) loss
    if 625 in run:
        dawn = run[625][0] if ax is axes[0] else run[625][1]
        ax.axhline(dawn, color="#7048e8", ls="-.", lw=1.2, label=f"Dawn ep625 ({dawn:.4f})")
    ax.legend(loc="lower left", fontsize=9)
ge = [e for e in sorted(gates) if e >= FROM]
gw = [gates[e][0] for e in ge]
gd = [gates[e][1] for e in ge]
axg = axes[2]
axg.bar(ge, gw, width=3, color="#2f9e44", label="gate win %")
axg.bar(ge, gd, width=3, bottom=gw, color="#adb5bd", label="gate draw %")
# 2026-10-11 (user): gate rule changed at ep911 -> draw a divider and use each era's score/threshold
GATE_RULE_EP = 911
lo, hi = (ge[0] - 3, ge[-1] + 3) if ge else (0, 1)
if lo < GATE_RULE_EP:
    axg.hlines(55, lo, min(hi, GATE_RULE_EP), color="#c92a2a", ls=":", lw=1, label="promotion line (W + D/2 >= 55%, until ep910)")
if hi > GATE_RULE_EP:
    axg.hlines(50, max(lo, GATE_RULE_EP), hi, color="#c92a2a", ls="-", lw=1.2, label="promotion line (W + D/2 - 0.75 L > 50%, from ep911)")
if lo < GATE_RULE_EP < hi:
    axg.axvline(GATE_RULE_EP, color="#c92a2a", ls="-", lw=1.6, alpha=0.7)
    axg.text(GATE_RULE_EP + 0.5, 101, "gate rule change", fontsize=7, color="#c92a2a", va="top")
gl = [gates[e][2] for e in ge]
gs = [w + d / 2 - (0.75 * l if e >= GATE_RULE_EP else 0) for e, w, d, l in zip(ge, gw, gd, gl)]
axg.scatter(ge, gs, marker="_", s=140, linewidths=2.5, color="#212529", zorder=3, label="gate score (rule of its era)")
if len(ge) <= 30:  # numbers only when they stay readable (short ranges)
    for e, s in zip(ge, gs):
        axg.text(e, s + 1.5, f"{s:.1f}", ha="center", va="bottom", fontsize=7, color="#212529")
for e in ge:
    if gates[e][3] == "PROMOTED":
        axg.plot(e, gates[e][0] + gates[e][1] + 4, marker="v", color="#c92a2a", ms=6)
axg.set_ylim(0, 105)
axg.set_ylabel("Gate vs best (%)  ▼ = promoted")
axg.grid(alpha=0.2)
for e, _ in EVENTS:
    if ge and ge[0] <= e <= ge[-1]:
        axg.axvline(e, color="#555", ls="--", lw=0.8)
axg.legend(loc="upper right", fontsize=8, ncol=2)
axes[2].set_xlabel("epoch")
fig.suptitle("NAPS run: losses with pipeline changes marked (loss drops after target changes are NOT strength gains)",
             fontsize=11)
fig.tight_layout()
fig.savefig(r"C:\Users\My PC\Desktop\final\15x15\model\loss_trend_annotated.png", dpi=110)
print("saved", len(ep), "epochs", ep[0], "-", ep[-1])
