"""Decisive vs draw rates over epochs (2026-10-13, user request).
Self-play (16 games/epoch), vs-League (12/epoch), gate (20 per 5 epochs) from training_report_11x11.txt.
Output: model/draw_trend.png"""
import re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FROM = int(sys.argv[1]) if len(sys.argv) > 1 else 850
sp, lg, gt = {}, {}, {}
for l in open("model/training_report_11x11.txt", encoding="utf-8", errors="ignore"):
    m = re.search(r"Epoch (\d+)/\d+", l)
    if not m: continue
    e = int(m[1])
    if e < FROM: continue
    a = re.search(r"Self-Play \(this epoch, (\d+) games\): X-win (\d+) / O-win (\d+) / Draw (\d+)", l)
    if a: sp[e] = (int(a[2]), int(a[3]), int(a[4]))
    b = re.search(r"vs-League \([^)]*\): MCTS-win (\d+) / League-win (\d+) / Draw (\d+)", l)
    if b: lg[e] = (int(b[1]), int(b[2]), int(b[3]))
    c = re.search(r"Gate vs best \((\d+) games\): Win (\d+)% / Draw (\d+)% / Loss (\d+)%", l)
    if c: gt[e] = (int(c[2]), int(c[3]), int(c[4]))
def roll(d, idx, k=5):
    xs = sorted(d); out = []
    for i, x in enumerate(xs):
        w = xs[max(0, i - k + 1):i + 1]
        out.append((x, 100 * sum(d[j][idx] for j in w) / sum(sum(d[j]) for j in w)))
    return out
fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for a_, d, title, labels in ((ax[0], sp, "Self-play (16 games/epoch, 5-epoch rolling %)", ("X win", "O win", "Draw")),
                             (ax[1], lg, "vs-League (12 games/epoch, 5-epoch rolling %)", ("training model win", "league win", "Draw"))):
    for i, (lab, col) in enumerate(zip(labels, ("#2a6fdb", "#e0663a", "#8a8f98"))):
        r = roll(d, i); a_.plot([x for x, _ in r], [y for _, y in r], label=lab, color=col, lw=2 if lab == "Draw" else 1.4)
    a_.set_title(title, loc="left"); a_.set_ylim(0, 100); a_.legend(loc="upper left", fontsize=8); a_.grid(alpha=.3)
xs = sorted(gt)
ax[2].bar(xs, [gt[x][0] for x in xs], 3, color="#2f9e44", label="gate win %")
ax[2].bar(xs, [gt[x][1] for x in xs], 3, bottom=[gt[x][0] for x in xs], color="#adb5bd", label="gate draw %")
ax[2].bar(xs, [gt[x][2] for x in xs], 3, bottom=[gt[x][0] + gt[x][1] for x in xs], color="#e03131", label="gate loss %")
ax[2].set_title("Gate vs best (20 games per 5 epochs)", loc="left"); ax[2].set_ylim(0, 100); ax[2].legend(loc="upper left", fontsize=8)
for a_ in ax:
    for e, t in ((911, "gate -0.75"), (966, "temp 6"), (1002, "960 ladder promo"), (1041, "clean rollback, temp 4")):
        if e >= FROM:
            a_.axvline(e, color="k", ls=":", lw=.8)
    a_.set_xlim(FROM - 2, max(list(sp) + list(gt)) + 3)
for e, t in ((911, "gate -0.75"), (966, "temp 6"), (1002, "960 promo"), (1041, "rollback, temp 4")):
    if e >= FROM: ax[0].text(e + 1, 95, t, fontsize=8, va="top")
ax[2].set_xlabel("epoch")
fig.tight_layout(); fig.savefig("model/draw_trend.png", dpi=105)
for name, d in (("self-play", sp), ("league", lg)):
    for lo, hi in ((FROM, 965), (966, 1040), (1047, 9999)):
        es = [e for e in d if lo <= e <= hi]
        if es:
            t = [sum(d[e][i] for e in es) for i in range(3)]; n = sum(t)
            print(f"{name:9s} ep{lo}-{min(hi, max(es))}: draw {100*t[2]/n:4.1f}%  decisive {100*(t[0]+t[1])/n:4.1f}%  (n={n})")
for lo, hi in ((FROM, 965), (966, 1040), (1047, 9999)):
    es = [e for e in gt if lo <= e <= hi]
    if es: print(f"gate      ep{lo}-{min(hi, max(es))}: draw {sum(gt[e][1] for e in es)/len(es):4.1f}%  (checks {len(es)})")
