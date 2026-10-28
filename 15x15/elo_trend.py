"""
Retrospective Elo trajectory for the "best" checkpoint lineage, computed
purely from the gate-check history already logged in
training_report_11x11.txt -- no new games played, no live training
touched.

Each gate check ("Gate vs best (20 games): Win X% / Draw Y% / Loss Z%")
is treated as a 20-game match between the candidate (that epoch's
snapshot) and the CURRENT tracked best. The match score gives a
maximum-likelihood Elo DIFFERENCE via the standard logistic formula:

    elo_diff = 400 * log10(W / (1 - W))      where W = score fraction

If the candidate was PROMOTED, it becomes the new "best" and its
computed Elo (best_elo + elo_diff) carries forward as the new baseline.
If not promoted, best_elo is unchanged (the candidate's Elo is still
computed/logged for reference, but doesn't affect the chain).

This is a standard method (used e.g. by AlphaZero-lineage papers' own
"vs previous best" Elo curves) for turning a chain of pairwise gate
matches into a single running rating -- not a full round-robin Elo pool,
so treat it as directionally correct/order-of-magnitude, not a precise
absolute rating.

Usage:
    python elo_trend.py
"""
import re
import math

REPORT_PATH = r"C:\Users\My PC\Desktop\final\15x15\model\training_report_11x11.txt"

# Only look at the CURRENT run -- reusing status_check.py's own boundary
# logic (any point where the epoch counter goes <= the previous line's
# epoch, not just an explicit "Epoch 1" reset -- a silent mid-run reset
# without its own "Epoch 1" line, e.g. a restart that resumed from a
# stale train_state.json, needs the same treatment) rather than a naive
# "Epoch 1" text match, which misses exactly that case.
with open(REPORT_PATH, encoding="utf-8") as f:
    all_lines = f.readlines()

epoch_line_re = re.compile(r"Epoch (\d+)/20000 \| Policy Loss:")
prev_epoch = None
boundary_idx = 0
for i, l in enumerate(all_lines):
    m = epoch_line_re.search(l)
    if not m:
        continue
    epoch = int(m.group(1))
    if prev_epoch is not None and epoch <= prev_epoch:
        boundary_idx = i
    prev_epoch = epoch

lines = all_lines[boundary_idx:]

gate_re = re.compile(
    r"Epoch\s+(\d+)/20000.*?Gate vs best \((\d+) games\): "
    r"Win\s+(\d+)% / Draw\s+(\d+)% / Loss\s+(\d+)% -> (PROMOTED|kept)"
)

matches = []
for l in lines:
    m = gate_re.search(l)
    if m:
        epoch, n_games, win_pct, draw_pct, loss_pct, outcome = m.groups()
        matches.append((int(epoch), int(n_games), int(win_pct), int(draw_pct), int(loss_pct), outcome))

print(f"Found {len(matches)} gate checks in the current run.\n")

BASELINE_ELO = 1500.0
best_elo = BASELINE_ELO
history = [(0, best_elo)]  # (epoch, elo) for the promoted-lineage chain

for epoch, n_games, win_pct, draw_pct, loss_pct, outcome in matches:
    score = (win_pct + 0.5 * draw_pct) / 100.0
    # Clamp away from exact 0/1 -- a 20-game 100%/0% match doesn't mean
    # "infinite" Elo gap, just that 20 games isn't enough to measure it;
    # clamping to a 1/(2N) style bound keeps the estimate finite and
    # roughly matches the resolution the sample size can actually support.
    score = min(max(score, 1 / (2 * n_games)), 1 - 1 / (2 * n_games))
    elo_diff = 400 * math.log10(score / (1 - score))
    candidate_elo = best_elo + elo_diff
    promoted = outcome == "PROMOTED"
    if promoted:
        best_elo = candidate_elo
        history.append((epoch, best_elo))
    marker = "PROMOTED ->" if promoted else "  (kept)  "
    print(f"  epoch {epoch:>5} | {win_pct:>3}%/{draw_pct:>3}%/{loss_pct:>3}% | "
          f"candidate_elo={candidate_elo:>8.1f} | {marker} best_elo={best_elo:>8.1f}")

print(f"\nBaseline Elo: {BASELINE_ELO:.0f} (arbitrary anchor -- only the DELTA "
      f"from here is meaningful, not the absolute number)")
print(f"Current best Elo: {best_elo:.1f}  (+{best_elo - BASELINE_ELO:.1f} over "
      f"{len(history) - 1} promotions across {matches[-1][0] if matches else 0} epochs)")

# Simple linear-regression slope of the promoted-lineage Elo vs epoch, to
# quantify "roughly linear" vs "accelerating" growth.
if len(history) >= 3:
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    # R^2 of the linear fit -- close to 1 means genuinely linear, well
    # below 1 means the growth curve is bending (accelerating or decaying).
    pred = [my + slope * (x - mx) for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    print(f"\nLinear fit over the promoted-lineage Elo curve: slope={slope:.3f} Elo/epoch, R^2={r2:.3f}")
    print("(R^2 close to 1 -> genuinely linear growth over this run; well below 1 -> curve is bending)")

# Optional: render a quick plot if matplotlib is available (project already
# uses it for loss_trend.png).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("Elo (arbitrary baseline 1500)")
    plt.title("Promoted-lineage Elo trajectory (from gate-check history)")
    plt.grid(alpha=0.3)
    out_path = r"C:\Users\My PC\Desktop\final\15x15\model\elo_trend.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
except ImportError:
    print("\n(matplotlib not available -- skipped plot)")

print("\nDONE")
