"""Attack-leaning final move pick, no retraining (2026-10-26, user: "white is too defensive, ease it by a
technical method, not training"). After the search, among candidates the search rates almost as good as
the best (visits >= SHARE x top and Q >= best Q - QTOL), play the one that builds the most of our own
lines instead of the most visited one. A proven win is never overridden and a proven loss is never picked."""
SHARE = 0.3    # candidate must have at least this fraction of the top move's visits
QTOL = 0.05    # and a Q value within this of the top move's Q
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def attack_score(board, r, c, me, n=11):
    """Sum over every 5-cell window through (r, c) free of opponent stones: 4 ** (our stones in it)."""
    opp = 'O' if me == 'X' else 'X'
    total = 0
    for dr, dc in DIRS:
        for s in range(-4, 1):
            cells = [(r + (s + k) * dr, c + (s + k) * dc) for k in range(5)]
            if not all(0 <= y < n and 0 <= x < n for y, x in cells):
                continue
            stones = [board[y][x] for y, x in cells]
            if opp in stones:
                continue
            total += 4 ** stones.count(me)
    return total


def pick(board, me, cands, n=11):
    """-> the chosen candidate dict (same shape as the engine's)."""
    best = max(cands, key=lambda c: c["visit_count"])
    # engine 'proven' is from the opponent's side like q_value: -1 = a proven win for us, +1 = a proven loss.
    # (2026-10-26 fix: any proven flag used to disable the pick, and far-off losing moves are almost always
    # flagged, so the pick hardly ever ran.) Keep the search's move when we have a proven win; never pick a loss.
    if any(c.get("proven") == -1 for c in cands):
        return best
    pool = [c for c in cands if c.get("proven") != 1 and c["visit_count"] >= SHARE * best["visit_count"]
            and c["q_value"] <= best["q_value"] + QTOL]   # engine q_value is from the opponent's side (lower = better for us)
    if not pool:   # every near-best move is a proven loss: leave it to the search
        return best
    return max(pool, key=lambda c: (attack_score(board, c["move"][0], c["move"][1], me, n), c["visit_count"]))
