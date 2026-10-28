"""Root-only VCT (victory by continuous threats: fours and open threes) for 11x11 freestyle (2026-10-23, user).

p18 is holdable (Rapfi 30k nodes defends it vs Rapfi 1M) but our model loses it: Rapfi's attack mixes threes with
fours, which the C++ VCF (fours only) cannot see. This is a prototype used only at the root: after a candidate
move, does the opponent have a VCT? If so the candidate loses and the next one is tried.

AND/OR search with a node budget; "False" means "no VCT found within budget", not a proof that none exists.
"""
N = 11
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _line(b, r, c, p, dr, dc):
    n = 1
    for s in (1, -1):
        rr, cc = r + s * dr, c + s * dc
        while 0 <= rr < N and 0 <= cc < N and b[rr][cc] == p:
            n += 1
            rr += s * dr
            cc += s * dc
    return n


def makes_five(b, r, c, p):
    return any(_line(b, r, c, p, dr, dc) >= 5 for dr, dc in DIRS)


def near(b, d=2):
    out = set()
    for r in range(N):
        for c in range(N):
            if b[r][c] != ' ':
                for rr in range(max(0, r - d), min(N, r + d + 1)):
                    for cc in range(max(0, c - d), min(N, c + d + 1)):
                        if b[rr][cc] == ' ':
                            out.add((rr, cc))
    return out


def win_moves(b, p, cells=None):
    return [(r, c) for r, c in (cells if cells is not None else near(b, 1)) if b[r][c] == ' ' and makes_five(b, r, c, p)]


def _span(r, c):
    """Empty-or-not cells within 4 along the four lines through (r, c)."""
    out = set()
    for dr, dc in DIRS:
        for k in range(-4, 5):
            rr, cc = r + k * dr, c + k * dc
            if 0 <= rr < N and 0 <= cc < N:
                out.add((rr, cc))
    return out


def _could_threaten(b, r, c, p):
    """Cheap filter: some 5-window through (r, c) holds >= 3 of p's stones (incl. (r, c)) and none of the other's."""
    for dr, dc in DIRS:
        for s in range(-4, 1):
            cnt, ok = 0, True
            for k in range(s, s + 5):
                rr, cc = r + k * dr, c + k * dc
                if not (0 <= rr < N and 0 <= cc < N):
                    ok = False
                    break
                v = b[rr][cc]
                if (rr, cc) == (r, c) or v == p:
                    cnt += 1
                elif v != ' ':
                    ok = False
                    break
            if ok and cnt >= 3:
                return True
    return False


def threats_after(b, r, c, p):
    """After p stands on (r, c): (win cells, open-four cells). Open-four cell = a move giving >=2 win cells."""
    if not _could_threaten(b, r, c, p):
        return [], []
    span = _span(r, c)
    wins = win_moves(b, p, span)
    if wins:
        return wins, []
    fours = []
    for rr, cc in span:
        if b[rr][cc] != ' ':
            continue
        b[rr][cc] = p
        if len(win_moves(b, p, _span(rr, cc))) >= 2:
            fours.append((rr, cc))
        b[rr][cc] = ' '
    return wins, fours


class Budget:
    def __init__(self, n):
        self.n = n


def has_vct(b, att, depth=8, budget=None):
    budget = budget or Budget(3000)
    return _vct(b, att, 'O' if att == 'X' else 'X', depth, budget)


def _vct(b, att, dfd, depth, bud):
    bud.n -= 1
    if bud.n <= 0 or depth <= 0:
        return False
    if win_moves(b, att):
        return True
    dw = win_moves(b, dfd)
    if len(dw) >= 2:
        return False
    # one cheap pass: classify every candidate; an open/double four wins at once, then fours before threes
    moves = []
    for r, c in (dw if dw else sorted(near(b, 2))):   # a defender four must be blocked first
        b[r][c] = att
        wins, fours = threats_after(b, r, c, att)
        b[r][c] = ' '
        if not wins and not fours:
            continue
        if len(wins) >= 2:
            b[r][c] = att
            safe = not win_moves(b, dfd)
            b[r][c] = ' '
            if safe:
                return True
            continue
        moves.append((0 if wins else 1, -len(fours), r, c, wins, fours))   # double threes first
    moves.sort()
    for _, _, r, c, wins, fours in moves:
        b[r][c] = att
        if win_moves(b, dfd):        # attacker's threat is slower than the defender's own five
            b[r][c] = ' '
            continue
        if wins:
            defs = wins
        else:                        # open three: every reply that leaves no open four, plus defender fours
            defs = []
            for rr, cc in _span(r, c) | set(fours):
                if b[rr][cc] != ' ':
                    continue
                b[rr][cc] = dfd
                stopped = not win_moves(b, att, _span(r, c)) and not any(
                    b[x][y] == ' ' and _gives_open_four(b, x, y, att) for x, y in fours)
                b[rr][cc] = ' '
                if stopped:
                    defs.append((rr, cc))
            for x, y in near(b, 2):
                if (x, y) not in defs:
                    b[x][y] = dfd
                    if win_moves(b, dfd, _span(x, y)):
                        defs.append((x, y))
                    b[x][y] = ' '
            if not defs:
                b[r][c] = ' '
                return True
        ok = True
        for x, y in defs:
            b[x][y] = dfd
            sub = _vct(b, att, dfd, depth - 1, bud)
            b[x][y] = ' '
            if not sub:
                ok = False
                break
        b[r][c] = ' '
        if ok:
            return True
        if bud.n <= 0:
            return False
    return False


def _gives_open_four(b, r, c, p):
    b[r][c] = p
    n = len(win_moves(b, p, _span(r, c)))
    b[r][c] = ' '
    return n >= 2
