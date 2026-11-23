"""
Attack-side counterpart of the defense ladder in fork_status_check.py
(2026-09-26). Measurement only -- nothing here feeds training.

  A1 (1 ply)   : X has a four -> must complete five (O also has a four,
                 so blocking instead of winning is a mistake).
  A2 (2-3 ply) : X has an open three -> must make an open four
                 (a move after which X has >= 2 distinct winning cells).
  A3 (4+ ply)  : X can make a double-four fork -> must play a fork move
                 (>= 2 distinct winning cells after the move).
  A3 T-shape   : X owns the T-shape (X _ X / X / X) -> must create a
                 double open three (or better).

Every position is built from a fixed seed so runs are comparable.
"""
import random
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
N_POS = 20
SIMS = 1100


def _inb(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _empty_board():
    return [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def _winning_cells(board, color):
    out = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = color
            if fast_check_win(board, color):
                out.append((r, c))
            board[r][c] = ' '
    return out


def _threat_after(board, move, color):
    """(wins_now, number of winning cells color has after playing move)."""
    r, c = move
    board[r][c] = color
    try:
        if fast_check_win(board, color):
            return True, 0
        return False, len(_winning_cells(board, color))
    finally:
        board[r][c] = ' '


def _scatter(board, rng, color, n, forbid):
    empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
               if board[r][c] == ' ' and (r, c) not in forbid]
    rng.shuffle(empties)
    for (r, c) in empties[:n]:
        board[r][c] = color


def _is_open_three(board, r0, c0, dr, dc, color):
    def count_dir(dr, dc):
        n = 0
        r, c = r0 + dr, c0 + dc
        while _inb(r, c) and board[r][c] == color:
            n += 1
            r, c = r + dr, c + dc
        return n, _inb(r, c) and board[r][c] == ' '
    n1, o1 = count_dir(dr, dc)
    n2, o2 = count_dir(-dr, -dc)
    return (1 + n1 + n2) == 3 and o1 and o2


def _top_move(engine_mod, eval_batch_fn, board):
    engine = engine_mod.MCTSEngine()
    _, rv, cands = engine.search_batched(
        board, 'X', eval_batch_fn, SIMS, 1.414, False, 0.3, 0.25,
        BOARD_SIZE, WIN_LENGTH, 16,
    )
    return tuple(max(cands, key=lambda c: c['visit_count'])['move'])


# ---------------------------------------------------------------- generators
def _gen_a1(rng):
    """X four (one gap) + O four elsewhere; X to move must win."""
    for _ in range(5000):
        board = _empty_board()
        used = set()
        ok = True
        for color in ('X', 'O'):
            dr, dc = rng.choice(DIRS)
            r0, c0 = rng.randrange(BOARD_SIZE), rng.randrange(BOARD_SIZE)
            line = [(r0 + k * dr, c0 + k * dc) for k in range(5)]
            if not all(_inb(*p) for p in line) or used & set(line):
                ok = False
                break
            gap = rng.randrange(5)
            for k, (r, c) in enumerate(line):
                if k != gap:
                    board[r][c] = color
            used |= set(line)
        if not ok:
            continue
        _scatter(board, rng, 'X', 2, used)
        _scatter(board, rng, 'O', 2, used)
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        xw = _winning_cells(board, 'X')
        if len(xw) != 1 or not _winning_cells(board, 'O'):
            continue
        return board, xw
    return None


def _gen_a2(rng):
    """X open three (_XXX_ with room), O has no four; correct = open-four makers."""
    for _ in range(5000):
        board = _empty_board()
        dr, dc = rng.choice(DIRS)
        r0, c0 = rng.randrange(BOARD_SIZE), rng.randrange(BOARD_SIZE)
        span = [(r0 + k * dr, c0 + k * dc) for k in range(-2, 5)]  # _ _ X X X _ _
        if not all(_inb(*p) for p in span):
            continue
        for (r, c) in span[2:5]:
            board[r][c] = 'X'
        _scatter(board, rng, 'O', 5, set(span))
        _scatter(board, rng, 'X', 1, set(span))
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        if _winning_cells(board, 'X') or _winning_cells(board, 'O'):
            continue
        correct = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == ' ':
                    won, n = _threat_after(board, (r, c), 'X')
                    if won or n >= 2:
                        correct.append((r, c))
        if correct:
            return board, correct
    return None


def _gen_a3(rng):
    """Two X threes on different lines meeting at an empty cell p -> fork at p."""
    for _ in range(5000):
        board = _empty_board()
        p = (rng.randrange(BOARD_SIZE), rng.randrange(BOARD_SIZE))
        d1, d2 = rng.sample(DIRS, 2)
        cells = []
        good = True
        for (dr, dc) in (d1, d2):
            sgn = rng.choice((1, -1))
            line = [(p[0] + sgn * k * dr, p[1] + sgn * k * dc) for k in (1, 2, 3)]
            if not all(_inb(*q) for q in line):
                good = False
                break
            cells += line
        if not good or len(set(cells)) != 6:
            continue
        for (r, c) in cells:
            board[r][c] = 'X'
        _scatter(board, rng, 'O', 7, set(cells) | {p})
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        if _winning_cells(board, 'X') or _winning_cells(board, 'O'):
            continue
        correct = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == ' ':
                    won, n = _threat_after(board, (r, c), 'X')
                    if won or n >= 2:
                        correct.append((r, c))
        if p in correct:
            return board, correct
    return None


def _run_level(name, gen, rng, engine_mod, eval_batch_fn):
    tested = ok = 0
    while tested < N_POS:
        pos = gen(rng)
        if pos is None:
            break
        board, correct = pos
        top = _top_move(engine_mod, eval_batch_fn, board)
        tested += 1
        ok += int(top in correct)
    print(f"{name}: {ok}/{tested} = {ok / max(tested, 1):.1%}")
    return f"{ok}/{tested} = {ok / max(tested, 1):.0%}"


def _tshape_attack(engine_mod, eval_batch_fn, model_prior_rank_fn):
    board = _empty_board()
    gap = (5, 5)
    for (r, c) in [(5, 4), (5, 6), (6, 5), (7, 5)]:
        board[r][c] = 'X'
    for (r, c) in [(3, 3), (8, 8), (3, 7), (9, 2)]:
        board[r][c] = 'O'
    top = _top_move(engine_mod, eval_batch_fn, board)
    r, c = top
    board[r][c] = 'X'
    wins = fast_check_win(board, 'X')
    n_open3 = sum(1 for (dr, dc) in DIRS if _is_open_three(board, r, c, dr, dc, 'X'))
    n_win_cells = 0 if wins else len(_winning_cells(board, 'X'))
    board[r][c] = ' '
    if top == gap:
        verdict = "BEST"
    elif wins or n_win_cells >= 2 or n_open3 >= 2:
        verdict = "OK"
    else:
        verdict = "WRONG"
    rank = model_prior_rank_fn(board, gap)
    print(f"A3 T-shape attack: top_move={top} -> {verdict} (prior rank of gap {rank}/121)")
    return f"{verdict} (prior rank of gap {rank}/121)"


def run_attack_ladder(engine_mod, eval_batch_fn, model_prior_rank_fn):
    summary = {}
    summary['A1'] = _run_level("A1 complete own four", _gen_a1, random.Random(101), engine_mod, eval_batch_fn)
    summary['A2'] = _run_level("A2 open three -> open four", _gen_a2, random.Random(202), engine_mod, eval_batch_fn)
    summary['A3'] = _run_level("A3 make double-four fork", _gen_a3, random.Random(303), engine_mod, eval_batch_fn)
    summary['A3_tshape'] = _tshape_attack(engine_mod, eval_batch_fn, model_prior_rank_fn)
    return summary
