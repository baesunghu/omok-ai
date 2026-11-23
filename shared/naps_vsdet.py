"""Games against the deterministic minimax opponent (deterministic_play): evaluation and training data.
"""
import deterministic_play
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor

from mcts_cpp import BOARD_SIZE, MCTS, PooledEvaluator, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH, POOLED_TARGET_BATCH
from board_encoding import augment_sample, board_to_input_array
from naps_common import mixed_value_target, sims_for_state
from naps_openings import ALL_CELLS, OPENING_SQUARES


def _play_vs_deterministic(model, device, num_simulations, mcts_plays_x, boosted_sims=None, boost_threshold=0,
                            c_puct=1.414, boost_when_narrow=False, opening=None,
                            eval_cache=None, pooled_evaluator=None):
    """
    Plays one game: the in-memory model (via fresh MCTS search, no noise,
    greedy) against deterministic_play's minimax AI. Returns 'mcts', 'det',
    or 'draw'.

    add_noise=False makes MCTS search deterministic, and deterministic_play is
    deterministic too, so without `opening` every game with the same
    mcts_plays_x is an identical replay of the same line -- num_games games
    then really only cover 2 distinct positions (one per side). Passing a
    (row, col) `opening` pre-places X's first move there before normal play
    resumes, fanning otherwise-identical matchups out into real variety.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    mcts_engine = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        if fast_check_win(board, mcts_player):
            return 'mcts'
        if fast_check_win(board, 'O' if mcts_player == 'X' else 'X'):
            return 'det'
        if fast_check_draw(board):
            return 'draw'

        if current_player == mcts_player:
            num_empty = sum(1 for row in board for cell in row if cell == ' ')
            sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
            root = mcts_engine.search(board, current_player, model, device,
                                       num_simulations=sims_this_move, add_noise=False)
            if not root.children:
                return 'draw'
            best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
            r, c = divmod(best_action, BOARD_SIZE)
        else:
            r, c = deterministic_play.play(board, current_player)
            if r == -1:
                return 'draw'

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


def evaluate_vs_deterministic(model, device, num_games=10, num_simulations=50, boosted_sims=None, boost_threshold=0,
                               c_puct=1.414, boost_when_narrow=False, max_workers=None, pooled_evaluator=None,
                               all_cells=None):
    """
    Plays num_games against deterministic_play (minimax/alpha-beta), alternating
    who moves first, and returns (win_rate, draw_rate, loss_rate) from the
    trained model's perspective. This is the direct metric for the research
    goal ("beat the deterministic algorithm").

    Tries the multiprocessing implementation (mp_eval.evaluate_vs_deterministic_mp)
    first -- each game gets a real OS process instead of a thread, which
    matters because deterministic_play (the pure-Python minimax opponent)
    holds the GIL for its whole call and was found to fully serialize the
    threaded implementation down to ~1 CPU core regardless of thread count
    (the C++ MCTS engine itself already releases the GIL correctly and was
    not the bottleneck). Verified against the threaded path at the real
    production shape (20 games, 600/1350 sims, live checkpoint): identical
    win/draw/loss rates (deterministic/no-noise/greedy + fixed openings is a
    pure function of model weights, so crossing a process boundary changes
    no numeric value), 2.35x wall-clock speedup. Falls back to the original
    threaded implementation on ANY exception from the mp path (process spawn
    failure, a worker dying mid-game, GPU OOM in the evaluator process,
    etc.) so a transient multiprocessing hiccup degrades this call to
    "slower" rather than crashing or hanging an unattended training run.

    pooled_evaluator is only used by the threaded fallback path; the mp path
    has no equivalent parameter (see _evaluate_vs_deterministic_threaded).
    """
    model.eval()
    all_cells = all_cells if all_cells is not None else ALL_CELLS
    try:
        from mp_eval import evaluate_vs_deterministic_mp
        return evaluate_vs_deterministic_mp(
            model, device, num_games=num_games, num_simulations=num_simulations,
            boosted_sims=boosted_sims, boost_threshold=boost_threshold, c_puct=c_puct,
            boost_when_narrow=boost_when_narrow, all_cells=all_cells,
        )
    except Exception as e:  # noqa: BLE001
        print(f"evaluate_vs_deterministic_mp failed ({e!r}), falling back to threaded implementation")
        return _evaluate_vs_deterministic_threaded(
            model, device, num_games=num_games, num_simulations=num_simulations,
            boosted_sims=boosted_sims, boost_threshold=boost_threshold, c_puct=c_puct,
            boost_when_narrow=boost_when_narrow, max_workers=max_workers, pooled_evaluator=pooled_evaluator,
            all_cells=all_cells,
        )


def _evaluate_vs_deterministic_threaded(model, device, num_games=10, num_simulations=50, boosted_sims=None,
                                         boost_threshold=0, c_puct=1.414, boost_when_narrow=False,
                                         max_workers=None, pooled_evaluator=None, all_cells=None):
    """
    Original ThreadPoolExecutor + PooledEvaluator implementation, kept as the
    safety-net fallback for evaluate_vs_deterministic (see its docstring).

    Games run concurrently across a ThreadPoolExecutor with a shared
    eval_cache and PooledEvaluator, the same pattern
    generate_mcts_vs_deterministic_data already uses (see its docstring) --
    this is behavior-preserving for game OUTCOMES: no-noise/greedy MCTS vs
    deterministic minimax is a pure function of (model weights, opening,
    settings), and eval_cache only memoizes that pure function (a cache hit
    returns the exact same value a fresh compute would have) -- only
    wall-clock time changes.

    pooled_evaluator: reuse a caller-owned PooledEvaluator wrapping this
    exact `model` object if the caller already has one running; when
    omitted (the default), a fresh one scoped to just this call is
    constructed and torn down before returning, so this function stays
    usable standalone. NOTE: the training loop's own pooled_evaluator is
    scoped to gen_model, not model, so the existing call site cannot reuse
    it.
    """
    model.eval()
    all_cells = all_cells if all_cells is not None else ALL_CELLS
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    eval_cache = {}
    owns_pooled_evaluator = pooled_evaluator is None and device.type == "cuda"
    if owns_pooled_evaluator:
        pooled_evaluator = PooledEvaluator(model, device, target_batch=POOLED_TARGET_BATCH, max_wait=0.05)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for i in range(num_games):
                mcts_plays_x = (i % 2 == 0)
                opening = all_cells[i % len(all_cells)]
                futures.append(pool.submit(_play_vs_deterministic, model, device, num_simulations, mcts_plays_x,
                                            boosted_sims, boost_threshold, c_puct, boost_when_narrow, opening,
                                            eval_cache, pooled_evaluator))
            wins = draws = losses = 0
            for future in futures:
                result = future.result()
                if result == 'mcts':
                    wins += 1
                elif result == 'draw':
                    draws += 1
                else:
                    losses += 1
    finally:
        if owns_pooled_evaluator:
            pooled_evaluator.stop()
    n = max(1, num_games)
    return wins / n, draws / n, losses / n


def sample_vs_det_max_depth(epoch):
    """
    Opponent-depth ladder for vs-Deterministic training-data generation
    ONLY (never evaluate_vs_deterministic or run_gating_check, which must
    stay a fixed, comparable depth-2 opponent forever). Samples a per-game
    minimax search depth for deterministic_play from a schedule keyed on the
    current training epoch, so the model faces a weak/shallow opponent
    early on and a deeper one later instead of a fixed difficulty from
    epoch 0. Depth 0 means "skip minimax, play a uniformly random legal
    move" (handled by the caller, not by deterministic_play itself).

    Deliberately capped at depth 3 for compute budget -- never sample
    beyond it (a deeper minimax is exponentially more expensive, and this
    project's 10x10/WIN_LENGTH=5 board is already the slowest, most
    compute-constrained board size here).
    """
    if epoch < 300:
        depths, weights = (0, 1, 2), (0.15, 0.35, 0.50)
    elif epoch < 1500:
        # Small depth-0 floor (diversity pass): depth>=1 minimax
        # opponents always play structurally coherent moves, so dropping
        # depth-0 entirely here permanently loses exposure to the
        # off-book/scattered configurations only a random opponent
        # produces, for the 1200-epoch stretch this bucket covers.
        depths, weights = (0, 1, 2, 3), (0.05, 0.10, 0.45, 0.40)
    else:
        depths, weights = (2, 3), (0.45, 0.55)
    return random.choices(depths, weights=weights, k=1)[0]


def format_by_depth(by_depth):
    """Render generate_mcts_vs_deterministic_data's per-opponent-depth tally as
    'd0 2W-0L-0D | d2 0W-5L-0D'. Depth 0 is the random-move opponent, so wins
    there say nothing about real play -- keeping the depths separate is the
    only way to see whether anything is improving against an actual searcher."""
    if not by_depth:
        return "by-depth n/a"
    return " | ".join(f"d{d} {t['w']}W-{t['l']}L-{t['d']}D"
                      for d, t in sorted(by_depth.items()))


def play_one_mcts_vs_deterministic_game(model, device, num_simulations, mcts_plays_x,
                                          boosted_sims=None, boost_threshold=0,
                                          temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                          c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                          pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                          opening=None, det_max_depth=deterministic_play.MAX_DEPTH):
    """
    Plays one training game between the in-memory model (MCTS with noise
    and temperature sampling -- same exploration settings as self-play) and
    deterministic_play's minimax AI. Only the MCTS side's decisions are
    recorded, in the same (board_3d, player_turn, target_policy_map) format
    play_one_mcts_game produces, so callers can mix both into one buffer.

    Pure self-play never has to defend against or attack the specific
    tactics deterministic_play actually uses, so the model can stay blind to
    patterns that opponent exploits no matter how long training runs.
    Mixing in games against it directly closes that loop.

    deterministic_play is deterministic, so without `opening` every game
    with the same mcts_plays_x replays the same line -- see
    _play_vs_deterministic's docstring. Passing a (row, col) `opening`
    pre-places X's first move there before normal play resumes.

    det_max_depth (see sample_vs_det_max_depth) sets the deterministic
    opponent's minimax search depth for this game; 0 means it plays a
    uniformly random legal move instead of calling minimax at all.

    Returns (history, outcome, mcts_result): outcome uses the same
    X=1/O=-1/draw=0 convention as self-play; mcts_result is
    'mcts'/'det'/'draw' from the trained model's perspective.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    history = []
    # MCTS-side decisions made so far (full or fast) -- drives temperature;
    # equals len(history) at full_search_prob=1.0 (PCR fix, see
    # play_one_mcts_game's docstring).
    mcts_decisions = 0
    outcome = 0

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        if fast_check_win(board, 'X'):
            outcome = 1
            break
        if fast_check_win(board, 'O'):
            outcome = -1
            break
        if fast_check_draw(board):
            outcome = 0
            break

        if current_player == mcts_player:
            num_empty = sum(1 for row in board for cell in row if cell == ' ')
            is_full_search = full_search_prob >= 1.0 or random.random() < full_search_prob
            if is_full_search:
                sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
            else:
                sims_this_move = fast_sims or max(1, num_simulations // 6)
            root = mcts.search(board, current_player, model, device,
                                num_simulations=sims_this_move, add_noise=is_full_search,
                                dirichlet_alpha=dirichlet_alpha, noise_weight=noise_weight)

            if not root.children:
                outcome = 0
                break

            total_visits = sum(child.n_visits for child in root.children.values())
            target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for act, child in root.children.items():
                r, c = divmod(act, BOARD_SIZE)
                target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0

            move_count = mcts_decisions
            actions = list(root.children.keys())
            visits = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)
            if move_count < temperature_moves:
                probs = visits / visits.sum()
                chosen_act = np.random.choice(actions, p=probs)
            else:
                chosen_act = actions[np.argmax(visits)]
            r, c = divmod(chosen_act, BOARD_SIZE)
            mcts_decisions += 1

            if is_full_search:
                board_3d = board_to_input_array(board, current_player)
                history.append((board_3d, current_player, target_policy_map, float(root.value())))
        else:
            if det_max_depth == 0:
                empty_spots = [(rr, cc) for rr in range(BOARD_SIZE) for cc in range(BOARD_SIZE)
                               if board[rr][cc] == ' ']
                if not empty_spots:
                    outcome = 0
                    break
                r, c = random.choice(empty_spots)
            else:
                r, c = deterministic_play.play(board, current_player, max_depth=det_max_depth)
                if r == -1:
                    outcome = 0
                    break

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'

    if outcome == 1:
        mcts_result = 'mcts' if mcts_player == 'X' else 'det'
    elif outcome == -1:
        mcts_result = 'mcts' if mcts_player == 'O' else 'det'
    else:
        mcts_result = 'draw'

    return history, outcome, mcts_result


def generate_mcts_vs_deterministic_data(model, device, num_games=6, num_simulations=40, max_workers=None,
                                         boosted_sims=None, boost_threshold=0,
                                         temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                         c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                         pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                         epoch=0, value_q_weight=0.0):
    """
    Same shape as generate_mcts_self_play_data, but MCTS plays against
    deterministic_play instead of itself (alternating X/O by game index), so
    the replay buffer includes direct experience of the exact opponent
    evaluate_vs_deterministic measures against.

    Each game cycles through OPENING_SQUARES by game index (deterministic_play
    is deterministic, so without this every game with the same side
    assignment would replay the same line -- see _play_vs_deterministic's
    docstring) and samples its own opponent search depth from the
    epoch-keyed ladder in sample_vs_det_max_depth -- evaluate_vs_deterministic
    and run_gating_check are untouched by this and always keep the fixed
    depth-2 opponent.
    """
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    print(f"Generating {num_games} MCTS-vs-Deterministic training games "
          f"(Simulations: {num_simulations}/move, workers: {max_workers})...")

    model.eval()
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'det_win': 0, 'draw': 0}
    # Per-depth tally. The aggregate "1-3 wins per 10 games" line
    # was measured to be 100% depth-0 (random opponent) wins and 0% against
    # both depth 1 and depth 2, so the aggregate was hiding a flat zero
    # against any real opponent. Split it so progress is actually visible.
    by_depth = {}
    game_depths = [sample_vs_det_max_depth(epoch) for _ in range(num_games)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(play_one_mcts_vs_deterministic_game, model, device, num_simulations,
                                i % 2 == 0, boosted_sims, boost_threshold,
                                temperature_moves, dirichlet_alpha, noise_weight, c_puct, eval_cache,
                                boost_when_narrow, pooled_evaluator, full_search_prob, fast_sims,
                                OPENING_SQUARES[(i + epoch) % len(OPENING_SQUARES)],
                                game_depths[i])
                   for i in range(num_games)]
        for depth, future in zip(game_depths, futures):
            history, outcome, mcts_result = future.result()
            tally = by_depth.setdefault(depth, {'w': 0, 'l': 0, 'd': 0})
            if mcts_result == 'mcts':
                outcome_counts['mcts_win'] += 1
                tally['w'] += 1
            elif mcts_result == 'det':
                outcome_counts['det_win'] += 1
                tally['l'] += 1
            else:
                outcome_counts['draw'] += 1
                tally['d'] += 1
            for (board_3d, player_turn, target_policy_map, q) in history:
                # Full +-1.0 scale (was 0.5). The halved scale
                # shrank Q's dynamic range against an unchanged c_puct, which
                # doubles the effective exploration constant and quadruples the
                # simulations needed for visit counts to overcome the prior. It
                # also fought tactical_refresh's own 0.97 value target.
                final_val = mixed_value_target(float(outcome if player_turn == 'X' else -outcome), q, value_q_weight)
                sample_weight = 1.5
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val, sample_weight))

    outcome_counts['by_depth'] = by_depth
    print(f"Generated {len(raw_samples)} augmented vs-Deterministic training samples. "
          f"Outcomes: MCTS {outcome_counts['mcts_win']} / Det {outcome_counts['det_win']} / "
          f"Draw {outcome_counts['draw']} | {format_by_depth(by_depth)}")
    return raw_samples, outcome_counts
