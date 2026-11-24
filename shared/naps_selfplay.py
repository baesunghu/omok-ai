"""One self-play game with the C++ MCTS (play_one_mcts_game) and a batch of them (generate_mcts_self_play_data).
The multiprocess pools (selfplay_pool_fallback.py, selfplay_pool.py) call play_one_mcts_game directly.
"""
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor

from fast_board import sims_for_state
from mcts_cpp import BOARD_SIZE, MCTS, fast_check_draw, fast_check_win
from run_config import EVAL_BATCH
from nets import SURVIVAL_CAP
from board_encoding import augment_sample, board_to_input_array
from naps_common import mixed_value_target
from naps_openings import sample_forced_opening_start, sample_tactical_start


def pick_training_move(mcts, board, player, model, device, decisions, num_simulations, boosted_sims, boost_threshold,
                       boost_when_narrow, full_search_prob, fast_sims, temperature_moves, dirichlet_alpha, noise_weight):
    """The model's move in a training game (self-play, vs-Det, vs-League). A full search -- always at
    full_search_prob=1.0, else with that probability (playout-cap randomization) -- gets sims_for_state's budget and
    Dirichlet noise and is the one callers record; a fast search only picks the move. While `decisions` (this side's
    moves so far) < temperature_moves the move is sampled in proportion to visits, later it is the most visited one.
    Returns None when the search found no move, else (r, c, is_full_search, target_policy_map, root)."""
    num_empty = sum(1 for row in board for cell in row if cell == ' ')
    is_full_search = full_search_prob >= 1.0 or random.random() < full_search_prob
    if is_full_search:
        sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
    else:
        sims_this_move = fast_sims or max(1, num_simulations // 6)
    root = mcts.search(board, player, model, device, num_simulations=sims_this_move, add_noise=is_full_search,
                       dirichlet_alpha=dirichlet_alpha, noise_weight=noise_weight)
    if not root.children:
        return None
    total_visits = sum(child.n_visits for child in root.children.values())
    target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for act, child in root.children.items():
        r, c = divmod(act, BOARD_SIZE)
        target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0
    actions = list(root.children.keys())
    visits = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)
    if decisions < temperature_moves:
        probs = visits / visits.sum()
        chosen_act = np.random.choice(actions, p=probs)
    else:
        chosen_act = actions[np.argmax(visits)]
    r, c = divmod(chosen_act, BOARD_SIZE)
    return r, c, is_full_search, target_policy_map, root


def play_one_mcts_game(model, device, num_simulations, boosted_sims=None, boost_threshold=0,
                         temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25, c_puct=1.414,
                         eval_cache=None, boost_when_narrow=False, start_board=None, start_player=None,
                         pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                         return_ply_info=False, max_plies=None):
    """
    One self-play game. Returns (history, outcome): history holds (board_3d, player, target_policy_map, root value)
    per recorded move, before augmentation. With return_ply_info=True it returns (history, outcome, ply_indices,
    total_plies): the ply at which each history entry was recorded and the game's length, which the survival target
    needs because skipped fast-search plies make len(history) undercount the plies left.

    eval_cache, when given, is shared by every game of the epoch; that is safe across threads (dict get/set is atomic
    under the GIL) and correct because the weights are frozen for the whole generation phase.

    start_board / start_player (both or neither): start from that position instead of an empty board (see
    sample_tactical_start), so self-play meets hard tactical positions more often than full games would produce.

    full_search_prob / fast_sims: playout-cap randomization (KataGo). Each move is a full search (recorded as a
    training target) with probability full_search_prob, else a cheap fast_sims search that only picks the move -- its
    visit counts are too noisy to train on. full_search_prob=1.0 (default) records every move. Temperature counts plies
    actually played, not recorded entries, and fast searches get no Dirichlet noise (they exist to play well cheaply,
    not to explore).
    """
    mcts = MCTS(c_puct=c_puct, eval_batch_size=EVAL_BATCH, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    if start_board is not None:
        board = [row[:] for row in start_board]
        current_player = start_player
    else:
        board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        current_player = random.choice(['X', 'O'])
    history = []
    ply_indices = []
    ply = 0
    outcome = 0
    # max_plies caps the game length (plies counted from this
    # game's start position). A capped game is scored a draw, and its
    # survival labels are dropped (total_plies returned as None) since the
    # real remaining length is unknown.
    truncated = False

    while True:
        if max_plies and ply >= max_plies:
            outcome = 0
            truncated = True
            break
        if fast_check_win(board, 'X'):
            outcome = 1
            break
        if fast_check_win(board, 'O'):
            outcome = -1
            break
        if fast_check_draw(board):
            outcome = 0
            break

        # temperature is keyed on plies actually played (equals len(history) at full_search_prob=1.0)
        move = pick_training_move(mcts, board, current_player, model, device, ply, num_simulations,
                                  boosted_sims, boost_threshold, boost_when_narrow, full_search_prob, fast_sims,
                                  temperature_moves, dirichlet_alpha, noise_weight)
        if move is None:
            break
        r, c, is_full_search, target_policy_map, root = move

        if is_full_search:
            # Store board state (3, BOARD_SIZE, BOARD_SIZE)
            board_3d = board_to_input_array(board, current_player)
            # 4th element: this search's root value from the mover's side
            # (for the z/q value-target mix, see mixed_value_target).
            history.append((board_3d, current_player, target_policy_map, float(root.value())))
            ply_indices.append(ply)

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        ply += 1

    if return_ply_info:
        return history, outcome, ply_indices, (None if truncated else ply)
    return history, outcome


def generate_mcts_self_play_data(model, device, num_games=10, num_simulations=40, max_workers=None,
                                  boosted_sims=None, boost_threshold=0,
                                  temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25, c_puct=1.414,
                                  eval_cache=None, boost_when_narrow=False, num_curriculum_games=0,
                                  num_opening_games=0,
                                  pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                  value_q_weight=0.0, max_plies=None):
    """
    Generates MCTS Self-Play games using full MCTS search and Dirichlet noise.

    Games are played concurrently across a thread pool. This isn't defeated
    by the GIL the way pure-Python parallel code would be: every neural-net
    forward pass (the expensive part of each MCTS simulation) runs inside
    PyTorch's C++ kernels, which release the GIL while they compute, so
    multiple games really do make progress on different CPU cores at once.
    The model is only read from (eval mode, no_grad, no state mutated), so
    sharing it across threads is safe.
    """
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    boost_note = f", boosted to {boosted_sims} while >{boost_threshold} cells empty" if boosted_sims else ""
    print(f"Generating {num_games} MCTS Self-Play games "
          f"(Simulations: {num_simulations}/move{boost_note}, workers: {max_workers})...")

    model.eval()
    raw_samples = []
    outcome_counts = {'x_win': 0, 'o_win': 0, 'draw': 0}
    opening_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(num_games):
            start_board, start_player = None, None
            if i < num_curriculum_games:
                sampled = sample_tactical_start()
                if sampled is not None:
                    start_board, start_player = sampled
            elif i < num_curriculum_games + num_opening_games:
                start_board, start_player = sample_forced_opening_start()
            futures.append(pool.submit(play_one_mcts_game, model, device, num_simulations, boosted_sims,
                                        boost_threshold, temperature_moves, dirichlet_alpha, noise_weight, c_puct,
                                        eval_cache, boost_when_narrow, start_board, start_player, pooled_evaluator,
                                        full_search_prob, fast_sims, True, max_plies))
        for future in futures:
            history, outcome, ply_indices, total_plies = future.result()
            if outcome == 1:
                outcome_counts['x_win'] += 1
            elif outcome == -1:
                outcome_counts['o_win'] += 1
            else:
                outcome_counts['draw'] += 1
            if history and ply_indices and ply_indices[0] == 0:
                # history[0]'s board planes are still empty (no stone has landed yet);
                # the first ACTUAL stone is recorded as target_policy_map's
                # single argmax entry (a one-hot visit distribution isn't
                # guaranteed, but for a diagnostic row-coverage count the
                # argmax cell is a fine proxy for "where game i opened").
                first_cell = int(np.argmax(history[0][2]))
                opening_rows.append(first_cell // BOARD_SIZE)
            # Assign outcomes and apply 8x Data Augmentation. Survival uses the
            # real ply index (PCR skips recording fast-search plies, so the
            # history index alone would undercount plies-to-end).
            for (board_3d, player_turn, target_policy_map, q), ply_idx in zip(history, ply_indices):
                survival = (None if total_plies is None
                            else min(total_plies - ply_idx, SURVIVAL_CAP) / SURVIVAL_CAP)
                # Full +-1.0 scale (was 0.5). The halved scale
                # shrank Q's dynamic range against an unchanged c_puct, which
                # doubles the effective exploration constant and quadruples the
                # simulations needed for visit counts to overcome the prior. It
                # also fought tactical_refresh's own 0.97 value target.
                final_val = mixed_value_target(float(outcome if player_turn == 'X' else -outcome), q, value_q_weight)
                mover_won = (outcome == 1 and player_turn == 'X') or (outcome == -1 and player_turn == 'O')
                sample_weight = 1.15 if mover_won else 1.0
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val, sample_weight, survival))

    print(f"Generated {len(raw_samples)} augmented MCTS training samples. "
          f"Outcomes: X {outcome_counts['x_win']} / O {outcome_counts['o_win']} / Draw {outcome_counts['draw']} "
          f"| Opening row-coverage this epoch: {len(set(opening_rows))}/{BOARD_SIZE}")
    return raw_samples, outcome_counts
