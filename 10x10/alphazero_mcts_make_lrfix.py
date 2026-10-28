import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
from collections import deque, Counter
import numpy as np
import random
import os
import pickle
import re
import time
from datetime import datetime

from alphago_lee import board_to_input_tensor, board_to_input_array, preferred_memory_format
# 2026-09-22 Gen12: swapped to the V2 architecture (GroupNorm+SE, see
# alphago_lee_v2_seblock.py) -- aliased as AlphaZeroTicTacToe so every
# existing instantiation site below (model/best_model/league_model/
# gen_model) picks it up without individual edits. NOT weight-compatible
# with the old alphago_lee.AlphaZeroTicTacToe checkpoints -- this run
# starts from the freshly-trained gen12_full_FINAL.pth, not an old one.
# 2026-09-25: V3 = V2 + auxiliary survival (remaining-plies) head, see
# alphago_lee_v3_survival.py. Loads V2 checkpoints; forward() unchanged.
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3 as AlphaZeroTicTacToe, SURVIVAL_CAP
SURVIVAL_LOSS_WEIGHT = 0.25
from alphazero_mcts_lrfix import MCTS, BOARD_SIZE, NUM_CELLS, fast_check_win, fast_check_draw, PooledEvaluator
from alphago_lee_make import augment_sample, logger
from fast_board import WIN_LENGTH
import determinisic_play
from selfplay_mp import generate_mcts_self_play_data_mp
from selfplay_mp_unified import generate_epoch_games_unified, get_task_runner

model = AlphaZeroTicTacToe()

# Keep 3x3 and other board sizes writing to separate checkpoint/log files so
# concurrent runs (e.g. a 3x3 job still training while a 5x5 job starts)
# never clobber each other's state on disk. 3x3 keeps its original unsuffixed
# paths for continuity with earlier runs.
_SUFFIX = "" if BOARD_SIZE == 3 else f"_{BOARD_SIZE}x{BOARD_SIZE}"

# ============================================================
# Report Logging (for research write-up: loss / entropy / win-rate vs. Deterministic)
# ============================================================
REPORT_LOG_PATH = f"model/training_report{_SUFFIX}.txt"


def report_logger(msg):
    os.makedirs(os.path.dirname(REPORT_LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(REPORT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def compute_policy_entropy(policies_np):
    """
    Mean Shannon entropy (nats) of the target policy distributions used for
    training this epoch. Higher entropy = MCTS visit counts spread across
    more candidate moves (more exploration/uncertainty); entropy trending
    down over training is expected as the model sharpens its move choices.
    """
    eps = 1e-8
    flat = policies_np.reshape(policies_np.shape[0], -1)
    flat = flat / (flat.sum(axis=1, keepdims=True) + eps)
    entropy_per_sample = -np.sum(flat * np.log(flat + eps), axis=1)
    return float(entropy_per_sample.mean())

# ============================================================
# Replay Buffer
# ============================================================
# True AlphaZero self-play does not train on a single epoch's fresh games
# alone -- it trains on a rolling window of recent self-play data. The
# original version here only ever saw the ~10 games (a few hundred augmented
# samples) generated that same epoch, which starves the optimizer of variety
# and is very likely why training_log.txt shows policy loss plateauing
# around 1.59-1.61 instead of trending down. A persisted replay buffer fixes
# that, and also survives across separate runs of this script.
REPLAY_BUFFER_PATH = f"model/mcts_replay_buffer{_SUFFIX}.pkl"
# 2026-10-08 (user): 24,000 -> 196,608 (~40 epochs of data instead of ~5). Training
# no longer sweeps the whole buffer: each epoch draws TRAIN_SAMPLES_PER_EPOCH at random.
REPLAY_BUFFER_MAX_SIZE = 196_608
TRAIN_SAMPLES_PER_EPOCH = 24_000
# History: was 50_000, cut to 8_000 -- ~66-generation-stale MCTS targets
# plateaued Policy Loss (see the 50k->8k comment this replaced, copied from
# the 5x5 sibling file). 2026-08-29: raised 8_000->12_000, mirroring the
# same change just applied to 5x5 -- this board's own vs-Deterministic eval
# has been sliding (Loss 50/100/95/75/85/80% over epochs 120-170, see
# model/training_report_10x10.txt) since the neighbor-radius search pruning
# went live, while its one gate check (epoch 155) still read 100% promoted;
# a too-fast-turnover buffer (measured ~979 fresh augmented samples/epoch
# from train_stdout.log -> 8_000 turned over in ~8 epochs) producing
# narrow, fast-rotating local optima is at least a plausible contributor,
# same mechanism as 5x5's 341-epoch promotion drought. 12_000 (~12-epoch
# turnover) is a deliberately small step, not a return to 50_000.
# tactical_refresh_interval lowered 31->21 at the call site to hold
# buffer_size * refresh_interval roughly constant (8_000*31 = 248_000 ~=
# 12_000*21 = 252_000), preserving the tactical curriculum's gradient share.


def get_devices():
    """
    That single-sample (batch=1) CPU-vs-GPU finding no longer applies here:
    self-play has used virtual-loss batched search (eval_batch_size=8) for
    a while now, and re-benchmarked at batch=8 under the actual concurrent
    access pattern (self_play_workers threads sharing one model), GPU wins
    -- 2.17x more samples/s than CPU (single-call latency alone suggested
    5.7x, but concurrent threads partly serialize on the GPU, a single
    shared resource unlike CPU cores). So self-play now also targets CUDA
    when available. batch=1 CPU still beats GPU per the original
    benchmark, so if eval_batch_size ever drops back to 1, revisit this.
    Training issues large batched forward+backward passes, where the GPU
    wins comfortably (benchmarked: batch=256 is roughly 3-4x faster on
    this GPU than on CPU). Both phases now prefer CUDA when available.
    """
    self_play_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Self-play device: {self_play_device} | Train device: {train_device}")
    return self_play_device, train_device


def sims_for_state(num_empty, base_sims, boosted_sims=None, boost_threshold=0, boost_when_narrow=False):
    """
    A sim-budget diagnostic on this project's 5x5 board (see training
    conversation) found a fixed budget badly under-converged on wide-open
    boards but already converged on narrower ones: at 25 empty cells, the
    top move's visit share was still climbing at 1600 sims (49% -> 79% from
    100 -> 1600); at 14 empty cells, it was flat from 100 all the way to
    1600 sims (~38%, unchanged). So only spend extra search while the board
    is still wide open -- past boost_threshold empty cells, more search
    doesn't change the answer, it just costs time.

    boost_when_narrow=False (default) keeps that exact behavior. On 3x3,
    a diagnosed critical defensive move stayed stuck at ~1/100 visits all
    session, in the mid/late game (few empty cells) -- the opposite end
    from what the 5x5 finding above boosts, since 3x3's branching factor
    (<=9) is already narrower than 5x5's "already converged" 14-empty case
    everywhere. boost_when_narrow=True inverts the comparison so extra sims
    go to low-num_empty states instead, targeting that specific bug.

    That inversion is all-or-nothing (narrow XOR open), so once 5x5 started
    passing boost_when_narrow=True (for its own late-game precision), it
    silently lost the original wide-open boost above -- confirmed still
    needed by a live diagnostic: at 23 empty cells a 100-sim search picked a
    move MCTS itself ranks far worse than the alternative it converges to
    given 600+ sims (visit share 24% -> 92% for the better move as sims
    100->2400, its q swinging from -0.37 to -0.90 in the eventual mover's
    own favor), and that misjudged move was directly upstream of a loss.
    So independently of the boost_when_narrow branch above, also always
    boost the wide-open end (num_empty > NUM_CELLS - 10, i.e. before the
    10th stone is placed) to boosted_sims -- covers both ends at once
    instead of forcing a choice between them.
    """
    if not boosted_sims:
        return base_sims
    if boost_when_narrow:
        sims = boosted_sims if num_empty <= boost_threshold else base_sims
    else:
        sims = boosted_sims if num_empty > boost_threshold else base_sims
    if num_empty > NUM_CELLS - 10:
        sims = max(sims, boosted_sims)
    return sims


def mixed_value_target(z, q, q_weight):
    """Value training target = (1 - q_weight) * z + q_weight * q (2026-09-27).

    z is the game result from the mover's side (+1/0/-1); q is the root
    value of the full search that produced this ply's policy target, from
    the same mover's side (the C++ engine's root_value). Pure z carries one
    noisy bit per game: in 11x11 freestyle the side on defense usually loses
    whatever the quality of an individual defensive move, so z alone kept
    re-teaching the value head "defending = lost" even after the value head
    was reset (pessimism probe: raw -0.42 vs empirical -0.09 at ep435, while
    the search value was already close, -0.14). Mixing in q (Oracle
    "Lessons from AlphaZero" pt.4 z/q averaging; Lc0 q_ratio) lowers that
    variance with the search's own look-ahead. q=None (e.g. plies searched
    by a league opponent) or q_weight<=0 falls back to pure z."""
    if q is None or q_weight <= 0.0:
        return z
    return (1.0 - q_weight) * z + q_weight * float(q)


def annealed_value(epoch, anneal_epochs, start_value, end_value):
    """Linearly anneals start_value -> end_value over anneal_epochs, then
    holds at end_value. anneal_epochs<=0 disables annealing (always
    start_value) -- matches this file's existing "0 disables" convention
    (tactical_refresh_interval, eval_interval). More exploration noise early
    in training, less late, is standard AlphaZero/KataGo practice; this
    project so far has kept noise_weight/dirichlet_alpha fixed for the
    entire run."""
    if anneal_epochs <= 0:
        return start_value
    t = min(1.0, epoch / anneal_epochs)
    return start_value + (end_value - start_value) * t


def _play_one_mcts_game(model, device, num_simulations, boosted_sims=None, boost_threshold=0,
                         temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25, c_puct=1.414,
                         eval_cache=None, boost_when_narrow=False, start_board=None, start_player=None,
                         pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                         return_ply_info=False, max_plies=None):
    """
    Plays a single MCTS self-play game to completion and returns
    (history, outcome), where history is a list of
    (board_3d, player_turn, target_policy_map) tuples, pre-augmentation.
    With return_ply_info=True it returns (history, outcome, ply_indices,
    total_plies) instead: ply_indices[k] is the ply number (0-based, counted
    from this game's start position) at which history[k] was recorded, and
    total_plies is how many plies the game actually lasted. Needed for the
    V3 survival target once Playout Cap Randomization skips recording
    fast-search plies -- len(history) - k then undercounts the true
    plies-to-end.

    Builds its own MCTS() instance, but eval_cache (when passed) is shared
    across every game and move in the calling epoch -- safe to call
    concurrently from multiple worker threads because it's read-mostly
    (dict get/set is GIL-atomic) and a pure function of the frozen weights
    for the whole generation phase.

    start_board/start_player (both or neither): start from this board
    instead of empty -- see _sample_tactical_start, used to occasionally
    start self-play from a reachable "hard" tactical position instead of
    always from scratch, so self-play encounters that family of positions
    (and must resolve it with its own search) more often than it would
    arise naturally from full-game rollouts.

    full_search_prob/fast_sims: Playout Cap Randomization (KataGo). At each
    move, with probability full_search_prob run the normal (possibly
    boosted) search and record it as a training target; otherwise run a
    cheap fast_sims-simulation search just to pick a move and advance the
    game, without recording it (a low-sim search's visit distribution is
    too noisy to be a good policy target). Same total game count spends
    less average compute per move, so the freed budget can go to more
    games/more full searches elsewhere. full_search_prob=1.0 (default)
    disables this -- every move is a full search and recorded, matching
    prior behavior exactly.

    2026-09-26 PCR fixes (no effect at full_search_prob=1.0):
      * temperature sampling is keyed on the number of plies actually played
        (`ply`), not len(history) -- with PCR, len(history) only counts the
        recorded (full-search) plies, which would have stretched the
        temperature_moves exploratory window over ~1/full_search_prob times
        as many real moves.
      * fast searches run without Dirichlet noise (KataGo: fast searches
        exist to play strong moves cheaply, not to explore -- they are never
        recorded as training targets).
    """
    mcts = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
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
    # 2026-09-28: max_plies caps the game length (plies counted from this
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

        num_empty = sum(1 for row in board for cell in row if cell == ' ')
        is_full_search = full_search_prob >= 1.0 or random.random() < full_search_prob
        if is_full_search:
            sims_this_move = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
        else:
            sims_this_move = fast_sims or max(1, num_simulations // 6)

        # Run MCTS search (Dirichlet noise on full searches only -- see docstring)
        root = mcts.search(board, current_player, model, device, num_simulations=sims_this_move,
                            add_noise=is_full_search,
                            dirichlet_alpha=dirichlet_alpha, noise_weight=noise_weight)

        if not root.children:
            break

        total_visits = sum(child.n_visits for child in root.children.values())
        target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        for act, child in root.children.items():
            r, c = divmod(act, BOARD_SIZE)
            target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0

        # Action selection: Temperature sampling for early moves, greedy for late moves
        # (keyed on plies actually played -- equals len(history) at full_search_prob=1.0)
        move_count = ply
        actions = list(root.children.keys())
        visits = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)

        if move_count < temperature_moves:
            probs = visits / visits.sum()
            chosen_act = np.random.choice(actions, p=probs)
        else:
            chosen_act = actions[np.argmax(visits)]

        r, c = divmod(chosen_act, BOARD_SIZE)

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
                sampled = _sample_tactical_start()
                if sampled is not None:
                    start_board, start_player = sampled
            elif i < num_curriculum_games + num_opening_games:
                start_board, start_player = _sample_forced_opening_start()
            futures.append(pool.submit(_play_one_mcts_game, model, device, num_simulations, boosted_sims,
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
                board_3d0 = history[0][0]
                # board_3d0's plane 0 is "current player's stones" -- at
                # history[0] no move has landed yet, so this is always empty;
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
                # Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale
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


def _canonical_symmetry_openings(board_size):
    """One representative (row, col) per dihedral-symmetry equivalence class
    of first moves on an empty board_size x board_size board. Used to fan a
    fixed no-noise/greedy matchup out into genuinely different games instead
    of one deterministic line replayed under different game-index labels."""
    def symmetries(r, c, n):
        yield (r, c)
        yield (c, n - 1 - r)
        yield (n - 1 - r, n - 1 - c)
        yield (n - 1 - c, r)
        yield (r, n - 1 - c)
        yield (n - 1 - r, c)
        yield (c, r)
        yield (n - 1 - c, n - 1 - r)

    seen = set()
    reps = []
    for r in range(board_size):
        for c in range(board_size):
            if (r, c) in seen:
                continue
            orbit = set(symmetries(r, c, board_size))
            seen |= orbit
            reps.append((r, c))
    return reps


_OPENING_SQUARES = _canonical_symmetry_openings(BOARD_SIZE)
# All board cells, not just one representative per symmetry class -- used by
# evaluate_vs_deterministic (see its docstring): with only len(_OPENING_SQUARES)
# distinct openings, requesting a game count not coprime with that count made
# many "games" exact repeats of an earlier one under a different index label.
# Cycling through every cell instead gives eval_games far more room before
# that collision recurs.
_ALL_CELLS = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
# Gen3 eval-diversification (2026-09-09): row-major order meant every vs-Det
# check (sliced to the first num_games) only ever sampled the top-left
# edge/corner region of the board, every check, for the whole project's
# history. Shuffle once with a fixed seed so a slice is a representative
# spread across the whole board, while keeping openings reproducible
# run-to-run (same index -> same cell every time, preserving the
# pure-function-of-weights property callers/caching rely on).
random.Random(1109).shuffle(_ALL_CELLS)

# ============================================================
# Promotion Gate (AlphaGo Zero style)
# ============================================================
# A self-play policy trained by continuous gradient descent isn't guaranteed
# to improve monotonically -- this project directly measured non-transitive
# strength between checkpoints (beats one from 500 epochs ago, loses to one
# from 100 epochs ago). AlphaGo Zero's fix was to never let raw, unvalidated
# weights become the new baseline: a candidate must actually beat the current
# best in fair (no-noise, full-search) play before it's promoted. This is the
# lightweight version -- training itself is untouched (self-play/vs-Det/
# vs-League still all use the continuously-updated `model` exactly as
# before); only a separate BEST_CHECKPOINT_PATH is tracked and periodically
# updated, giving a "most robustly-validated so far" checkpoint for
# monitoring/deployment (e.g. play_vs_ai.py) independent of whatever the
# latest raw weights happen to be doing.
BEST_CHECKPOINT_PATH = f"model/best_mcts{_SUFFIX}.pth"
GATE_PROMOTION_THRESHOLD = 0.35  # 2026-10-13 (user): 0.25 -> 0.35 (6-8-6, an even result, scored 0.275 and promoted ep1005). 2026-10-12: 0.50 -> 0.25. 2026-10-11: was 0.55 (AlphaGo Zero, draws 0.5)
# 2026-10-11 (user): "a model that does not lose" -- a loss now costs points:
# score = (wins + 0.5*draws - GATE_LOSS_PENALTY*losses) / n, promoted only if score > 0.50 (strictly).
# With 0.75 that is exactly "wins > 2.5 x losses"; draws alone can never promote (all draws = 0.50).
GATE_LOSS_PENALTY = 0.75
# 2026-10-13 (user): two-stage promotion. The gate (vs the current best only) can pick a model that is
# weaker by the Rapfi ladder (ep1050 beat ep960 12-5-3 but measured 2,404 vs 4,432). A gate pass now only
# makes a CANDIDATE: its weights go to model/gate_candidates/ and model/gate_candidate.json; the ladder
# watcher measures it and promotes it (model/ladder_promote.json) only if it is not weaker than the best.
GATE_CANDIDATE_MODE = True
RECORD_LEAGUE_SIDE = False   # 2026-10-13 (user): see _play_one_mcts_vs_league_game
_LAST_GATE_CANDIDATE = {"path": None}


def _save_gate_candidate(model, score):
    import json as _json
    os.makedirs("model/gate_candidates", exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    path = f"model/gate_candidates/cand_{tag}{_SUFFIX}.pth"
    torch.save(model.state_dict(), path)
    with open("model/gate_candidate.json", "w", encoding="utf-8") as f:
        _json.dump({"ckpt": path, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gate_score": round(score, 4),
                    "status": "pending"}, f, indent=1)
    _LAST_GATE_CANDIDATE["path"] = path
    return path


def _play_gate_match_game(model_a, model_b, device, num_simulations, boosted_sims=None, boost_threshold=0,
                           boost_when_narrow=False, c_puct=1.414, a_plays_x=True, opening=None,
                           eval_cache_a=None, eval_cache_b=None,
                           pooled_evaluator_a=None, pooled_evaluator_b=None):
    """
    One fair (no-noise, greedy, full-search) game between two frozen models,
    with a forced opening (see _canonical_symmetry_openings -- without one,
    no-noise MCTS vs no-noise MCTS is fully deterministic and every game
    with the same side assignment would just replay the same line). Returns
    'a', 'b', or 'draw'.
    """
    a_player = 'X' if a_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    mcts_a = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache_a, pooled_evaluator=pooled_evaluator_a)
    mcts_b = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache_b, pooled_evaluator=pooled_evaluator_b)

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        if fast_check_win(board, a_player):
            return 'a'
        if fast_check_win(board, 'O' if a_player == 'X' else 'X'):
            return 'b'
        if fast_check_draw(board):
            return 'draw'

        num_empty = sum(1 for row in board for cell in row if cell == ' ')
        sims = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
        active_model, active_mcts = (model_a, mcts_a) if current_player == a_player else (model_b, mcts_b)
        root = active_mcts.search(board, current_player, active_model, device,
                                   num_simulations=sims, add_noise=False)
        if not root.children:
            return 'draw'
        best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        r, c = divmod(best_action, BOARD_SIZE)
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


# Gate-check speed optimization (2026-09-10): the periodic tactical-refresh
# epoch (every tactical_refresh_interval) was measured taking 15-25x a
# normal epoch's wall-clock -- ~42% of total training time in a 232-epoch
# stretch -- and the 42-game (21 openings x 2 sides) gate check was the
# dominant cost within it. Cut to 10 stratified openings (20 games): a
# fixed-seed shuffle of _OPENING_SQUARES so the subset stays spatially
# spread rather than clustering in whatever order
# _canonical_symmetry_openings happened to emit them (same principle as the
# vs-Det opening fix). Trades some promotion-decision precision (10%-point
# granularity instead of ~2.4%-point) for roughly half the gate-check cost.
_GATE_OPENING_SQUARES = list(_OPENING_SQUARES)
random.Random(910).shuffle(_GATE_OPENING_SQUARES)
_GATE_OPENING_SQUARES = _GATE_OPENING_SQUARES[:10]


def run_gating_check(model, gen_model, device, num_simulations, boosted_sims=None, boost_threshold=0,
                      boost_when_narrow=False, c_puct=1.414, max_workers=None, runner=None):
    """
    Plays the currently-training `model` against the tracked best checkpoint
    across all board-symmetry openings x both sides (fair games). If model's
    score (win=1, draw=0.5) reaches GATE_PROMOTION_THRESHOLD, model's weights
    become the new best. Bootstraps on first call (no best yet). `gen_model`
    (the frozen model self-play/vs-Det/vs-League data generation actually
    uses) is synced to the tracked best's weights on every promotion,
    including the bootstrap case, so generation always reflects the latest
    validated checkpoint.

    Games run concurrently across a ThreadPoolExecutor (same pattern as
    evaluate_vs_deterministic / the three generate_mcts_*_data functions).
    Two frozen models are in play here (`model`, `best_model`), unlike the
    single-model generation functions, so each gets its OWN PooledEvaluator
    and its OWN shared eval_cache -- a cache entry or pooled batch is only
    ever valid for the exact weights that produced it, so model's and
    best_model's calls are never pooled or cached together. Behavior-
    preserving for game OUTCOMES (see evaluate_vs_deterministic docstring
    for why); only wall-clock time changes.
    runner (2026-09-27): optional selfplay_mp_unified task runner -- plays
    the 20 gate games in the persistent multi-process pool (the two models
    under separate eval tags/caches, same isolation as the two
    PooledEvaluators below); any failure falls back to the threaded path.
    Returns (win_rate, draw_rate, loss_rate, promoted).
    """
    if not os.path.exists(BEST_CHECKPOINT_PATH):
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
        gen_model.load_state_dict(model.state_dict())
        return 1.0, 0.0, 0.0, True

    best_model = AlphaZeroTicTacToe()
    try:
        best_model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=device))
    except Exception as e:
        print(f"Gate check notice ({e}), skipping this cycle.")
        return 0.0, 0.0, 0.0, False
    best_model.to(device, memory_format=preferred_memory_format(device))
    best_model.eval()

    # model arrives here straight out of tactical_refresh(), which leaves it
    # in train() mode -- with eval_batch_size=1 every MCTS leaf eval is a
    # single-sample forward pass, so BatchNorm would normalize against that
    # one sample's own stats instead of the accumulated running mean/var,
    # systematically distorting every policy/value output for this model
    # only (2026-08-29: same bug confirmed in the 5x5 sibling's gate check).
    model.eval()

    results = None
    if runner is not None:
        try:
            results = runner.gate(model, best_model, device, dict(
                num_simulations=num_simulations, boosted_sims=boosted_sims, boost_threshold=boost_threshold,
                boost_when_narrow=boost_when_narrow, c_puct=c_puct),
                [(opening, latest_plays_x) for opening in _GATE_OPENING_SQUARES for latest_plays_x in (True, False)])
        except Exception as e:  # noqa: BLE001
            print(f"  [gate] unified-pool gate failed ({e!r}) -- threaded fallback", flush=True)
            results = None
    if results is not None:
        wins = sum(1 for r in results if r == 'a')
        draws = sum(1 for r in results if r == 'draw')
        losses = len(results) - wins - draws
        n = max(1, wins + draws + losses)
        score = (wins + 0.5 * draws - GATE_LOSS_PENALTY * losses) / n
        promoted = score > GATE_PROMOTION_THRESHOLD
        if promoted and GATE_CANDIDATE_MODE:
            _save_gate_candidate(model, score)
            promoted = False
        elif promoted:
            torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
            gen_model.load_state_dict(model.state_dict())
        return wins / n, draws / n, losses / n, promoted

    max_workers = max_workers or max(1, min(len(_GATE_OPENING_SQUARES) * 2, os.cpu_count() or 1))
    eval_cache_model = {}
    eval_cache_best = {}
    owns_pooled = device.type == "cuda"
    pooled_model = PooledEvaluator(model, device, target_batch=24, max_wait=0.05) if owns_pooled else None
    pooled_best = PooledEvaluator(best_model, device, target_batch=24, max_wait=0.05) if owns_pooled else None

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for opening in _GATE_OPENING_SQUARES:
                for latest_plays_x in (True, False):
                    futures.append(pool.submit(_play_gate_match_game, model, best_model, device, num_simulations,
                                                boosted_sims, boost_threshold, boost_when_narrow, c_puct,
                                                latest_plays_x, opening,
                                                eval_cache_model, eval_cache_best, pooled_model, pooled_best))
            wins = draws = losses = 0
            for future in futures:
                result = future.result()
                if result == 'a':
                    wins += 1
                elif result == 'draw':
                    draws += 1
                else:
                    losses += 1
    finally:
        if pooled_model is not None:
            pooled_model.stop()
        if pooled_best is not None:
            pooled_best.stop()

    n = max(1, wins + draws + losses)
    score = (wins + 0.5 * draws - GATE_LOSS_PENALTY * losses) / n
    promoted = score > GATE_PROMOTION_THRESHOLD
    if promoted and GATE_CANDIDATE_MODE:
        _save_gate_candidate(model, score)
        promoted = False
    elif promoted:
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
        gen_model.load_state_dict(model.state_dict())
    return wins / n, draws / n, losses / n, promoted


# ============================================================
# Simulation-budget ramp (train cheap early, deepen on proven improvement)
# ============================================================
# Early in training the network is close to random, so a deep 900-simulation
# search mostly amplifies that network's own noise rather than finding
# genuinely better moves -- this board's search is also the single biggest
# cost driver of this project's slowest training loop. Start self-play/vs-Det/
# vs-League data generation at a cheaper budget and only deepen it when the
# promotion gate (run_gating_check) actually confirms real improvement,
# rather than assuming compute should scale with epoch count alone. Persisted
# the same way as cumulative_epoch/best_checkpoint so it survives restarts.
SIM_RAMP_PATH = f"model/sim_ramp_state{_SUFFIX}.json"
SIM_RAMP_START = 150
SIM_RAMP_STEP = 100
SIM_RAMP_CEILING = 1600  # 2026-10-13 (user): 1100 -> 1600 (deeper full searches = better policy targets)
BOOST_RAMP_START = 300
BOOST_RAMP_STEP = 200
BOOST_RAMP_CEILING = 2200
# If the gate keeps 'kept'-ing (no promotion) for this many checks in a row,
# bump the sim budget anyway by the same step -- otherwise a long stagnation
# streak leaves self-play stuck at a low budget indefinitely, since normally
# the ramp only advances on an actual promotion.
STAGNATION_RAMP_LIMIT = 5
# 2026-09-15: a blind calendar-based LR warm-restart (CosineAnnealingWarmRestarts
# below) can land in the middle of a fresh promotion's destabilized window --
# measured on this exact run: the epoch-273 promotion (first ever, after 252
# straight 0%-loss gate checks) was followed by vs-Det formal win rate
# collapsing from an oscillating 15-45% to a 0-10% floor, and the epoch-300
# warm restart (LR 5e-5 -> 0.001) landed right inside that still-unrecovered
# window, 27 epochs later, making it worse rather than helping the model
# re-explore. LR_COOLDOWN_EPOCHS holds the scheduler frozen at its current
# (already-decayed) LR for this many epochs right after any promotion, so a
# calendar restart can't collide with a promotion-induced shift again.
LR_COOLDOWN_EPOCHS = 25


def load_sim_budget():
    if os.path.exists(SIM_RAMP_PATH):
        try:
            import json
            with open(SIM_RAMP_PATH) as f:
                d = json.load(f)
            return (int(d.get("num_simulations", SIM_RAMP_START)),
                    int(d.get("boosted_sims", BOOST_RAMP_START)),
                    int(d.get("stagnation_count", 0)),
                    int(d.get("lr_cooldown_remaining", 0)))
        except Exception as e:
            print(f"Sim-ramp state load notice ({e}). Starting at base tier.")
    return SIM_RAMP_START, BOOST_RAMP_START, 0, 0


def save_sim_budget(num_simulations, boosted_sims, stagnation_count=0, lr_cooldown_remaining=0):
    # Atomic write -- see save_cumulative_epoch's comment for why (same
    # class of empty-file corruption risk on a kill mid-write).
    import json
    os.makedirs(os.path.dirname(SIM_RAMP_PATH) or ".", exist_ok=True)
    tmp_path = SIM_RAMP_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"num_simulations": num_simulations, "boosted_sims": boosted_sims,
                   "stagnation_count": stagnation_count,
                   "lr_cooldown_remaining": lr_cooldown_remaining}, f)
    os.replace(tmp_path, SIM_RAMP_PATH)


def _play_vs_deterministic(model, device, num_simulations, mcts_plays_x, boosted_sims=None, boost_threshold=0,
                            c_puct=1.414, boost_when_narrow=False, opening=None,
                            eval_cache=None, pooled_evaluator=None):
    """
    Plays one game: the in-memory model (via fresh MCTS search, no noise,
    greedy) against determinisic_play's minimax AI. Returns 'mcts', 'det',
    or 'draw'.

    add_noise=False makes MCTS search deterministic, and determinisic_play is
    deterministic too, so without `opening` every game with the same
    mcts_plays_x is an identical replay of the same line -- num_games games
    then really only cover 2 distinct positions (one per side). Passing a
    (row, col) `opening` pre-places X's first move there before normal play
    resumes, fanning otherwise-identical matchups out into real variety.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    mcts_engine = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)

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
            r, c = determinisic_play.play(board, current_player)
            if r == -1:
                return 'draw'

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


def evaluate_vs_deterministic(model, device, num_games=10, num_simulations=50, boosted_sims=None, boost_threshold=0,
                               c_puct=1.414, boost_when_narrow=False, max_workers=None, pooled_evaluator=None,
                               all_cells=None):
    """
    Plays num_games against determinisic_play (minimax/alpha-beta), alternating
    who moves first, and returns (win_rate, draw_rate, loss_rate) from the
    trained model's perspective. This is the direct metric for the research
    goal ("beat the deterministic algorithm").

    Tries the multiprocessing implementation (mp_eval.evaluate_vs_deterministic_mp)
    first -- each game gets a real OS process instead of a thread, which
    matters because determinisic_play (the pure-Python minimax opponent)
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
    all_cells = all_cells if all_cells is not None else _ALL_CELLS
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
    all_cells = all_cells if all_cells is not None else _ALL_CELLS
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    eval_cache = {}
    owns_pooled_evaluator = pooled_evaluator is None and device.type == "cuda"
    if owns_pooled_evaluator:
        pooled_evaluator = PooledEvaluator(model, device, target_batch=24, max_wait=0.05)
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


def _sample_vs_det_max_depth(epoch):
    """
    Opponent-depth ladder for vs-Deterministic training-data generation
    ONLY (never evaluate_vs_deterministic or run_gating_check, which must
    stay a fixed, comparable depth-2 opponent forever). Samples a per-game
    minimax search depth for determinisic_play from a schedule keyed on the
    current training epoch, so the model faces a weak/shallow opponent
    early on and a deeper one later instead of a fixed difficulty from
    epoch 0. Depth 0 means "skip minimax, play a uniformly random legal
    move" (handled by the caller, not by determinisic_play itself).

    Deliberately capped at depth 3 for compute budget -- never sample
    beyond it (a deeper minimax is exponentially more expensive, and this
    project's 10x10/WIN_LENGTH=5 board is already the slowest, most
    compute-constrained board size here).
    """
    if epoch < 300:
        depths, weights = (0, 1, 2), (0.15, 0.35, 0.50)
    elif epoch < 1500:
        # Small depth-0 floor (2026-09-09 diversity pass): depth>=1 minimax
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


def _play_one_mcts_vs_deterministic_game(model, device, num_simulations, mcts_plays_x,
                                          boosted_sims=None, boost_threshold=0,
                                          temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                          c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                          pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                          opening=None, det_max_depth=determinisic_play.MAX_DEPTH):
    """
    Plays one training game between the in-memory model (MCTS with noise
    and temperature sampling -- same exploration settings as self-play) and
    determinisic_play's minimax AI. Only the MCTS side's decisions are
    recorded, in the same (board_3d, player_turn, target_policy_map) format
    _play_one_mcts_game produces, so callers can mix both into one buffer.

    Pure self-play never has to defend against or attack the specific
    tactics determinisic_play actually uses, so the model can stay blind to
    patterns that opponent exploits no matter how long training runs.
    Mixing in games against it directly closes that loop.

    determinisic_play is deterministic, so without `opening` every game
    with the same mcts_plays_x replays the same line -- see
    _play_vs_deterministic's docstring. Passing a (row, col) `opening`
    pre-places X's first move there before normal play resumes.

    det_max_depth (see _sample_vs_det_max_depth) sets the deterministic
    opponent's minimax search depth for this game; 0 means it plays a
    uniformly random legal move instead of calling minimax at all.

    Returns (history, outcome, mcts_result): outcome uses the same
    X=1/O=-1/draw=0 convention as self-play; mcts_result is
    'mcts'/'det'/'draw' from the trained model's perspective.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    history = []
    # MCTS-side decisions made so far (full or fast) -- drives temperature;
    # equals len(history) at full_search_prob=1.0 (2026-09-26 PCR fix, see
    # _play_one_mcts_game's docstring).
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
                r, c = determinisic_play.play(board, current_player, max_depth=det_max_depth)
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
    determinisic_play instead of itself (alternating X/O by game index), so
    the replay buffer includes direct experience of the exact opponent
    evaluate_vs_deterministic measures against.

    Each game cycles through _OPENING_SQUARES by game index (determinisic_play
    is deterministic, so without this every game with the same side
    assignment would replay the same line -- see _play_vs_deterministic's
    docstring) and samples its own opponent search depth from the
    epoch-keyed ladder in _sample_vs_det_max_depth -- evaluate_vs_deterministic
    and run_gating_check are untouched by this and always keep the fixed
    depth-2 opponent.
    """
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    print(f"Generating {num_games} MCTS-vs-Deterministic training games "
          f"(Simulations: {num_simulations}/move, workers: {max_workers})...")

    model.eval()
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'det_win': 0, 'draw': 0}
    # Per-depth tally (2026-09-10). The aggregate "1-3 wins per 10 games" line
    # was measured to be 100% depth-0 (random opponent) wins and 0% against
    # both depth 1 and depth 2, so the aggregate was hiding a flat zero
    # against any real opponent. Split it so progress is actually visible.
    by_depth = {}
    game_depths = [_sample_vs_det_max_depth(epoch) for _ in range(num_games)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_play_one_mcts_vs_deterministic_game, model, device, num_simulations,
                                i % 2 == 0, boosted_sims, boost_threshold,
                                temperature_moves, dirichlet_alpha, noise_weight, c_puct, eval_cache,
                                boost_when_narrow, pooled_evaluator, full_search_prob, fast_sims,
                                _OPENING_SQUARES[(i + epoch) % len(_OPENING_SQUARES)],
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
                # Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale
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


# ============================================================
# League Training (self-play vs past checkpoints)
# ============================================================
# Pure self-play (vs current-self) + vs-Deterministic (one fixed opponent)
# risks overfitting to that single opponent's style -- a weakness never
# gets tested against unless that one opponent happens to exploit it. League
# training (AlphaStar/OpenAI Five) mixes in games against a pool of the
# model's OWN past checkpoints (already saved every 10 epochs as
# model_MCTS_ep{N}{_SUFFIX}.pth snapshots -- no new storage needed), so
# self-play sees a diversity of strategies instead of one fixed target.


LEAGUE_ANCHOR_PROB = 0.0   # random anchor draw (2026-09-30 first try) -- superseded by the rotation below
LEAGUE_ANCHOR_ROTATION = True  # 2026-09-30: epoch%3 -> Dawn / ep645 / normal draw
_LEAGUE_SNAPSHOT_RE = re.compile(r"model_MCTS_ep(\d+)" + re.escape(_SUFFIX) + r"\.pth$")


def _load_league_model(path, device):
    """Shared checkpoint-load helper for _pick_league_opponent's two
    branches (tracked-best / snapshot-pool). Returns the loaded, eval()'d
    model, or None on any load failure (caller treats None as "skip league
    games this epoch", same as the pre-existing failure behavior)."""
    league_model = AlphaZeroTicTacToe()
    try:
        league_model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        print(f"League opponent load notice ({e}), skipping league games this epoch.")
        return None
    league_model.to(device, memory_format=preferred_memory_format(device))
    league_model.eval()
    return league_model


import glob


def _pick_league_opponent(device, pool_size=20, best_checkpoint_prob=0.28, epoch=0,
                           deep_history_prob=0.10, deep_history_min_epoch=100):
    """Picks a frozen league opponent for this epoch's vs-League games.

    With probability best_checkpoint_prob (checked first), loads the
    tracked BEST_CHECKPOINT_PATH -- the strongest validated checkpoint,
    always kept up to date by run_gating_check's promotion logic -- instead
    of sampling from the snapshot pool at all. This guarantees a real,
    strong sparring partner shows up in league play regardless of how the
    plain recency-weighted draw below happens to fall.

    Otherwise (or if that file doesn't exist), with probability
    deep_history_prob (once epoch >= deep_history_min_epoch) samples
    UNIFORMLY from the FULL snapshot history instead of just the most
    recent pool_size -- the recency-weighted pool below permanently
    excludes a snapshot once pool_size*10 epochs have passed it, so without
    this the model would only ever face styles from its own recent past,
    never its much-earlier (structurally weaker/more exploratory) self.
    Kept low-probability and floored at deep_history_min_epoch so it stays
    a small, occasional addition, not a replacement for the recency-weighted
    pool that already does most of the work.

    Otherwise, falls back to the existing snapshot pool: a random past
    checkpoint snapshot from the most recent pool_size snapshots BY EPOCH
    NUMBER, not filesystem mtime -- mtime can get scrambled by anything
    that touches the files without touching training progress, e.g. a
    backup pass or several snapshots written in quick succession right
    around a restart. Sampling within that pool is recency-weighted
    (linearly favoring the newest snapshot), not uniform, since the newest
    snapshots are the most representative of the model's current style --
    the whole point of league play is testing against strategies close to
    what self-play might otherwise overfit to.

    Returns (league_model, path), or None if the best-checkpoint load
    failed, or if fewer than 2 snapshots exist yet -- too early in training
    for league play to add anything over the fixed deterministic opponent
    already in use."""
    # 2026-09-30 (user, option A): deterministic rotation over epochs --
    # epoch % 3 == 0 -> first anchor (Dawn ep625), == 1 -> second anchor (ep645),
    # == 2 -> the normal draw below (best / deep history / recency pool).
    anchors = sorted(glob.glob("model/league_anchors/*.pth")) if LEAGUE_ANCHOR_ROTATION else []
    if anchors and epoch % 3 < len(anchors) and epoch % 3 < 2:
        chosen = anchors[epoch % 3]
        league_model = _load_league_model(chosen, device)
        if league_model is not None:
            return league_model, chosen

    if best_checkpoint_prob > 0 and random.random() < best_checkpoint_prob and os.path.exists(BEST_CHECKPOINT_PATH):
        league_model = _load_league_model(BEST_CHECKPOINT_PATH, device)
        if league_model is None:
            return None
        return league_model, BEST_CHECKPOINT_PATH

    # 2026-09-30 (user): fixed "anchor" opponents -- the graduated Dawn (ep625) and the
    # long-standing best ep645 -- so league play keeps facing the strongest known models
    # while the gate is stuck. Checked right after the best-checkpoint draw.
    anchors = [p for p in sorted(glob.glob("model/league_anchors/*.pth"))] if LEAGUE_ANCHOR_PROB > 0 else []
    if anchors and random.random() < LEAGUE_ANCHOR_PROB:
        chosen = random.choice(anchors)
        league_model = _load_league_model(chosen, device)
        if league_model is None:
            return None
        return league_model, chosen

    pattern = f"model/model_MCTS_ep*{_SUFFIX}.pth"
    numbered = []
    for path in glob.glob(pattern):
        m = _LEAGUE_SNAPSHOT_RE.search(os.path.basename(path))
        if m:  # excludes the "_final" end-of-run checkpoint, which doesn't match this pattern
            numbered.append((int(m.group(1)), path))
    numbered.sort(key=lambda t: t[0])
    snapshots = [path for _, path in numbered]
    if len(snapshots) < 2:
        return None

    if epoch >= deep_history_min_epoch and random.random() < deep_history_prob:
        chosen = random.choice(snapshots)
        league_model = _load_league_model(chosen, device)
        if league_model is None:
            return None
        return league_model, chosen

    candidates = snapshots[-pool_size:]
    # candidates is oldest-to-newest; a linear ramp (1, 2, ..., len) makes
    # the newest snapshot len(candidates) times as likely as the oldest,
    # while still leaving every snapshot in the pool a nonzero chance.
    weights = list(range(1, len(candidates) + 1))
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    league_model = _load_league_model(chosen, device)
    if league_model is None:
        return None
    return league_model, chosen


LEAGUE_NUM_OPPONENTS = 12  # 2026-10-08 (user): many opponents per epoch instead of one


def _pick_league_opponents_multi(device, k, pool_size=40, epoch=0):
    """2026-10-08 (user, option A): k distinct league opponents for ONE epoch --
    the fixed anchors (Dawn ep625, ep645), the current best, one uniform pick from
    the whole snapshot history, and the rest recency-weighted from the newest
    pool_size snapshots without replacement. Returns [(model, path), ...]."""
    chosen = []
    seen = set()

    def add(path):
        if path in seen or len(chosen) >= k:
            return
        m = _load_league_model(path, device)
        if m is not None:
            chosen.append((m, path))
            seen.add(path)

    for a in sorted(glob.glob("model/league_anchors/*.pth")):
        add(a)
    if os.path.exists(BEST_CHECKPOINT_PATH):
        add(BEST_CHECKPOINT_PATH)
    numbered = []
    for path in glob.glob(f"model/model_MCTS_ep*{_SUFFIX}.pth"):
        mm = _LEAGUE_SNAPSHOT_RE.search(os.path.basename(path))
        if mm:
            numbered.append((int(mm.group(1)), path))
    numbered.sort(key=lambda t: t[0])
    snapshots = [p for _, p in numbered]
    if snapshots:
        add(random.choice(snapshots))
    cands = snapshots[-pool_size:]
    weights = list(range(1, len(cands) + 1))
    while len(chosen) < k and any(c not in seen for c in cands):
        pool = [(c, w) for c, w in zip(cands, weights) if c not in seen]
        add(random.choices([c for c, _ in pool], weights=[w for _, w in pool], k=1)[0])
    return chosen


def _play_one_mcts_vs_league_game(model, league_model, device, num_simulations, mcts_plays_x,
                                   boosted_sims=None, boost_threshold=0,
                                   temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                   c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                   pooled_evaluator=None, full_search_prob=1.0, fast_sims=None,
                                   opening=None, league_pooled_evaluator=None, league_eval_cache=None,
                                   max_plies=None):
    """
    Like _play_one_mcts_vs_deterministic_game, but the opponent is a frozen
    past checkpoint of this same model instead of determinisic_play. Both
    sides' moves are recorded (the league opponent runs a real, always-full
    search -- see league_sims below -- so its visit distribution is just as
    valid a policy target as the live model's own): the live model was
    previously only ever training on its own moves against the league pool,
    never on the moves of the checkpoints that actually beat it, which
    starved league-derived data of the one thing it could uniquely teach.

    opening (see _play_one_mcts_vs_deterministic_game): pre-places X's first
    move there before normal play resumes. The league side's own moves use
    add_noise=False (greedy), so when it moves first, every game with the
    same side assignment would otherwise replay the identical opening move.

    league_pooled_evaluator/league_eval_cache (2026-09-26, unified MP
    game pool): route the LEAGUE side's leaf evaluations through a separate
    evaluator/cache -- a worker process holds no weights, so it needs a
    proxy that the GPU coordinator maps to the league model. Both default
    to None, which keeps the original behavior (league side calls
    league_model directly with a private per-search cache).

    Returns (history, outcome, mcts_result): mcts_result is
    'mcts'/'league'/'draw' from the live model's perspective.
    """
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    league_mcts = MCTS(c_puct=c_puct, eval_batch_size=16, shared_cache=league_eval_cache,
                       pooled_evaluator=league_pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    history = []
    # Decisions that record (or, for a PCR fast search, would have recorded)
    # a history entry -- drives temperature; equals len(history) at
    # full_search_prob=1.0 (2026-09-26 PCR fix).
    decisions = 0
    outcome = 0

    if opening is not None:
        r0, c0 = opening
        board[r0][c0] = 'X'
        current_player = 'O'

    while True:
        # 2026-09-28: game-length cap (stones on the board); capped = draw.
        if max_plies and sum(1 for row in board for cell in row if cell != ' ') >= max_plies:
            outcome = 0
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

        num_empty = sum(1 for row in board for cell in row if cell == ' ')

        if current_player == mcts_player:
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

            move_count = decisions
            actions = list(root.children.keys())
            visits = np.array([root.children[a].n_visits for a in actions], dtype=np.float32)
            if move_count < temperature_moves:
                probs = visits / visits.sum()
                chosen_act = np.random.choice(actions, p=probs)
            else:
                chosen_act = actions[np.argmax(visits)]
            r, c = divmod(chosen_act, BOARD_SIZE)
            decisions += 1

            if is_full_search:
                board_3d = board_to_input_array(board, current_player)
                history.append((board_3d, current_player, target_policy_map, float(root.value())))
        else:
            # 2026-09-28: PCR on the league side too (same full_search_prob
            # as the MCTS side): a fast search only picks the move and is not
            # recorded. At full_search_prob=1.0 this is the old behavior.
            league_full = full_search_prob >= 1.0 or random.random() < full_search_prob
            if league_full:
                league_sims = sims_for_state(num_empty, num_simulations, boosted_sims, boost_threshold, boost_when_narrow)
            else:
                league_sims = fast_sims or max(1, num_simulations // 6)
            root = league_mcts.search(board, current_player, league_model, device,
                                       num_simulations=league_sims, add_noise=False)
            if not root.children:
                outcome = 0
                break
            total_visits = sum(child.n_visits for child in root.children.values())
            target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            for act, child in root.children.items():
                r, c = divmod(act, BOARD_SIZE)
                target_policy_map[r, c] = child.n_visits / total_visits if total_visits > 0 else 0.0

            best_action = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
            r, c = divmod(best_action, BOARD_SIZE)

            # 2026-10-13 (user): league-side plies are no longer training targets -- the opponents are older,
            # weaker checkpoints (some ~700-1,000 equal nodes), so their visit counts taught weaker moves.
            if league_full and RECORD_LEAGUE_SIDE:
                board_3d = board_to_input_array(board, current_player)
                # League-side ply: its search value comes from a different
                # (older) network, so this ply keeps a pure-z value target.
                history.append((board_3d, current_player, target_policy_map, None))
            decisions += 1

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'

    if outcome == 1:
        mcts_result = 'mcts' if mcts_player == 'X' else 'league'
    elif outcome == -1:
        mcts_result = 'mcts' if mcts_player == 'O' else 'league'
    else:
        mcts_result = 'draw'

    return history, outcome, mcts_result


def generate_mcts_vs_league_data(model, league_model, device, num_games=6, num_simulations=40, max_workers=None,
                                  boosted_sims=None, boost_threshold=0,
                                  temperature_moves=6, dirichlet_alpha=0.3, noise_weight=0.25,
                                  c_puct=1.414, eval_cache=None, boost_when_narrow=False,
                                  pooled_evaluator=None, full_search_prob=1.0, fast_sims=None, epoch=0,
                                  value_q_weight=0.0, max_plies=None):
    """
    Same shape as generate_mcts_vs_deterministic_data, but the opponent is a
    frozen past checkpoint of this model (see _play_one_mcts_vs_league_game).
    """
    max_workers = max_workers or max(1, min(num_games, os.cpu_count() or 1))
    print(f"Generating {num_games} MCTS-vs-League training games "
          f"(Simulations: {num_simulations}/move, workers: {max_workers})...")

    model.eval()
    raw_samples = []
    outcome_counts = {'mcts_win': 0, 'league_win': 0, 'draw': 0}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_play_one_mcts_vs_league_game, model, league_model, device, num_simulations,
                                i % 2 == 0, boosted_sims, boost_threshold,
                                temperature_moves, dirichlet_alpha, noise_weight, c_puct, eval_cache,
                                boost_when_narrow, pooled_evaluator, full_search_prob, fast_sims,
                                _OPENING_SQUARES[(i + epoch) % len(_OPENING_SQUARES)],
                                None, None, max_plies)
                   for i in range(num_games)]
        for future in futures:
            history, outcome, mcts_result = future.result()
            if mcts_result == 'mcts':
                outcome_counts['mcts_win'] += 1
            elif mcts_result == 'league':
                outcome_counts['league_win'] += 1
            else:
                outcome_counts['draw'] += 1
            for (board_3d, player_turn, target_policy_map, q) in history:
                # Full +-1.0 scale (was 0.5 until 2026-09-10). The halved scale
                # shrank Q's dynamic range against an unchanged c_puct, which
                # doubles the effective exploration constant and quadruples the
                # simulations needed for visit counts to overcome the prior. It
                # also fought tactical_refresh's own 0.97 value target.
                final_val = mixed_value_target(float(outcome if player_turn == 'X' else -outcome), q, value_q_weight)
                sample_weight = 1.25
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val, sample_weight))

    print(f"Generated {len(raw_samples)} augmented vs-League training samples. "
          f"Outcomes: MCTS {outcome_counts['mcts_win']} / League {outcome_counts['league_win']} / "
          f"Draw {outcome_counts['draw']}")
    return raw_samples, outcome_counts


def load_replay_buffer():
    if os.path.exists(REPLAY_BUFFER_PATH):
        try:
            with open(REPLAY_BUFFER_PATH, "rb") as f:
                stored = pickle.load(f)
            buf = deque(stored, maxlen=REPLAY_BUFFER_MAX_SIZE)
            print(f"Loaded replay buffer with {len(buf)} samples from {REPLAY_BUFFER_PATH}")
            return buf
        except Exception as e:
            print(f"Replay buffer load notice ({e}). Starting with an empty buffer.")
    return deque(maxlen=REPLAY_BUFFER_MAX_SIZE)


def save_replay_buffer(buffer):
    os.makedirs(os.path.dirname(REPLAY_BUFFER_PATH), exist_ok=True)
    with open(REPLAY_BUFFER_PATH, "wb") as f:
        pickle.dump(list(buffer), f)


# ============================================================
# Cumulative epoch state (survives process restarts)
# ============================================================
# The LR step-down (step_size=2000) and noise/dirichlet annealing
# (noise_anneal_epochs) are both keyed off the training loop's `epoch`
# variable -- previously always 0 at the start of every call, so every
# restart (crash, manual restart, hyperparameter change) reset the model
# back to full LR and near-max exploration noise, regardless of how much
# real training had already accumulated in the weights. Persisting the
# cumulative epoch count here and offsetting the loop by it means those
# schedules (and snapshot/eval/tactical-refresh timing) track real
# cumulative progress instead of restarting every time.
TRAIN_STATE_PATH = f"model/train_state{_SUFFIX}.json"


def load_start_epoch():
    if os.path.exists(TRAIN_STATE_PATH):
        try:
            import json
            with open(TRAIN_STATE_PATH) as f:
                return int(json.load(f).get("cumulative_epoch", 0))
        except Exception as e:
            print(f"Train-state load notice ({e}). Starting epoch offset at 0.")
    return 0


def save_cumulative_epoch(n):
    # Atomic write (temp file + os.replace, same directory so the replace
    # is a single filesystem-level rename): a plain open(path, "w") truncates
    # the file immediately, so a process kill between the truncate and the
    # json.dump landing (e.g. an external restart) leaves an empty/corrupt
    # file -- 2026-09-09 this exact failure (empty train_state file ->
    # "Expecting value: line 1 column 1" -> silently restarted the epoch
    # counter at 0 while the checkpoint/buffer were still the real ~73-epoch
    # state) actually happened live. os.replace is atomic on both Windows
    # and POSIX, so a reader only ever sees the old complete file or the new
    # complete file, never a partial one.
    import json
    os.makedirs(os.path.dirname(TRAIN_STATE_PATH) or ".", exist_ok=True)
    tmp_path = TRAIN_STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"cumulative_epoch": n}, f)
    os.replace(tmp_path, TRAIN_STATE_PATH)


# ============================================================
# Tactical Curriculum (periodic refresh)
# ============================================================
# Diagnosis from tonight's session: pure self-play plateaus at 0% win rate
# against the deterministic opponent because it keeps missing obvious
# "3-filled-of-4 + 1 empty" line threats (open-3s and internal gaps alike).
# A supervised sanity check showed the network CAN learn this pattern
# instantly (94% val accuracy in 15 epochs on 448 synthetic examples) --
# capacity isn't the bottleneck, self-play's own search just rarely
# generates strong enough targets for it. A one-time pretrain on this
# curriculum gave a real but temporary boost (loss broke through its old
# plateau, entropy kept climbing) that faded over ~30 epochs as the much
# larger stream of ordinary self-play/vs-det samples diluted it back out.
# So instead of a single injection, refresh periodically throughout training.
_TACTICAL_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _tactical_windows():
    windows = []
    for dr, dc in _TACTICAL_DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = [(r + dr * k, c + dc * k) for k in range(WIN_LENGTH)]
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    windows.append(cells)
    return windows


def _tactical_clutter(board, exclude_cells, n_stones=3, cluster_radius=3):
    """2026-09-14: clutter now clusters NEAR the tactical window (within
    cluster_radius of any exclude_cells cell) instead of scattering
    uniformly across the whole 11x11 board. Real games don't produce
    stones scattered board-wide around an isolated pattern -- play
    naturally clusters around the area of active contest, which is
    exactly the shape a real mined position has. Uniform-random clutter
    was a distribution the model could learn to recognize (and exploit)
    that never matches real games, plausibly a contributor to synthetic-
    trained candidates underperforming even the real-data label-conflict
    ceiling on held-out real positions (see
    scratch/multiproc_investigation/mine_tactical_conflict_check.py).
    Falls back to anywhere-empty if the local neighborhood is too crowded
    to fit n_stones (rare)."""
    near = set()
    for (r0, c0) in exclude_cells:
        for dr in range(-cluster_radius, cluster_radius + 1):
            for dc in range(-cluster_radius, cluster_radius + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == ' ' and (r, c) not in exclude_cells:
                    near.add((r, c))
    candidates = list(near)
    random.shuffle(candidates)
    if len(candidates) < n_stones:
        rest = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                if board[r][c] == ' ' and (r, c) not in exclude_cells and (r, c) not in near]
        random.shuffle(rest)
        candidates += rest
    chosen = candidates[:n_stones]
    for (r, c) in chosen:
        board[r][c] = random.choice(['X', 'O'])
    # Never keep a clutter placement that accidentally creates/completes a
    # 4-in-a-row elsewhere -- that would make the intended target wrong.
    if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
        for (r, c) in chosen:
            board[r][c] = ' '


def _sample_tactical_start(max_attempts=10):
    """
    Builds one REACHABLE board matching a random tactical curriculum pattern
    (same "3-in-a-row + 1 gap" shapes as generate_tactical_examples), for
    self-play to occasionally start a game from instead of always from an
    empty board -- so self-play actually encounters this tactical family
    (and must resolve it with its own search) more often than it arises
    naturally from full-game rollouts. Unlike _tactical_clutter, clutter
    here is piece-count constrained: exactly 2 stones of the color NOT used
    to fill the window, so count(X) == count(O) after placement -- the
    parity a real alternating game reaching this position (with 'X' to
    move next) requires. Mirrors _get_tactical_tensors' own convention
    (board_to_input_tensor(board, 'X', ...)): 'X' is always the mover,
    whether about to block an 'O' open line or complete its own.

    This changes what positions self-play explores FROM, not what move is
    labeled correct -- the move actually played is still whatever this
    epoch's MCTS search (with its own noise/exploration) decides, same as
    any other self-play game.

    Returns (board, 'X'), or None if no valid placement is found within
    max_attempts (caller should fall back to a normal empty-board start).
    """
    windows = _tactical_windows()
    for _ in range(max_attempts):
        cells = random.choice(windows)
        gap_idx = random.randrange(WIN_LENGTH)
        filled_cells = [cells[k] for k in range(WIN_LENGTH) if k != gap_idx]
        gap_r, gap_c = cells[gap_idx]
        stone = random.choice(['O', 'X'])
        clutter_stone = 'O' if stone == 'X' else 'X'

        board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for (r, c) in filled_cells:
            board[r][c] = stone

        empties = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                   if board[r][c] == ' ' and (r, c) != (gap_r, gap_c)]
        if len(empties) < 2:
            continue
        random.shuffle(empties)
        for (r, c) in empties[:2]:
            board[r][c] = clutter_stone

        if board[gap_r][gap_c] != ' ':
            continue
        if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
            continue
        return board, 'X'
    return None


def _sample_forced_opening_start():
    """
    Places a single stone at a stratified board-region sample (reusing
    _OPENING_SQUARES -- the dihedral-symmetry-class representatives already
    used for eval/gate openings, so this picks up their spread across the
    whole board for free) as self-play's starting position, for a small
    forced fraction of games -- see generate_mcts_self_play_data's
    num_opening_games. Unlike _sample_tactical_start (a mid-game tactical
    shape), this only forces the FIRST move, guaranteeing literal
    board-region coverage independent of whatever the network's own root
    prior plus Dirichlet noise would have picked on an empty board.

    Returns (board, next_player_to_move).
    """
    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    r, c = random.choice(_OPENING_SQUARES)
    first_player = random.choice(['X', 'O'])
    board[r][c] = first_player
    next_player = 'O' if first_player == 'X' else 'X'
    return board, next_player


def generate_tactical_examples():
    """
    Every length-WIN_LENGTH window on the board, in all 4 directions, with
    exactly 3 cells filled by one player and 1 empty -- covers open-3s and
    internal gaps uniformly, for both the deterministic AI's typical
    attacking shapes ('O' filled -> X must block) and the mirror offensive
    case ('X' filled -> X should complete its own win). Each base pattern
    also gets a couple of "clutter" variants with a few extra random stones
    elsewhere, so the model sees the shape embedded in messier positions
    instead of only clean isolated ones.
    """
    examples = []
    for cells in _tactical_windows():
        for gap_idx in range(WIN_LENGTH):
            filled_cells = [cells[k] for k in range(WIN_LENGTH) if k != gap_idx]
            gap_r, gap_c = cells[gap_idx]
            # clutter_variants was 2; measured held-out generalization on the
            # exact same abstract pattern was only ~17.5% argmax-correct (avg
            # rank 2.4/9) despite ~95%+ *train* accuracy in tactical_refresh --
            # the model was memorizing the few clutter variants, not learning
            # a clutter-robust "block/complete the gap" feature. Raised to 12,
            # then to 16 for even broader clutter coverage of the same generic
            # patterns going into the next full reset.
            for stone, clutter_variants in (('O', 16), ('X', 16)):
                for variant in range(clutter_variants):
                    board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                    for (r, c) in filled_cells:
                        board[r][c] = stone
                    if variant > 0:
                        _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                    if board[gap_r][gap_c] != ' ':
                        continue  # clutter collided with the gap cell, skip
                    # The mover is always 'X' in these synthetic boards (see
                    # board_to_input_tensor(board, 'X', ...) below), so
                    # stone == 'X' is the "mover completes their own line
                    # here" case -- a provable +1-for-the-mover outcome.
                    # stone == 'O' is the "must block" case: correct play,
                    # but not itself a provable value label (blocking one
                    # threat says nothing about who ultimately wins), so it
                    # must NOT get a value target.
                    mover_completes = (stone == 'X')
                    examples.append((board, (gap_r, gap_c), mover_completes))
    return examples


# ------------------------------------------------------------
# Closed-three and double-three-fork curricula
# ------------------------------------------------------------
# Two more tactical blind spots, diagnosed separately (see
# scratch/multiproc_investigation/closed_three_diagnostic_newrun.py and
# double_three_fork_diagnostic.py) as weak even on the latest checkpoint,
# and NOT covered by the 4-of-5-gap family above (that family is always 4
# stones + 1 gap; both shapes here are one filled square short of it):
#   - closed-three: 3-in-a-row blocked on one end, open (with room to
#     extend) on the other. Offense = extend into the open end. Defense =
#     block the open end first.
#   - double-three-fork: a single point that completes an open-three in
#     two directions at once (unstoppable -- the defender can only block
#     one line). Offense = play the shared point to create the fork.
#     Defense = play it first to deny the opponent's fork.
# Unlike the 4-of-5-gap family, NEITHER pattern's correct move is ever a
# provable win on its own (extending/blocking a closed-three, or
# creating/denying a fork, doesn't complete a 5-in-a-row) -- so both
# offense and defense are policy-only here, mover_completes always False.
# This keeps the two new curricula symmetric by construction, unlike the
# existing family's offense/defense value-supervision asymmetry.
_FORK_DIRECTION_PAIRS = [(_TACTICAL_DIRECTIONS[i], _TACTICAL_DIRECTIONS[j])
                         for i in range(len(_TACTICAL_DIRECTIONS))
                         for j in range(i + 1, len(_TACTICAL_DIRECTIONS))]


def _closed_three_spans():
    """
    Every length-5 (blocker, 3-in-a-row, target) span on the board, in all 4
    directions -- same exhaustive per-direction/per-cell sweep as
    _tactical_windows(), just anchored one cell later along the line (cells
    run k=-1..3 relative to the loop cell instead of k=0..4), so index 0 is
    the blocker, indices 1-3 are the three-in-a-row, and index 4 is the
    target (the empty cell past the open end, with room to extend into).
    Translation-invariant along each line, so this covers the same total
    count of length-5 windows as _tactical_windows() despite the different
    anchor.
    """
    spans = []
    for dr, dc in _TACTICAL_DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = [(r + dr * k, c + dc * k) for k in range(-1, 4)]
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in cells):
                    spans.append(cells)
    return spans


def generate_closed_three_examples():
    """
    OFFENSE: the mover ('X', same "mover is always 'X'" convention as
    generate_tactical_examples) owns the three-stone line; 'O' holds the
    blocking stone. Correct move = the target cell -- extends the three
    into a four (still capped on one end by 'O', so NOT a 5-in-a-row).
    DEFENSE: 'O' owns the three-stone line and its blocked end; the mover
    ('X') owns the blocking stone (played earlier). Correct move = the SAME
    target cell -- denies the extension before 'O' can play it.

    Each span gets a clean variant plus a couple of clutter variants (reusing
    _tactical_clutter, same pattern as generate_tactical_examples).
    """
    examples = []
    for cells in _closed_three_spans():
        blocker_cell = cells[0]
        three_cells = cells[1:4]
        target_cell = cells[4]
        for offense in (True, False):
            three_color = 'X' if offense else 'O'
            blocker_color = 'O' if offense else 'X'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                board[blocker_cell[0]][blocker_cell[1]] = blocker_color
                for (r, c) in three_cells:
                    board[r][c] = three_color
                if variant > 0:
                    _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                tr, tc = target_cell
                if board[tr][tc] != ' ':
                    continue  # clutter collided with the target cell, skip
                examples.append((board, target_cell, False))
    return examples


def _fork_geometries():
    """
    Every (direction-pair, center) combo where a single move at the shared
    center cell would complete an open-three ("_XXX_") in BOTH directions at
    once -- the double-three fork. For each of the 2 directions, needs
    stones at +-1 step from center and empty cells at +-2 step (the open
    ends); the two directions' cells never overlap since they only meet at
    the center itself, which stays empty until the move is made. Exhaustive
    over all 6 direction pairs and every center cell whose full 8-cell
    footprint (4 per direction) stays in bounds -- same spirit as
    _tactical_windows()'s exhaustive sweep, over direction PAIRS instead of
    single directions.
    """
    combos = []
    for d1, d2 in _FORK_DIRECTION_PAIRS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone_cells, open_cells, ok = [], [], True
                for dr, dc in (d1, d2):
                    s1, s2 = (r - dr, c - dc), (r + dr, c + dc)
                    o1, o2 = (r - 2 * dr, c - 2 * dc), (r + 2 * dr, c + 2 * dc)
                    if not all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE for rr, cc in (s1, s2, o1, o2)):
                        ok = False
                        break
                    stone_cells += [s1, s2]
                    open_cells += [o1, o2]
                if ok:
                    combos.append(((r, c), stone_cells, open_cells))
    return combos


def generate_fork_examples():
    """
    OFFENSE: the mover ('X') owns all 4 stones (2 per direction); playing
    the shared center cell creates the fork. DEFENSE: 'O' owns all 4 stones
    (would create the fork if 'O' got to move next); the mover ('X') must
    occupy the shared center FIRST to deny it -- same target cell either
    way.

    Each geometry gets a clean variant plus a couple of clutter variants
    (reusing _tactical_clutter, same pattern as generate_tactical_examples),
    excluding both the stone cells and the required-open end cells so
    clutter can't collide with either.
    """
    examples = []
    for center, stone_cells, open_cells in _fork_geometries():
        for offense in (True, False):
            stone_color = 'X' if offense else 'O'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in stone_cells:
                    board[r][c] = stone_color
                if variant > 0:
                    exclude = set(stone_cells) | set(open_cells) | {center}
                    _tactical_clutter(board, exclude, n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                cr, cc = center
                if board[cr][cc] != ' ' or any(board[r][c] != ' ' for (r, c) in open_cells):
                    continue  # clutter collided with the target or a required-open end, skip
                examples.append((board, center, False))
    return examples


def _double_four_geometries():
    """
    Every (direction-pair, center) combo where a single move at the shared
    center cell would complete a FOUR-in-a-row (one cell short of the
    winning 5) in BOTH directions at once -- a double-four. For each of the
    2 directions, needs 3 consecutive stones immediately adjacent to center
    (center+d, center+2d, center+3d) and an empty cell one step past them
    (center+4d) for the resulting four to still be completable next move.
    The two directions' cells never overlap since they only meet at the
    center itself, which stays empty until the move is made. Exhaustive
    over all 6 direction pairs and every center cell whose full footprint
    (4 per direction) stays in bounds -- same spirit as _fork_geometries().
    """
    combos = []
    for d1, d2 in _FORK_DIRECTION_PAIRS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone_cells, open_cells, ok = [], [], True
                for dr, dc in (d1, d2):
                    line_stones = [(r + dr * k, c + dc * k) for k in (1, 2, 3)]
                    line_open = (r + dr * 4, c + dc * 4)
                    if not all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                               for rr, cc in line_stones + [line_open]):
                        ok = False
                        break
                    stone_cells += line_stones
                    open_cells.append(line_open)
                if ok:
                    combos.append(((r, c), stone_cells, open_cells))
    return combos


def generate_double_four_examples():
    """
    OFFENSE: the mover ('X') owns all 6 stones (3 per direction); playing
    the shared center cell completes a four in both directions at once --
    each threatens a win next move via its own open end, and the defender
    can only block one, a strictly more decisive threat than the open-three
    fork (generate_fork_examples) even though, like that fork, it is one
    move short of an outright win itself. DEFENSE: 'O' owns all 6 stones
    (would create the double-four if 'O' got to move next); the mover ('X')
    must occupy the shared center FIRST to deny it -- same target cell
    either way. mover_completes is always False for the same reason
    generate_fork_examples' is: completing/denying this shape isn't itself
    a provable 5-in-a-row.

    Each geometry gets a clean variant plus a couple of clutter variants
    (reusing _tactical_clutter, same pattern as generate_fork_examples),
    excluding both the stone cells and the required-open end cells so
    clutter can't collide with either.
    """
    examples = []
    for center, stone_cells, open_cells in _double_four_geometries():
        for offense in (True, False):
            stone_color = 'X' if offense else 'O'
            for variant in range(3):  # 1 clean + a couple of clutter variants
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in stone_cells:
                    board[r][c] = stone_color
                if variant > 0:
                    exclude = set(stone_cells) | set(open_cells) | {center}
                    _tactical_clutter(board, exclude, n_stones=random.randint(3, 36))  # 2026-09-14: was (2,4) -- clustered-near-window clutter can afford more stones without spreading unrealistically thin
                cr, cc = center
                if board[cr][cc] != ' ' or any(board[r][c] != ' ' for (r, c) in open_cells):
                    continue  # clutter collided with the target or a required-open end, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # geometry/clutter combo already completes a win elsewhere, skip
                examples.append((board, center, False))
    return examples


# ------------------------------------------------------------
# Open-three curriculum
# ------------------------------------------------------------
# 2026-09-15 (added after re-diagnosing the HLT-loss value/MCTS blind spot --
# see scratch/multiproc_investigation/rediag_value_blindspot.py, run against
# the live epoch443 checkpoint): a genuine OPEN three (3-in-a-row with BOTH
# immediately-flanking cells empty) is a shape this curriculum never actually
# covered. generate_tactical_examples' "4-of-5-gap" family is always one move
# from a proven outcome (mover_completes or a single forced block);
# generate_closed_three_examples' shape has only ONE open end. An open three
# has TWO equally-correct response cells (either flank creates/denies an open
# four), and re-testing 6 real/synthetic open-three positions against the
# live checkpoint found the model still gets roughly half of them wrong --
# MCTS converging on an unrelated move despite the raw policy prior usually
# still ranking at least one true response cell near the top -- the same
# "good-ish prior, MCTS/value diverges" pattern diagnosed earlier, just never
# targeted by a dedicated curriculum shape until now.
def generate_open_three_examples():
    """
    Reuses _closed_three_spans()'s exact 5-cell (k=-1..3) windows, but
    treats BOTH end cells (index 0 and index 4) as empty and equally
    correct, instead of one being an occupied blocker and the other the
    sole target -- so each variant yields TWO rows for the SAME board (one
    per valid end), which _multi_label_targets_from_rows groups back into a
    single 50/50 target over both cells, exactly like the two symmetric
    block cells in generate_tactical_examples' gap family.

    OFFENSE: mover ('X') owns the three-stone line; either end extends it
    into an open four. DEFENSE: 'O' owns the three-stone line; the mover
    ('X') must occupy either end to prevent the open four. Neither is a
    provable 5-in-a-row on its own, so mover_completes is always False --
    same policy-only convention as generate_closed_three_examples/
    generate_fork_examples/generate_double_four_examples.
    """
    examples = []
    for cells in _closed_three_spans():
        end1, three_cells, end2 = cells[0], cells[1:4], cells[4]
        # 2026-09-23: edge/corner oversampling (added earlier today, REMOVED
        # here) -- the organic self-play open-three test
        # (open3_organic_test.py) found the model missing threats located
        # centrally while repeatedly playing near row 0 instead, matching
        # exactly the kind of "edge habit" this oversampling could produce
        # if it overshot: skewing the network toward edge/corner moves as a
        # general prior rather than genuinely improving edge-tactic
        # recognition. Reverting to a flat, position-independent variant
        # count (matches the window-count distribution the board's own
        # geometry produces, not artificially reweighted) to isolate
        # whether the organic-test miss rate improves without it.
        n_variants = 14
        for offense in (True, False):
            three_color = 'X' if offense else 'O'
            # 2026-09-21: 3->8 variants (1 clean + more clutter diversity) --
            # open-three defense is the pattern the passive open3 tracker
            # has shown persistently flat/noisy (0.2-0.7 miss rate) despite
            # training, so this deliberately over-represents it relative to
            # the other four tactical families in the combined curriculum.
            # 2026-09-22: 8->14 -- tactical_refresh() (the periodic burst
            # supervised re-injection this curriculum used to feed) was
            # removed the same day: isolated testing
            # (scratch/.../tactical_refresh_destroy_test.py) measured ONE
            # refresh call immediately dropping vs-Det win rate 65%->25%,
            # and a separate retention check
            # (scratch/.../tactical_retention_check.py) found that gain
            # decayed from ~100% back to ~70% tactical accuracy within just
            # 2 epochs anyway -- net negative. Replaced by continuous
            # per-epoch injection into the ordinary replay buffer (see
            # _get_open_three_injection_samples), so this pool now needs
            # more inherent variant diversity since it's no longer refreshed
            # by a dedicated fine-tuning pass, just sampled from directly.
            for variant in range(n_variants):  # was: flat range(14); now edge/corner-weighted
                board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
                for (r, c) in three_cells:
                    board[r][c] = three_color
                if variant > 0:
                    _tactical_clutter(board, set(cells), n_stones=random.randint(3, 36))
                if board[end1[0]][end1[1]] != ' ' or board[end2[0]][end2[1]] != ' ':
                    continue  # clutter collided with one of the two open ends, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # clutter combo already completes a win elsewhere, skip
                examples.append((board, end1, False))
                examples.append((board, end2, False))
    return examples


# ------------------------------------------------------------
# Diagnosed-position hard-mining
# ------------------------------------------------------------
# 2026-09-15: Position A (epoch291 HLT loss) is a real open-three-defense
# instance whose raw POLICY prior (not just the value head/MCTS search) has
# drifted onto the historically-wrong move after repeated self-play/HLT
# reinforcement -- confirmed by direct re-check against the live checkpoint
# (rank 1/121, 12.6% prior on the wrong cell, vs the two correct blocks at
# 3-5% each). A diffuse generic curriculum (generate_open_three_examples
# above) teaches the general shape but has no special weight on this EXACT,
# already-entrenched association -- unlearning it needs direct, repeated
# counter-supervision on the specific pattern, not just broader coverage.
# 2026-09-16: Position B (live test game, Claude vs the epoch462-promoted
# checkpoint) -- a DIAGONAL open-three missed in a denser, real-game-like
# position (5 X + 4 O stones already on the board, unlike Position A's
# emptier board) despite the model correctly blocking a near-identical
# horizontal open-three earlier in the SAME game. Confirms the open-three
# fix from generate_open_three_examples works on the shape in isolation but
# does not yet fully generalize to busier boards -- clutter density was
# widened 3-36 (was 3-6, see _tactical_clutter calls throughout this file)
# specifically to close this gap generically, and this exact position is
# also hard-mined directly, same rationale as Position A.
_DIAGNOSED_POSITIONS = [
    # (moves, current_player, correct_response_cells)
    (
        [('X', 5, 5), ('O', 5, 4), ('X', 4, 5), ('O', 6, 5), ('X', 4, 3), ('O', 7, 6), ('X', 4, 4)],
        'O', [(4, 6), (4, 2)],
    ),
    (
        [('X', 5, 5), ('O', 4, 6), ('X', 5, 6), ('O', 2, 4), ('X', 5, 7), ('O', 5, 4),
         ('X', 7, 5), ('O', 6, 5), ('X', 6, 6)],
        'O', [(8, 4), (4, 8)],
    ),
    # 2026-09-16: Positions C/D/E found by the autonomous tactical_watch.py
    # loop's very first run -- randomly-generated open-three test boards
    # with dense (heavy-clutter) surrounding stones, all missed by the
    # live checkpoint (block_visit_share 2.0-6.5%) despite the widened
    # 3-36 clutter curriculum deployed the SAME epoch (expected: that
    # curriculum hasn't been trained on yet, only code-deployed -- these
    # are exactly the kind of busy-board misses it should start fixing
    # once the next tactical_refresh cycle actually trains on it).
    (
        [('X', 1, 0), ('X', 1, 2), ('X', 1, 3), ('O', 2, 4), ('X', 3, 6), ('X', 4, 0), ('X', 4, 5), ('O', 4, 6), ('O', 5, 2), ('O', 5, 4), ('X', 5, 5), ('X', 5, 6), ('O', 5, 8), ('O', 6, 3), ('X', 7, 0), ('X', 7, 1), ('X', 7, 2), ('O', 7, 4), ('X', 7, 6), ('X', 7, 8), ('O', 8, 3), ('X', 8, 7), ('X', 8, 8), ('X', 9, 0), ('X', 9, 3), ('X', 9, 6), ('X', 10, 2), ('O', 10, 3), ('O', 10, 4), ('X', 10, 8)],
        'X', [(4, 1), (8, 5)],
    ),
    (
        [('X', 0, 2), ('O', 0, 5), ('O', 1, 2), ('O', 1, 4), ('X', 1, 5), ('O', 1, 6), ('X', 1, 7), ('X', 1, 9), ('O', 2, 3), ('O', 2, 6), ('O', 2, 7), ('O', 2, 8), ('O', 3, 2), ('O', 3, 4), ('X', 3, 6), ('O', 3, 7), ('O', 3, 9), ('O', 4, 2), ('O', 4, 3), ('O', 4, 5), ('O', 4, 9), ('X', 5, 4), ('O', 5, 8), ('O', 5, 9)],
        'X', [(2, 5), (2, 9)],
    ),
    (
        [('O', 2, 2), ('X', 2, 3), ('X', 3, 1), ('X', 3, 3), ('X', 4, 4), ('O', 4, 6), ('X', 5, 0), ('X', 5, 1), ('O', 5, 5), ('O', 5, 6), ('O', 5, 7), ('X', 6, 0), ('O', 6, 3), ('O', 6, 4), ('X', 6, 5), ('X', 7, 3), ('O', 7, 4), ('X', 7, 9), ('O', 8, 2), ('O', 8, 5), ('O', 8, 9), ('O', 9, 0), ('O', 9, 1), ('O', 9, 4), ('O', 10, 1), ('O', 10, 4), ('X', 10, 5)],
        'X', [(5, 2), (9, 6)],
    ),
    # 2026-09-16: 2nd tactical_watch.py manual run (~7h after the 1st,
    # loop wakeup chain appears to have stopped firing autonomously --
    # see chat) -- 3 more dense-board misses, block_visit_share
    # 0.0-1.2%. 2/5 tested passed this time (vs 0/3 the first run),
    # consistent with gradual, partial improvement rather than solved.
    (
        [('X', 0, 0), ('X', 0, 1), ('O', 0, 2), ('X', 0, 3), ('X', 1, 4), ('O', 2, 2), ('X', 2, 3), ('X', 2, 5), ('O', 3, 3), ('O', 4, 3), ('O', 4, 4), ('O', 4, 5), ('X', 4, 7), ('O', 5, 1), ('O', 5, 2), ('O', 5, 4), ('O', 5, 6), ('X', 5, 7), ('O', 6, 5), ('O', 6, 7), ('X', 6, 9), ('O', 7, 0), ('X', 7, 3), ('X', 7, 4), ('O', 8, 1), ('O', 8, 3), ('X', 8, 6), ('X', 8, 7), ('X', 8, 9), ('X', 9, 5), ('X', 10, 6), ('X', 10, 7), ('O', 10, 9)],
        'X', [(3, 2), (7, 6)],
    ),
    (
        [('O', 0, 1), ('X', 0, 5), ('X', 0, 6), ('X', 0, 7), ('O', 1, 2), ('O', 1, 4), ('O', 1, 7), ('X', 2, 1), ('X', 2, 2), ('O', 2, 3), ('O', 2, 4), ('O', 2, 6), ('O', 3, 1), ('X', 3, 3), ('O', 3, 4), ('O', 3, 5), ('X', 4, 2), ('O', 4, 3), ('X', 4, 5), ('X', 4, 6), ('O', 5, 1), ('X', 5, 2), ('O', 5, 3), ('X', 5, 4), ('O', 5, 5), ('O', 5, 7), ('O', 6, 1), ('X', 6, 2), ('O', 6, 3), ('X', 6, 6), ('O', 6, 7), ('O', 7, 1), ('O', 7, 2), ('X', 7, 3), ('O', 7, 6)],
        'X', [(0, 4), (4, 4)],
    ),
    (
        [('X', 0, 6), ('X', 0, 10), ('X', 1, 8), ('O', 1, 10), ('X', 3, 4), ('O', 3, 6), ('O', 3, 7), ('X', 3, 8), ('X', 4, 3), ('X', 4, 4), ('O', 4, 6), ('O', 4, 9), ('X', 5, 2), ('O', 5, 5), ('O', 6, 1), ('X', 6, 3), ('X', 6, 6), ('X', 7, 2), ('X', 7, 4), ('O', 7, 5), ('O', 7, 6), ('O', 7, 8), ('O', 8, 2), ('O', 8, 3), ('X', 8, 5), ('O', 8, 6), ('O', 8, 8), ('X', 9, 1), ('O', 9, 3), ('O', 9, 6)],
        'X', [(2, 8), (6, 4)],
    ),
    # 2026-09-23 Position F: HLT (human) loss for Gen13/NAPS, epoch129,
    # cumulative_epoch 129 checkpoint. Root value swung to +0.977 (AI very
    # confident of winning) at move 24, then collapsed to -0.921 by move 36
    # as the human found the refutation -- see the session's HLT-analysis
    # discussion. Root cause traced by replaying row 8's stone history:
    # after X's move at (8,3) (ply 35), row 8 reads "...X.XXXOX." -- X at
    # cols 3,5,6,7 with a SINGLE empty gap at col 4 and O blocking col 8.
    # This is a plain 4-of-5-gap shape (the OLDEST tactical curriculum
    # family in this file, generate_tactical_examples) with exactly one
    # winning cell (8,4) -- yet the live model's actual move-36 search
    # never surfaced (8,4) in its top-5 candidates at any simulation
    # checkpoint (only (3,7)/(8,4-adjacent) cells the network's prior
    # never weighted), meaning this wasn't a search-depth miss but a
    # policy-prior blind spot on a pattern the model has trained on
    # extensively. Distinct from Positions A-E (all open-three shapes) --
    # this is a gap-four shape missed on a busy, asymmetric real board.
    (
        [('X', 5, 5), ('O', 4, 6), ('X', 5, 4), ('O', 5, 6), ('X', 4, 4), ('O', 6, 6), ('X', 3, 6), ('O', 6, 5),
         ('X', 6, 3), ('O', 4, 5), ('X', 3, 4), ('O', 6, 4), ('X', 3, 5), ('O', 6, 7), ('X', 6, 8), ('O', 7, 8),
         ('X', 8, 9), ('O', 7, 6), ('X', 8, 6), ('O', 3, 3), ('X', 7, 7), ('O', 5, 9), ('X', 8, 5), ('O', 4, 7),
         ('X', 7, 4), ('O', 3, 8), ('X', 2, 9), ('O', 4, 8), ('X', 4, 9), ('O', 5, 2), ('X', 5, 3), ('O', 2, 6),
         ('X', 8, 7), ('O', 8, 8), ('X', 8, 3)],
        'O', [(8, 4)],
    ),
]


def _dihedral_transform(r, c, k, size):
    """Applies the k-th (0-7) of the 8 square-board symmetries (4 rotations,
    optionally preceded by a horizontal reflection) to cell (r, c). k=0 is
    the identity. Verified self-consistent: applying k=1..3 four times in a
    row returns the original cell, and k=4..7 apply the same rotations after
    an initial column-flip -- together the standard dihedral group of order 8."""
    if k >= 4:
        c = size - 1 - c
        k -= 4
    for _ in range(k):
        r, c = c, size - 1 - r
    return r, c


def generate_diagnosed_positions_examples():
    """
    Hard-mines the exact real diagnosed position(s) in _DIAGNOSED_POSITIONS,
    each replicated across all 8 dihedral symmetries and a batch of clutter
    variants (reusing _tactical_clutter), so this specific pattern gets
    meaningfully more representation than a single clean board would --
    comparable in order of magnitude to how other targeted families in this
    file are weighted, while staying a small fraction of the full curriculum.
    Policy-only (mover_completes always False), same convention as
    generate_open_three_examples, whose shape this position is an instance
    of.

    Returns (board, current_player, (r, c), mover_completes) 4-tuples --
    current_player varies per position, not always 'X' (same shape
    generate_ignored_threat_examples used to return).
    """
    DIAGNOSED_CLUTTER_VARIANTS = 20
    examples = []
    for moves, current_player, correct_cells in _DIAGNOSED_POSITIONS:
        for k in range(8):
            base_board = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            placed_cells = set()
            for p, r, c in moves:
                tr, tc = _dihedral_transform(r, c, k, BOARD_SIZE)
                base_board[tr][tc] = p
                placed_cells.add((tr, tc))
            correct_transformed = [_dihedral_transform(r, c, k, BOARD_SIZE) for (r, c) in correct_cells]
            for variant in range(DIAGNOSED_CLUTTER_VARIANTS):
                board = [row[:] for row in base_board]
                if variant > 0:
                    _tactical_clutter(board, placed_cells | set(correct_transformed), n_stones=random.randint(3, 36))  # 2026-09-16: widened 2-5 -> 3-36 for consistency with the other families (see generate_open_three_examples' header comment)
                if any(board[r][c] != ' ' for (r, c) in correct_transformed):
                    continue  # clutter collided with a correct response cell, skip
                if fast_check_win(board, 'X') or fast_check_win(board, 'O'):
                    continue  # clutter combo already completes a win elsewhere, skip
                for (r, c) in correct_transformed:
                    examples.append((board, current_player, (r, c), False))
    return examples


_synthetic_tactical_tensors_cache = None
_mined_tactical_tensors_cache = None
_open_three_injection_cache = None

# Real positions mined from actual self-play/vs-Deterministic games (150
# sims/move, live checkpoint), scanned for the same "4-of-5+gap" pattern
# family as generate_tactical_examples() and labeled with the identical
# mover_completes semantics -- see scratch/multiproc_investigation/
# mine_tactical_examples.py, mine_tactical_parallel.py, and
# verify_mined_examples.py for provenance and independent label
# verification. Copied here (rather than read from scratch/) so the live
# training code doesn't depend on a scratch directory.
MINED_TACTICAL_EXAMPLES_PATH = f"model/mined_tactical_examples{_SUFFIX}.pkl"


def _multi_label_targets_from_rows(rows_by_key):
    """2026-09-14 multi-label fix. rows_by_key: dict mapping a board(+player)
    key -> list of ((gap_r, gap_c), mover_completes) tuples all scanned from
    that EXACT SAME input position (it commonly matches more than one
    tactical window at once, e.g. two simultaneous open threats). The old
    scheme emitted one single-hot row per matching window, so the identical
    network input got DIFFERENT "correct" one-hot targets across rows --
    an argmax target that no single-output model can ever satisfy for all
    of them. Measured on the live mined pool (2026-09-14,
    scratch/multiproc_investigation/mine_tactical_conflict_check.py): 27.1%
    of unique boards carried >=2 conflicting single-cell labels, capping the
    theoretical MAX achievable argmax accuracy on the raw pool at 64.6% --
    this is very likely the dominant reason tactical_refresh's own reported
    accuracy has been flat at ~80% for weeks (98% synthetic + ~65% real,
    averaged in the deployed 50/50 mix, ~= 80%).

    Fix: one row per unique board(+player). If ANY matching window for that
    board is a provable mover_completes=True win, the target concentrates
    ONLY on those winning cells (playing a non-winning move when a win is
    available is never correct, so diluting the target with defensive
    cells would be actively wrong). Otherwise the target spreads uniformly
    over every distinct labeled (defensive) gap cell -- all of them are
    legitimately urgent, and this removes the artificial ceiling without
    claiming false certainty about which single one is "the" answer.
    Yields (key, target_vec[NUM_CELLS] summing to 1, has_winning_cell)."""
    for key, entries in rows_by_key.items():
        winning = {gap for gap, mc in entries if mc}
        chosen = winning if winning else {gap for gap, mc in entries}
        target = np.zeros(NUM_CELLS, dtype=np.float32)
        w = 1.0 / len(chosen)
        for (r, c) in chosen:
            target[r * BOARD_SIZE + c] = w
        yield key, target, bool(winning)


def _get_synthetic_tactical_tensors(device):
    global _synthetic_tactical_tensors_cache
    if _synthetic_tactical_tensors_cache is None:
        # Five curricula combined here: the original 4-of-5-gap family (only
        # one with a provable value label, see mover_completes below), the
        # closed-three/double-three-fork/double-four/open-three families
        # (policy-only by construction, see their own docstrings). All five
        # use 'X' as mover -- normalized to (board, player, (r,c),
        # mover_completes) 4-tuples here so this flows through the same
        # grouping/tensorize path as the mined examples below. The hard-mined
        # diagnosed positions (generate_diagnosed_positions_examples) carry
        # their own current_player per position, so they're added separately
        # after this normalization instead of through fixed_x_examples.
        fixed_x_examples = (generate_tactical_examples()
                             + generate_closed_three_examples()
                             + generate_fork_examples()
                             + generate_double_four_examples()
                             + generate_open_three_examples())
        examples = [(board, 'X', gap, is_mc) for (board, gap, is_mc) in fixed_x_examples]
        examples += generate_diagnosed_positions_examples()

        # Group by (board, player) first (see _multi_label_targets_from_rows)
        # -- a board can be produced by more than one generator above, or
        # matched by more than one window/gap_idx within a single generator
        # once clutter is involved.
        board_by_key, rows_by_key = {}, {}
        for board, player, (gr, gc), is_mc in examples:
            key = (tuple(tuple(row) for row in board), player)
            board_by_key.setdefault(key, (board, player))
            rows_by_key.setdefault(key, []).append(((gr, gc), is_mc))

        inputs, targets, value_targets = [], [], []
        for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
            board, player = board_by_key[key]
            t = board_to_input_tensor(board, player, device).squeeze(0).cpu().numpy()
            inputs.append(t)
            targets.append(target)
            value_targets.append(0.97 if has_win else float('nan'))

        _synthetic_tactical_tensors_cache = (
            torch.tensor(np.array(inputs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
            torch.tensor(np.array(value_targets), dtype=torch.float32),
        )
        print(f"Tactical curriculum: {len(examples)} synthetic rows -> {len(inputs)} rows "
              f"(4-of-5-gap + closed-three + fork + double-four + open-three + diagnosed-positions, "
              f"multi-label, cached for reuse)")
    return _synthetic_tactical_tensors_cache


def _get_mined_tactical_tensors(device):
    """Loads and tensorizes the real mined examples (board, current_player,
    (gap_r, gap_c), mover_completes) tuples once and caches. Unlike the
    synthetic boards (always encoded from 'X' mover's perspective), each
    mined example carries its own real current_player -- passed through
    as-is to board_to_input_tensor, whose perspective-relative encoding
    (curr_plane = current_player's stones) needs no color canonicalization.

    2026-09-09: an 8-way dihedral-augmented version of this (same
    augment_sample every other data source gets) was tried and reverted --
    it reproducibly stalled the live process for 2+ hours at the third
    tactical_refresh cycle (ep63; the first two, ep21/ep42, ran fine), most
    likely GPU-memory fragmentation from repeatedly building/discarding an
    8x-larger (150K+) tensor cache every refresh compounding across cycles.
    Not re-attempted here; a safer version would need to build the
    per-example encoding via board_to_input_array (pure CPU numpy, no
    per-item .to(device) round-trip) instead of board_to_input_tensor, and
    ideally batch rather than rebuild-from-scratch every refresh.

    Returns empty tensors (falls back to synthetic-only) if the pickle is
    missing, so a fresh checkout without the mined-examples file still runs."""
    global _mined_tactical_tensors_cache
    if _mined_tactical_tensors_cache is None:
        if os.path.exists(MINED_TACTICAL_EXAMPLES_PATH):
            with open(MINED_TACTICAL_EXAMPLES_PATH, "rb") as f:
                data = pickle.load(f)

            # Group by (board, player) first -- see _multi_label_targets_from_rows.
            # Real positions match multiple tactical windows at once far more
            # often than synthetic ones (measured: 27.1% of unique mined
            # boards, 65.2% of all rows), so this matters even more here.
            board_by_key, rows_by_key = {}, {}
            for board, current_player, (gr, gc), is_mover_completion in data:
                key = (tuple(tuple(row) for row in board), current_player)
                board_by_key.setdefault(key, (board, current_player))
                rows_by_key.setdefault(key, []).append(((gr, gc), is_mover_completion))

            inputs, targets, value_targets = [], [], []
            for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
                board, current_player = board_by_key[key]
                t = board_to_input_tensor(board, current_player, device).squeeze(0).cpu().numpy()
                inputs.append(t)
                targets.append(target)
                value_targets.append(0.97 if has_win else float('nan'))

            _mined_tactical_tensors_cache = (
                torch.tensor(np.array(inputs), dtype=torch.float32),
                torch.tensor(np.array(targets), dtype=torch.float32),
                torch.tensor(np.array(value_targets), dtype=torch.float32),
            )
            print(f"Tactical curriculum: {len(data)} real mined rows -> {len(inputs)} unique (board,player) "
                  f"loaded from {MINED_TACTICAL_EXAMPLES_PATH} (multi-label, cached for reuse)")
        else:
            print(f"Tactical curriculum: no mined examples found at {MINED_TACTICAL_EXAMPLES_PATH}, "
                  f"falling back to synthetic-only")
            _mined_tactical_tensors_cache = (
                torch.empty((0, 5, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32),
                torch.empty((0, NUM_CELLS), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
    return _mined_tactical_tensors_cache


def _root_entropy_info(root):
    """(entropy, n_candidates) of a search root's visit distribution, for the
    hard-example flat-distribution detector (Hard-6, see
    scratch/multiproc_investigation/entropy_complexity_probe.py -- 2026-09-11
    finding: ~19% of live buffer entries are visit distributions EXACTLY
    uniform over their candidate set, a PUCT round-robin degeneracy when
    prior+value are both flat, most common at 17-31 stones)."""
    n = len(root.children)
    if n == 0:
        return 0.0, 0
    tot = sum(ch.n_visits for ch in root.children.values())
    if tot == 0:
        return float(np.log(n)), n
    probs = np.array([ch.n_visits / tot for ch in root.children.values()])
    probs = np.clip(probs, 1e-12, None)
    return float(-(probs * np.log(probs)).sum()), n


def _mine_one_selfplay_game(model, device, sims, pooled_evaluator=None, eval_cache=None):
    """Plays one self-play game (noise on, no training-data recording) to
    collect (board, current_player) snapshots for refresh_mined_tactical_pool
    to scan -- mirrors mine_tactical_parallel.py's _play_one_selfplay.

    2026-09-11: also returns per-ply (act, entropy_info) and the game outcome
    so refresh_mined_tactical_pool can additionally run outcome-based hard-
    example detection (Hard-1/2/3/5/6) on top of the original pattern scan --
    every ply here is a model ply (self-play is the model against itself).

    pooled_evaluator/eval_cache (2026-09-27, unified MP pool): optional
    routing for the leaf evaluations; None keeps the original behavior."""
    mcts = MCTS(c_puct=3.2, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = random.choice(['X', 'O'])
    plies = []
    moves = 0
    outcome = 0
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
        board_snapshot = [row[:] for row in board]
        root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                            dirichlet_alpha=1.0, noise_weight=0.85)
        if not root.children:
            break
        act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        ent_info = _root_entropy_info(root)
        r, c = divmod(act, BOARD_SIZE)
        plies.append((board_snapshot, current_player, (r, c), True, ent_info))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


def _mine_one_vsdet_game(model, device, sims, mcts_plays_x, det_max_depth=determinisic_play.MAX_DEPTH,
                         pooled_evaluator=None, eval_cache=None):
    """vs-Deterministic counterpart to _mine_one_selfplay_game -- mirrors
    mine_tactical_parallel.py's _play_one_vsdet.

    2026-09-11 bugfix: this used to call determinisic_play.play(board,
    current_player) with no depth argument, silently mining exclusively
    against the fixed depth-2 opponent -- the same fixed opponent
    evaluate_vs_deterministic/run_gating_check hold as an untouched
    promotion metric. Callers now pass _sample_vs_det_max_depth(epoch) so
    mining draws from the same 0-3 depth ladder training data generation
    already uses, keeping the fixed-depth-2 metric uncontaminated.

    pooled_evaluator/eval_cache: see _mine_one_selfplay_game."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    plies = []
    moves = 0
    outcome = 0
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
        board_snapshot = [row[:] for row in board]
        if current_player == mcts_player:
            root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                                dirichlet_alpha=0.3, noise_weight=0.25)
            if not root.children:
                break
            act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
            ent_info = _root_entropy_info(root)
            r, c = divmod(act, BOARD_SIZE)
            plies.append((board_snapshot, current_player, (r, c), True, ent_info))
        else:
            r, c = determinisic_play.play(board, current_player, max_depth=det_max_depth)
            if r == -1:
                break
            plies.append((board_snapshot, current_player, (r, c), False, None))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


def _mine_one_vsleague_game(model, league_model, device, sims, mcts_plays_x,
                            pooled_evaluator=None, eval_cache=None,
                            league_pooled_evaluator=None, league_eval_cache=None):
    """vs-League counterpart to _mine_one_vsdet_game -- the opponent is a
    frozen past checkpoint's own greedy (no-noise) search instead of
    determinisic_play, mirroring _play_one_mcts_vs_league_game's
    opponent-side logic. Mining from vs-League games closes a gap the
    self-play/vs-Det mining above never reaches: whatever tactical patterns
    specifically arise when the model plays against its own past selves.

    pooled_evaluator/eval_cache route the model side, league_* the league
    side (see _mine_one_selfplay_game); all None = original behavior."""
    mcts_player = 'X' if mcts_plays_x else 'O'
    mcts = MCTS(c_puct=3.2, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    league_mcts = MCTS(c_puct=1.6, eval_batch_size=16, shared_cache=league_eval_cache,
                       pooled_evaluator=league_pooled_evaluator)
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'
    plies = []
    moves = 0
    outcome = 0
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
        board_snapshot = [row[:] for row in board]
        if current_player == mcts_player:
            root = mcts.search(board, current_player, model, device, num_simulations=sims, add_noise=True,
                                dirichlet_alpha=0.3, noise_weight=0.25)
            is_model_ply = True
        else:
            root = league_mcts.search(board, current_player, league_model, device, num_simulations=sims,
                                       add_noise=False)
            is_model_ply = False
        if not root.children:
            break
        act = max(root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        ent_info = _root_entropy_info(root) if is_model_ply else None
        r, c = divmod(act, BOARD_SIZE)
        plies.append((board_snapshot, current_player, (r, c), is_model_ply, ent_info))
        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'
        moves += 1
        if moves > 130:
            break
    return plies, outcome


# ============================================================
# Outcome-based hard-example mining (2026-09-11)
# ============================================================
# Adds a second, outcome/disagreement-based mining criterion on top of the
# pattern-only scan below (_threat_gaps mirrors that scan's own "4-of-5+gap"
# test exactly, factored out so both share one implementation). Approved
# design (see this session's diagnosis workflow, agent "diag:hard-mining-spec"):
#   Hard-1: a winning move was available (own threat_gaps) and not played.
#   Hard-2: opponent had an immediate winning threat and it was not blocked.
#   Hard-3: a fork (_fork_geometries) was available/threatened and not taken/denied.
#   Hard-5: position within the last HARD5_LOOKBACK of the mover's own plies
#           in a game that mover actually lost.
#   Hard-6 (2026-09-11 addition, not in the original spec): the search's own
#           visit distribution came out suspicion-of-degenerate flat for its
#           candidate count -- see entropy_complexity_probe.py's finding that
#           ~19% of the live buffer is EXACTLY uniform over its candidate set,
#           a PUCT round-robin degeneracy when prior+value are both flat.
# Relabeling deliberately does NOT imitate the opponent's move (that review
# rejected minimax-as-teacher: a shallow, fixed depth-2 opponent is most
# likely to be WRONG exactly where the model's own search disagrees with it
# most). Instead every flagged position is re-searched with the model's OWN
# MCTS at a boosted sim count and the resulting visit distribution becomes
# the policy target -- purely a bigger dose of the exact fix already applied
# this morning (more search), targeted at the positions where today's smaller
# production budget still isn't enough.
HARD_EXAMPLE_POOL_PATH = f"model/hard_mined_examples{_SUFFIX}.pkl"
RELABEL_CAP = 256
# 800 -> 1800 (2026-09-12): production's own sim budget auto-ramped past this
# (currently 1100 base) since RELABEL_SIMS was first set, so the "bigger dose
# of search" this relabeling is supposed to provide had quietly fallen BELOW
# production depth. Direct measurement today (live_model_tactical_check.py)
# found BLOCK-class positions (must defend a threat) still regress relative
# to the raw prior at production's own 1100 sims (85%->80%), while H2 (missed
# block) is the single largest hard-mining category by volume (~20-35/cycle)
# -- meaning the relabeling pass meant to fix exactly this was itself running
# at a shallower depth (800) than the level already shown to be insufficient
# (1100). Raised to 1800 (matching the current boosted_sims ceiling) so
# relabeled targets are reliably deeper than whatever production is doing.
RELABEL_SIMS = 1800
HARD_EXAMPLE_CAP = 8000
HARD5_LOOKBACK = 6
FLAT_ENTROPY_MARGIN = 0.05  # within this of ln(n_candidates) counts as degenerate-flat


def _threat_gaps(board, color, windows):
    """Every (r, c) where `color` already has 4-of-5 filled and exactly one
    empty cell in some tactical window -- the same test
    refresh_mined_tactical_pool's own pattern scan uses, factored out so the
    hard-example detectors below (Hard-1/Hard-2) reuse it exactly rather than
    re-deriving it."""
    gaps = []
    for cells in windows:
        vals = [board[r][c] for (r, c) in cells]
        if sum(v == color for v in vals) == 4 and vals.count(' ') == 1:
            gaps.append(cells[vals.index(' ')])
    return gaps


def _fork_hits(board, color, forks):
    """Every fork center (see _fork_geometries) where `color` already holds
    all 4 stone cells with both open ends still empty and the center itself
    still empty -- i.e. a live, not-yet-taken/denied fork."""
    hits = []
    for center, stone_cells, open_cells in forks:
        cr, cc = center
        if board[cr][cc] != ' ':
            continue
        if (all(board[r][c] == color for (r, c) in stone_cells)
                and all(board[r][c] == ' ' for (r, c) in open_cells)):
            hits.append(center)
    return hits


def _opponent_can_fork_next(board, opponent):
    """True if `opponent` has SOME reply (placed right now) that either wins
    outright or creates 2+ simultaneous immediate-winning cells for
    `opponent` -- i.e. the position already contains a next-move fork
    `opponent` can walk into unopposed. This is the check that actually
    matters for Hard-7 (see its call site): a fork isn't visible by
    inspecting the board as-is (that only shows CURRENTLY-live winning
    cells, which is what _immediate_win_cells checks) -- it's visible one
    ply further, in whether the opponent HAS a move that manufactures a
    fork out of stones already on the board. O(BOARD_SIZE^2) candidate
    replies x O(BOARD_SIZE^2) fast_check_win each -- only run on already-
    mined game plies at refresh_mined_tactical_pool's cadence, not in the
    self-play hot loop."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = opponent
            if fast_check_win(board, opponent):
                board[r][c] = ' '
                return True
            if len(_immediate_win_cells(board, opponent)) >= 2:
                board[r][c] = ' '
                return True
            board[r][c] = ' '
    return False


def _immediate_win_cells(board, color):
    """Every empty cell where placing `color` right now completes 5-in-a-row.
    Template-free (unlike _fork_hits/_fork_geometries, which only match the
    clean symmetric double-open-three shape) -- brute-forces every empty
    cell through fast_check_win, so it also catches irregular, real-game
    multi-threat shapes _fork_geometries structurally can't represent.
    len(...) >= 2 means an UNSTOPPABLE fork for whoever must move next (no
    single move blocks two simultaneous winning replies).

    2026-09-23: added after 4 straight HLT (human) losses this session, all
    ending in this exact kind of fork, NONE of which were caught by the
    existing (narrow, template-based) Hard-3 detector below -- see the
    session's find_fork_points.py/test_fork_prevention.py scratch analysis.
    O(BOARD_SIZE^2) fast_check_win calls per invocation; only run on
    already-mined game plies (a few hundred per refresh cycle), not inside
    the hot self-play loop itself.
    """
    cells = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = color
            if fast_check_win(board, color):
                cells.append((r, c))
            board[r][c] = ' '
    return cells


def _fork_safe_cells(board, player, other):
    """Every empty cell where `player` moving there does NOT hand `other` an
    unstoppable next-move fork (winning outright always counts as safe too)
    -- the Hard-7 analogue of _threat_gaps' g_p/g_o: a verified, brute-force
    ground truth for "which replies are actually correct here", not just
    "is a fork possible". Used by _relabel_hard_examples to force the
    Hard-7 policy target onto a real safe cell when the re-searched MCTS
    visit distribution doesn't already favor one -- mirrors how hard_class
    1/2 force their policy onto g_p/g_o. Without this, Hard-7's policy
    target was just whatever a RELABEL_SIMS-budget search happened to
    converge to, which an artificial verified-answer puzzle test
    (2026-09-24 session) showed can itself fail to concentrate visits on
    the best-Q move -- i.e. the relabel step could silently inherit and
    reinforce the same policy-prior blind spot it's meant to correct.
    O(BOARD_SIZE^2) candidate moves x _opponent_can_fork_next's own
    O(BOARD_SIZE^2)-worst-case scan; only run inside
    _relabel_hard_examples's already-capped (RELABEL_CAP) pass, not the
    self-play hot loop."""
    safe = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                continue
            board[r][c] = player
            if fast_check_win(board, player):
                safe.append((r, c))
            elif not _opponent_can_fork_next(board, other):
                safe.append((r, c))
            board[r][c] = ' '
    return safe


def _classify_hard_examples(games, windows, forks):
    """games: list of (plies, outcome) from _mine_one_*_game -- each ply is
    (board, player, act_rc, is_model_ply, entropy_info_or_None). Scans only
    the mover's OWN plies (is_model_ply) and returns a deduped list of
    (board, player, hard_class, outcome_for_player, g_p, g_o) -- g_p/g_o are
    kept for the relabel pass's one-hot safety net (see below)."""
    seen = set()
    flagged = []
    for plies, outcome in games:
        model_ply_positions = [i for i, p in enumerate(plies) if p[3]]
        for rank, i in enumerate(model_ply_positions):
            board, player, act, _, ent_info = plies[i]
            other = 'O' if player == 'X' else 'X'
            outcome_for_player = outcome if player == 'X' else -outcome

            g_p = _threat_gaps(board, player, windows)
            g_o = _threat_gaps(board, other, windows)
            forked = _fork_hits(board, player, forks) + _fork_hits(board, other, forks)

            # Hard-7 (2026-09-23): general, template-free fork check -- does
            # THIS move (act) leave `other` with SOME reply that manufactures
            # an unstoppable fork (or outright win)? Must look one ply past
            # the move itself (_opponent_can_fork_next), not just inspect
            # the board as-is -- see that function's docstring for why a
            # same-ply check (an earlier version of this) misses the real
            # case. Checked separately from g_p/g_o/forked above since none
            # of those catch irregular real-game fork shapes (see
            # _immediate_win_cells docstring).
            after = [row[:] for row in board]
            ar, ac = act
            after[ar][ac] = player
            allowed_fork = (not fast_check_win(after, player)
                             and _opponent_can_fork_next(after, other))

            hard_class = None
            if g_p and act not in g_p:
                hard_class = 1
            elif g_o and not g_p and act not in g_o:
                hard_class = 2
            elif allowed_fork:
                hard_class = 7
            elif forked and not g_p and not g_o and act not in forked:
                hard_class = 3
            elif outcome_for_player == -1 and rank >= len(model_ply_positions) - HARD5_LOOKBACK:
                hard_class = 5
            elif ent_info is not None:
                ent, n_cand = ent_info
                if n_cand >= 8 and ent > float(np.log(n_cand)) - FLAT_ENTROPY_MARGIN:
                    hard_class = 6

            if hard_class is None:
                continue
            key = (''.join(''.join(row) for row in board), player)
            if key in seen:
                continue
            seen.add(key)
            flagged.append((board, player, hard_class, outcome_for_player, g_p, g_o))
    return flagged


def _relabel_hard_examples(flagged, model, device, cap=RELABEL_CAP, sims=RELABEL_SIMS):
    """Re-searches each flagged position with the CURRENT model at a boosted
    sim count (no noise) and turns the resulting visit distribution into the
    policy target. Value target: Hard-1 gets a provable +1 (the flagged gap
    IS a winning completion for `player`, independent of how the actual
    recorded game later turned out); Hard-5 gets the real recorded outcome
    (that's its whole definition); Hard-2/3/6 stay policy-only (no value
    supervision), matching generate_tactical_examples' own block/fork
    convention -- none of those three license a value claim.
    Returns a list of (board, player, policy[NUM_CELLS] float32, value_or_None, hard_class)."""
    ordered = _order_relabel_candidates(flagged, cap)
    was_training = model.training
    model.eval()
    relabeled = []
    try:
        for item in ordered:
            r = _relabel_one(item, model, device, sims)
            if r is not None:
                relabeled.append(r)
    finally:
        model.train(was_training)
    return relabeled


def _order_relabel_candidates(flagged, cap=RELABEL_CAP):
    """Class-priority order + per-class shuffle + cap -- the selection step of
    _relabel_hard_examples, factored out (2026-09-27) so the unified MP pool
    can relabel the SAME selection in parallel."""
    by_class = {}
    for f in flagged:
        by_class.setdefault(f[2], []).append(f)
    ordered = []
    for cls in (1, 2, 7, 3, 5, 6):
        items = by_class.get(cls, [])
        random.shuffle(items)
        ordered.extend(items)
    ordered = ordered[:cap]
    return ordered


def _relabel_one(item, model, device, sims=RELABEL_SIMS, pooled_evaluator=None, eval_cache=None):
    """Relabels ONE flagged item (see _relabel_hard_examples); returns the
    (board, player, policy, value, hard_class) tuple, or None if the search
    found no children. Factored out 2026-09-27 so the unified MP pool can
    run relabels in parallel worker processes (pooled_evaluator/eval_cache
    route the leaf evaluations there); None/None = original behavior."""
    board, player, hard_class, outcome_for_player, g_p, g_o = item
    mcts = MCTS(c_puct=1.6, eval_batch_size=16, shared_cache=eval_cache, pooled_evaluator=pooled_evaluator)
    root = mcts.search(board, player, model, device, num_simulations=sims, add_noise=False)
    if not root.children:
        return None
    total_visits = sum(ch.n_visits for ch in root.children.values())
    policy = np.zeros(NUM_CELLS, dtype=np.float32)
    for act, ch in root.children.items():
        policy[act] = (ch.n_visits / total_visits) if total_visits > 0 else 0.0

    if hard_class in (1, 2):
        gaps = g_p if hard_class == 1 else g_o
        argmax_rc = divmod(int(np.argmax(policy)), BOARD_SIZE)
        if gaps and argmax_rc not in gaps:
            forced = random.choice(gaps)
            policy[:] = 0.0
            policy[forced[0] * BOARD_SIZE + forced[1]] = 1.0
    elif hard_class == 7:
        # 2026-09-24: same forcing as Hard-1/2, but the "known
        # correct cells" have to be computed here (not stored at
        # mining time) via _fork_safe_cells -- see that function's
        # docstring for why this was needed (the re-searched policy
        # target could otherwise inherit the same weak-prior
        # blind spot it's supposed to correct).
        other = 'O' if player == 'X' else 'X'
        safe_cells = _fork_safe_cells(board, player, other)
        argmax_rc = divmod(int(np.argmax(policy)), BOARD_SIZE)
        if safe_cells and argmax_rc not in safe_cells:
            forced = random.choice(safe_cells)
            policy[:] = 0.0
            policy[forced[0] * BOARD_SIZE + forced[1]] = 1.0

    if hard_class == 1:
        # 2026-09-23: 1.0 -> 0.95 (label smoothing). value_overfit_check.py
        # found the live value head badly overconfident on FRESH,
        # never-trained-on self-play games (held-out MSE 1.36 --
        # worse than a constant-0 predictor; 61% of predictions
        # saturated past |0.9|; the model's own "very confident"
        # bucket only actually won ~67% of the time, not ~100%).
        # Confident artificial labels (this class and Hard-7) are
        # the most likely amplifier -- softening them off the
        # tanh's saturated tail should reduce that overconfidence
        # without giving up the corrective signal's direction.
        value = 0.95
    elif hard_class == 7:
        # 2026-09-23: unlike Hard-1/5, this is a confident NEGATIVE
        # label, not a policy-only class -- letting `other` reach an
        # unstoppable fork is a near-certain loss for `player`
        # regardless of what happens afterward (session evidence:
        # test_fork_prevention.py found 0/3 real HLT forks
        # preventable even with a fresh 10000-sim search, so this
        # needs a direct value correction, not just more search at
        # relabel time). -1.0 -> -0.95: same label-smoothing
        # rationale as Hard-1 above.
        # 2026-09-26: REMOVED -- policy-only now (value=None). The
        # board stored here is the position BEFORE the defender
        # moves, where fork-safe replies exist, so "near-certain
        # loss" was the value of the position AFTER a bad reply, not
        # this one. pessimism_probe (14 Hard-7 positions, safe reply
        # then best-vs-best playouts) measured raw value -0.78 vs an
        # empirical outcome of -0.04: this label was teaching the
        # value head that any fork threat means the game is lost.
        value = None
    elif hard_class == 5:
        value = float(outcome_for_player)
    else:
        value = None

    return ([row[:] for row in board], player, policy, value, hard_class)


def _load_hard_example_pool():
    if os.path.exists(HARD_EXAMPLE_POOL_PATH):
        try:
            with open(HARD_EXAMPLE_POOL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Hard-example pool load notice ({e}). Starting with an empty pool.")
    return []


def _save_hard_example_pool(pool, cap=HARD_EXAMPLE_CAP):
    if len(pool) > cap:
        pool = pool[-cap:]
    os.makedirs(os.path.dirname(HARD_EXAMPLE_POOL_PATH), exist_ok=True)
    with open(HARD_EXAMPLE_POOL_PATH, "wb") as f:
        pickle.dump(pool, f)
    return pool


def _get_hard_example_samples(device, n_positions=46, sample_weight=1.5):
    """Per-epoch draw from the hard-mined pool for the regular replay buffer
    (see train_mcts_alpha_zero's call site, right after the league samples).
    n_positions=46 x 8-way augmentation ~= 368 samples/epoch, matching the
    spec's target self/league/hard ~= 0.70/0.15/0.15 buffer mix. value=None
    entries (Hard-2/3/6, policy-only) get 0.0 so they never receive value
    gradient signal (value_criterion is computed the same as every other
    source; 0.0 is a placeholder, not a claim -- see how tactical_refresh
    keeps a boolean mask instead if this ever needs stricter separation)."""
    pool = _load_hard_example_pool()
    if not pool:
        return []
    chosen = random.sample(pool, min(n_positions, len(pool)))
    samples = []
    for board, player, policy, value, hard_class in chosen:
        board_3d = board_to_input_array(board, player)
        target_policy_map = policy.reshape(BOARD_SIZE, BOARD_SIZE)
        final_val = float('nan') if value is None else float(value)  # NaN = no value label, masked in training
        for aug_b, aug_p in augment_sample(board_3d, target_policy_map):
            samples.append((aug_b, aug_p, final_val, sample_weight))
    return samples


def _get_fork_mistake_samples(n_positions=60, sample_weight=3.0):
    """Per-epoch DEDICATED draw from the Hard-7 (general, template-free
    fork) slice of the hard-mined pool, separate from
    _get_hard_example_samples' whole-pool draw.

    2026-09-23: Hard-7 is currently a small fraction of the shared 8000-cap
    pool (261/8000 the first time it was measured), so
    _get_hard_example_samples' n_positions=46 draw only pulls ~1-2 Hard-7
    examples per epoch by chance -- measured directly
    (hard7_adaptation_test.py): the live model still walked into 18/20
    freshly-mined Hard-7 positions one epoch after mining, essentially no
    real training exposure yet. This dedicated draw guarantees Hard-7 gets
    real per-epoch volume regardless of how large the rest of the pool
    (mostly Hard-5, outcome-based) grows around it. sample_weight=2.0
    (higher than the shared draw's 1.5) since Hard-7 carries a confident
    value=-1.0 label (see _relabel_hard_examples), not a policy-only one.
    """
    pool = _load_hard_example_pool()
    h7 = [item for item in pool if item[4] == 7]
    if not h7:
        return []
    chosen = random.sample(h7, min(n_positions, len(h7)))
    samples = []
    for board, player, policy, value, hard_class in chosen:
        board_3d = board_to_input_array(board, player)
        target_policy_map = policy.reshape(BOARD_SIZE, BOARD_SIZE)
        final_val = float('nan') if value is None else float(value)  # NaN = no value label, masked in training
        for aug_b, aug_p in augment_sample(board_3d, target_policy_map):
            samples.append((aug_b, aug_p, final_val, sample_weight))
    return samples


def _get_open_three_injection_samples(n_examples=120, sample_weight=1.0):
    """Per-epoch draw of open-three + open-3-3-fork (+ diagnosed-position)
    examples for the regular replay buffer, same pattern as
    _get_hard_example_samples right below this.

    2026-09-22: replaces tactical_refresh()'s periodic burst supervised
    fine-tune as this curriculum's delivery mechanism. Isolated before/after
    testing (scratch/.../tactical_refresh_destroy_test.py) found ONE
    tactical_refresh() call immediately dropped vs-Det win rate 65%->25%
    (a single 20-game/600-sim sample -- noisy, but the direction matches the
    loss-spike pattern already documented in tactical_refresh's own
    docstring at ep22/ep43/ep64). A separate retention check
    (scratch/.../tactical_retention_check.py) found the ~100% tactical
    accuracy right after a refresh had already decayed to ~70% just 2
    epochs later, before the next refresh even fired -- so each burst was
    paying a real, measured cost for a gain that mostly didn't survive to
    the next cycle anyway. Blending a smaller amount of this curriculum into
    EVERY epoch's ordinary training batch (like the hard-mined pool
    already does) avoids both problems: no isolated-batch overfit spike,
    and no burst to decay away between refreshes since there's no longer a
    gap between applications.

    n_examples=120 positions x up to 8-way dihedral augmentation ~= up to
    ~960 samples/epoch -- deliberately a larger share of the epoch's buffer
    than the hard-mined pool's ~368/epoch (n_positions=46), since this is
    now the sole delivery path for the open-three curriculum rather than a
    supplement to periodic bursts.
    """
    global _open_three_injection_cache
    if _open_three_injection_cache is None:
        # 2026-09-24: generate_fork_examples() (the open-3-3 / double-open-
        # three geometry -- see _fork_geometries) added to this SAME
        # continuous pool. It was already implemented and already used to
        # get delivered via tactical_refresh()'s burst fine-tune, but that
        # call was removed from the main loop this session (see the
        # tactical_refresh_interval guard's comment) and nothing replaced
        # it for this specific curriculum the way open-three's own got
        # migrated here on 2026-09-22 -- so it silently went from
        # "burst-delivered" to "never delivered". A held-out T-shape probe
        # this session (test_t_shape_33fork_realistic.py) found this
        # measurably regressing over the ~200 epochs since: raw policy
        # rank for the correct denial cell 3rd/121 at the Gen12 reset
        # point -> 47th/121 at the current live checkpoint, MCTS solving
        # it correctly at epoch 0 and failing every checkpoint since --
        # i.e. real forgetting of knowledge the supervised-pretrain
        # checkpoint already had, not just slow acquisition. No new
        # examples are being authored here -- this is reconnecting an
        # existing generator to an existing delivery mechanism.
        examples = [(board, 'X', gap, is_mc) for (board, gap, is_mc) in generate_open_three_examples()]
        examples += [(board, 'X', gap, is_mc) for (board, gap, is_mc) in generate_fork_examples()]
        examples += generate_diagnosed_positions_examples()
        board_by_key, rows_by_key = {}, {}
        for board, player, (gr, gc), is_mc in examples:
            key = (tuple(tuple(row) for row in board), player)
            board_by_key.setdefault(key, (board, player))
            rows_by_key.setdefault(key, []).append(((gr, gc), is_mc))
        pool = []
        for key, target, has_win in _multi_label_targets_from_rows(rows_by_key):
            board, player = board_by_key[key]
            board_arr = board_to_input_array(board, player)
            target_policy_map = target.reshape(BOARD_SIZE, BOARD_SIZE)
            value = 0.97 if has_win else 0.0
            pool.append((board_arr, target_policy_map, value))
        _open_three_injection_cache = pool
        print(f"Open-three(+3-3-fork) injection pool built: {len(pool)} unique positions "
              f"(continuous per-epoch draw, replaces tactical_refresh burst)")
    pool = _open_three_injection_cache
    if not pool:
        return []
    chosen = random.sample(pool, min(n_examples, len(pool)))
    samples = []
    for board_arr, target_policy_map, value in chosen:
        for aug_b, aug_p in augment_sample(board_arr, target_policy_map):
            samples.append((aug_b, aug_p, value, sample_weight))
    return samples


def refresh_mined_tactical_pool(model, device, num_selfplay_games=16, num_vsdet_games=16,
                                 sims=150, max_workers=6, cap=20000, num_vsleague_games=16,
                                 epoch=0, league_pool_size=20, league_best_checkpoint_prob=0.28,
                                 runner=None):
    """
    Periodic top-up for the real-game-mined tactical pool (see
    _get_mined_tactical_tensors), called once per tactical_refresh cycle
    (right before tactical_refresh itself) so the pool keeps tracking the
    model's CURRENT play style instead of staying frozen at whatever epoch
    it was first mined at.

    Why this matters: the mined pool's board positions come from games the
    model itself plays, so a one-time-only pool reflects only that
    snapshot's blind spots (a self-referential limitation -- see project
    discussion). As self-play's own exploration (Dirichlet noise + MCTS's
    PUCT exploration bonus, which visits under-prior-weighted moves and
    feeds back into training when they turn out good) gradually corrects
    those blind spots over many epochs, re-mining periodically lets the
    tactical curriculum's data source ride along with that improvement
    instead of being stuck at its original snapshot forever.

    Plays num_selfplay_games + num_vsdet_games real games with the CURRENT
    model (threaded, mirroring generate_mcts_self_play_data's own
    ThreadPoolExecutor pattern -- torch releases the GIL during the forward
    pass so this is real concurrency), scans every ply of every game for
    the same "4-of-5+gap" pattern _tactical_windows() defines, appends any
    matches to the on-disk pool at MINED_TACTICAL_EXAMPLES_PATH (capped at
    `cap` total examples -- oldest examples dropped first once exceeded, so
    the pool doesn't grow unboundedly over a 20000-epoch run), and
    invalidates the in-memory tensor cache so the very next
    _get_tactical_tensors() call (from the tactical_refresh() call right
    after this one) picks up the fresh pool instead of stale cached tensors.

    Uses a modest sims count (150, not the live training sim-ramp budget)
    since realistic stone clustering matters more here than search
    strength -- matches the original one-time mining's own choice (see
    scratch/multiproc_investigation/mine_tactical_examples.py).

    2026-09-11: the same games also feed a SECOND, outcome-based mining pass
    (_classify_hard_examples / _relabel_hard_examples) writing to the
    separate HARD_EXAMPLE_POOL_PATH -- see the block of comments/functions
    just above this one for the full design. Kept in the same function
    (rather than a second refresh call) so the expensive game-generation
    itself is only paid once per tactical_refresh cycle.

    runner (2026-09-27): optional selfplay_mp_unified task runner. When
    given, the mining games (+ their per-game hard-example classification)
    and the relabel searches run in the persistent multi-process game pool
    instead of threads in this process; any runner failure falls back to
    the original threaded/sequential code below for that step.
    """
    global _mined_tactical_tensors_cache

    was_training = model.training
    model.eval()
    windows = _tactical_windows()
    forks = _fork_geometries()
    games = []  # (plies, outcome) per game, for the hard-example pass
    games_played = 0
    flagged = None
    # Opponent picks / depth draws happen here, once, for either path.
    mine_depths = [_sample_vs_det_max_depth(epoch) for _ in range(num_vsdet_games)]
    mine_league_model = None
    if num_vsleague_games > 0:
        # deep_history_prob=0.0: mining wants the model's CURRENT blind
        # spots (see docstring above), so this pick stays biased toward
        # recent/strong opponents only.
        _mine_pick = _pick_league_opponent(device, league_pool_size, league_best_checkpoint_prob,
                                           epoch=epoch, deep_history_prob=0.0)
        if _mine_pick is not None:
            mine_league_model = _mine_pick[0]
    if runner is not None:
        try:
            games, flagged = runner.mine(model, mine_league_model, device, sims,
                                         num_selfplay_games, mine_depths,
                                         num_vsleague_games if mine_league_model is not None else 0)
            games_played = len(games)
        except Exception as e:  # noqa: BLE001
            print(f"  [refresh] unified-pool mining failed ({e!r}) -- threaded fallback", flush=True)
            games, flagged, games_played = [], None, 0
            runner = None
    try:
        if runner is None:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_mine_one_selfplay_game, model, device, sims)
                           for _ in range(num_selfplay_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_selfplay_games
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_mine_one_vsdet_game, model, device, sims, (g % 2 == 0), mine_depths[g])
                           for g in range(num_vsdet_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_vsdet_games
            if mine_league_model is not None:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_mine_one_vsleague_game, model, mine_league_model, device, sims, (g % 2 == 0))
                               for g in range(num_vsleague_games)]
                    for fut in futures:
                        games.append(fut.result())
                games_played += num_vsleague_games
    finally:
        model.train(was_training)

    all_positions = [(board, player) for plies, _ in games for (board, player, _, _, _) in plies]

    new_examples = []
    for board, current_player in all_positions:
        for cells in windows:
            vals = [board[r][c] for (r, c) in cells]
            for color in ('X', 'O'):
                if sum(v == color for v in vals) == 4 and vals.count(' ') == 1:
                    gap_cell = cells[vals.index(' ')]
                    new_examples.append(([row[:] for row in board], current_player,
                                          gap_cell, color == current_player))

    existing = []
    if os.path.exists(MINED_TACTICAL_EXAMPLES_PATH):
        with open(MINED_TACTICAL_EXAMPLES_PATH, "rb") as f:
            existing = pickle.load(f)

    combined = existing + new_examples
    if len(combined) > cap:
        combined = combined[-cap:]  # drop oldest first, keep the pool's most recent snapshot bounded

    os.makedirs(os.path.dirname(MINED_TACTICAL_EXAMPLES_PATH), exist_ok=True)
    with open(MINED_TACTICAL_EXAMPLES_PATH, "wb") as f:
        pickle.dump(combined, f)

    _mined_tactical_tensors_cache = None  # force _get_mined_tactical_tensors to reload from disk next call

    if flagged is None:
        flagged = _classify_hard_examples(games, windows, forks)
    relabeled = None
    if runner is not None:
        try:
            relabeled = runner.relabel(model, device, _order_relabel_candidates(flagged), RELABEL_SIMS)
        except Exception as e:  # noqa: BLE001
            print(f"  [refresh] unified-pool relabel failed ({e!r}) -- sequential fallback", flush=True)
            relabeled = None
    if relabeled is None:
        relabeled = _relabel_hard_examples(flagged, model, device)
    hard_pool = _load_hard_example_pool()
    hard_pool = _save_hard_example_pool(hard_pool + relabeled)
    class_counts = Counter(r[4] for r in relabeled)
    class_note = " ".join(f"H{c}={class_counts.get(c,0)}" for c in (1, 2, 7, 3, 5, 6))

    msg = (f"Tactical pool re-mined: +{len(new_examples)} new (from {len(all_positions)} positions "
           f"across {games_played} games) -> {len(combined)} total"
           f"{' (capped, oldest dropped)' if len(existing) + len(new_examples) > cap else ''}"
           f" | Hard examples: {len(flagged)} flagged -> {len(relabeled)} relabeled ({class_note}) "
           f"-> {len(hard_pool)} total")
    print(msg)
    return len(new_examples), len(combined)


def _get_tactical_tensors(device):
    """
    Builds this refresh's tactical training pool as an EQUAL-WEIGHTED mix:
    ALL currently-available real mined examples (naturally-clustered
    positions mined from actual played games) combined with a FRESH random
    subsample of the synthetic pool of equal size (drawn without
    replacement, independently on every call -- not cached/fixed). This is
    "candidate d" from the held-out generalization study (Phase 2:
    scratch/multiproc_investigation/tactical_mix_experiment.py; Phase 3
    independent re-verification: scratch/multiproc_investigation/
    phase3_independent_reverify.py) -- the only mix among 5 tested that
    materially improved held-out-REAL argmax accuracy (Phase 2: 0.5079 ->
    0.6947; independent re-verification with a different seed/split:
    0.3135 -> 0.6239) with no measurable cost to held-out-SYNTHETIC
    accuracy (both runs: within ~0.2pp of the synthetic-only baseline).
    Naive full-concatenation and real-upsampled-to-synthetic-volume mixes
    were both measurably WORSE than synthetic-only on held-out real
    accuracy in Phase 2 -- do not "simplify" this back to a plain
    concatenation.

    Re-drawing the synthetic subsample fresh every call (rather than fixing
    it once) means the full synthetic pool's window/direction/color
    coverage guarantee is still achieved in expectation across many refresh
    cycles (every tactical_refresh_interval epochs), while any single
    refresh's pool stays balanced 50/50 instead of being diluted the way a
    naive full-concatenation was measured to be.
    """
    syn_inputs, syn_targets, syn_mc = _get_synthetic_tactical_tensors(device)
    real_inputs, real_targets, real_mc = _get_mined_tactical_tensors(device)

    n_real = real_inputs.shape[0]
    if n_real == 0:
        return syn_inputs, syn_targets, syn_mc

    n_syn = syn_inputs.shape[0]
    n_down = min(n_real, n_syn)
    syn_idx = torch.from_numpy(np.random.choice(n_syn, size=n_down, replace=False))

    inputs = torch.cat([syn_inputs[syn_idx], real_inputs], dim=0)
    targets = torch.cat([syn_targets[syn_idx], real_targets], dim=0)
    mover_completes = torch.cat([syn_mc[syn_idx], real_mc], dim=0)
    return inputs, targets, mover_completes


def tactical_refresh(model, device, epochs=3, batch_size=32):
    """
    Runs a few epochs of supervised fine-tuning on the tactical curriculum.

    Uses its OWN local Adam (2026-09-10) rather than the main RL loop's
    optimizer. Sharing it meant 8 supervised epochs on a different objective
    wrote into the shared Adam moment estimates, and every refresh was
    followed by a 2-3 epoch loss spike as the main loop recovered (ep22:
    P 0.98->2.04, V 0.06->0.26; ep43 and ep64 the same shape) -- roughly 14%
    of all epochs spent undoing it. Meant to be called
    periodically (every N epochs) during train_mcts_alpha_zero so this
    signal doesn't get diluted back out by the much larger volume of
    ordinary self-play samples between refreshes.

    Policy loss is computed over the whole batch, as before. Value loss is
    added ONLY over the subset of the batch carrying a real value label --
    a per-row float value_target (NaN = no label, e.g. the pure-block-only
    case, which gets none: "the defender played correctly" alone doesn't
    prove a value by itself). 0.97 (not literal 1.0) stays off the tanh
    output's saturated-gradient tail.
    """
    inputs, targets, value_targets = _get_tactical_tensors(device)
    inputs, targets, value_targets = inputs.to(device), targets.to(device), value_targets.to(device)
    n = inputs.shape[0]

    # 2026-09-21 lr-fix restart: 5e-4 -> 5e-5. This was tuned (2026-09-10)
    # against the OLD main-loop lr=1e-3 (a 0.5x ratio); the main loop is now
    # 1e-4 (this restart's lr-fix), and tactical_refresh_interval dropped
    # 21->5 for denser gate-check signal, so this refresh now fires ~4.2x
    # more often. Left at 5e-4 it would repeatedly overpower the just-
    # lowered main LR with the same-ratio-violating nudge, ~4x more often --
    # likely why gate checks are still stuck at ~0% 25+ epochs into this
    # restart. Scaling down to preserve the original 0.5x-of-main-lr ratio.
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    model.train()
    last_acc = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_in, b_tgt, b_val = inputs[idx], targets[idx], value_targets[idx]
            b_mask = ~torch.isnan(b_val)
            optimizer.zero_grad()
            logits, value_pred = model(b_in)
            policy_loss = -(b_tgt * logits).sum(dim=1).mean()
            if b_mask.any():
                # value_pred is (batch,1); b_val[b_mask] is (n,) -- must match
                # shape or mse_loss silently broadcasts to (n,n) and computes
                # the wrong (inflated) loss instead of an elementwise one.
                value_loss = nn.functional.mse_loss(value_pred[b_mask], b_val[b_mask].unsqueeze(-1))
                loss = policy_loss + 1.0 * value_loss
            else:
                loss = policy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                pred = torch.exp(logits)
                # Multi-label-correct: the predicted cell counts as right if
                # it's ANY of the valid cells for this board (b_tgt > 0), not
                # only if it matches b_tgt's own argmax -- b_tgt can now carry
                # several equally-valid cells (see _multi_label_targets_from_rows),
                # and tie-breaking on argmax would wrongly mark the others wrong.
                pred_idx = pred.argmax(dim=1, keepdim=True)
                correct += (b_tgt.gather(1, pred_idx).squeeze(1) > 0).sum().item()
        last_acc = correct / n
    return last_acc


def train_mcts_alpha_zero(
    num_epochs=3,
    num_self_play_games=10,
    num_simulations=40,
    batch_size=64,
    self_play_workers=None,
    eval_interval=10,
    eval_games=50,
    eval_simulations=50,
    boosted_sims=None,
    boost_threshold=0,
    num_vs_deterministic_games=0,
    temperature_moves=6,
    dirichlet_alpha=0.3,
    noise_weight=0.25,
    c_puct=1.414,
    # -- Search sharpness for vs-Det/vs-League generation and all fair
    # (no-noise) evaluation/gating play, kept independent of self-play's own
    # (annealed) c_puct/noise/dirichlet_alpha so those exploratory self-play
    # settings don't leak into matches meant to measure/generate against a
    # fixed external opponent.
    c_puct_exploit=1.6,
    vs_det_noise_weight=0.4,
    vs_det_dirichlet_alpha=0.3,
    tactical_refresh_interval=0,
    tactical_refresh_epochs=3,
    boost_when_narrow=False,
    num_curriculum_games=0,
    num_opening_games=0,
    # -- League training (self-play vs past checkpoints, see the "League
    # Training" section above): num_league_games=0 disables it (backward
    # compatible default).
    num_league_games=0,
    league_pool_size=20,
    league_best_checkpoint_prob=0.28,
    # -- Playout Cap Randomization (KataGo): full_search_prob=1.0 disables
    # it (backward compatible default -- every move is a full search).
    full_search_prob=1.0,
    fast_sims=None,
    # 2026-09-26: PCR for vs-Det/vs-League games, kept separate so PCR can
    # be applied to self-play only. None = same as full_search_prob (the
    # previous behavior, where one value was passed to all three).
    full_search_prob_other=None,
    # 2026-09-26: play self-play + vs-Det + vs-League games in ONE
    # multi-process pool (selfplay_mp_unified) instead of three sequential
    # phases. False = the previous sequential path (which is also the
    # automatic fallback whenever the unified pool fails).
    use_unified_game_pool=False,
    # 2026-09-27: value target = (1-w)*z + w*q for generated games (see
    # mixed_value_target). 0.0 = pure game result (previous behavior).
    value_target_q_weight=0.0,
    # 2026-10-10 (user): half-life decay of the q weight. From epoch
    # value_target_q_decay_start on, w(epoch) = value_target_q_weight *
    # 0.5 ** ((epoch - start) / value_target_q_halflife); below
    # value_target_q_floor it becomes exactly 0 (pure z). halflife 0 = off.
    value_target_q_halflife=0,
    value_target_q_decay_start=0,
    value_target_q_floor=0.01,
    # 2026-09-28: cap self-play / league game length (plies); None = no cap.
    max_game_plies=None,
    # 2026-09-28: vs-Deterministic games inside the refresh-epoch MINING pass
    # (the old hard-coded default was 16). 0 = mining uses self-play and
    # league games only.
    refresh_vsdet_games=16,
    # -- Exploration noise annealing: noise_anneal_epochs=0 disables it
    # (backward compatible default -- noise_weight/dirichlet_alpha stay
    # fixed at their passed-in value for the whole run, as before).
    noise_weight_end=None,
    dirichlet_alpha_end=None,
    noise_anneal_epochs=0,
    # -- Prioritized replay (error-weighted loss, see training loop below).
    use_prioritized_replay=False,
):
    self_play_device, train_device = get_devices()
    # 2026-09-21 lr-fix restart: base_lr 0.001->0.0001 and T_0 450->150.
    # Weight-drift measurement (see conversation/workflow findings) showed
    # lr=0.001 drifts the post-supervised-pretrain checkpoint 1.86x farther
    # (matched-epoch comparison) than lr=0.0001, and drift never reverses
    # even during low-instantaneous-LR windows of the T_0=450 schedule --
    # consistent with the gate-check having stayed stuck at ~0% for 564+
    # epochs after the Gen10 supervised-pretrain restart. The prior Gen7->
    # Gen8 supervised-pretrain restart broke through gate-check right around
    # the point its T_0=150 schedule decayed to ~1.2e-4 -- this restart
    # starts there directly instead of spending hundreds of epochs decaying
    # down to it.
    # 2026-09-30 (user): gate stuck 15 checks (650-720) while losses kept falling ->
    # halve the LR band (1e-4..5e-5 -> 5e-5..1e-5) to consolidate instead of drifting.
    # 2026-10-11 (user): ep937 cooldown ended -> LR jumped 1e-5 -> 4.4e-5 and ep940 fell to ~1,092 Rapfi nodes.
    # Cosine peak 5e-5 -> 2e-5 (band 2e-5..1e-5) to polish rather than shake before the festival.
    optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1, eta_min=1e-5)
    start_epoch = load_start_epoch()
    for _ in range(start_epoch):
        scheduler.step()
    # 2026-10-12 (user): restore Adam's moments across restarts. Every restart used to reset them
    # (loss spikes at 905/911/913/945). Loaded only if saved right after the epoch we resume at;
    # the LR stays whatever the schedule/cooldown says now (the saved param_groups LR is discarded).
    OPT_STATE_PATH = f"model/optimizer_state{_SUFFIX}.pth"
    # 2026-10-13: a fresh Adam moves every weight by ~LR on its first steps (bias-corrected m/sqrt(v) ~ sign(g)),
    # which caused the loss spikes after every restart/rollback (ep1003 0.38, ep1047 0.41). If the moments
    # are not restored, ramp the LR over the first epochs instead.
    _fresh_opt_warmup = [0.1, 0.3, 0.6]
    if os.path.exists(OPT_STATE_PATH):
        try:
            _os = torch.load(OPT_STATE_PATH, map_location="cpu")
            if _os.get("next_epoch") == start_epoch:
                _keep = [(g["lr"], g.get("initial_lr")) for g in optimizer.param_groups]
                optimizer.load_state_dict(_os["optimizer"])
                for g, (lr_, ilr) in zip(optimizer.param_groups, _keep):
                    g["lr"] = lr_
                    if ilr is not None:
                        g["initial_lr"] = ilr
                print(f"Optimizer state restored (saved after epoch {start_epoch - 1}).")
                _fresh_opt_warmup = []
            else:
                print(f"Optimizer state not restored (saved for epoch {_os.get('next_epoch')}, resuming at {start_epoch}).")
        except Exception as e:
            print(f"Optimizer state load notice ({e}); starting with a fresh optimizer.")
    if _fresh_opt_warmup:   # first trained epoch uses the first factor; the rest follow each scheduler step
        _f0 = _fresh_opt_warmup.pop(0)
        for _pg in optimizer.param_groups:
            _pg['lr'] = _pg['lr'] * _f0
        print(f"Fresh optimizer: LR warm-up x{_f0} for the first trained epoch, then {_fresh_opt_warmup}", flush=True)
    # reduction='none' so prioritized replay (below) can reweight per-sample
    # before averaging; when use_prioritized_replay=False the weights are
    # all 1.0 so this is numerically identical to 'batchmean'/'mean'.
    policy_criterion = nn.KLDivLoss(reduction='none')
    value_criterion = nn.MSELoss(reduction='none')
    # Minibatch shape is fixed (batch_size,3,3,3) for every batch but the last,
    # so cudnn can safely autotune its conv algorithm once instead of
    # re-picking a safe default every call.
    torch.backends.cudnn.benchmark = True
    BUFFER_SAVE_INTERVAL = 10  # see save_replay_buffer call below

    self_play_workers = self_play_workers or max(1, (os.cpu_count() or 4) - 2)

    os.makedirs('model', exist_ok=True)
    checkpoint_path = f"model/alphazero_mcts{_SUFFIX}.pth"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"Loaded existing MCTS checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Checkpoint load notice ({e})")

    replay_buffer = load_replay_buffer()

    # Persistent "generation" model used for self-play/vs-Det/vs-League data
    # generation, kept separate from `model` (mutated in place by every
    # optimizer.step()) so training data always comes from a validated,
    # gate-promoted checkpoint (AlphaGo Zero style) instead of raw,
    # not-yet-validated weights. Synced with the tracked best via
    # run_gating_check on every promotion (including bootstrap) -- never
    # trained directly (.train() must never be called on it).
    gen_model = AlphaZeroTicTacToe()
    if os.path.exists(BEST_CHECKPOINT_PATH):
        try:
            gen_model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location="cpu"))
        except Exception as e:
            print(f"gen_model checkpoint load notice ({e}), falling back to current model weights.")
            gen_model.load_state_dict(model.state_dict())
    else:
        gen_model.load_state_dict(model.state_dict())
    gen_model.to(self_play_device, memory_format=preferred_memory_format(self_play_device))
    gen_model.eval()

    # Pools eval_batch_fn calls from concurrent self-play worker threads into
    # fewer, larger GPU forward passes -- see PooledEvaluator docstring.
    # Constructed once (not per-epoch): holds a reference to `gen_model`, not
    # a weight snapshot, so it keeps working correctly as run_gating_check's
    # promotions mutate those same weights in place across epochs.
    # target_batch=24, max_wait=0.05 (2026-09-15, was 48/0.005): live stats
    # logging showed this PooledEvaluator hitting its target batch 0% of the
    # time across 300K+ flushes (avg batch ~4, timing out every single time)
    # -- PooledEvaluator._run's `timeout = self.max_wait if pending else None`
    # re-arms a fresh wait window on every arrival (a silence-gap debounce,
    # not a deadline from the first pending item), and with 10 CPU-bound
    # self-play worker threads the gap between arrivals almost never stays
    # under the old 5ms window. Benchmarked on this exact self-play path
    # (scratch/multiproc_investigation/bench_pooled_evaluator.py, 3-run
    # average): target_batch=24/max_wait=0.015 grows the achieved batch
    # ~3.8->5.0 (+32%) and self-play games/sec +19% over the old operating
    # point; CPU-bound MCTS tree search, not GPU eval, still dominates epoch
    # wall-clock, so this is a real but modest gain, not a step-change. Kept
    # consistent with evaluate_vs_deterministic / run_gating_check's own
    # scoped PooledEvaluators below, which use the same values.
    pooled_evaluator = PooledEvaluator(gen_model, self_play_device, target_batch=24, max_wait=0.05) \
        if self_play_device.type == "cuda" else None

    cur_num_simulations, cur_boosted_sims, cur_stagnation_count, cur_lr_cooldown = load_sim_budget()

    print(f"\n--- Starting MCTS AlphaZero Model Training ({num_epochs} Epochs) ---")
    if start_epoch > 0:
        print(f"Resuming cumulative epoch/schedule state at epoch {start_epoch} (from {TRAIN_STATE_PATH})")
    print(f"Self-play/vs-Det/vs-League simulation budget: {cur_num_simulations} "
          f"(boosted {cur_boosted_sims}) -- ramps toward {SIM_RAMP_CEILING}/{BOOST_RAMP_CEILING} on promotion")

    # Noise-anneal safety valve (2026-09-10): noise_anneal_epochs was just
    # cut way down (3000->600) so dirichlet_alpha/noise_weight actually
    # finish annealing within a realistic run length instead of still being
    # ~85% un-annealed after 470+ epochs. A much faster anneal risks the
    # policy sharpening (Policy Entropy dropping) faster than intended --
    # if the last few epochs' entropy falls below a floor well under the
    # ~2.4-3.1 range observed so far, freeze noise_weight/dirichlet_alpha at
    # their last-known-safe level instead of continuing to anneal down into
    # whatever is causing the drop; resumes automatically once entropy
    # recovers above the floor. In-memory only (resets on a process
    # restart), which is fine -- a restart just re-checks from scratch.
    _recent_entropy_window = []
    # Measured on the TARGET (MCTS visit) distributions, not network output,
    # and the candidate set is the radius-2 neighborhood (~45 cells), so the
    # uniform ceiling is ~ln(45)=3.8 and a healthy sharp target sits near 0.7.
    # The old 1.5 would have frozen noise annealing exactly when targets
    # finally started sharpening.
    ENTROPY_SAFETY_FLOOR = 0.5
    ENTROPY_SAFETY_WINDOW = 3
    _last_safe_noise_weight = noise_weight
    _last_safe_dirichlet_alpha = dirichlet_alpha

    epoch = start_epoch
    for _local_i in range(num_epochs):
        epoch = start_epoch + _local_i
        cur_q_weight = value_target_q_weight
        if value_target_q_halflife and epoch >= value_target_q_decay_start:
            cur_q_weight = value_target_q_weight * 0.5 ** ((epoch - value_target_q_decay_start) / value_target_q_halflife)
            if cur_q_weight < value_target_q_floor:
                cur_q_weight = 0.0
        print(f"  [value target] epoch {epoch}: q weight {cur_q_weight:.4f} (z weight {1 - cur_q_weight:.4f})", flush=True)
        # 2026-10-12 (user): second promotion path -- Rapfi ladder. The gate (vs the current best only)
        # can be blocked by a matchup ("상성"); if the ladder watcher finds a measured checkpoint stronger
        # than the best with P >= 80% (bootstrap), it writes model/ladder_promote.json and the checkpoint
        # becomes the best AND the self-play generation model here, at the start of the next epoch.
        try:
            _lp_path = "model/ladder_promote.json"
            if os.path.exists(_lp_path):
                import json as _json
                with open(_lp_path, encoding="utf-8") as _f:
                    _lp = _json.load(_f)
                if not _lp.get("applied") and os.path.exists(_lp["ckpt"]):
                    _sd = torch.load(_lp["ckpt"], map_location="cpu")
                    if os.path.exists(BEST_CHECKPOINT_PATH):
                        __import__("shutil").copy2(BEST_CHECKPOINT_PATH, BEST_CHECKPOINT_PATH + f".before_ladder_promo_ep{epoch}")
                    torch.save(_sd, BEST_CHECKPOINT_PATH)
                    gen_model.load_state_dict(_sd)
                    gen_model.to(self_play_device, memory_format=preferred_memory_format(self_play_device))
                    gen_model.eval()
                    _lp["applied"] = True; _lp["applied_at_epoch"] = epoch
                    with open(_lp_path, "w", encoding="utf-8") as _f:
                        _json.dump(_lp, _f, ensure_ascii=False, indent=1)
                    report_logger(f"Epoch {epoch} | Ladder promotion: {_lp.get('name', _lp['ckpt'])} -> best "
                                  f"(eq {_lp.get('eq', '?')} vs best {_lp.get('best_eq', '?')}, P {_lp.get('p', '?')}) -> PROMOTED (ladder)")
                    print(f"  [ladder promotion] {_lp['ckpt']} is now best + generation model", flush=True)
        except Exception as _e:
            print(f"  [ladder promotion] notice: {_e!r}", flush=True)
        if epoch % 10 == 0:
            snap_path = f"model/model_MCTS_ep{epoch}{_SUFFIX}.pth"
            torch.save(model.state_dict(), snap_path)
            print(f"Snapshot saved to {snap_path}")

        # Self-play phase: CPU, one intra-op thread per call so the
        # ThreadPoolExecutor workers (not torch's internal thread pool)
        # supply the parallelism -- see get_devices() for why. Memory format
        # matches board_to_input_tensor's output format per device: on CUDA
        # that's contiguous_format (channels_last measured 1.4x-3.4x SLOWER
        # on this GPU -- see preferred_memory_format's docstring in
        # alphago_lee.py), on CPU it stays channels_last as before.
        model.to(self_play_device, memory_format=preferred_memory_format(self_play_device))
        torch.set_num_threads(1)
        # Fresh per-epoch NN-eval cache: weights are frozen for this whole
        # generation phase (model.eval(), no optimizer.step() until after both
        # calls below return), so (board, player) -> (probs, value) is a pure
        # function of the current weights across every game and every move in
        # this epoch, self-play and vs-Det alike. Sharing one cache is exact,
        # not approximate; it must be rebuilt every epoch since training below
        # changes the weights.
        epoch_eval_cache = {}
        _target_noise_weight = annealed_value(epoch, noise_anneal_epochs, noise_weight,
                                               noise_weight_end if noise_weight_end is not None else noise_weight)
        _target_dirichlet_alpha = annealed_value(epoch, noise_anneal_epochs, dirichlet_alpha,
                                                  dirichlet_alpha_end if dirichlet_alpha_end is not None else dirichlet_alpha)
        _recent_entropy_avg = (sum(_recent_entropy_window) / len(_recent_entropy_window)
                                if len(_recent_entropy_window) >= ENTROPY_SAFETY_WINDOW else None)
        if _recent_entropy_avg is not None and _recent_entropy_avg < ENTROPY_SAFETY_FLOOR:
            cur_noise_weight = _last_safe_noise_weight
            cur_dirichlet_alpha = _last_safe_dirichlet_alpha
            print(f"  [noise-anneal safety] recent Policy Entropy avg {_recent_entropy_avg:.3f} < floor "
                  f"{ENTROPY_SAFETY_FLOOR} -- holding noise_weight={cur_noise_weight:.3f}, "
                  f"dirichlet_alpha={cur_dirichlet_alpha:.3f} instead of annealing further")
        else:
            cur_noise_weight = _target_noise_weight
            cur_dirichlet_alpha = _target_dirichlet_alpha
            _last_safe_noise_weight = cur_noise_weight
            _last_safe_dirichlet_alpha = cur_dirichlet_alpha
        _t_epoch_start = time.monotonic()
        _fsp_other = full_search_prob if full_search_prob_other is None else full_search_prob_other

        # League opponent is picked up front (it used to be picked after the
        # self-play/vs-Det phases) so the unified pool can play its games
        # alongside everything else.
        league_model = None
        league_opponent_name = None
        if num_league_games > 0 and LEAGUE_NUM_OPPONENTS > 1:
            _picks = _pick_league_opponents_multi(self_play_device, min(LEAGUE_NUM_OPPONENTS, num_league_games),
                                                  league_pool_size, epoch=epoch)
            if _picks:
                league_model = [m for m, _ in _picks]
                league_opponent_name = f"{len(_picks)} opponents: " + ", ".join(
                    os.path.basename(p).replace("model_MCTS_", "").replace("_11x11.pth", "").replace(".pth", "")
                    for _, p in _picks)
        elif num_league_games > 0:
            league_pick = _pick_league_opponent(self_play_device, league_pool_size, league_best_checkpoint_prob,
                                                 epoch=epoch)
            if league_pick is not None:
                league_model, league_path = league_pick
                league_opponent_name = os.path.basename(league_path)

        def _sequential_generation(reason=None):
            """The original three-phase generation (self-play MP, then
            vs-Det threaded, then vs-League threaded). Used directly when the
            unified pool is off, and as its automatic fallback."""
            if reason:
                print(f"  [generation] sequential path: {reason}", flush=True)
            _sp, _sp_counts = generate_mcts_self_play_data_mp(
                gen_model, self_play_device, num_games=num_self_play_games, num_simulations=cur_num_simulations,
                max_workers=self_play_workers, boosted_sims=cur_boosted_sims, boost_threshold=boost_threshold,
                temperature_moves=temperature_moves, dirichlet_alpha=cur_dirichlet_alpha, noise_weight=cur_noise_weight,
                c_puct=c_puct, eval_cache=epoch_eval_cache, boost_when_narrow=boost_when_narrow,
                num_curriculum_games=num_curriculum_games, num_opening_games=num_opening_games,
                pooled_evaluator=pooled_evaluator,
                full_search_prob=full_search_prob, fast_sims=fast_sims,
                value_q_weight=cur_q_weight, max_plies=max_game_plies
            )
            _vd, _vd_counts = None, None
            if num_vs_deterministic_games > 0:
                _vd, _vd_counts = generate_mcts_vs_deterministic_data(
                    gen_model, self_play_device, num_games=num_vs_deterministic_games, num_simulations=cur_num_simulations,
                    max_workers=self_play_workers, boosted_sims=cur_boosted_sims, boost_threshold=boost_threshold,
                    temperature_moves=temperature_moves, dirichlet_alpha=vs_det_dirichlet_alpha, noise_weight=vs_det_noise_weight,
                    c_puct=c_puct_exploit, eval_cache=epoch_eval_cache, boost_when_narrow=boost_when_narrow,
                    pooled_evaluator=pooled_evaluator, full_search_prob=_fsp_other, fast_sims=fast_sims,
                    epoch=epoch, value_q_weight=cur_q_weight
                )
            _lg, _lg_counts = None, None
            _lm = league_model[0] if isinstance(league_model, list) else league_model  # fallback: first opponent only
            if _lm is not None:
                _lg, _lg_counts = generate_mcts_vs_league_data(
                    gen_model, _lm, self_play_device, num_games=num_league_games, num_simulations=cur_num_simulations,
                    max_workers=self_play_workers, boosted_sims=cur_boosted_sims, boost_threshold=boost_threshold,
                    temperature_moves=temperature_moves, dirichlet_alpha=vs_det_dirichlet_alpha, noise_weight=vs_det_noise_weight,
                    c_puct=c_puct_exploit, eval_cache=epoch_eval_cache, boost_when_narrow=boost_when_narrow,
                    pooled_evaluator=pooled_evaluator, full_search_prob=_fsp_other, fast_sims=fast_sims,
                    epoch=epoch, value_q_weight=cur_q_weight, max_plies=max_game_plies
                )
            return _sp, _sp_counts, _vd, _vd_counts, _lg, _lg_counts

        if use_unified_game_pool:
            _selfplay_cfg = dict(
                num_simulations=cur_num_simulations, boosted_sims=cur_boosted_sims, boost_threshold=boost_threshold,
                temperature_moves=temperature_moves, dirichlet_alpha=cur_dirichlet_alpha, noise_weight=cur_noise_weight,
                c_puct=c_puct, boost_when_narrow=boost_when_narrow,
                full_search_prob=full_search_prob, fast_sims=fast_sims,
                max_plies=max_game_plies,
            )
            _vsdet_cfg = dict(
                num_simulations=cur_num_simulations, boosted_sims=cur_boosted_sims, boost_threshold=boost_threshold,
                temperature_moves=temperature_moves, dirichlet_alpha=vs_det_dirichlet_alpha, noise_weight=vs_det_noise_weight,
                c_puct=c_puct_exploit, boost_when_narrow=boost_when_narrow,
                full_search_prob=_fsp_other, fast_sims=fast_sims,
                max_plies=max_game_plies,  # used by the league games (vs-Det has no cap)
            )
            (raw_samples, outcome_counts, vs_det_samples, vs_det_outcome_counts,
             league_samples, league_outcome_counts) = generate_epoch_games_unified(
                gen_model, self_play_device, epoch=epoch, max_workers=self_play_workers,
                selfplay_cfg=_selfplay_cfg, num_self_play_games=num_self_play_games,
                num_curriculum_games=num_curriculum_games, num_opening_games=num_opening_games,
                vsdet_cfg=_vsdet_cfg, num_vs_det_games=num_vs_deterministic_games,
                league_model=league_model, num_league_games=num_league_games,
                fallback=_sequential_generation, value_q_weight=cur_q_weight,
            )
        else:
            (raw_samples, outcome_counts, vs_det_samples, vs_det_outcome_counts,
             league_samples, league_outcome_counts) = _sequential_generation()

        replay_buffer.extend(raw_samples)
        if vs_det_samples:
            replay_buffer.extend(vs_det_samples)
        if league_samples:
            replay_buffer.extend(league_samples)
        _t_gen_done = time.monotonic()

        # Continuous sampling from the hard-mined pool (2026-09-11) -- the
        # mining/relabeling itself stays batched on the expensive
        # tactical_refresh_interval cadence (see refresh_mined_tactical_pool),
        # but drawing from the already-mined pool costs nothing like a fresh
        # search, so this runs every epoch instead of waiting for the next
        # refresh. n_positions=46 -> ~368 augmented samples/epoch, in the same
        # ballpark as this epoch's vs-Det/league volume (the target ~0.70/
        # 0.15/0.15 self/league/hard split from the approved mining spec).
        hard_samples = _get_hard_example_samples(self_play_device, n_positions=46)
        if hard_samples:
            replay_buffer.extend(hard_samples)

        # 2026-09-22: continuous per-epoch open-three curriculum injection,
        # replacing tactical_refresh()'s periodic burst (see
        # _get_open_three_injection_samples docstring for why).
        open_three_samples = _get_open_three_injection_samples(n_examples=120)
        if open_three_samples:
            replay_buffer.extend(open_three_samples)

        # 2026-09-23: dedicated Hard-7 (general fork) quota, on top of
        # hard_samples' whole-pool draw above -- see
        # _get_fork_mistake_samples docstring (measured: without this,
        # Hard-7 got ~1-2 exposures/epoch by chance, essentially no real
        # training signal one full mining cycle after being added).
        fork_mistake_samples = _get_fork_mistake_samples(n_positions=60)
        if fork_mistake_samples:
            replay_buffer.extend(fork_mistake_samples)

        # Interval-saved (not every epoch): the buffer only changes by ~1-2%
        # per epoch, so pickling all 50,000 samples every single epoch was
        # ~0.5s of pure I/O for no learning benefit. Worst case on a crash
        # mid-interval is losing that epoch's fresh self-play data, not any
        # trained weights -- the model checkpoint below still saves every epoch.
        if (epoch + 1) % BUFFER_SAVE_INTERVAL == 0 or epoch == num_epochs - 1:
            save_replay_buffer(replay_buffer)

        vs_det_note = ""
        if vs_det_outcome_counts is not None:
            vs_det_note = (f" | vs-Det (this epoch, {num_vs_deterministic_games} games): "
                            f"MCTS-win {vs_det_outcome_counts['mcts_win']} / "
                            f"Det-win {vs_det_outcome_counts['det_win']} / "
                            f"Draw {vs_det_outcome_counts['draw']} "
                            f"[{format_by_depth(vs_det_outcome_counts.get('by_depth', {}))}]")

        league_note = ""
        if league_outcome_counts is not None:
            league_note = (f" | vs-League ({league_opponent_name}, {num_league_games} games): "
                           f"MCTS-win {league_outcome_counts['mcts_win']} / "
                           f"League-win {league_outcome_counts['league_win']} / "
                           f"Draw {league_outcome_counts['draw']}")

        # 2026-10-13 (user, clean rollback to ep960): with an emptied buffer, only generate data until
        # NAPS_MIN_BUFFER samples exist, so the first training steps do not overfit a tiny buffer.
        if len(replay_buffer) < max(batch_size, int(os.environ.get("NAPS_MIN_BUFFER", "0") or 0)):
            print(f"  Replay buffer warming up: {len(replay_buffer)}/{batch_size} samples "
                  f"(need at least one full batch). Skipping the training step this epoch.")
            report_logger(f"Epoch {epoch + 1}/{num_epochs} | Warming up ({len(replay_buffer)}/{batch_size}) | "
                          f"Self-Play (this epoch, {num_self_play_games} games): "
                          f"X-win {outcome_counts['x_win']} / O-win {outcome_counts['o_win']} / "
                          f"Draw {outcome_counts['draw']}{vs_det_note}{league_note}")
            continue

        buffer_list = list(replay_buffer)
        if len(buffer_list) > TRAIN_SAMPLES_PER_EPOCH:
            buffer_list = random.sample(buffer_list, TRAIN_SAMPLES_PER_EPOCH)
        inputs_np = np.array([s[0] for s in buffer_list], dtype=np.float32)
        policies_np = np.array([s[1] for s in buffer_list], dtype=np.float32)
        values_np = np.array([[s[2]] for s in buffer_list], dtype=np.float32)
        # 4th tuple element is a static per-sample weight (see raw_samples.append
        # call sites: self-play's mover-won bonus, fixed vs-Det/vs-League
        # weights); default to 1.0 for older 3-element samples already sitting
        # in a loaded replay buffer pickle from before this feature existed.
        weights_np = np.array([[s[3] if len(s) > 3 else 1.0] for s in buffer_list], dtype=np.float32)
        # 5th element: V3 survival target (self-play samples only); NaN =
        # no label (vs-Det/vs-League/hard/injection samples, older buffer
        # entries), masked out of the auxiliary loss below.
        survival_np = np.array([[s[4] if len(s) > 4 and s[4] is not None else np.nan] for s in buffer_list],
                               dtype=np.float32)

        # Training phase: GPU (batched forward+backward is faster there
        # even though single-sample self-play forward passes are not).
        # contiguous_format matches the NCHW layout torch.from_numpy(inputs_np)
        # below produces -- leaving this as channels_last (inherited from the
        # self-play .to() above) made every conv convert layout on the fly.
        model.to(train_device, memory_format=torch.contiguous_format)
        if train_device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        model.train()

        # Whole buffer (~7MB for 50,000 3x3 samples) lives resident on
        # train_device, indexed each epoch by one torch.randperm -- the same
        # shuffled full-buffer pass a DataLoader(shuffle=True) already did
        # (RandomSampler draws one permutation, BatchSampler chunks it
        # contiguously, keeping the final short batch), just without 50,000
        # individual __getitem__ calls, 391 collate/stack calls, and 1,173
        # separate host->device copies per epoch.
        in_t = torch.from_numpy(inputs_np).to(train_device)
        pol_t = torch.from_numpy(policies_np).to(train_device)
        val_t = torch.from_numpy(values_np).to(train_device)
        weight_t = torch.from_numpy(weights_np).to(train_device)
        surv_t = torch.from_numpy(survival_np).to(train_device)
        n_samples = in_t.shape[0]
        total_s_loss = 0.0
        num_s_b = 0
        perm = torch.randperm(n_samples, device=train_device)

        total_p_loss = torch.zeros((), device=train_device, dtype=torch.float64)
        total_v_loss = torch.zeros((), device=train_device, dtype=torch.float64)
        num_b = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_in, b_pol, b_val, b_weight = in_t[idx], pol_t[idx], val_t[idx], weight_t[idx]
            b_surv = surv_t[idx]
            optimizer.zero_grad()

            pred_pol_logits, pred_val, pred_surv = model.forward_all(b_in)
            # reduction='none': policy_criterion gives (B, NUM_CELLS), sum
            # over dim=1 to match KLDivLoss(reduction='batchmean')'s
            # per-sample contribution; value_criterion gives (B, 1).
            p_loss_per_sample = policy_criterion(pred_pol_logits, b_pol).sum(dim=1)
            # 2026-09-26: value=None samples (policy-only hard classes) carry
            # NaN and are masked out here. Before this they were stored as
            # 0.0 and silently trained the value head toward "draw".
            v_mask = ~torch.isnan(b_val.squeeze(-1))
            v_loss_per_sample = value_criterion(pred_val, torch.nan_to_num(b_val, nan=0.0)).squeeze(-1) * v_mask
            # Logged/tracked regardless of use_prioritized_replay, so the
            # Policy/Value Loss numbers in training_report stay comparable
            # to every epoch before this feature existed.
            p_loss_unweighted = p_loss_per_sample.mean()
            v_loss_unweighted = v_loss_per_sample.sum() / v_mask.sum().clamp_min(1)

            # Prioritized replay (Schaul et al. 2015), adapted to this
            # project's full-buffer-pass-per-epoch training loop instead
            # of a sampled minibatch replay: reweight each batch's loss
            # by its own current policy/value-prediction "surprise"
            # (computed fresh from this same forward pass, no stored/stale
            # priority table needed) so samples the model is most surprised
            # by contribute more gradient. b_weight is each sample's static,
            # source-based weight (self-play's mover-won bonus, fixed
            # vs-Det/vs-League weights -- see the raw_samples.append call
            # sites) and is always applied; the surprise term on top of it
            # only kicks in when use_prioritized_replay is True. NOTE: when
            # prioritized, weights correlate with p_loss_per_sample/
            # v_loss_per_sample by construction (weight = f(loss)), so the
            # weighted mean is *not* comparable to the plain mean -- it's
            # mechanically inflated (Cov(weight, loss) > 0). That's
            # expected/intended (harder samples deliberately dominate the
            # gradient), not a regression -- hence tracking the unweighted
            # loss above separately for logging.
            policy_surprise = p_loss_per_sample.detach()
            value_surprise = v_loss_per_sample.detach().sqrt()
            if use_prioritized_replay:
                priority = (policy_surprise + value_surprise + 1e-3) * b_weight.squeeze(-1)
            else:
                priority = b_weight.squeeze(-1)
            weights = priority / priority.mean().clamp_min(1e-8)
            p_loss = (weights * p_loss_per_sample).mean()
            v_loss = (weights * v_loss_per_sample).mean()

            loss = p_loss + 1.0 * v_loss

            # V3 auxiliary survival loss (KataGo-style dense signal: how many
            # plies remain), only on rows that carry a label. Weighted low
            # (0.25) so it shapes the shared trunk without competing with
            # the policy/value objectives.
            s_mask = ~torch.isnan(b_surv)
            if s_mask.any():
                s_loss = nn.functional.mse_loss(pred_surv[s_mask], b_surv[s_mask])
                loss = loss + SURVIVAL_LOSS_WEIGHT * s_loss
                total_s_loss += float(s_loss.detach())
                num_s_b += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate on-device (float64, matching Python-float precision
            # well past the 4 logged decimals) instead of a .item() per batch,
            # which forced a full CUDA sync every single minibatch. Always
            # the *unweighted* losses (see note above) for report continuity.
            total_p_loss += p_loss_unweighted.detach().double()
            total_v_loss += v_loss_unweighted.detach().double()
            num_b += 1

        avg_p = (total_p_loss / max(1, num_b)).item()
        avg_v = (total_v_loss / max(1, num_b)).item()
        entropy = compute_policy_entropy(policies_np)
        _recent_entropy_window.append(entropy)
        if len(_recent_entropy_window) > ENTROPY_SAFETY_WINDOW:
            _recent_entropy_window.pop(0)
        scheduler.step()
        if cur_lr_cooldown > 0:
            # Post-promotion LR cooldown (see LR_COOLDOWN_EPOCHS): hold at
            # eta_min instead of wherever the cosine schedule's own math
            # landed this epoch. scheduler.step() above still runs every
            # epoch unconditionally so its internal (T_cur, T_i) state stays
            # exactly reconstructible by the unchanged fast-forward loop on
            # restart; only the LR actually applied to the optimizer is
            # overridden here, and only for the countdown's duration.
            for _pg in optimizer.param_groups:
                _pg['lr'] = scheduler.eta_min
            cur_lr_cooldown -= 1
        if _fresh_opt_warmup:
            _f = _fresh_opt_warmup.pop(0)
            for _pg in optimizer.param_groups:
                _pg['lr'] = _pg['lr'] * _f
            print(f"  [fresh optimizer] LR warm-up x{_f} this epoch ({len(_fresh_opt_warmup)} warm-up epochs left)", flush=True)

        # 2026-09-22: tactical_refresh() (the periodic burst supervised
        # re-injection that used to run here) was REMOVED -- see
        # _get_open_three_injection_samples docstring for the measured
        # before/after evidence (immediate vs-Det win-rate drop, decays back
        # within ~2 epochs anyway). The open-three/diagnosed-position
        # curriculum is now injected continuously every epoch instead (see
        # the open_three_samples call above, right after hard_samples).
        # refresh_mined_tactical_pool (mining fresh REAL tactical examples
        # from current self-play, unrelated to the burst fine-tune) and the
        # promotion gate check both still run on this same interval cadence
        # -- neither was implicated by the destroy/retention tests, and the
        # gate check in particular still needs a periodic "checkpoint" cadence
        # to compare against, unrelated to whether tactical_refresh runs.
        _t_train_done = time.monotonic()
        tactical_note = ""
        gate_note = ""
        if tactical_refresh_interval > 0 and (epoch + 1) % tactical_refresh_interval == 0:
            _runner = get_task_runner(self_play_workers) if use_unified_game_pool else None
            _t_refresh0 = time.monotonic()
            refresh_mined_tactical_pool(model, train_device, epoch=epoch,
                                         num_vsdet_games=refresh_vsdet_games,
                                         league_pool_size=league_pool_size,
                                         league_best_checkpoint_prob=league_best_checkpoint_prob,
                                         runner=_runner)
            _t_refresh1 = time.monotonic()

            # Promotion gate (see run_gating_check docstring): reuses this
            # same periodic cadence since it's the natural "checkpoint" point
            # already established for this loop, and running it every epoch
            # would add unnecessary cost for no extra signal at this pace.
            model.to(self_play_device, memory_format=preferred_memory_format(self_play_device))
            torch.set_num_threads(1)
            _LAST_GATE_CANDIDATE["path"] = None
            gate_win, gate_draw, gate_loss, promoted = run_gating_check(
                model, gen_model, self_play_device, eval_simulations, boosted_sims=boosted_sims,
                boost_threshold=boost_threshold, boost_when_narrow=boost_when_narrow, c_puct=c_puct_exploit,
                runner=_runner
            )
            # 2026-10-11 (user): keep the weights of every gate where the OLD rule (W + D/2 >= 55%) would
            # have promoted but the new loss-penalty rule did not, so the new rule's judgement can be checked
            # later (e.g. Rapfi ladder). Only these disagreements are saved (~34 MB each).
            if not promoted and gate_win + 0.5 * gate_draw >= 0.55:
                os.makedirs("model/gate_disagree", exist_ok=True)
                _gd_path = f"model/gate_disagree/ep{epoch}{_SUFFIX}.pth"
                torch.save(model.state_dict(), _gd_path)
                print(f"  [gate] old rule would have promoted ep{epoch}; weights kept at {_gd_path}", flush=True)
            print(f"  [refresh-timing] mining+relabel {_t_refresh1 - _t_refresh0:.1f}s | "
                  f"gate {time.monotonic() - _t_refresh1:.1f}s", flush=True)
            ramp_note = ""
            if promoted:
                cur_stagnation_count = 0
                cur_lr_cooldown = LR_COOLDOWN_EPOCHS
                new_sims = min(SIM_RAMP_CEILING, cur_num_simulations + SIM_RAMP_STEP)
                new_boost = min(BOOST_RAMP_CEILING, cur_boosted_sims + BOOST_RAMP_STEP)
                if new_sims != cur_num_simulations or new_boost != cur_boosted_sims:
                    cur_num_simulations, cur_boosted_sims = new_sims, new_boost
                    ramp_note = f" -> sim budget now {cur_num_simulations}/{cur_boosted_sims}"
                ramp_note += f" | LR cooldown: {LR_COOLDOWN_EPOCHS} epochs (held at eta_min)"
                save_sim_budget(cur_num_simulations, cur_boosted_sims, cur_stagnation_count, cur_lr_cooldown)
            else:
                cur_stagnation_count += 1
                if cur_stagnation_count >= STAGNATION_RAMP_LIMIT:
                    cur_stagnation_count = 0
                    new_sims = min(SIM_RAMP_CEILING, cur_num_simulations + SIM_RAMP_STEP)
                    new_boost = min(BOOST_RAMP_CEILING, cur_boosted_sims + BOOST_RAMP_STEP)
                    if new_sims != cur_num_simulations or new_boost != cur_boosted_sims:
                        cur_num_simulations, cur_boosted_sims = new_sims, new_boost
                        ramp_note = f" -> sim budget now {cur_num_simulations}/{cur_boosted_sims} (stagnation bump)"
                save_sim_budget(cur_num_simulations, cur_boosted_sims, cur_stagnation_count, cur_lr_cooldown)
            gate_note = (f" | Gate vs best ({len(_GATE_OPENING_SQUARES) * 2} games): Win {gate_win:.0%} / "
                         f"Draw {gate_draw:.0%} / Loss {gate_loss:.0%} -> {'PROMOTED' if promoted else ('CANDIDATE (ladder check pending)' if _LAST_GATE_CANDIDATE['path'] else 'kept')}{ramp_note}")
            model.train()

        _t_gate_done = time.monotonic()
        msg = f"MCTS Epoch {epoch + 1}: Policy Loss: {avg_p:.4f}, Value Loss: {avg_v:.4f}, Buffer: {len(replay_buffer)}"
        if num_s_b:
            msg += f", Survival Loss: {total_s_loss / num_s_b:.4f}"
        logger(msg)
        print(f"  {msg}")

        report_msg = (f"Epoch {epoch + 1}/{num_epochs} | Policy Loss: {avg_p:.4f} | "
                       f"Value Loss: {avg_v:.4f} | Policy Entropy: {entropy:.4f} | "
                       f"Buffer Size: {len(replay_buffer)} | "
                       f"Self-Play (this epoch, {num_self_play_games} games): "
                       f"X-win {outcome_counts['x_win']} / O-win {outcome_counts['o_win']} / "
                       f"Draw {outcome_counts['draw']}{vs_det_note}{league_note}{tactical_note}{gate_note}")
        line = report_logger(report_msg)
        print(f"  {line}")

        # Save after every trained epoch (not just every 10th) so a crash or
        # early stop never loses more than one epoch's worth of training.
        torch.save(model.state_dict(), checkpoint_path)

        # 2026-09-11 bugfix: this used to be called at the TOP of the loop
        # (save_cumulative_epoch(epoch), before this epoch's own work ran) --
        # a lagging save that only confirmed epoch N complete once epoch N+1
        # had already STARTED. A manual stop anytime between epoch N's report
        # line printing and epoch N+1 beginning left the state file at N-1,
        # so a restart silently redid the whole already-completed epoch N
        # (confirmed live: two differently-valued "Epoch 240" report lines
        # from today's stop/restart, status_check's reset-heuristic then
        # misread the non-monotonic epoch number as a full history reset).
        # Saving epoch+1 here, right after this epoch's own checkpoint write,
        # closes that window: a kill after this line correctly resumes at
        # epoch+1; a kill before it correctly redoes only epoch (never more).
        try:   # 2026-10-12: optimizer moments for the next restart (atomic write)
            torch.save({"next_epoch": epoch + 1, "optimizer": optimizer.state_dict()}, OPT_STATE_PATH + ".tmp")
            os.replace(OPT_STATE_PATH + ".tmp", OPT_STATE_PATH)
        except Exception as e:
            print(f"Optimizer state save notice ({e})")
        save_cumulative_epoch(epoch + 1)

        # Periodically measure the actual research goal: win rate against the
        # deterministic (minimax) opponent. Logged separately since it's much
        # more expensive per-sample than a loss computation.
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.to(self_play_device, memory_format=preferred_memory_format(self_play_device))
            torch.set_num_threads(1)
            # Seed by epoch (not a fresh draw each call) so a fresh 20-opening
            # sample is used at every formal check -- averaging out the
            # narrow/edge-heavy bias a fixed opening set had -- while a given
            # epoch's result stays exactly reproducible (re-evaluating the
            # same saved checkpoint later draws the same openings).
            eval_opening_cells = list(_ALL_CELLS)
            random.Random(epoch + 1).shuffle(eval_opening_cells)
            win_rate, draw_rate, loss_rate = evaluate_vs_deterministic(
                model, self_play_device, num_games=eval_games, num_simulations=eval_simulations,
                boosted_sims=boosted_sims, boost_threshold=boost_threshold, c_puct=c_puct_exploit,
                boost_when_narrow=boost_when_narrow, all_cells=eval_opening_cells
            )
            eval_msg = (f"Epoch {epoch + 1}/{num_epochs} | Eval vs Deterministic "
                        f"({eval_games} games, {eval_simulations} sims/move, openings seed={epoch + 1}): "
                        f"Win {win_rate:.0%} | Draw {draw_rate:.0%} | Loss {loss_rate:.0%}")
            line = report_logger(eval_msg)
            print(f"  {line}")
            model.train()

        # Per-stage wall-clock (2026-09-26): games = all generation games,
        # train = injections + the training pass, refresh+gate = mining +
        # promotion gate (every tactical_refresh_interval epochs), eval =
        # formal vs-Det eval (every eval_interval epochs).
        _t_end = time.monotonic()
        print(f"  [stage-timing] epoch {epoch + 1}: games {_t_gen_done - _t_epoch_start:.1f}s | "
              f"train {_t_train_done - _t_gen_done:.1f}s | refresh+gate {_t_gate_done - _t_train_done:.1f}s | "
              f"eval {_t_end - _t_gate_done:.1f}s | total {_t_end - _t_epoch_start:.1f}s", flush=True)

    final_path = f"model/model_MCTS_ep{epoch + 1}_final{_SUFFIX}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), final_path)
    print(f"MCTS Model saved to {checkpoint_path} and {final_path}")


def load_model(model, path="model/model_ep2.pth"):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"Loaded model checkpoint from {path}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Starting fresh.")
    else:
        print(f"No checkpoint found at {path}. Starting fresh.")
# load_model(model, "model/model_MCTS_ep70.pth")

if __name__ == '__main__':
    train_mcts_alpha_zero(num_epochs=500, num_self_play_games=10, num_simulations=30, batch_size=512)
