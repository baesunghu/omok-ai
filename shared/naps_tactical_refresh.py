"""Tactical refresh: a few epochs of supervised fine-tuning on tactical positions (synthetic ones from
naps_tactics, real ones mined from games), plus the per-epoch open-three injection.
"""
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from mcts_cpp import BOARD_SIZE, NUM_CELLS
from naps_common import FILE_SUFFIX
from naps_tactics import (
    fork_geometries,
    generate_closed_three_examples,
    generate_diagnosed_positions_examples,
    generate_double_four_examples,
    generate_fork_examples,
    generate_open_three_examples,
    generate_tactical_examples,
)
from naps_league import pick_league_opponent
from naps_vsdet import sample_vs_det_max_depth
from naps_openings import tactical_windows
from board_encoding import augment_sample, board_to_input_array, board_to_input_tensor
from naps_hard_examples import (
    RELABEL_SIMS, classify_hard_examples, load_hard_example_pool, mine_one_selfplay_game, mine_one_vsdet_game,
    mine_one_vsleague_game, order_relabel_candidates, relabel_hard_examples, save_hard_example_pool,
)


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
MINED_TACTICAL_EXAMPLES_PATH = f"model/mined_tactical_examples{FILE_SUFFIX}.pkl"


def _multi_label_targets_from_rows(rows_by_key):
    """One training target per unique position. rows_by_key maps a board(+player) key to the ((gap_r, gap_c),
    mover_completes) rows found on it; a position often matches several tactical windows at once.

    Why: one one-hot row per matching window gave the same input different "correct" answers. On the mined pool 27.1%
    of positions had two or more conflicting labels, capping reachable argmax accuracy at 64.6% -- likely why
    tactical_refresh's accuracy sat at ~80% (98% synthetic, ~65% real, mixed 50/50).

    Now: if any matching window is a win for the mover, the target covers only the winning cells (a non-winning move
    is wrong when a win exists); otherwise it is spread evenly over every distinct defensive gap cell, all of which
    are urgent. Yields (key, target_vec[NUM_CELLS] summing to 1, has_winning_cell)."""
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
    """Loads the real mined examples ((board, current_player, (gap_r, gap_c), mover_completes) tuples) as tensors,
    once, and caches them. Unlike the synthetic boards (always from X's side), each keeps its real current_player;
    the encoding is relative to the mover, so no colour flipping is needed.

    No 8-way symmetry augmentation here: that version stalled the live process for 2+ hours at the third refresh
    (ep63), most likely GPU-memory fragmentation from rebuilding an 8x larger (150K+) tensor cache every refresh. A
    safe version would encode with board_to_input_array (CPU numpy) and avoid rebuilding from scratch.

    Returns empty tensors (synthetic-only training) if the pickle is missing, so a fresh checkout still runs."""
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


def get_open_three_injection_samples(n_examples=120, sample_weight=1.0):
    """Per-epoch draw of open-three, open-3-3-fork and diagnosed-position examples into the regular replay buffer
    (like get_hard_example_samples).

    This replaced tactical_refresh's periodic burst as the way this curriculum is delivered: one burst dropped the
    vs-Det win rate 65% -> 25% (one noisy 20-game sample, but matching the loss spikes after ep22/43/64), and the
    ~100% tactical accuracy right after a burst had decayed to ~70% two epochs later. A smaller share in every epoch
    has neither the spike nor the decay.

    n_examples=120 positions x up to 8 symmetries ~ 960 samples per epoch -- more than the hard-example pool's ~368,
    since this is now the only way the open-three curriculum is delivered.
    """
    global _open_three_injection_cache
    if _open_three_injection_cache is None:
        # generate_fork_examples (the open 3-3 / double-open-three shape) goes into this same per-epoch pool.
        # It used to be delivered only by tactical_refresh's burst; when that was removed nothing took its place,
        # and a held-out T-shape probe showed real forgetting over the next ~200 epochs: the correct denial cell
        # fell from policy rank 3/121 at the Gen12 reset to 47/121, and the search went from solving it at epoch 0
        # to failing on every later checkpoint. No new examples -- an existing generator reconnected.
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
    Re-mine the tactical pools from fresh games, once per tactical_refresh cycle (right before tactical_refresh),
    so they follow the model's current play instead of the epoch they were first mined at.

    Plays num_selfplay_games + num_vsdet_games (+ league) games with the current model at a modest 150 sims (realistic
    stone patterns matter more than strength), then from the same games:
      * appends every position with the "4-of-5 plus a gap" pattern (tactical_windows) to the mined pool at
        MINED_TACTICAL_EXAMPLES_PATH, capped at `cap` (oldest dropped first), and clears the tensor cache so the next
        tactical_refresh uses the new pool;
      * runs the outcome-based hard-example pass (classify_hard_examples, relabel) into HARD_EXAMPLE_POOL_PATH -- in
        the same function so the games are paid for once.

    runner: the selfplay_pool task runner. When given, the games (with their per-game classification) and the
    relabel searches run in the worker pool; any runner failure falls back to the threaded code for that step.
    """
    global _mined_tactical_tensors_cache

    was_training = model.training
    model.eval()
    windows = tactical_windows()
    forks = fork_geometries()
    games = []  # (plies, outcome) per game, for the hard-example pass
    games_played = 0
    flagged = None
    # Opponent picks / depth draws happen here, once, for either path.
    mine_depths = [sample_vs_det_max_depth(epoch) for _ in range(num_vsdet_games)]
    mine_league_model = None
    if num_vsleague_games > 0:
        # deep_history_prob=0.0: mining wants the model's CURRENT blind
        # spots (see docstring above), so this pick stays biased toward
        # recent/strong opponents only.
        _mine_pick = pick_league_opponent(device, league_pool_size, league_best_checkpoint_prob,
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
                futures = [pool.submit(mine_one_selfplay_game, model, device, sims)
                           for _ in range(num_selfplay_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_selfplay_games
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(mine_one_vsdet_game, model, device, sims, (g % 2 == 0), mine_depths[g])
                           for g in range(num_vsdet_games)]
                for fut in futures:
                    games.append(fut.result())
            games_played += num_vsdet_games
            if mine_league_model is not None:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(mine_one_vsleague_game, model, mine_league_model, device, sims, (g % 2 == 0))
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
        flagged = classify_hard_examples(games, windows, forks)
    relabeled = None
    if runner is not None:
        try:
            relabeled = runner.relabel(model, device, order_relabel_candidates(flagged), RELABEL_SIMS)
        except Exception as e:  # noqa: BLE001
            print(f"  [refresh] unified-pool relabel failed ({e!r}) -- sequential fallback", flush=True)
            relabeled = None
    if relabeled is None:
        relabeled = relabel_hard_examples(flagged, model, device)
    hard_pool = load_hard_example_pool()
    hard_pool = save_hard_example_pool(hard_pool + relabeled)
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
    This refresh's tactical training set: ALL real mined examples plus an equal-size random sample of the synthetic
    pool, drawn anew on every call. Of five mixes tested on held-out data, only this one raised held-out REAL argmax
    accuracy (0.51 -> 0.69; re-checked with another seed and split: 0.31 -> 0.62) without hurting held-out synthetic
    accuracy (within ~0.2 points). Plain concatenation and upsampling the real examples were both WORSE than
    synthetic-only on real positions -- do not "simplify" this to a concatenation. Re-drawing the synthetic half each
    call keeps every refresh 50/50 while still covering the whole synthetic pool over many refreshes.
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
    A few epochs of supervised fine-tuning on the tactical curriculum, called every N epochs so this signal is not
    diluted away by the much larger volume of self-play samples in between.

    It uses its own Adam, not the main loop's: with a shared optimizer, 8 supervised epochs on another objective
    rewrote the Adam moment estimates and every refresh was followed by a 2-3 epoch loss spike (ep22: policy
    0.98 -> 2.04, value 0.06 -> 0.26; ep43 and ep64 the same) -- about 14% of all epochs spent recovering.

    Policy loss covers the whole batch; value loss only the rows with a value label (NaN = none, e.g. a pure block,
    which proves no value). The value target is 0.97, not 1.0, to stay off tanh's flat tail.
    """
    inputs, targets, value_targets = _get_tactical_tensors(device)
    inputs, targets, value_targets = inputs.to(device), targets.to(device), value_targets.to(device)
    n = inputs.shape[0]

    # lr-fix restart: 5e-4 -> 5e-5. This was tuned
    # against the OLD main-loop lr=1e-3 (a 0.5x ratio); the main loop is now
    # 1e-4 (the lr-fix restart), and tactical_refresh_interval dropped
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
