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


def get_open_three_injection_samples(n_examples=120, sample_weight=1.0):
    """Per-epoch draw of open-three + open-3-3-fork (+ diagnosed-position)
    examples for the regular replay buffer, same pattern as
    get_hard_example_samples right below this.

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
        # generate_fork_examples() (the open-3-3 / double-open-
        # three geometry -- see fork_geometries) added to this SAME
        # continuous pool. It was already implemented and already used to
        # get delivered via tactical_refresh()'s burst fine-tune, but that
        # call was removed from the main loop this session (see the
        # tactical_refresh_interval guard's comment) and nothing replaced
        # it for this specific curriculum the way open-three's own got
        # migrated here -- so it silently went from
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
    the same "4-of-5+gap" pattern tactical_windows() defines, appends any
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
    (classify_hard_examples / relabel_hard_examples) writing to the
    separate HARD_EXAMPLE_POOL_PATH -- see the block of comments/functions
    just above this one for the full design. Kept in the same function
    (rather than a second refresh call) so the expensive game-generation
    itself is only paid once per tactical_refresh cycle.

    runner (2026-09-27): optional selfplay_pool task runner. When
    given, the mining games (+ their per-game hard-example classification)
    and the relabel searches run in the persistent multi-process game pool
    instead of threads in this process; any runner failure falls back to
    the original threaded/sequential code below for that step.
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

    # lr-fix restart: 5e-4 -> 5e-5. This was tuned
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
