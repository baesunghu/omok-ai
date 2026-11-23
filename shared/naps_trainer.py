"""The NAPS training loop (train_mcts_alpha_zero): generate games, fill the replay buffer, train, gate, measure.
Started by shared/train_run.py (each run folder's _restart_train.py). Until 2026-11 this file also held
everything now in naps_common / naps_openings / naps_selfplay / naps_gate / naps_vsdet / naps_league /
naps_tactics / naps_hard_examples.
"""
# The self-play pools (selfplay_pool_fallback.py, selfplay_pool.py) import this module by name and use
# prod._play_one_mcts_game, prod._OPENING_SQUARES, ... -- those names are imported here for them too.
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from run_config import EVAL_BATCH, LR_MIN, LR_PEAK, POOLED_TARGET_BATCH
from nets import GomokuNetV3 as GomokuNet
from mcts_cpp import PooledEvaluator
from selfplay_pool import generate_epoch_games_unified, get_task_runner
from selfplay_pool_fallback import generate_mcts_self_play_data_mp
from board_encoding import preferred_memory_format
from naps_common import (
    BOOST_RAMP_CEILING, BOOST_RAMP_STEP, LR_COOLDOWN_EPOCHS, SIM_RAMP_CEILING, SIM_RAMP_STEP, STAGNATION_RAMP_LIMIT,
    SURVIVAL_LOSS_WEIGHT, TRAIN_SAMPLES_PER_EPOCH, TRAIN_STATE_PATH, _SUFFIX, annealed_value, compute_policy_entropy,
    get_devices, load_replay_buffer, logger, load_sim_budget, load_start_epoch, model, report_logger,
    save_cumulative_epoch, save_replay_buffer, save_sim_budget,
)
from naps_openings import _ALL_CELLS
from naps_gate import BEST_CHECKPOINT_PATH, _GATE_OPENING_SQUARES, _LAST_GATE_CANDIDATE, run_gating_check
from naps_vsdet import evaluate_vs_deterministic, format_by_depth, generate_mcts_vs_deterministic_data
from naps_league import (
    LEAGUE_NUM_OPPONENTS, _pick_league_opponent, _pick_league_opponents_multi, generate_mcts_vs_league_data,
)
from naps_hard_examples import _get_fork_mistake_samples, _get_hard_example_samples
from naps_tactical_refresh import _get_open_three_injection_samples, refresh_mined_tactical_pool
# for the self-play pools, which use them as prod.<name> (see the note at the top)
from naps_common import (  # noqa: F401
    mixed_value_target,
)
from naps_openings import (  # noqa: F401
    _OPENING_SQUARES, _sample_forced_opening_start, _sample_tactical_start, _tactical_windows,
)
from naps_selfplay import (  # noqa: F401
    _play_one_mcts_game, generate_mcts_self_play_data,
)
from naps_gate import (  # noqa: F401
    RECORD_LEAGUE_SIDE, _play_gate_match_game,
)
from naps_vsdet import (  # noqa: F401
    _play_one_mcts_vs_deterministic_game, _sample_vs_det_max_depth,
)
from naps_league import (  # noqa: F401
    _play_one_mcts_vs_league_game,
)
from naps_tactics import (  # noqa: F401
    _fork_geometries,
)
from naps_hard_examples import (
    _classify_hard_examples, _mine_one_selfplay_game, _mine_one_vsdet_game, _mine_one_vsleague_game, _relabel_one,
)


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
    # PCR for vs-Det/vs-League games, kept separate so PCR can
    # be applied to self-play only. None = same as full_search_prob (the
    # previous behavior, where one value was passed to all three).
    full_search_prob_other=None,
    # Play self-play + vs-Det + vs-League games in ONE
    # multi-process pool (selfplay_pool) instead of three sequential
    # phases. False = the previous sequential path (which is also the
    # automatic fallback whenever the unified pool fails).
    use_unified_game_pool=False,
    # Value target = (1-w)*z + w*q for generated games (see
    # mixed_value_target). 0.0 = pure game result (previous behavior).
    value_target_q_weight=0.0,
    # half-life decay of the q weight. From epoch
    # value_target_q_decay_start on, w(epoch) = value_target_q_weight *
    # 0.5 ** ((epoch - start) / value_target_q_halflife); below
    # value_target_q_floor it becomes exactly 0 (pure z). halflife 0 = off.
    value_target_q_halflife=0,
    value_target_q_decay_start=0,
    value_target_q_floor=0.01,
    # Cap self-play / league game length (plies); None = no cap.
    max_game_plies=None,
    # vs-Deterministic games inside the refresh-epoch MINING pass
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
    # lr-fix restart: base_lr 0.001->0.0001 and T_0 450->150.
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
    # Gate stuck 15 checks (650-720) while losses kept falling ->
    # halve the LR band (1e-4..5e-5 -> 5e-5..1e-5) to consolidate instead of drifting.
    # ep937 cooldown ended -> LR jumped 1e-5 -> 4.4e-5 and ep940 fell to ~1,092 Rapfi nodes.
    # Cosine peak 5e-5 -> 2e-5 (band 2e-5..1e-5) to polish rather than shake before the festival.
    # LR band per run (run_config.py): 11x11 NAPS 2e-5..1e-5, 11x11_v2 5e-5..1e-5, 15x15 1e-4..2e-5
    # (the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat).
    optimizer = optim.Adam(model.parameters(), lr=LR_PEAK, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=1, eta_min=LR_MIN)
    start_epoch = load_start_epoch()
    for _ in range(start_epoch):
        scheduler.step()
    # Restore Adam's moments across restarts. Every restart used to reset them
    # (loss spikes at 905/911/913/945). Loaded only if saved right after the epoch we resume at;
    # the LR stays whatever the schedule/cooldown says now (the saved param_groups LR is discarded).
    OPT_STATE_PATH = f"model/optimizer_state{_SUFFIX}.pth"
    # A fresh Adam moves every weight by ~LR on its first steps (bias-corrected m/sqrt(v) ~ sign(g)),
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
    gen_model = GomokuNet()
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
    # target_batch=24, max_wait=0.05 (was 48/0.005): live stats
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
    pooled_evaluator = PooledEvaluator(gen_model, self_play_device, target_batch=POOLED_TARGET_BATCH, max_wait=0.05) \
        if self_play_device.type == "cuda" else None

    cur_num_simulations, cur_boosted_sims, cur_stagnation_count, cur_lr_cooldown = load_sim_budget()

    print(f"\n--- Starting MCTS AlphaZero Model Training ({num_epochs} Epochs) ---")
    if start_epoch > 0:
        print(f"Resuming cumulative epoch/schedule state at epoch {start_epoch} (from {TRAIN_STATE_PATH})")
    print(f"Self-play/vs-Det/vs-League simulation budget: {cur_num_simulations} "
          f"(boosted {cur_boosted_sims}) -- ramps toward {SIM_RAMP_CEILING}/{BOOST_RAMP_CEILING} on promotion")

    # Noise-anneal safety valve: noise_anneal_epochs was just
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
        # Second promotion path -- Rapfi ladder. The gate (vs the current best only)
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
        # board_encoding.py), on CPU it stays channels_last as before.
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
                    os.path.basename(p).replace("model_MCTS_", "").replace(f"{_SUFFIX}.pth", "").replace(".pth", "")
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

        # Continuous sampling from the hard-mined pool -- the
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

        # Continuous per-epoch open-three curriculum injection,
        # replacing tactical_refresh()'s periodic burst (see
        # _get_open_three_injection_samples docstring for why).
        open_three_samples = _get_open_three_injection_samples(n_examples=120)
        if open_three_samples:
            replay_buffer.extend(open_three_samples)

        # Dedicated Hard-7 (general fork) quota, on top of
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

        # Clean rollback to ep960: with an emptied buffer, only generate data until
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
        # 15x15: Adam moments restored on resume were loaded while the model sat on the CPU;
        # keep them on the training device or the first step fails (cuda:0 vs cpu).
        for _st in optimizer.state.values():
            for _k, _v in _st.items():
                if torch.is_tensor(_v) and _v.device != train_device and _k != "step":
                    _st[_k] = _v.to(train_device)
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
            # value=None samples (policy-only hard classes) carry
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

        # tactical_refresh() (the periodic burst supervised
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
            # Keep the weights of every gate where the OLD rule (W + D/2 >= 55%) would
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

        # Bugfix: this used to be called at the TOP of the loop
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
        try:   # Optimizer moments for the next restart (atomic write)
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

        # Per-stage wall-clock: games = all generation games,
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
