"""Settings and state every part of the NAPS trainer shares: paths of the run's files (model/*), the report
logger, devices, simulation-budget and LR-cooldown settings, value targets, replay buffer and epoch counter.
"""
import numpy as np
import os
import pickle
import torch
from collections import deque
from datetime import datetime

from mcts_cpp import BOARD_SIZE, NUM_CELLS
from nets import GomokuNetV3 as GomokuNet
# (until 2026-11 the trainer was one file, alphazero_mcts_make_lrfix.py; see naps_trainer.py)

# Gen12: swapped to the V2 architecture (GroupNorm+SE, see
# nets.py) -- aliased as GomokuNet so every
# existing instantiation site below (model/best_model/league_model/
# gen_model) picks it up without individual edits. NOT weight-compatible
# with the old nets.GomokuNetV1 checkpoints -- this run
# starts from the freshly-trained gen12_full_FINAL.pth, not an old one.
# V3 = V2 + auxiliary survival (remaining-plies) head, see
# nets.py. Loads V2 checkpoints; forward() unchanged.
SURVIVAL_LOSS_WEIGHT = 0.25

model = GomokuNet()

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
# 24,000 -> 196,608 (~40 epochs of data instead of ~5). Training
# no longer sweeps the whole buffer: each epoch draws TRAIN_SAMPLES_PER_EPOCH at random.
REPLAY_BUFFER_MAX_SIZE = 196_608
TRAIN_SAMPLES_PER_EPOCH = 24_000
# History: was 50_000, cut to 8_000 -- ~66-generation-stale MCTS targets
# plateaued Policy Loss (see the 50k->8k comment this replaced, copied from
# the 5x5 sibling file). Raised 8_000->12_000, mirroring the
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
SIM_RAMP_CEILING = 1600  # 1100 -> 1600 (deeper full searches = better policy targets)
BOOST_RAMP_START = 300
BOOST_RAMP_STEP = 200
BOOST_RAMP_CEILING = 2200
# If the gate keeps 'kept'-ing (no promotion) for this many checks in a row,
# bump the sim budget anyway by the same step -- otherwise a long stagnation
# streak leaves self-play stuck at a low budget indefinitely, since normally
# the ramp only advances on an actual promotion.
STAGNATION_RAMP_LIMIT = 5
# A blind calendar-based LR warm-restart (CosineAnnealingWarmRestarts
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
    # file -- this exact failure (empty train_state file ->
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
