"""Settings of the 10x10 run: 11x11 NAPS (the folder name is historical; the board is 11x11).

Everything that differs between runs lives in this file; the code is shared (../shared/). The values are the ones
the run ended with (NAPS-960, 2026-10-13). How they got there is in docs/NAPS_lineage.md.
"""
BOARD_SIZE = 11
LABEL = "NAPS"                       # run name in phone notifications (promotion_sms.py)

# engine: this folder's built mcts_core module (C++ source cpp_mcts/mcts_core_solver.cpp)
ENGINE_V2 = False                    # True only for the 11x11_v2 build (VCF, MEGA, extra search arguments)
VCF = None                           # (attacker-four depth, node budget) of the VCF check; v2 engine only
MEGA_VISITS, MEGA_MIN_SIMS = (0, 0), 1000   # MEGA stage-1 root visits; v2 engine only

# trainer: shared/alphazero_mcts_make_lrfix.py, started by _restart_train.py
SELF_PLAY_GAMES = 16                 # self-play games per epoch
GATE_LOSS_PENALTY = 0.75             # gate score = (wins + 0.5 draws - penalty x losses) / games
GATE_OPENING_RADIUS = None           # gate openings only within this distance of the centre (None: all)
LR_PEAK, LR_MIN = 2e-5, 1e-5         # cosine warm restarts (T_0 = 150 epochs)
EVAL_BATCH = 16                      # leaves per MCTS evaluation request
POOLED_TARGET_BATCH = 24             # GPU batch of the pooled evaluator
CPP_SELFPLAY = False                 # self-play and league games in C++ threads (11x11_v2/selfplay_cpp.py)

# watchers
TRAIN_SCRIPT = "_restart_train.py"   # the process watchdog_.py keeps alive
LADDER_GAMES = 20                    # games per Rapfi level (rapfi_ladder_watch.py)
LADDER_LEVELS = "1000,2000,3000,5000,10000"   # Rapfi node levels, or "adaptive" (follow the best's equal point)
