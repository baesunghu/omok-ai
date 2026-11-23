"""Settings of the 11x11_v2 run: NAPS v2 (restarted from NAPS-960 on 2026-10-21; VCF engine, C++ self-play).
v3 (11x11_v2/v3/) has its own trainer and settings (v3/train_v3.py, v3/arms.py) and only takes BOARD_SIZE from here.

Everything that differs between runs lives in this file; the code is shared (../shared/). How the values got there
is in docs/NAPS_lineage.md (act 4, "NAPS v2").
"""
BOARD_SIZE = 11
LABEL = "NAPS"                       # run name in phone notifications (promotion_sms.py)

# engine: this folder's built mcts_core module (the v2 build; v3/cpp/mcts_core_solver.cpp plays the same games)
ENGINE_V2 = True                     # VCF, MEGA, extra search arguments
VCF = (10, 3000)                     # (attacker-four depth, node budget) checked at every node expansion
MEGA_VISITS, MEGA_MIN_SIMS = (0, 0), 1000   # MEGA off (Rapfi 3000: VCF 60% vs VCF+MEGA 31%); was (30, 15)

# trainer: shared/naps_trainer.py, started by _restart_train.py
SELF_PLAY_GAMES = 32                 # self-play games per epoch (like 15x15)
GATE_LOSS_PENALTY = 0.75             # gate score = (wins + 0.5 draws - penalty x losses) / games
GATE_OPENING_RADIUS = None           # gate openings only within this distance of the centre (None: all)
LR_PEAK, LR_MIN = 5e-5, 1e-5         # restarted from a mature net: peak between run-2's 1e-4 and the 2e-5 polish
EVAL_BATCH = 32                      # Raise GPU use: leaves per MCTS evaluation request (was 16)
POOLED_TARGET_BATCH = 64             # GPU batch of the pooled evaluator (was 24)
CPP_SELFPLAY = True                  # self-play and league games in C++ threads (selfplay_cpp.py)

# watchers
TRAIN_SCRIPT = "_restart_train.py"   # the process watchdog_.py keeps alive
LADDER_GAMES = 40                    # games per Rapfi level (rapfi_ladder_watch.py)
LADDER_LEVELS = "adaptive"           # best equal point x (1/3, 1, 3, 10): the run started near ~10 Rapfi nodes
