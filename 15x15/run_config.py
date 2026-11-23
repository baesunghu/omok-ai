"""Settings of the 15x15 run (2026-10-13 ~): epoch 0 from Rapfi-labelled supervised data, then self-play only.

Everything that differs between runs lives in this file; the code is shared (../shared/). How the values got there
is in docs/NAPS_lineage.md (act 4, "15x15") and docs/archive/15x15_migration_plan.md.
"""
BOARD_SIZE = 15
LABEL = "15x15"                      # run name in phone notifications (promotion_sms.py)

# engine: this folder's built mcts_core module (C++ source cpp_mcts/mcts_core_solver.cpp)
ENGINE_V2 = False                    # True only for the 11x11_v2 build (VCF, MEGA, extra search arguments)
VCF = None                           # (attacker-four depth, node budget) of the VCF check; v2 engine only
MEGA_VISITS, MEGA_MIN_SIMS = (0, 0), 1000   # MEGA stage-1 root visits; v2 engine only

# trainer: shared/alphazero_mcts_make_lrfix.py, started by _restart_train15.py
SELF_PLAY_GAMES = 32                 # self-play games per epoch
GATE_LOSS_PENALTY = 0.4              # 2026-10-16 (user, ep93): 0.75 -> 0.4 (no draws on 15x15; 11-9 passes, 10-10 does not)
GATE_OPENING_RADIUS = 3              # 2026-10-13: canonical openings include far corners on 15x15; keep the gate near the centre
LR_PEAK, LR_MIN = 1e-4, 2e-5         # 2026-10-15 (user): the 2e-5 polish band of the mature 11x11 net kept the young 15x15 net flat
EVAL_BATCH = 16                      # leaves per MCTS evaluation request
POOLED_TARGET_BATCH = 24             # GPU batch of the pooled evaluator
CPP_SELFPLAY = False                 # self-play and league games in C++ threads (11x11_v2/selfplay_cpp.py)

# watchers
TRAIN_SCRIPT = "_restart_train15.py"   # 2026-10-14: a name distinct from the 11x11 trainer on the same laptop
LADDER_GAMES = 40                    # 2026-10-14 (user): 20 -> 40 per level for a tighter Elo estimate
LADDER_LEVELS = "30,100,300,1000,3000"   # 2026-10-16: run 3 starts from a Rapfi-10 epoch 0
