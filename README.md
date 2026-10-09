# NAPS: AlphaZero 11x11 Gomoku on a Weak Notebook

Private research record. To be made public together with the paper.

**Research question:** Can an AI that beats ordinary humans be trained in a harsh, resource-poor
environment? Everything here was trained on one laptop (Intel i5-8300H, GTX 1050 4GB, 16GB RAM).

**Graduated model:** NAPS-625 "여명" (DAWN = Defensive AlphaZero on a Weak Notebook),
graduated 2026-09-29 14:22:13 KST. Rapfi equal-strength point about 1,150 nodes at 1,100
simulations per move; graduation problems 3/3; about 90% wins against humans on the web board.

## Layout

| Path | Contents |
|---|---|
| `10x10/` | 11x11 project (the folder name is historical). Trainer, MCTS wrapper, self-play pool, measurement tools, web board |
| `10x10/docs/NAPS_lineage.md` | Model lineage from Gen0 to the current NAPS run |
| `10x10/tactical_problems/` | Graduation and stage-2 problems (measurement only, never training data) |
| `cpp_mcts/` | C++ MCTS engine (`mcts_core_solver.cpp`) and the depth-2 minimax baseline |
| `manual/` | Overhaul manual sources (Korean) and build scripts |

Not in this repository: model weights, replay buffers, logs, compiled binaries, Rapfi, cloudflared,
and the earlier 3x3 / 5x5 projects that live in the same working folder.

## Main entry points

- Training: `10x10/_restart_train.py` → `10x10/alphazero_mcts_make_lrfix.py`
- Engine build: `cpp_mcts/build_mcts_core_solver.bat` (MSVC + pybind11, Python 3.11)
- Measurement: `10x10/rapfi_ladder.py`, `10x10/tactical_problem_check.py`
- Web board: `10x10/hlt/web/server.py`; spectator board: `10x10/hlt/web/spectate.py`

Author: 배성훈 (Sunghun Bae)
