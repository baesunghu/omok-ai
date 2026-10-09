"""
Thin delegating shim to the C++ port (deterministic_play_core.pyd), which
releases the GIL for the whole search (see deterministic_play_core.cpp) --
the original pure-Python play()/_minimax() held the GIL for its entire
computation, serializing self-play worker threads during vs-Deterministic
game generation. Verified behaviorally identical to the original Python
implementation across 585 move-for-move comparisons (0 mismatches) before
this swap; original kept at determinisic_play.py.bak_before_cpp_deploy.
Public API (play(board, turn, max_depth)) and MAX_DEPTH are preserved
unchanged so no call site elsewhere in the codebase needs to change.
"""
import deterministic_play_core

MAX_DEPTH = 2

play = deterministic_play_core.play
