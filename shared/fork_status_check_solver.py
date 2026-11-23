"""fork_status_check.py with the solver engine (mcts_core_solver) instead of the production mcts_core.

    cd 11x11_naps      (or 11x11_v2 / 15x15)
    python ../shared/fork_status_check_solver.py
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fork_status_check.py"),
               init_globals={"ENGINE_MODULE": "mcts_core_solver"}, run_name="__main__")
