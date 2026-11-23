"""Start (or resume) the 11x11_v2 run's training: `python _restart_train.py` in this folder (watchdog_.py does the same).
Settings: run_config.py. Code: ../shared/train_run.py and ../shared/alphazero_mcts_make_lrfix.py.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))   # shared/: code common to every run
from train_run import main  # noqa: E402

# The guard matters: on Windows every multiprocessing child re-imports this file.
if __name__ == "__main__":
    main()
