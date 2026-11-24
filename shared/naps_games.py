"""The game functions the self-play pool workers run, in one place: selfplay_pool_tasks, selfplay_pool,
selfplay_pool_fallback and 11x11_v2/selfplay_cpp.py import this module (as naps_games) instead of the whole trainer.
Every name here is the trainer's own object, imported from the module that defines it.
"""
from run_config import EVAL_BATCH  # noqa: F401
from naps_common import mixed_value_target  # noqa: F401
from naps_openings import OPENING_SQUARES, sample_forced_opening_start, sample_tactical_start, tactical_windows  # noqa: F401
from naps_selfplay import generate_mcts_self_play_data, play_one_mcts_game  # noqa: F401
from naps_gate import RECORD_LEAGUE_SIDE, play_gate_match_game  # noqa: F401
from naps_vsdet import format_by_depth, play_one_mcts_vs_deterministic_game, sample_vs_det_max_depth  # noqa: F401
from naps_league import play_one_mcts_vs_league_game  # noqa: F401
from naps_tactics import fork_geometries  # noqa: F401
from naps_hard_examples import (  # noqa: F401
    classify_hard_examples, mine_one_selfplay_game, mine_one_vsdet_game, mine_one_vsleague_game, relabel_one,
)
