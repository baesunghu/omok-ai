"""Experiment arms of the CATFISH controlled experiment (2026-10-31 ~ 11-02), all run by the one trainer v3/train_v3.py.

    python v3/train_v3.py B_catfish_s2          (or NAPS_ARM=B_catfish_s2 python v3/train_v3.py)

An arm runs in v3/ab_scratch/<arm>/ (model/, training_report_v3.txt, STOP) and differs from the v3 main run only in
the fields below; every other setting is train_v3.py's. Without an arm, train_v3.py is the v3 main run in v3/ itself.

Until 2026-11 each arm was a full copy of train_v3.py (kaggle/train_<arm>.py) with these lines edited by hand; the
copies were replaced by this table. The values are exactly those of the copies.

Fields
    catfish   {kind: scale}: crippled copies of the current net played each epoch (LEAGUE_GAMES games each).
              kind "value*" scales the last value Linear, "policy*" the last policy Linear; scale 0 = switched off.
              Empty = no catfish, every game is self-play.
    games     games per epoch (self-play = games - LEAGUE_GAMES x number of catfish)
    seed      torch / random / numpy seed set before the network is built (same init in both arms of a seed)
    adaptive  G arms: after every epoch each catfish scale moves toward a target score (see train_v3.adapt_catfish)

Seed 1 (arms A / B) was run on the laptop before this table existed; its trainer copies are not in the repository.
"""

V3_CATFISH = {"value0": 0.0, "value_half": 0.5, "policy_half": 0.5}   # v3 main run from ep130 (= arm B)
SEED = {2: 20261231, 3: 20261331, 4: 20261431, 5: 20261531}

ARMS = {}
for s in (2, 3, 4, 5):
    ARMS[f"A_nocatfish_s{s}"] = dict(catfish={}, games=192, seed=SEED[s])                  # control: self-play only
    ARMS[f"B_catfish_s{s}"] = dict(catfish=dict(V3_CATFISH), games=192, seed=SEED[s])     # v3 catfish setting
# 11-01 why does catfish help/hurt? (seed 2, against A_nocatfish_s2)
ARMS["E_cf192_s2"] = dict(catfish=dict(V3_CATFISH), games=216, seed=SEED[2])   # data cost: self-play kept at 192 + 24 catfish
ARMS["F_strongcat_s2"] = dict(catfish={"value_half": 0.75, "policy_half": 0.75}, games=192, seed=SEED[2])  # stronger catfish
# 11-02 G arm (user: find the conditions where catfish works): B's catfish lost 21/24 to the main net (too weak),
# so each catfish's scale adapts until the main net scores about 65% against it.
for s in (2, 3):
    ARMS[f"G_adaptcat_s{s}"] = dict(catfish=dict(V3_CATFISH), games=192, seed=SEED[s], adaptive=True)
