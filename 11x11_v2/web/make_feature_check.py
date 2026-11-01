"""Reference input planes for checking the browser feature encoder (web/site/app.js planes()) against
alphago_lee.board_to_input_array_from_numeric. Writes web/site/feature_check.json (dev only, not needed by the demo).
Usage: python web/make_feature_check.py   (run from 11x11_v2)
"""
import json
import os
import sys

import numpy as np

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)
os.chdir(V2)
from alphago_lee import board_to_input_array_from_numeric

rng = np.random.default_rng(1)
cases = []
for k in range(300):
    b = np.zeros((11, 11), np.int8)
    n = int(rng.integers(4, 60))
    if k % 3 == 0:   # dense local clusters make fours and forks common
        r0, c0 = rng.integers(2, 9, size=2)
        cells = [(int(r0 + dr), int(c0 + dc)) for dr in range(-2, 3) for dc in range(-2, 3)]
        rng.shuffle(cells)
        for i, (r, c) in enumerate(cells[:n]):
            if 0 <= r < 11 and 0 <= c < 11:
                b[r, c] = 1 + i % 2
    else:
        for i, c in enumerate(rng.choice(121, n, replace=False)):
            b.flat[c] = 1 + i % 2
    who = "X" if k % 2 == 0 else "O"
    x = board_to_input_array_from_numeric(b, who)
    cases.append({"board": b.reshape(-1).tolist(), "to_move": 1 if who == "X" else 2, "planes": x.reshape(-1).astype(int).tolist()})
json.dump(cases, open("web/site/feature_check.json", "w"))
print(len(cases), "cases; forks present in", sum(any(c["planes"][363:]) for c in cases))
