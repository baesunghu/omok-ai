"""Reference VCF results from the C++ engine (mcts_core.vcf_move after set_vcf(10, 3000), as in head2head.py)
for checking the browser VCF in web/site/app.js. Writes web/site/vcf_check.json (dev only, not deployed).
Usage: python web/make_vcf_check.py   (run from 11x11_v2)
"""
import json
import os
import random
import sys

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2)
os.chdir(V2)
import mcts_core
mcts_core.set_vcf(10, 3000)

random.seed(3)
cases = []
for k in range(400):
    board = [[" "] * 11 for _ in range(11)]
    n = random.randint(10, 40)
    for i, c in enumerate(random.sample(range(121), n)):
        board[c // 11][c % 11] = "X" if i % 2 == 0 else "O"
    who = "X" if k % 2 == 0 else "O"
    move, plies = mcts_core.vcf_move(board, who, 11, 5)
    cases.append({"board": [0 if v == " " else (1 if v == "X" else 2) for row in board for v in row],
                  "who": 1 if who == "X" else 2, "move": move, "plies": plies})
json.dump(cases, open("web/site/vcf_check.json", "w"))
print(len(cases), "cases;", sum(c["move"] >= 0 for c in cases), "with a C++ VCF")
