"""15x15 supervised pretraining from Rapfi-100-node labels (2026-10-13, user decisions):
  policy target = Rapfi's move (one-hot), value target = Rapfi's evaluation mapped to [-1, 1].

Eval mapping: v = tanh(eval / S), with S fitted so tanh(eval / S) best predicts the game result from the
side to move (z = +1 win, -1 loss, 0 draw); mate scores ("M"/"+M"/"-M") map to +-1.
Inputs use the production encoder (alphago_lee.board_to_input_array, 5 planes); each batch applies a random
dihedral symmetry. 5% of the GAMES are held out.

Writes model/model_MCTS_ep0_15x15.pth (the epoch-0 starting point) plus the training checkpoint and best
(model/alphazero_mcts_15x15.pth, model/best_mcts_15x15.pth), and model/sl_pretrain_report.json.
Usage: python sl_pretrain.py [epochs]
"""
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphago_lee import board_to_input_array
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE as N

assert N == 15, N
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BATCH = 256
LR = 1e-3
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def parse_eval(e):
    if e is None:
        return None
    if "M" in e.upper():
        return 1e9 if not e.startswith("-") else -1e9
    try:
        return float(e)
    except ValueError:
        return None


# ---- load games -> (moves prefix, player to move, target cell, eval, result for player to move)
games = []
for fn in sorted(glob.glob(os.environ.get("SL_GLOB", "sl_data/rapfi100_w*.jsonl"))):
    for line in open(fn, encoding="utf-8"):
        try:
            games.append(json.loads(line))
        except ValueError:
            pass
random.shuffle(games)
n_hold = max(1, len(games) // 20)
splits = {"held": games[:n_hold], "train": games[n_hold:]}
print(f"games: {len(games)} (held-out {n_hold})", flush=True)


def build(gs):
    X, P, E, Z = [], [], [], []
    for g in gs:
        moves = [tuple(m) for m in g["moves"]]
        for ply, r, c, ev in g["labels"]:
            b = [[' '] * N for _ in range(N)]
            for p, rr, cc in moves[:ply]:
                b[rr][cc] = p
            me = 'X' if ply % 2 == 0 else 'O'
            X.append(board_to_input_array(b, me).astype(np.float32))
            P.append(r * N + c)
            E.append(parse_eval(ev))
            Z.append(0.0 if g["winner"] is None else (1.0 if g["winner"] == me else -1.0))
    return np.stack(X), np.array(P, np.int64), E, np.array(Z, np.float32)


t0 = time.time()
data = {k: build(v) for k, v in splits.items()}
print(f"positions: train {len(data['train'][1])}, held-out {len(data['held'][1])} ({time.time() - t0:.0f}s)", flush=True)

# ---- fit eval scale S on the training split
ev_tr = np.array([e if e is not None else 0.0 for e in data["train"][2]], np.float64)
z_tr = data["train"][3]
best_s, best_mse = None, 9.0
for s in [50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]:
    mse = float(np.mean((np.tanh(np.clip(ev_tr / s, -50, 50)) - z_tr) ** 2))
    print(f"  eval scale {s:5d}: MSE vs game result {mse:.4f}")
    if mse < best_mse:
        best_s, best_mse = s, mse
print(f"eval scale S = {best_s} (MSE {best_mse:.4f})", flush=True)


def value_targets(evs):
    return np.array([math.tanh(max(-50.0, min(50.0, (e if e is not None else 0.0) / best_s))) for e in evs], np.float32)


V = {k: value_targets(v[2]) for k, v in data.items()}

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
net = AlphaZeroTicTacToeV3().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
Xtr, Ptr = data["train"][0], data["train"][1]
steps = EPOCHS * ((len(Ptr) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)


def sym(x, p, k, flip):
    """dihedral symmetry on input planes (B,C,N,N) and flat target indices"""
    x = torch.rot90(x, k, (2, 3))
    pm = torch.zeros(len(p), N * N, device=x.device); pm[torch.arange(len(p)), p] = 1
    pm = torch.rot90(pm.view(-1, 1, N, N), k, (2, 3))
    if flip:
        x = torch.flip(x, (3,)); pm = torch.flip(pm, (3,))
    return x, pm.reshape(-1, N * N).argmax(1)


def evaluate():
    net.eval()
    Xh, Ph, Vh = data["held"][0], data["held"][1], V["held"]
    top1 = top3 = 0; vm = 0.0; n = 0
    with torch.inference_mode():
        for i in range(0, len(Ph), 1024):
            x = torch.from_numpy(Xh[i:i + 1024]).to(dev)
            lp, v = net(x)
            legal = (x[:, 0] + x[:, 1]).view(len(x), -1) == 0
            lp = lp.masked_fill(~legal, -1e9)
            tgt = torch.from_numpy(Ph[i:i + 1024]).to(dev)
            top1 += (lp.argmax(1) == tgt).sum().item()
            top3 += (lp.topk(3, 1).indices == tgt[:, None]).any(1).sum().item()
            vm += ((v.view(-1) - torch.from_numpy(Vh[i:i + 1024]).to(dev)) ** 2).sum().item(); n += len(x)
    net.train()
    return top1 / n, top3 / n, vm / n


hist = []
t0 = time.time()
for ep in range(EPOCHS):
    perm = np.random.permutation(len(Ptr)); tot_p = tot_v = 0.0; nb = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        x = torch.from_numpy(Xtr[idx]).to(dev); p = torch.from_numpy(Ptr[idx]).to(dev)
        v = torch.from_numpy(V["train"][idx]).to(dev)
        x, p = sym(x, p, random.randint(0, 3), random.random() < 0.5)
        lp, vp = net(x)
        loss_p = F.nll_loss(lp, p); loss_v = F.mse_loss(vp.view(-1), v)
        opt.zero_grad(); (loss_p + loss_v).backward(); opt.step(); sched.step()
        tot_p += loss_p.item(); tot_v += loss_v.item(); nb += 1
    t1, t3, vm = evaluate()
    hist.append(dict(epoch=ep + 1, policy_loss=tot_p / nb, value_loss=tot_v / nb, held_top1=t1, held_top3=t3, held_value_mse=vm))
    print(f"epoch {ep + 1}/{EPOCHS} ({time.time() - t0:.0f}s): policy {tot_p / nb:.4f} value {tot_v / nb:.4f} | "
          f"held-out top1 vs Rapfi {t1:.3f} top3 {t3:.3f} value MSE {vm:.4f}", flush=True)

os.makedirs("model", exist_ok=True)
sd = {k: v.cpu() for k, v in net.state_dict().items()}
for path in ("model/model_MCTS_ep0_15x15.pth", "model/alphazero_mcts_15x15.pth", "model/best_mcts_15x15.pth"):
    torch.save(sd, path)
json.dump(dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), games=len(games), positions_train=int(len(Ptr)),
               positions_held=int(len(data["held"][1])), eval_scale=best_s, eval_scale_mse=best_mse, epochs=hist),
          open("model/sl_pretrain_report.json", "w"), indent=1)
print("saved model/model_MCTS_ep0_15x15.pth (+ training checkpoint and best)", flush=True)
