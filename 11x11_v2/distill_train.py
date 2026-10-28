"""Small-net distillation, round 2 (11x11_v2, 2026-10-21, user: "more and better data").

Trains small nets of the main architecture on model/distill_data.npz (distill_data_gen.py: ~1M
distinct teacher-play positions labelled with the teacher's raw policy and value). Held-out test set =
whole games (last 5% of game ids). A random one of the 8 board symmetries is applied to every batch
(planes and policy target together). Reports, per size:
  top1   share of test positions where the student's best move = the teacher's
  KL     policy KL(teacher || student)
  v_corr correlation of student and teacher values
  v_rel  value MSE / variance of the teacher's values (1.0 = no better than predicting the mean)
  v_sign sign agreement on positions the teacher calls clearly (|v| > 0.2)
  rows/s forward throughput on this GPU (batch 256)
Usage: DISTILL_SIZES=128x6,96x14 python distill_train.py [samples_per_model]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import mcts_core
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3 as Net
from fast_board import BOARD_SIZE as N, WIN_LENGTH

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000
SIZES = [tuple(int(v) for v in t.split("x"))
         for t in os.environ.get("DISTILL_SIZES", "64x6,128x6,128x10,64x16,96x14").split(",")]
BS = 256
dev = torch.device("cuda")
torch.manual_seed(0)

d = np.load(os.path.join(HERE, "model", "distill_data.npz"))
boards, player = d["boards"], d["player"]
t0 = time.time()
planes = np.empty((len(boards), 5, N, N), np.uint8)
for i in range(len(boards)):
    planes[i] = mcts_core.encode_planes(boards[i].reshape(N, N), "X" if player[i] == 1 else "O", N, WIN_LENGTH)
print(f"{len(boards):,} positions, planes built in {time.time() - t0:.0f}s", flush=True)

game = d["game"]
test = game >= np.quantile(game, 0.95)
P = torch.from_numpy(planes).to(dev)
LP = torch.from_numpy(d["logp"]).to(dev)
V = torch.from_numpy(d["value"]).to(dev)
tr_idx = torch.from_numpy(np.flatnonzero(~test)).to(dev)
te_idx = torch.from_numpy(np.flatnonzero(test)).to(dev)
print(f"train {len(tr_idx):,} / test {len(te_idx):,} positions (test = whole held-out games)", flush=True)


def sym(x, k, flip):
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def outputs(m, idx):
    Ps, Vs = [], []
    with torch.inference_mode():
        for i in range(0, len(idx), 1024):
            p, v = m(P[idx[i:i + 1024]].float())
            Ps.append(p.float()); Vs.append(v.float().view(-1))
    return torch.cat(Ps), torch.cat(Vs)


def throughput(m, secs=2.0):
    x = torch.rand(BS, 5, N, N, device=dev)
    with torch.inference_mode():
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        t, n = time.time(), 0
        while time.time() - t < secs:
            p, v = m(x); p.cpu(); n += 1
    return n * BS / (time.time() - t)


lt, vt = LP[te_idx].float(), V[te_idx].float()
for ch, nb in SIZES:
    m = Net(num_channels=ch, num_res_blocks=nb, value_hidden=max(64, ch)).to(dev)
    steps = SAMPLES // BS
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    m.train()
    for s in range(steps):
        idx = tr_idx[torch.randint(len(tr_idx), (BS,), device=dev)]
        k, flip = int(torch.randint(4, ())), bool(torch.randint(2, ()))
        x = sym(P[idx].float(), k, flip)
        tlp = sym(LP[idx].float().view(-1, N, N), k, flip).reshape(-1, N * N)
        p, v = m(x)
        loss = (tlp.exp() * (tlp - p)).sum(1).mean() + F.mse_loss(v.view(-1), V[idx].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    sp, sv = outputs(m, te_idx)
    clear = vt.abs() > 0.2
    r = dict(size=f"{ch}x{nb}", params=sum(p.numel() for p in m.parameters()),
             top1=(sp.argmax(1) == lt.argmax(1)).float().mean().item(),
             kl=(lt.exp() * (lt - sp)).sum(1).mean().item(),
             v_corr=torch.corrcoef(torch.stack([sv, vt]))[0, 1].item(),
             v_rel=(((sv - vt) ** 2).mean() / vt.var()).item(),
             v_sign=((sv[clear] > 0) == (vt[clear] > 0)).float().mean().item(),
             rows=throughput(m), mins=(time.time() - t0) / 60)
    print(f"{r['size']:>7} {r['params']:>10,} params | top1 {r['top1']:.3f} KL {r['kl']:.3f} | v_corr {r['v_corr']:.3f} "
          f"v_rel {r['v_rel']:.3f} v_sign {r['v_sign']:.3f} | {r['rows']:.0f} rows/s | {r['mins']:.1f} min", flush=True)
    torch.save(m.state_dict(), os.path.join(HERE, "model", f"small_distill2_{ch}x{nb}.pth"))
