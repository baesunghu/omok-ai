"""Small-network capacity test (11x11_v2, 2026-10-21, user: "can a smaller net cover this board?").

Distills the current big net (256 ch x 7 blocks, ~8.6M params) into smaller nets of the same
architecture (GroupNorm + SE res blocks, same heads) and measures, on held-out positions:
  * top-1 agreement with the teacher's policy, policy KL(teacher || student), value MSE and sign agreement,
  * forward throughput (rows/s) on this GPU.
Positions: the last saved replay buffer of the festival run (10x10/model/mcts_replay_buffer_11x11.pkl);
the first 90% (in buffer order, so augmentations of one position stay together) train, the last 10% test.
Targets are the teacher's raw outputs (no search), so this isolates network capacity.
Measurement only: the live training run is not touched (it shares the GPU while this runs).
Usage: python small_net_capacity.py [epochs] [teacher.pth]
"""
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphago_lee_v3_survival import AlphaZeroTicTacToeV3 as Net

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TEACHER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "model", "best_mcts_11x11.pth")
BUFFER = os.path.join(HERE, "..", "10x10", "model", "mcts_replay_buffer_11x11.pkl")
SIZES = [(64, 6), (96, 6), (128, 6), (128, 10), (192, 7)]
if os.environ.get("CAP_SIZES"):   # e.g. CAP_SIZES=64x16,96x14 (thin-and-deep variants)
    SIZES = [tuple(int(v) for v in t.split("x")) for t in os.environ["CAP_SIZES"].split(",")]
dev = torch.device("cuda")
torch.manual_seed(0)


def params(m):
    return sum(p.numel() for p in m.parameters())


def throughput(m, bs=256, secs=2.0):
    x = torch.rand(bs, 5, 11, 11, device=dev)
    with torch.inference_mode():
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        t, n = time.time(), 0
        while time.time() - t < secs:
            p, v = m(x)
            p.cpu()
            n += 1
    return n * bs / (time.time() - t)


def outputs(m, X, bs=512):
    P, V = [], []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            p, v = m(X[i:i + bs])
            P.append(p.float())
            V.append(v.float().view(-1))
    return torch.cat(P), torch.cat(V)


def metrics(logp_s, v_s, logp_t, v_t):
    kl = (logp_t.exp() * (logp_t - logp_s)).sum(1).mean().item()
    top1 = (logp_s.argmax(1) == logp_t.argmax(1)).float().mean().item()
    mse = ((v_s - v_t) ** 2).mean().item()
    sign = ((v_s > 0) == (v_t > 0)).float().mean().item()
    return dict(kl=kl, top1=top1, v_mse=mse, v_sign=sign)


buf = pickle.load(open(BUFFER, "rb"))
X = torch.from_numpy(np.stack([b[0] for b in buf])).to(dev)
n_tr = int(len(X) * 0.9)
teacher = Net()
teacher.load_state_dict(torch.load(TEACHER, map_location=dev))
teacher.to(dev).eval()
TP, TV = outputs(teacher, X)
print(f"positions {len(X)} (train {n_tr}, test {len(X) - n_tr}), teacher {os.path.basename(TEACHER)} "
      f"{params(teacher):,} params, {throughput(teacher):.0f} rows/s", flush=True)

rows = []
for ch, nb in SIZES:
    m = Net(num_channels=ch, num_res_blocks=nb, value_hidden=max(64, ch)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * (n_tr // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr - 255, 256):
            idx = perm[i:i + 256]
            p, v = m(X[idx])
            loss = (TP[idx].exp() * (TP[idx] - p)).sum(1).mean() + F.mse_loss(v.view(-1), TV[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    sp, sv = outputs(m, X[n_tr:])
    r = metrics(sp, sv, TP[n_tr:], TV[n_tr:])
    r.update(size=f"{ch}x{nb}", params=params(m), rows_s=throughput(m), train_min=(time.time() - t0) / 60)
    rows.append(r)
    print(f"{r['size']:>7} {r['params']:>10,} params | top1 {r['top1']:.3f} KL {r['kl']:.3f} | "
          f"v_mse {r['v_mse']:.4f} v_sign {r['v_sign']:.3f} | {r['rows_s']:.0f} rows/s | train {r['train_min']:.1f} min",
          flush=True)
    torch.save(m.state_dict(), os.path.join(HERE, "model", f"small_distill_{ch}x{nb}.pth"))
