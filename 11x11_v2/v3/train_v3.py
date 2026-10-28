"""11x11_v3 trainer (2026-10-22, user: "도박"): pure cold start, no outside data.

  * network      net_v3.NetV3, 128 ch x 8 blocks, fresh random init
  * search       C++ SelfPlayRunner with the Gumbel AlphaZero root (Danihelka et al. 2022):
                 Gumbel-top-k candidates + sequential halving, policy target
                 softmax(logit + sigma(completed q)); VCF at every expansion (as v2)
  * strong moves playout-cap randomization (KataGo): only FULL searches are recorded as
                 training rows; fast searches just move the game along
  * auxiliary    opponent's next move, final cell ownership, remaining plies (survival)

One epoch = GAMES self-play games -> 8-fold dihedral augmentation -> replay window ->
STEPS_PER_NEW x (new rows / BATCH) optimizer steps. Self-play only (no league, gate or
outside opponent); strength is measured separately on the Rapfi ladder.
Stop: create the file STOP next to this script (the epoch in progress finishes first).
"""
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cpp"))       # v3 mcts_core (Gumbel) before v2's
sys.path.insert(1, os.path.dirname(HERE))
os.chdir(HERE)

import numpy as np
import torch
import torch.nn.functional as F

import copy

import mcts_core
from net_v3 import NetV3
from alphago_lee_v3_survival import SURVIVAL_CAP

assert hasattr(mcts_core.SelfPlayRunner(), "gumbel_m_full"), "wrong mcts_core (needs the v3 Gumbel build)"

N = 11
NC = N * N
mcts_core.set_vcf(10, 3000)

GAMES = 192              # self-play games per epoch
# 2026-10-22 21:00 (user): CATFISH league -- crippled copies of the CURRENT net, rebuilt every epoch:
# value0 (last value Linear zeroed), value_half (x0.5), policy0 (uniform prior). The catfish side
# searches with plain PUCT and its moves are not recorded; only the main net's full-search moves are.
# 2026-10-23 02:10 (user): policy_half added (policy logits x0.5 = flatter prior) to poke policy blind spots;
# tier-3 failures at ep120 were policy blind spots, not value overconfidence.
# Same time (user): policy0 dropped -- the main net beat it 95-97% (ep79-128), too weak to teach anything.
# 3 x 16 = 48 league games, 144 self-play (same split as before).
CATFISH = ("value0", "value_half", "policy_half")
LEAGUE_GAMES = 8         # 2026-10-23 03:xx (user): 16 -> 8 per catfish (league 25% -> 12.5%); ep120 lost to ep100 40% (4/8/8), suspected catfish over-fit
SELF_GAMES = GAMES - LEAGUE_GAMES * len(CATFISH)
SIMS_FULL = 400          # Gumbel full search (recorded). 2026-10-23 03h (user): 200 -> 400, stronger policy targets to win back the policy
SIMS_FAST = 48           # fast search (not recorded)
FULL_PROB = 0.25         # playout-cap randomization
GUMBEL_M_FULL, GUMBEL_M_FAST = 32, 16   # 2026-10-23 (user): 16/8 -> 32/16, low-prior moves reach the candidate set more often
# 2026-10-23 (user): exploration for self-play ("창발적 수"): random opening stones, early-move temperature
# (sampled from the Gumbel improved policy), Dirichlet noise on the Gumbel candidate sampling.
OPEN_FRAC = 0.25         # share of self-play games that start from a random opening (04:55 user: 0.5 -> 0.25, black-win drift)
OPEN_STONES = (2, 4)     # random stones placed (alternating X, O) within OPEN_RADIUS of the centre
OPEN_RADIUS = 3
TEMP_MOVES = 4           # first moves of every self-play game sampled from the improved policy
DIR_ALPHA, DIR_WEIGHT = 0.15, 0.25
LAST_STARTS = {}         # game index -> pre-placed stones {cell: 1/2}, for the ownership target
N_THREADS, TARGET_ROWS, MAX_WAIT_MS = 64, 512, 4.0
WINDOW = 300_000         # augmented rows kept
BATCH = 256
STEPS_PER_NEW = 2.0      # each new row is seen ~2x on average
LR, WD = 3e-4, 1e-4    # 2026-10-23 07:55 (user): 1e-3 -> 3e-4; policy loss minus target entropy flat at ~0.71 for 20 epochs
Q_WEIGHT = 0.2           # value target = (1-Q_WEIGHT) z + Q_WEIGHT q
W_OPP, W_OWN, W_SURV = 0.15, 0.15, 0.1
SAVE_EVERY = 10

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = "training_report_v3.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_catfish(model):
    out = []
    for kind in CATFISH:
        c = copy.deepcopy(model).eval()
        with torch.no_grad():
            if kind == "value0":
                c.value_head[6].weight.zero_(); c.value_head[6].bias.zero_()
            elif kind == "value_half":
                c.value_head[6].weight.mul_(0.5); c.value_head[6].bias.mul_(0.5)
            elif kind == "policy_half":
                c.policy_head[4].weight.mul_(0.5); c.policy_head[4].bias.mul_(0.5)
            else:
                c.policy_head[4].weight.zero_(); c.policy_head[4].bias.zero_()
        out.append(c)
    return out


def selfplay(model):
    R = mcts_core.SelfPlayRunner()
    R.board_size, R.win_length = N, 5
    R.eval_batch = 32
    R.forced_k = 0.0
    R.mega_d1 = R.mega_d2 = 0
    R.gumbel_m_full, R.gumbel_m_fast = GUMBEL_M_FULL, GUMBEL_M_FAST
    cfg = dict(num_simulations=SIMS_FULL, temperature_moves=TEMP_MOVES, full_search_prob=FULL_PROB, fast_sims=SIMS_FAST,
               dirichlet_alpha=DIR_ALPHA, noise_weight=DIR_WEIGHT)
    LAST_STARTS.clear()
    c0 = N // 2
    near = [(r, c) for r in range(c0 - OPEN_RADIUS, c0 + OPEN_RADIUS + 1) for c in range(c0 - OPEN_RADIUS, c0 + OPEN_RADIUS + 1)]
    for gi in range(SELF_GAMES):
        if random.random() < OPEN_FRAC:
            k = random.choice((2, 4))   # 04:55 (user): even only, equal stones for both sides
            cells = random.sample(near, k)
            board = [[" "] * N for _ in range(N)]
            st = {}
            for i, (r, c) in enumerate(cells):
                board[r][c] = "X" if i % 2 == 0 else "O"
                st[r * N + c] = 1 if i % 2 == 0 else 2
            LAST_STARTS[gi] = st
            R.add_task(0, board, "X" if k % 2 == 0 else "O", True, -1, 0, cfg)
        else:
            R.add_task(0, None, "", True, -1, 0, cfg)
    for ci in range(len(CATFISH)):
        for g in range(LEAGUE_GAMES):
            R.add_task(1, None, "", g % 2 == 0, -1, ci + 1, cfg)
    model.eval()
    nets = [model] + make_catfish(model)

    def eval_fn(tag, planes):
        x = torch.from_numpy(planes).to(DEV)
        with torch.inference_mode():
            lp, v = nets[tag](x)
        return (torch.exp(lp).double().cpu().numpy().reshape(planes.shape[0], -1),
                v.squeeze(-1).double().cpu().numpy().reshape(-1))

    t0 = time.monotonic()
    raw = R.run(eval_fn, min(N_THREADS, GAMES), TARGET_ROWS, MAX_WAIT_MS)
    # per-catfish tally from the main net's side: [win, loss, draw]
    tally = {k: [0, 0, 0] for k in CATFISH}
    for i, r in enumerate(raw[SELF_GAMES:]):
        tally[CATFISH[i // LEAGUE_GAMES]][{1: 0, -1: 1}.get(r[4], 2)] += 1
    stats = dict(secs=time.monotonic() - t0, rows=R.n_rows, batches=R.n_batches, cache=int(R.cache_hits), tally=tally)
    return raw, stats


_SYM = [(k, f) for f in (False, True) for k in range(4)]


def _t(a, k, f):           # dihedral transform on the last two axes
    a = np.rot90(a, k, axes=(-2, -1))
    return np.flip(a, axis=-1) if f else a


def build_rows(raw):
    """-> list of numpy arrays: planes, policy, value, survival, opp-reply map (+mask), ownership."""
    X, P, V, S, O, OM, W = [], [], [], [], [], [], []
    wins = [0, 0, 0]
    lengths = []
    for gi, (hist, outcome, plies, total_plies, _mres, moves) in enumerate(raw):
        if gi < SELF_GAMES:
            wins[{1: 0, -1: 1}.get(outcome, 2)] += 1
            lengths.append(len(moves))
        if not hist:
            continue
        p0 = 1 if hist[0][1] == "X" else 2          # who made the first move (games start with X or O)
        first = p0 if plies[0] % 2 == 0 else 3 - p0
        final = np.zeros(NC, np.int8)
        for cell, who in LAST_STARTS.get(gi, {}).items():
            final[cell] = who
        for i, a in enumerate(moves):
            final[a] = first if i % 2 == 0 else 3 - first
        for (board, player, pol, q), ply in zip(hist, plies):
            me = 1 if player == "X" else 2
            z = 0.0 if outcome == 0 else (1.0 if (outcome == 1) == (me == 1) else -1.0)
            v = (1 - Q_WEIGHT) * z + Q_WEIGHT * (q if q is not None else z)
            remaining = (total_plies if total_plies is not None else len(moves)) - ply
            surv = min(remaining, SURVIVAL_CAP) / SURVIVAL_CAP
            b = np.asarray(board, np.int8)
            owner = np.where(final.reshape(N, N) == 0, 0, np.where(final.reshape(N, N) == me, 1, 2))
            opp = np.zeros((N, N), np.float32)
            has_opp = ply + 1 < len(moves)
            if has_opp:
                opp.flat[moves[ply + 1]] = 1.0
            pol = np.asarray(pol, np.float32)
            for k, f in _SYM:
                bt = np.ascontiguousarray(_t(b, k, f))
                X.append(mcts_core.encode_planes(bt, player, N, 5))
                P.append(np.ascontiguousarray(_t(pol, k, f)).reshape(-1))
                O.append(np.ascontiguousarray(_t(opp, k, f)).reshape(-1))
                W.append(np.ascontiguousarray(_t(owner, k, f)).astype(np.int64))
                V.append(v)
                S.append(surv)
                OM.append(1.0 if has_opp else 0.0)
    rows = dict(x=np.stack(X).astype(np.float32), p=np.stack(P), v=np.array(V, np.float32),
                s=np.array(S, np.float32), o=np.stack(O), om=np.array(OM, np.float32), w=np.stack(W))
    return rows, wins, lengths


class Buffer:
    KEYS = ("x", "p", "v", "s", "o", "om", "w")

    def __init__(self):
        self.d = None

    def add(self, rows):
        if self.d is None:
            self.d = {k: rows[k] for k in self.KEYS}
        else:
            self.d = {k: np.concatenate([self.d[k], rows[k]])[-WINDOW:] for k in self.KEYS}

    def __len__(self):
        return 0 if self.d is None else len(self.d["v"])

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        return {k: torch.from_numpy(self.d[k][idx]).to(DEV) for k in self.KEYS}

    def save(self, path):
        np.savez(path, **self.d)

    def load(self, path):
        z = np.load(path)
        self.d = {k: z[k] for k in self.KEYS}


def train(model, opt, buf, n_new):
    model.train()
    steps = max(1, int(STEPS_PER_NEW * n_new / BATCH))
    acc = np.zeros(6)
    for _ in range(steps):
        b = buf.sample(BATCH)
        lp, v, s, olp, own = model.forward_all(b["x"])
        lp_pol = -(b["p"] * lp).sum(1).mean()
        l_v = F.mse_loss(v.squeeze(-1), b["v"])
        l_s = F.mse_loss(s.squeeze(-1), b["s"])
        l_o = (-(b["o"] * olp).sum(1) * b["om"]).sum() / b["om"].sum().clamp(min=1)
        l_w = F.nll_loss(own, b["w"])
        loss = lp_pol + l_v + W_SURV * l_s + W_OPP * l_o + W_OWN * l_w
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc += [loss.item(), lp_pol.item(), l_v.item(), l_o.item(), l_w.item(), l_s.item()]
    return steps, acc / steps


def main():
    model = NetV3().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    buf = Buffer()
    epoch = 0
    if os.path.exists("model/v3_latest.pth"):
        ck = torch.load("model/v3_latest.pth", map_location=DEV)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:      # the saved optimizer state carries the old lr; LR above wins
            g["lr"] = LR
        epoch = ck["epoch"]
        if os.path.exists("model/v3_buffer.npz"):
            buf.load("model/v3_buffer.npz")
        log(f"resumed at epoch {epoch}, buffer {len(buf)}")
    else:
        log(f"cold start: NetV3 {sum(p.numel() for p in model.parameters()):,} params, random init")
    while not os.path.exists("STOP"):
        epoch += 1
        raw, st = selfplay(model)
        rows, wins, lengths = build_rows(raw)
        buf.add(rows)
        steps, L = train(model, opt, buf, len(rows["v"]))
        ent = float(-(torch.from_numpy(rows["p"]).clamp(min=1e-12).log() * torch.from_numpy(rows["p"])).sum(1).mean())
        log(f"Epoch {epoch} | loss {L[0]:.4f} policy {L[1]:.4f} value {L[2]:.4f} opp {L[3]:.4f} own {L[4]:.4f} "
            f"surv {L[5]:.4f} | target entropy {ent:.3f} | games {SELF_GAMES}: X {wins[0]} O {wins[1]} D {wins[2]}, "
            f"avg len {np.mean(lengths):.1f} | rows +{len(rows['v'])} buffer {len(buf)} steps {steps} | "
            f"selfplay {st['secs']:.0f}s, GPU rows {st['rows']} in {st['batches']} batches, cache {st['cache']} | "
            f"catfish W/L/D " + ", ".join(f"{k} {w}/{l}/{d}" for k, (w, l, d) in st["tally"].items()))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, "model/v3_latest.pth")
        if epoch % SAVE_EVERY == 0 or epoch % 25 == 0:   # 10/25 (user): ladder every 25 epochs needs ep575 etc.
            torch.save(model.state_dict(), f"model/v3_ep{epoch}.pth")
            buf.save("model/v3_buffer.npz")
    log(f"STOP file found: stopped after epoch {epoch}")
    buf.save("model/v3_buffer.npz")


if __name__ == "__main__":
    main()
