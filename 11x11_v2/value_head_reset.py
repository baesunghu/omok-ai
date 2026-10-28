"""
Value-head reset (2026-09-27, user decision).

Hard-7 hard-mined positions carried a confident -1.0/-0.95 value label for
~300+ epochs (removed 2026-09-26). The label sat on the position BEFORE the
defender's reply, and it taught the value head that any fork threat means
the game is lost. The pessimism probe measured raw value -0.78 against an
empirical -0.04 on such positions. Diluting a learned function at the
current (cooldown, eta_min) learning rate would take a very long time, so
the user chose to throw away ONLY the value head:

  1. Load the tracked best checkpoint (the generation model).
  2. Re-initialize value_head (trunk, policy head and survival head are
     kept exactly as they are).
  3. Warm the new head up (trunk frozen) on the current replay buffer, which
     no longer contains Hard-7 value labels (NaN/None targets are masked).
     Hold out 10% of the buffer and early-stop on it.
  4. Report old vs new head on the held-out split: value MSE, saturation
     (|v| > 0.9), and a calibration table.
  5. Only if the new head is not clearly worse, write the result to BOTH
     the training checkpoint and the best checkpoint, so generation uses it
     from the first epoch after the restart.

Must be run with the trainer STOPPED (it overwrites both checkpoints).
Backups are made by the caller.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\11x11_v2")
os.chdir(r"C:\Users\My PC\Desktop\final\11x11_v2")

from alphago_lee_v3_survival import AlphaZeroTicTacToeV3  # noqa: E402

BEST = r"model\best_mcts_11x11.pth"
TRAIN_CKPT = r"model\alphazero_mcts_11x11.pth"
BUFFER = r"model\mcts_replay_buffer_11x11.pkl"
REPORT = r"model\value_head_reset_20260927.json"
HARD_POOL = r"model\hard_mined_examples_11x11.pkl"

SEED = 20260927
BATCH = 256
LR = 1e-3
MAX_PASSES = 40
PATIENCE = 4
ACCEPT_RATIO = 1.25  # new held-out MSE must be <= 1.25x the old head's


def value_stats(model, x, y, dev):
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x), 1024):
            _, v = model(x[i:i + 1024].to(dev))
            preds.append(v.squeeze(-1).float().cpu())
    p = torch.cat(preds).numpy()
    t = y.numpy()
    mse = float(np.mean((p - t) ** 2))
    sat = float(np.mean(np.abs(p) > 0.9))
    calib = []
    for lo, hi in ((-1.01, -0.9), (-0.9, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            calib.append({"bucket": f"[{lo:+.1f},{hi:+.1f})", "n": int(m.sum()),
                          "mean_pred": round(float(p[m].mean()), 3), "mean_target": round(float(t[m].mean()), 3)})
    return mse, sat, calib, p


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(BUFFER, "rb") as f:
        buf = list(pickle.load(f))
    x = torch.from_numpy(np.array([s[0] for s in buf], dtype=np.float32))
    y = torch.from_numpy(np.array([s[2] for s in buf], dtype=np.float32))
    w = torch.from_numpy(np.array([s[3] if len(s) > 3 else 1.0 for s in buf], dtype=np.float32))
    keep = ~torch.isnan(y)
    x, y, w = x[keep], y[keep], w[keep]
    n = len(y)
    perm = torch.randperm(n)
    n_val = n // 10
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"buffer: {len(buf)} samples, {n} with a value target -> train {len(tr_idx)} / held-out {n_val}")

    old = AlphaZeroTicTacToeV3()
    old.load_state_dict(torch.load(BEST, map_location=dev))
    old.to(dev)
    old_mse, old_sat, old_cal, _ = value_stats(old, x[val_idx], y[val_idx], dev)
    print(f"OLD head: held-out MSE {old_mse:.4f}, saturated(|v|>0.9) {old_sat:.1%}")

    model = AlphaZeroTicTacToeV3()
    model.load_state_dict(torch.load(BEST, map_location=dev))
    model.to(dev)
    for mod in model.value_head.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith("value_head.")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"value_head re-initialized; trainable params {n_trainable} (trunk/policy/survival frozen)")
    init_mse, init_sat, _, _ = value_stats(model, x[val_idx], y[val_idx], dev)
    print(f"NEW head at init: held-out MSE {init_mse:.4f}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.monotonic()
    for ep in range(MAX_PASSES):
        # trunk has no train/eval-dependent layers (GroupNorm/SE only), so
        # train() only matters for the head; keep it consistent anyway.
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        tot, nb = 0.0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            xb, yb, wb = x[idx].to(dev), y[idx].to(dev), w[idx].to(dev)
            _, v = model(xb)
            loss = (wb * (v.squeeze(-1) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        val_mse, val_sat, _, _ = value_stats(model, x[val_idx], y[val_idx], dev)
        print(f"  pass {ep + 1:2d}: train {tot / nb:.4f} | held-out MSE {val_mse:.4f} sat {val_sat:.1%}", flush=True)
        if val_mse < best_val - 1e-4:
            best_val, bad = val_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    new_mse, new_sat, new_cal, _ = value_stats(model, x[val_idx], y[val_idx], dev)
    print(f"warm-up {time.monotonic() - t0:.0f}s -> NEW head held-out MSE {new_mse:.4f} (old {old_mse:.4f}), "
          f"saturated {new_sat:.1%} (old {old_sat:.1%})")

    # policy must be bit-identical (frozen) -- sanity check
    with torch.inference_mode():
        xs = x[val_idx[:512]].to(dev)
        pol_old, _ = old(xs)
        pol_new, _ = model(xs)
    pol_diff = float((pol_old - pol_new).abs().max())
    print(f"policy max |diff| old vs new: {pol_diff:.2e}")

    # Hard-7 pessimism spot check (raw value, defender to move)
    from alphago_lee import board_to_input_array
    with open(HARD_POOL, "rb") as f:
        pool = pickle.load(f)
    h7 = [it for it in pool if it[4] == 7]
    rng = np.random.RandomState(42)
    pick = [h7[i] for i in rng.choice(len(h7), size=min(200, len(h7)), replace=False)]
    xb = torch.from_numpy(np.stack([board_to_input_array(b, pl) for b, pl, *_ in pick]).astype(np.float32)).to(dev)
    with torch.inference_mode():
        v_old = old(xb)[1].squeeze(-1).mean().item()
        v_new = model(xb)[1].squeeze(-1).mean().item()
    print(f"Hard-7 positions ({len(pick)}), mean raw value for the defender: old {v_old:+.3f} -> new {v_new:+.3f} "
          f"(empirical outcome after a safe reply measured -0.04)")

    accepted = new_mse <= old_mse * ACCEPT_RATIO and pol_diff < 1e-5
    report = dict(date="2026-09-27", base=BEST, n_train=int(len(tr_idx)), n_val=int(n_val),
                  old=dict(mse=old_mse, saturated=old_sat, calibration=old_cal),
                  new=dict(mse=new_mse, saturated=new_sat, calibration=new_cal, init_mse=init_mse),
                  hard7_mean_value=dict(old=v_old, new=v_new), policy_max_abs_diff=pol_diff,
                  accepted=bool(accepted))
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    print("calibration OLD:", old_cal)
    print("calibration NEW:", new_cal)
    if not accepted:
        print("NOT ACCEPTED -- checkpoints left untouched.")
        sys.exit(2)
    for prm in model.parameters():
        prm.requires_grad = True
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(sd, TRAIN_CKPT)
    torch.save(sd, BEST)
    print(f"ACCEPTED -- wrote new weights to {TRAIN_CKPT} and {BEST}")


if __name__ == "__main__":
    main()
