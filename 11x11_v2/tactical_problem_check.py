"""
11x11 "graduation" problems (2026-09-28), MEASUREMENT ONLY -- never used as
training data. From the 58-ply web game in which a human (with engine help)
found a 7-ply forced win against the model.

  attack  web58_p7_attack : X to move after ply 28 -- forced win in 7 plies.
  attack  web58_p5_attack : X to move after ply 30 -- forced win in 5 plies.
  defense web58_d13_defense: O to move after ply 25 -- the model's real move
          (8,9) walks into a 13-ply forced loss.

Pass criteria (model = tracked best checkpoint, production engine, 1100
sims, no noise; opponent = Rapfi at 3 s/move, i.e. near-perfect play here):
  attack : the model, playing X from the position, wins the game.
  defense: the model, playing O from the position, does not play (8,9)
           and is not lost within 40 plies.

Usage: python tactical_problem_check.py [checkpoint]
Appends one line to model/tactical_problem_log.jsonl.
"""
import json
import os
import sys
import time

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\11x11_v2")
os.chdir(r"C:\Users\My PC\Desktop\final\11x11_v2")
import numpy as np
import torch

import mcts_core
# 2026-10-23 (user): VCF on everywhere (was off: engine default depth 0); VCF=0 turns it off
if os.environ.get("VCF", "1") != "0":
    mcts_core.set_vcf(10, 3000)
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win
from rapfi_match import Rapfi

N = BOARD_SIZE
PROBLEMS = os.path.join("tactical_problems", "web_game58_problems.json")
LOG = os.path.join("model", "tactical_problem_log.jsonl")
SIMS = 1100
TRIAL_SIMS = (900, 1100, 1300)
RAPFI_MS = 3000
RAPFI_NODES = 1000000  # 2026-10-11: node limit replaces the time limit (see main)
MAX_PLIES = 40


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "model/best_mcts_11x11.pth"
    dev = torch.device("cuda" if torch.cuda.is_available() and not os.environ.get("FORCE_CPU") else "cpu")  # 2026-10-23: FORCE_CPU=1 keeps the training GPU free
    sd = torch.load(ckpt, map_location=dev)
    if any(k.startswith("opp_policy_head.") for k in sd):   # 2026-10-22: 11x11_v3 checkpoint (NetV3, 128x8)
        sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3"))
        from net_v3 import NetV3
        model = NetV3()
    else:
        model = AlphaZeroTicTacToeV3()
    model.load_state_dict(sd)
    model.to(dev).eval()

    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
        with torch.inference_mode():
            pl, v = model(t)
        return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)

    probs = json.load(open(PROBLEMS, encoding="utf-8"))["problems"]
    # 2026-10-22 (user): PROBLEM_IDS=id1,id2 grades only those problems (all as tier 1, nothing logged)
    only = [x for x in os.environ.get("PROBLEM_IDS", "").split(",") if x]
    if only:
        probs = [dict({k: v for k, v in p.items() if k != "tier"}) for p in probs if p["id"] in only]
    # 2026-10-11 (user): node-limited opponent so a grading run gives the same result every time (the 3 s
    # time limit made results depend on PC load; ep920 got 3/3 in one run and 1/3 in another with the same moves).
    # 1,000,000 nodes ~ what 3 s searched on an idle PC (~417k nodes/s), so difficulty stays comparable.
    rapfi = Rapfi(30000, RAPFI_NODES)
    results = []
    try:
        # 2026-10-08 (user): 3-tier check. Problems without "tier" always run (tier 1 includes the
        # stage-1 graduation set). Tier 2 (5 representative loss-backtrack problems) runs only if every
        # stage-1 problem passed; tier 3 (the remaining ones) only if every tier-2 problem passed.
        stage_of = {p["id"]: p.get("stage", 1) for p in probs}
        tier_of = {p["id"]: p.get("tier", 1) for p in probs}
        todo = [p for p in probs if "tier" not in p]
        i, tier_next = 0, 2
        while i < len(todo):
            pr = todo[i]
            i += 1
            # 2026-09-28 (user, option A): 3 trials at slightly different search budgets,
            # pass if >= 2 of 3 pass -- one deterministic game per problem was too noisy
            trials = []
            for sims in TRIAL_SIMS:
                moves = [tuple(m) for m in pr["moves"]]
                board = [[' '] * N for _ in range(N)]
                for p, r, c in moves:
                    board[r][c] = p
                ai = pr["side_to_move"]
                turn = ai
                engine = mcts_core.MCTSEngine()
                first_ai = None
                first_value = None
                winner = None
                for _ in range(MAX_PLIES):
                    if turn == ai:
                        _, rv, cands = engine.search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, WIN_LENGTH, 16)
                        r, c = max(cands, key=lambda x: x["visit_count"])["move"]
                        if first_ai is None:
                            first_ai, first_value = [r, c], round(float(rv), 3)
                    else:
                        r, c = rapfi.move(list(moves), turn)
                    board[r][c] = turn
                    moves.append((turn, r, c))
                    if fast_check_win(board, turn):
                        winner = turn
                        break
                    turn = 'O' if turn == 'X' else 'X'
                if pr["type"] == "attack":
                    ok = winner == ai
                else:
                    ok = first_ai != pr.get("bad_move") and winner != ('X' if ai == 'O' else 'O')
                trials.append({"sims": sims, "passed": ok, "ai_first_move": first_ai, "ai_root_value": first_value,
                               "winner": winner, "plies_played": len(moves) - len(pr["moves"])})
            n_ok = sum(t["passed"] for t in trials)
            passed = n_ok >= 2
            results.append({"id": pr["id"], "name": pr["name"], "type": pr["type"], "passed": passed,
                            "trials_passed": n_ok, "trials": trials})
            detail = ", ".join(f"{t['sims']}:{'O' if t['passed'] else 'X'} {tuple(t['ai_first_move'])} {t['ai_root_value']:+.2f}" for t in trials)
            print(f"{'PASS' if passed else 'FAIL'} ({n_ok}/3)  {pr['name']}: {detail}", flush=True)
            if i == len(todo) and tier_next <= 3:
                if tier_next == 2:
                    ok = all(r["passed"] for r in results if stage_of[r["id"]] == 1)
                else:
                    ok = all(r["passed"] for r in results if tier_of[r["id"]] == tier_next - 1)
                nxt = [p for p in probs if p.get("tier") == tier_next]
                if ok and nxt:
                    print(f"--- tier {tier_next}: {len(nxt)} problems (previous tier fully passed)", flush=True)
                    todo += nxt
                    tier_next += 1
                else:
                    if nxt:
                        print(f"--- tier {tier_next} skipped (previous tier not fully passed)", flush=True)
                    tier_next = 4
    finally:
        rapfi.close()
    n_pass = sum(r["passed"] for r in results)
    # 2026-09-28: stage 1 (graduation) problems vs stage 2 (stretch goal, together with Rapfi 2000)
    stage = {p["id"]: p.get("stage", 1) for p in probs}
    for r in results:
        r["stage"] = stage[r["id"]]
    g = [r for r in results if r["stage"] == 1]
    g_pass = sum(r["passed"] for r in g)
    tiers = {}
    for r in results:
        r["tier"] = tier_of[r["id"]]
        if r["tier"] > 1:
            t = tiers.setdefault(r["tier"], [0, 0])
            t[0] += r["passed"]
            t[1] += 1
    tier_txt = " | ".join(f"tier {k}: {v[0]}/{v[1]}" for k, v in sorted(tiers.items()))
    print(f"\n{n_pass}/{len(results)} problems passed | graduation (stage 1): {g_pass}/{len(g)}"
          + (f" | {tier_txt}" if tier_txt else ""))
    if only:
        return
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checkpoint": os.path.basename(ckpt),
                            "passed": n_pass, "total": len(results), "grad_passed": g_pass, "grad_total": len(g),
                            "tiers": {str(k): v for k, v in tiers.items()}, "results": results}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
