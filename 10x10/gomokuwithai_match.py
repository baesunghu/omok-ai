"""
NAPS-625 "Dawn" vs the Gomoku With AI 4 site AI (https://youngin05.github.io/gomokuwithai/),
2026-09-29. MEASUREMENT ONLY.

The site is a TurboWarp-packaged Scratch project. We load it in headless Chromium
(Playwright), stop the GUI sprite's scripts (so it cannot place stones or start its
own AI), and call the GWA4 sprite's AI directly:
  board list (size*size, 0 empty / 1 black / 2 white), "✅ Board_size" = 11,
  Ai_Difficulty = [black level, white level, side to move]  (0 You, 1 Easy, 2 Medium, 3 Hard),
  GWA4_run = 0, broadcast "GWA4 run" to the GWA4 sprite, wait until GWA4_run == 1,
  read "✅ AI_NextMove" (1-based index = col + 1 + row * size).
Freestyle (Rule 0). Dawn: production engine, SIMS sims, no noise; the first
OPENING_SAMPLE plies are sampled among near-best candidates (same as the web board)
so that the 50 games are not copies of each other. Colours alternate.

Usage: python -u gomokuwithai_match.py [games] [sims] [site_level]
Log:   model/gomokuwithai_match_log.jsonl (one line per game)
"""
import json
import os
import sys
import time

sys.path.insert(0, r"C:\Users\My PC\Desktop\final\10x10")
os.chdir(r"C:\Users\My PC\Desktop\final\10x10")
import numpy as np
import torch
from playwright.sync_api import sync_playwright

import mcts_core_solver as engine_mod
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import fast_check_win

N = 11
URL = "https://youngin05.github.io/gomokuwithai/"
CKPT = os.path.join("model", "graduated_11x11", "ep625_graduated_20260929.pth")
LOG = os.path.join("model", "gomokuwithai_match_log.jsonl")
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # 3 = Hard
# 2026-09-30: time-matched mode (SIMS = 0): Dawn searches until it has used as much
# wall time as the site AI's average move time so far in this match (min 1100 sims).
TIME_MATCH = SIMS == 0
if TIME_MATCH:
    LOG = os.path.join("model", "gomokuwithai_match_timematched_log.jsonl")
OPENING_SAMPLE, MIN_SHARE = 8, 0.3
SITE_TIMEOUT = 180  # seconds per site move

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroTicTacToeV3()
model.load_state_dict(torch.load(CKPT, map_location=dev))
model.to(dev).eval()


def ev(bs, ps):
    t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
    with torch.inference_mode():
        pl, v = model(t)
    return (torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1),
            v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1))


rng = np.random.default_rng(20260929)


def dawn_move(board, player, ply, engine, target_secs=None):
    t0 = time.time()
    if target_secs is None:
        _, rv, cands = engine.search_batched(board, player, ev, SIMS, 1.414, False, 0.3, 0.25, N, 5, 16, 0.0)
    else:
        sims = 1100
        while True:  # identical-board tree reuse: each call only tops the root up to `sims`
            _, rv, cands = engine.search_batched(board, player, ev, sims, 1.414, False, 0.3, 0.25, N, 5, 16, 0.0)
            if time.time() - t0 >= target_secs or any(c.get("proven") for c in cands) or sims >= 20000:
                break
            sims += 256
    best = max(cands, key=lambda c: c["visit_count"])
    if ply < OPENING_SAMPLE and not any(c.get("proven") for c in cands):
        top = best["visit_count"]
        pool = [c for c in cands if c["visit_count"] >= MIN_SHARE * top]
        w = np.array([c["visit_count"] for c in pool], dtype=np.float64)
        best = pool[int(rng.choice(len(pool), p=w / w.sum()))]
    return tuple(best["move"]), float(rv), time.time() - t0, sum(c["visit_count"] for c in cands)


SETUP_JS = """() => {
  const gui = vm.runtime.targets.find(t => t.isOriginal && t.getName() === 'GUI');
  vm.runtime.stopForTarget(gui);
  return true;
}"""

ASK_JS = """([cells, side, level]) => {
  const st = vm.runtime.getTargetForStage();
  const V = n => Object.values(st.variables).find(v => v.name === n);
  const gw = vm.runtime.targets.find(t => t.isOriginal && t.getName() === 'GWA4');
  V('✅ Board_size').value = %d;
  V('✅ Rule (0==Gomoku, 1==Renju)').value = 0;
  V('board').value = cells;
  V('Board_turn').value = side;
  V('Ai_Difficulty').value = [side === 1 ? level : 0, side === 2 ? level : 0, side];
  V('GWA4_run').value = 0;
  vm.runtime.startHats('event_whenbroadcastreceived', {BROADCAST_OPTION: 'GWA4 run'}, gw);
  return true;
}""" % N

POLL_JS = """() => {
  const st = vm.runtime.getTargetForStage();
  const V = n => Object.values(st.variables).find(v => v.name === n);
  const gw = vm.runtime.targets.find(t => t.isOriginal && t.getName() === 'GWA4');
  const nv = Object.values(gw.variables).find(v => v.name === '✅ AI_NextMove');
  return [String(V('GWA4_run').value), String(nv.value)];
}"""


def site_move(page, board, player):
    cells = [0 if x == ' ' else (1 if x == 'X' else 2) for row in board for x in row]
    side = 1 if player == 'X' else 2
    page.evaluate(ASK_JS, [cells, side, LEVEL])
    t0 = time.time()
    while time.time() - t0 < SITE_TIMEOUT:
        time.sleep(0.3)
        run, mv = page.evaluate(POLL_JS)
        if run in ("1", "1.0"):
            m = int(float(mv))
            r, c = (m - 1) // N, (m - 1) % N
            if 0 <= r < N and board[r][c] == ' ':
                return (r, c), time.time() - t0
            raise RuntimeError(f"site returned illegal move {m}")
    raise RuntimeError("site AI timed out")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="load")
        page.wait_for_function("() => window.vm && vm.runtime && vm.runtime.targets.length > 3", timeout=120000)
        time.sleep(5)
        page.evaluate(SETUP_JS)
        res = {"dawn": 0, "site": 0, "draw": 0}
        all_site = []
        for g in range(GAMES):
            dawn = 'X' if g % 2 == 0 else 'O'
            board = [[' '] * N for _ in range(N)]
            moves, turn, winner, engine = [], 'X', None, engine_mod.MCTSEngine()
            site_times, values, dawn_secs, dawn_sims = [], [], [], []
            while len(moves) < N * N:
                if turn == dawn:
                    target = (float(np.mean(all_site)) if all_site else 6.0) if TIME_MATCH else None
                    (r, c), v, secs, nv = dawn_move(board, turn, len(moves), engine, target)
                    values.append(round(v, 3))
                    dawn_secs.append(round(secs, 1))
                    dawn_sims.append(int(nv))
                else:
                    (r, c), dt = site_move(page, board, turn)
                    site_times.append(round(dt, 1))
                    all_site.append(dt)
                board[r][c] = turn
                moves.append((turn, r, c))
                if fast_check_win(board, turn):
                    winner = turn
                    break
                turn = 'O' if turn == 'X' else 'X'
            result = "draw" if winner is None else ("dawn" if winner == dawn else "site")
            res[result] += 1
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dawn": dawn, "site_level": LEVEL,
                                    "sims": SIMS, "result": result, "plies": len(moves), "moves": moves,
                                    "dawn_values": values, "site_secs": site_times,
                                    "dawn_secs": dawn_secs, "dawn_sims": dawn_sims, "time_matched": TIME_MATCH}, ensure_ascii=False) + "\n")
            n = g + 1
            score = (res["dawn"] + 0.5 * res["draw"]) / n
            print(f"{time.strftime('%H:%M')} game {n}/{GAMES}: Dawn={dawn} -> {result} ({len(moves)} plies, "
                  f"site avg {np.mean(site_times) if site_times else 0:.1f}s, Dawn avg {np.mean(dawn_secs) if dawn_secs else 0:.1f}s "
                  f"/ {int(np.mean(dawn_sims)) if dawn_sims else 0} sims) | "
                  f"Dawn W{res['dawn']} D{res['draw']} L{res['site']} score {score:.0%}", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
