"""
Model-vs-model spectator board (2026-10-08). Local only (http://127.0.0.1:8766).

Pick two players (or "random") from: 여명 (ep625, graduated), the current best,
and the 30 most recent epoch snapshots; watch them play move by move.
A separate process from server.py so the public analysis board is untouched.
Results go to model/spectate_log.jsonl; the record table is rebuilt from it.

Run:  python -u spectate.py   (from 10x10/hlt/web)
"""
import collections
import glob
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TENX = r"C:\Users\My PC\Desktop\final\10x10"
sys.path.insert(0, TENX)
os.chdir(TENX)
import numpy as np
import torch

# 2026-10-26 (user): VCF engine from 11x11_v2, same as the NOON grading (was mcts_core_solver, no VCF)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mcts_core", r"C:\Users\My PC\Desktop\final\11x11_v2\mcts_core.pyd")
engine_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(engine_mod)
engine_mod.set_vcf(10, 3000)
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(TENX, "model")
LOG = os.path.join(MODEL_DIR, "spectate_log.jsonl")
# 2026-10-23 (user): "기록 초기화" hides older games from the table; the log itself is kept
RESET = os.path.join(MODEL_DIR, "spectate_reset.json")
N = BOARD_SIZE
RECENT = 30              # number of recent snapshots offered
NET_CACHE = 4            # nets kept on the GPU at once (training shares the card)
OPENING_PLIES = 8        # sample near-best moves early so repeated pairings differ (2026-10-11: 6 -> 8)
OPENING_MIN_SHARE = 0.15  # 2026-10-11 (user: "same positions over and over"): 0.3 -> 0.15, wider pool
FIRST_STONE_RADIUS = 2    # 2026-10-11: black's first stone is a random cell within 2 of the centre (25 cells)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_players():
    snaps = []
    for f in glob.glob(os.path.join(MODEL_DIR, "model_MCTS_ep*_11x11.pth")):
        m = re.search(r"ep(\d+)_11x11", f)
        if m:
            snaps.append((int(m.group(1)), f))
    snaps.sort()
    players = {"dawn": {"name": "여명 (ep625 졸업)",
                        "path": os.path.join(MODEL_DIR, "graduated_11x11", "ep625_graduated_20260929.pth")},
               "best": {"name": "현재 best", "path": os.path.join(MODEL_DIR, "best_mcts_11x11.pth")}}
    # 2026-10-11 (user): frozen festival candidates (e.g. ep905, a best with no 10-epoch snapshot),
    # listed right after dawn/best so they are easy to find; newest first
    snap_eps = {ep for ep, _ in snaps[-RECENT:]}
    for f in sorted(glob.glob(os.path.join(MODEL_DIR, "festival_candidates", "ep*.pth")), reverse=True):
        k = os.path.splitext(os.path.basename(f))[0]
        if k[2:].isdigit() and int(k[2:]) not in snap_eps:
            players[k] = {"name": f"epoch {k[2:]} (보관본)", "path": f}
    # 2026-10-23 (user): v2 strongest and v3 (Gumbel, NetV3 128x8) checkpoints for head-to-head viewing
    V2M = r"C:\Users\My PC\Desktop\final\11x11_v2\model"
    V3M = r"C:\Users\My PC\Desktop\final\11x11_v2\v3\model"
    extra = {"v3_latest": ("v3 현재 (최신)", os.path.join(V3M, "v3_latest.pth")),
             "v2_best": ("v2 최강 (후보 10/22 05:50)", os.path.join(V2M, "_rapfi_cand_20261022_055020_11x11.pth"))}
    for ep in (530, 500, 400, 300, 200, 170, 140, 120, 100):   # 10/26: 530 = NOON (정오), festival pick
        extra[f"v3_ep{ep}"] = (f"v3 ep{ep}" + (" (정오)" if ep == 530 else ""), os.path.join(V3M, f"v3_ep{ep}.pth"))
    for k, (nm, p) in extra.items():
        if os.path.exists(p):
            players[k] = {"name": nm, "path": p}
    for ep, f in reversed(snaps[-RECENT:]):
        players[f"ep{ep}"] = {"name": f"epoch {ep}", "path": f}
    return players


PLAYERS = list_players()
_nets = collections.OrderedDict()
_net_lock = threading.Lock()


def get_net(key):
    with _net_lock:
        if key in _nets:
            _nets.move_to_end(key)
            return _nets[key]
        sd = torch.load(PLAYERS[key]["path"], map_location=dev)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):   # v3_latest holds model+opt
            sd = sd["model"]
        if any(k.startswith("opp_policy_head.") for k in sd):   # 2026-10-23: v3 NetV3 checkpoint
            sys.path.insert(1, r"C:\Users\My PC\Desktop\final\11x11_v2\v3")
            from net_v3 import NetV3
            net = NetV3()
        else:
            net = AlphaZeroTicTacToeV3()
        net.load_state_dict(sd)
        net = net.to(dev).eval()
        _nets[key] = net
        while len(_nets) > NET_CACHE:
            _nets.popitem(last=False)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return net


def make_ev(net):
    def ev(bs, ps):
        t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
        with torch.inference_mode():
            pl, v = net(t)
        return (torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1),
                v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1))
    return ev


# ---- match state -------------------------------------------------------
S = {"running": False, "auto": False, "swap": False, "sims": 300, "delay": 0.8, "gen": 0,
     "black": None, "white": None, "board": [[' '] * N for _ in range(N)], "moves": [],
     "turn": 'X', "over": False, "result": None, "winrate": [], "last": None, "games_played": 0,
     "pick": {"black": "random", "white": "random"}, "error": None}
LOCK = threading.Lock()


def resolve(pick, other=None):
    if pick in PLAYERS:
        return pick
    return random.choice([k for k in PLAYERS if k != other])


def save(result):
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "black": S["black"], "white": S["white"],
           "black_name": PLAYERS[S["black"]]["name"], "white_name": PLAYERS[S["white"]]["name"],
           "sims": S["sims"], "result": result, "plies": len(S["moves"]), "moves": S["moves"]}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record_table():
    tab = collections.defaultdict(lambda: {"w": 0, "d": 0, "l": 0})
    skip = json.load(open(RESET)).get("lines", 0) if os.path.exists(RESET) else 0
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                try:
                    g = json.loads(line)
                except Exception:
                    continue
                b, w, r = g["black_name"], g["white_name"], g["result"]
                if r == "draw":
                    tab[b]["d"] += 1
                    tab[w]["d"] += 1
                elif r == "black":
                    tab[b]["w"] += 1
                    tab[w]["l"] += 1
                elif r == "white":
                    tab[w]["w"] += 1
                    tab[b]["l"] += 1
    rows = [{"name": k, **v, "n": v["w"] + v["d"] + v["l"]} for k, v in tab.items()]
    for r in rows:
        r["score"] = (r["w"] + 0.5 * r["d"]) / r["n"] if r["n"] else 0
    return sorted(rows, key=lambda r: (-r["score"], -r["n"]))


def play_one(gen):
    """Play one game; True if it finished, False if it was stopped/replaced."""
    with LOCK:
        if S.get("rematch"):  # colour-swapped replay of the previous pairing
            S["black"], S["white"] = S.pop("rematch")
        else:
            S["black"] = resolve(S["pick"]["black"])
            S["white"] = resolve(S["pick"]["white"], S["black"])
        S["board"] = [[' '] * N for _ in range(N)]
        S.update(moves=[], turn='X', over=False, result=None, winrate=[], last=None, error=None)
        sims, keys = S["sims"], {'X': S["black"], 'O': S["white"]}
    evs = {p: make_ev(get_net(k)) for p, k in keys.items()}
    engines = {'X': engine_mod.MCTSEngine(), 'O': engine_mod.MCTSEngine()}
    while True:
        with LOCK:
            if S["gen"] != gen:
                return False
            board, player, ply = [r[:] for r in S["board"]], S["turn"], len(S["moves"])
        _, rv, cands = engines[player].search_batched(board, player, evs[player], sims, 1.414, False, 0.3, 0.25,
                                                      N, WIN_LENGTH, 16, 0.0)
        best = max(cands, key=lambda c: c["visit_count"])
        if ply == 0:   # random first stone near the centre so games start from different shapes
            ctr = N // 2
            best = {"move": (ctr + random.randint(-FIRST_STONE_RADIUS, FIRST_STONE_RADIUS),
                             ctr + random.randint(-FIRST_STONE_RADIUS, FIRST_STONE_RADIUS))}
        elif ply < OPENING_PLIES and not any(c.get("proven") for c in cands):
            pool = [c for c in cands if c["visit_count"] >= OPENING_MIN_SHARE * best["visit_count"]]
            w = np.array([c["visit_count"] for c in pool], dtype=np.float64)
            best = pool[int(np.random.choice(len(pool), p=w / w.sum()))]
        r, c = best["move"]
        mover_wr = (float(rv) + 1) / 2  # root value is from the side to move
        with LOCK:
            if S["gen"] != gen:
                return False
            S["board"][r][c] = player
            S["moves"].append([player, r, c])
            S["last"] = [r, c]
            S["winrate"].append({"ply": len(S["moves"]), "player": player,
                                 "black_wr": mover_wr if player == 'X' else 1 - mover_wr})
            if fast_check_win(S["board"], player):
                S["over"], S["result"] = True, ("black" if player == 'X' else "white")
            elif len(S["moves"]) >= N * N:
                S["over"], S["result"] = True, "draw"
            S["turn"] = 'O' if player == 'X' else 'X'
            if S["over"]:
                save(S["result"])
                S["games_played"] += 1
                # swap mode: every pairing is played twice, once with each colour
                S["pair_leg"] = 1 - S.get("pair_leg", 0) if S["swap"] else 0
                if S["pair_leg"] == 1:
                    S["rematch"] = (S["white"], S["black"])
                return True
            delay = S["delay"]
        time.sleep(delay)


def runner(gen):
    try:
        while play_one(gen):
            with LOCK:
                if not S["auto"] or S["gen"] != gen:
                    break
            time.sleep(4)  # show the final position before the next game
    except Exception as e:  # surface errors on the page instead of dying silently
        with LOCK:
            S["error"] = repr(e)
    with LOCK:
        if S["gen"] == gen:
            S["running"] = False


def start(body):
    with LOCK:
        S["gen"] += 1
        S["pick"] = {"black": body.get("black", "random"), "white": body.get("white", "random")}
        S["sims"] = max(50, min(2000, int(body.get("sims", 300))))
        S["delay"] = max(0.0, min(5.0, float(body.get("delay", 0.8))))
        S["auto"] = bool(body.get("auto", False))
        S["swap"] = bool(body.get("swap", False))
        S["pair_leg"] = 0
        S.pop("rematch", None)
        S["running"] = True
        gen = S["gen"]
    threading.Thread(target=runner, args=(gen,), daemon=True).start()


def stop():
    with LOCK:
        S["gen"] += 1
        S["running"] = False
        S["auto"] = False


def state():
    with LOCK:
        d = {k: S[k] for k in ("running", "auto", "swap", "sims", "delay", "board", "moves", "turn", "over",
                               "result", "winrate", "last", "games_played", "pick", "error")}
        d["black"] = S["black"] and {"key": S["black"], "name": PLAYERS.get(S["black"], {}).get("name", S["black"])}
        d["white"] = S["white"] and {"key": S["white"], "name": PLAYERS.get(S["white"], {}).get("name", S["white"])}
    return d


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        global PLAYERS
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "spectate.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self.send(200, state())
        if self.path == "/api/players":
            with LOCK:
                if not S["running"]:
                    PLAYERS = list_players()  # pick up new snapshots without restarting
                return self.send(200, [{"key": k, "name": v["name"]} for k, v in PLAYERS.items()])
        if self.path == "/api/records":
            return self.send(200, record_table())
        self.send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/start":
            start(body)
            return self.send(200, {"ok": True})
        if self.path == "/api/stop":
            stop()
            return self.send(200, {"ok": True})
        if self.path == "/api/reset_records":
            n = sum(1 for _ in open(LOG, encoding="utf-8")) if os.path.exists(LOG) else 0
            with open(RESET, "w") as f:
                json.dump({"lines": n, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            return self.send(200, {"ok": True})
        self.send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"spectator board: http://127.0.0.1:8766  ({len(PLAYERS)} players, device {dev})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8766), H).serve_forever()
