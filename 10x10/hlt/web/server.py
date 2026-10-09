"""
HLT web analysis board (2026-09-28). Local only (http://127.0.0.1:8765).

Play the tracked best checkpoint on 11x11 freestyle in the browser while
seeing its search: a win-rate curve (left) and, on the board, the most
visited candidate moves coloured by visit share with visits / Q shown.
Standard library only (no Flask). One game at a time.

Run:  python -u server.py   (from 10x10/hlt/web)
"""
import json
import re
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TENX = r"C:\Users\My PC\Desktop\final\10x10"
sys.path.insert(0, TENX)
os.chdir(TENX)
import numpy as np
import torch

import mcts_core_solver as engine_mod  # current engine build (vlfix + mate distance + solver), module_local
from alphago_lee import board_to_input_array_from_numeric
from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(TENX, "model", "best_mcts_11x11.pth")
N = BOARD_SIZE
TOP_K = 12
PV_TOP = 5    # candidates that get an expected continuation
PV_LEN = 10   # plies shown

DAWN_CKPT = os.path.join(TENX, "model", "graduated_11x11", "ep625_graduated_20260929.pth")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_net(path):
    net = AlphaZeroTicTacToeV3()
    net.load_state_dict(torch.load(path, map_location=dev))
    return net.to(dev).eval()


def last_promoted_epoch():
    """Epoch of the most recent gate promotion, read from the trainer logs
    (the best checkpoint file itself does not record which epoch it is)."""
    import glob
    best = None
    for f in sorted(glob.glob(os.path.join(TENX, "train_stdout_*.log")), key=os.path.getmtime):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "PROMOTED" in line:
                        m = re.search(r"Epoch (\d+)/", line)
                        if m:
                            best = int(m.group(1))
        except Exception:
            pass
    return best


def _latest_info(mtime):
    e = last_promoted_epoch()
    return {"key": "latest", "name": f"최신 모델 (epoch {e})" if e else "최신 모델",
            "checkpoint": os.path.basename(CKPT), "promoted_epoch": e,
            "saved": __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(mtime)), "_mtime": mtime}


# 2026-09-30 (user): visitors choose the model. "dawn" = graduated NAPS-625 (frozen),
# "latest" = the newest gate-promoted best checkpoint (reloaded when a game starts).
MODELS = {
    "dawn": {"net": _load_net(DAWN_CKPT),
             "info": {"key": "dawn", "name": "여명 (DAWN, 졸업 모델)", "checkpoint": os.path.basename(DAWN_CKPT),
                      "promoted_epoch": 625, "saved": "2026-09-29 14:22"}},
    "latest": {"net": _load_net(CKPT), "info": _latest_info(os.path.getmtime(CKPT))},
}
DEFAULT_MODEL = "dawn"
CKPT_INFO = MODELS["latest"]["info"]  # kept for the stats page


def public_info(key):
    return {k: v for k, v in MODELS[key]["info"].items() if not k.startswith("_")}


def ev(bs, ps, key=DEFAULT_MODEL):
    net = MODELS[key]["net"]
    t = torch.from_numpy(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)])).float().to(dev)
    with torch.inference_mode():
        pl, v = net(t)
    return torch.exp(pl).cpu().numpy().astype(np.float64).reshape(len(bs), -1), v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)


def num(board):
    return np.array([[1 if x == 'X' else 2 if x == 'O' else 0 for x in r] for r in board], dtype=np.int8)


def maybe_reload_model():
    """Pick up a newly promoted best checkpoint (called when a game starts,
    so a model never changes in the middle of a game)."""
    global CKPT_INFO
    mtime = os.path.getmtime(CKPT)
    if mtime == MODELS["latest"]["info"].get("_mtime"):
        return
    try:
        sd = torch.load(CKPT, map_location=dev)
    except Exception:
        return  # file being written right now; try again next game
    with EV_LOCK:
        MODELS["latest"]["net"].load_state_dict(sd)
        MODELS["latest"]["net"].eval()
    MODELS["latest"]["info"] = CKPT_INFO = _latest_info(mtime)
    print(f"reloaded best checkpoint: {CKPT_INFO['name']} (saved {CKPT_INFO['saved']})", flush=True)


LOCK = threading.Lock()      # guards GAMES
EV_LOCK = threading.Lock()   # one GPU forward pass at a time (several games may think at once)
GAMES = {}          # session id -> game dict (one game per browser)
LAST_SEEN = {}      # session id -> last request time (for the /stats page)
ACTIVE_WINDOW = 120  # seconds: a browser counts as "connected" if it polled within this
MAX_SESSIONS = 200
MAX_SIMS = 2000     # public link: cap the per-move search
CHUNK = 16          # stream the search every CHUNK simulations = the engine batch size (keeps GUI speed)


OPENING_SAMPLE_PLIES = 8        # first N plies of the game: sample the AI move instead of argmax
OPENING_SAMPLE_MIN_SHARE = 0.3  # only candidates with >= 30% of the top move's visits
BATCH_Q = []               # pending eval requests from all thinking games
BATCH_CV = threading.Condition()


def ev_locked(bs, ps, key=DEFAULT_MODEL):
    """Cross-game batching: queue this request; one worker merges every game's
    pending leaves (per model) into a single GPU forward pass and hands each game its rows."""
    req = {"bs": bs, "ps": ps, "key": key, "out": None, "done": threading.Event()}
    with BATCH_CV:
        BATCH_Q.append(req)
        BATCH_CV.notify()
    req["done"].wait()
    if isinstance(req["out"], Exception):
        raise req["out"]
    return req["out"]


def batch_worker():
    while True:
        with BATCH_CV:
            while not BATCH_Q:
                BATCH_CV.wait()
            reqs = BATCH_Q[:]
            BATCH_Q.clear()
        for key in {r["key"] for r in reqs}:
            group = [r for r in reqs if r["key"] == key]
            try:
                with EV_LOCK:
                    p, v = ev([b for r in group for b in r["bs"]], [x for r in group for x in r["ps"]], key)
                i = 0
                for r in group:
                    n = len(r["bs"])
                    r["out"] = (p[i:i + n], v[i:i + n])
                    i += n
            except Exception as e:
                for r in group:
                    r["out"] = e
        for r in reqs:
            r["done"].set()


threading.Thread(target=batch_worker, daemon=True).start()


def get_game(sid):
    sid = (sid or "anon")[:64]
    with LOCK:
        LAST_SEEN[sid] = __import__("time").time()
        g = GAMES.get(sid)
        if g is None:
            if len(GAMES) >= MAX_SESSIONS:
                GAMES.pop(next(iter(GAMES)))
            g = GAMES[sid] = {"lock": threading.Lock()}
            new_game(g, "O", 1100)
            g["started"] = False  # nothing happens until the visitor presses "new game"
        return g


def new_game(g, human, sims, model_key=DEFAULT_MODEL):
    lock, gen = g["lock"], g.get("gen", 0)
    model_key = model_key if model_key in MODELS else DEFAULT_MODEL
    g.clear()
    g.update(lock=lock, board=[[' '] * N for _ in range(N)], turn='X', human=human, ai='O' if human == 'X' else 'X',
             sims=sims, moves=[], winrate=[], analysis=None, over=False, result=None, thinking=False,
             engine=engine_mod.MCTSEngine(), gen=gen + 1, started=True,
             model_key=model_key, model_info=public_info(model_key))


def snapshot(g, engine, player, cands, rv, raw_p, raw_v, sims_done):
    total = sum(c["visit_count"] for c in cands) or 1
    top = sorted(cands, key=lambda c: -c["visit_count"])[:TOP_K]
    return {
        "player": player, "sims_done": sims_done, "sims_total": g["sims"],
        "root_value": float(rv), "raw_value": float(raw_v), "winrate": (float(rv) + 1) / 2,
        "candidates": [{"r": c["move"][0], "c": c["move"][1], "visits": int(c["visit_count"]),
                        "share": c["visit_count"] / total,
                        "q": float(-c["q_value"]),  # from the searching player's side
                        "prior": float(raw_p[c["move"][0] * N + c["move"][1]]),
                        "proven": int(c.get("proven", 0)),
                        # expected continuation below this move (most-visited line of the search tree)
                        "pv": ([[act // N, act % N, int(v)] for act, v in
                                engine.principal_variation(c["action_idx"], PV_LEN)] if i < PV_TOP else [])}
                       for i, c in enumerate(top)],
    }


def place(g, r, c, player):
    b = g["board"]
    if g["over"] or b[r][c] != ' ' or player != g["turn"]:
        return False
    b[r][c] = player
    g["moves"].append([player, r, c])
    if fast_check_win(b, player):
        g["over"], g["result"] = True, ("human" if player == g["human"] else "ai")
    elif len(g["moves"]) >= N * N:
        g["over"], g["result"] = True, "draw"
    g["turn"] = 'O' if player == 'X' else 'X'
    if g["over"]:
        save_game(g)
    return True


GAME_LOG = os.path.join(HERE, "games.jsonl")


def save_game(g):
    """Append every finished game (moves, result, AI win-rate curve) for later analysis."""
    try:
        with open(GAME_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                                "model": g.get("model_info") or public_info(DEFAULT_MODEL), "human": g["human"], "ai": g["ai"], "sims": g["sims"],
                                "result": g["result"], "moves": g["moves"], "ai_winrate": g["winrate"], "undos": g.get("undos", 0)}) + "\n")
    except Exception:
        pass


def ai_think(g, gen):
    """Search in CHUNK-simulation steps on the same tree (the engine keeps
    its tree for an identical position), publishing a snapshot after each
    step, then play the most visited move."""
    player, engine, sims = g["ai"], g["engine"], g["sims"]
    board = [row[:] for row in g["board"]]
    key = g.get("model_key", DEFAULT_MODEL)
    evk = lambda bs, ps: ev_locked(bs, ps, key)
    raw_p, raw_v = evk([num(board)], [player])
    done, cands, rv = 0, [], 0.0
    while done < sims:
        done = min(sims, done + CHUNK)
        _, rv, cands = engine.search_batched(board, player, evk, done, 1.414, False, 0.3, 0.25,
                                                  N, WIN_LENGTH, 16, 0.0)
        snap = snapshot(g, engine, player, cands, rv, raw_p[0], raw_v[0], done)
        with g["lock"]:
            if g.get("gen") != gen:
                return  # a new game was started meanwhile
            g["analysis"] = snap
        if any(c.get("proven") == -1 for c in cands) or (cands and all(c.get("proven") == 1 for c in cands)):
            break  # result proven: nothing left to learn
    best = max(cands, key=lambda c: c["visit_count"])["move"]
    if len(g["moves"]) < OPENING_SAMPLE_PLIES and not any(c.get("proven") for c in cands):
        # opening variety (2026-09-28): pick among near-best candidates in proportion to visits,
        # so humans cannot win by replaying a memorised line against a deterministic AI
        top = max(c["visit_count"] for c in cands)
        pool = [c for c in cands if c["visit_count"] >= OPENING_SAMPLE_MIN_SHARE * top]
        w = np.array([c["visit_count"] for c in pool], dtype=np.float64)
        best = pool[int(np.random.choice(len(pool), p=w / w.sum()))]["move"]
    with g["lock"]:
        if g.get("gen") != gen:
            return
        g["analysis"]["move_no"] = len(g["moves"]) + 1
        g["winrate"].append({"move_no": len(g["moves"]) + 1, "ai_winrate": (float(rv) + 1) / 2})
        place(g, best[0], best[1], player)
        g["thinking"] = False


def undo(g):
    """Take back the human's last move (and the AI reply that followed it).
    The search tree is discarded so the next AI search starts clean."""
    if g["thinking"] or not g.get("started") or not any(m[0] == g["human"] for m in g["moves"]):
        return False
    while g["moves"]:
        pl, r, c = g["moves"].pop()
        g["board"][r][c] = ' '
        if pl == g["human"]:
            break
    g["turn"] = 'X' if len(g["moves"]) % 2 == 0 else 'O'
    g["over"], g["result"] = False, None
    g["winrate"] = [w for w in g["winrate"] if w["move_no"] <= len(g["moves"])]
    g["analysis"] = None
    g["engine"] = engine_mod.MCTSEngine()
    g["undos"] = g.get("undos", 0) + 1
    return True


def start_ai(g):
    if g["over"] or g["turn"] != g["ai"] or g["thinking"]:
        return
    g["thinking"] = True
    threading.Thread(target=ai_think, args=(g, g["gen"]), daemon=True).start()


LEVEL_NAMES = [(150, "입문"), (400, "쉬움"), (800, "보통"), (1500, "어려움"), (10 ** 9, "최강")]


def level_of(sims):
    return next(name for lim, name in LEVEL_NAMES if sims < lim)


def stats():
    now = __import__("time").time()
    with LOCK:
        active = sum(1 for t in LAST_SEEN.values() if now - t <= ACTIVE_WINDOW)
        games = list(GAMES.values())
    playing = sum(1 for g in games if g.get("started") and not g.get("over") and g.get("moves"))
    thinking = sum(1 for g in games if g.get("thinking"))
    rows = []
    if os.path.exists(GAME_LOG):
        with open(GAME_LOG, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    by = {}
    for r in rows:
        key = (level_of(r["sims"]), "사람 선공" if r["human"] == "X" else "사람 후공")
        b = by.setdefault(key, {"ai": 0, "human": 0, "draw": 0, "plies": 0})
        b[r["result"]] += 1
        b["plies"] += len(r["moves"])
    order = [n for _, n in LEVEL_NAMES]
    table = [{"level": k[0], "side": k[1], **v, "games": v["ai"] + v["human"] + v["draw"],
              "avg_plies": round(v["plies"] / max(1, v["ai"] + v["human"] + v["draw"]), 1)}
             for k, v in sorted(by.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1]))]
    total = {"games": len(rows), "ai": sum(r["result"] == "ai" for r in rows),
             "human": sum(r["result"] == "human" for r in rows), "draw": sum(r["result"] == "draw" for r in rows)}
    human_wins = [{"time": r["time"], "level": level_of(r["sims"]), "sims": r["sims"],
                   "side": "선공" if r["human"] == "X" else "후공", "plies": len(r["moves"])}
                  for r in rows if r["result"] == "human"][-20:]
    return {"active_connections": active, "sessions": len(games), "games_in_progress": playing,
            "ai_thinking": thinking, "total": total, "by_level_side": table, "human_wins": human_wins,
            "model": {k: public_info(k) for k in MODELS}}


STATS_HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>오목 분석판 통계</title>
<style>:root{--bg:#f3f1ec;--panel:#fbfaf7;--ink:#1f2328;--muted:#6b6f76;--line:#d9d4c9;--accent:#c2410c}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--panel:#1e2125;--ink:#e7e5e1;--muted:#9aa0a8;--line:#33373d;--accent:#fb923c}}
body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,sans-serif;font-size:14px}
.w{max-width:900px;margin:0 auto;padding:18px 16px;display:grid;gap:14px}
.p{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.n{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}.l{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}h1{font-size:18px;margin:0}</style></head>
<body><div class="w"><div class="p"><h1>오목 분석판 통계</h1><div class="l" id="m"></div></div>
<div class="p k"><div><div class="n" id="a">-</div><div class="l">현재 접속 (최근 2분)</div></div>
<div><div class="n" id="g">-</div><div class="l">진행 중인 게임</div></div>
<div><div class="n" id="t">-</div><div class="l">AI 생각 중</div></div>
<div><div class="n" id="w">-</div><div class="l">완료 게임 AI 승률</div></div></div>
<div class="p"><div class="l">난이도·선후공별 결과 (완료된 게임)</div><table><thead><tr><th>난이도</th><th>사람</th><th>판</th><th>AI 승</th><th>사람 승</th><th>무</th><th>AI 승률</th><th>평균 수</th></tr></thead><tbody id="b"></tbody></table></div>
<div class="p"><div class="l">사람이 이긴 판 (최근 20)</div><div id="h"></div></div></div>
<script>
async function go(){const s=await (await fetch('/api/stats')).json();
document.getElementById('m').textContent=`모델 ${s.model.checkpoint} · 저장 ${s.model.saved} · 5초마다 갱신`;
document.getElementById('a').textContent=s.active_connections;document.getElementById('g').textContent=s.games_in_progress;
document.getElementById('t').textContent=s.ai_thinking;
const T=s.total;document.getElementById('w').textContent=T.games?Math.round((T.ai+0.5*T.draw)/T.games*100)+'% ('+T.games+'판)':'-';
document.getElementById('b').innerHTML=s.by_level_side.map(r=>`<tr><td>${r.level}</td><td>${r.side}</td><td>${r.games}</td><td>${r.ai}</td><td>${r.human}</td><td>${r.draw}</td><td>${Math.round((r.ai+0.5*r.draw)/r.games*100)}%</td><td>${r.avg_plies}</td></tr>`).join('')||'<tr><td colspan=8>아직 없음</td></tr>';
document.getElementById('h').innerHTML=s.human_wins.map(r=>`${r.time} · ${r.level}(${r.sims}회) · 사람 ${r.side} · ${r.plies}수`).join('<br>')||'아직 없음';}
go();setInterval(go,5000);</script></body></html>"""


def state(g):
    return {k: g[k] for k in ("board", "turn", "human", "ai", "sims", "moves", "winrate", "analysis", "over",
                              "result", "thinking", "started")} | {"undos": g.get("undos", 0),
            "model": g.get("model_info") or public_info(DEFAULT_MODEL),
            "models": {k: public_info(k) for k in MODELS}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/stats":
            return self._send(200, STATS_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/stats":
            return self._send(200, stats())
        if self.path == "/api/state":
            g = get_game(self.headers.get("X-Session"))
            with g["lock"]:
                return self._send(200, state(g))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        g = get_game(self.headers.get("X-Session"))
        with g["lock"]:
            if self.path == "/api/new":
                maybe_reload_model()
                human = body.get("human", "O")
                sims = max(50, min(MAX_SIMS, int(body.get("sims", 1100))))
                new_game(g, human if human in ("X", "O") else "O", sims, body.get("model", DEFAULT_MODEL))
                start_ai(g)
                return self._send(200, state(g))
            if self.path == "/api/undo":
                if not undo(g):
                    return self._send(400, {"error": "nothing to undo"} | state(g))
                return self._send(200, state(g))
            if self.path == "/api/move":
                if not g.get("started") or g["thinking"] or not place(g, int(body["r"]), int(body["c"]), g["human"]):
                    return self._send(400, {"error": "illegal move"} | state(g))
                start_ai(g)
                return self._send(200, state(g))
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print("HLT web board on http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
