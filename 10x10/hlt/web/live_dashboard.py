"""Local live dashboard of the web board's games (2026-10-16, user: "see the games in progress like a dashboard
before opening the spectator view").

Reads live_games.json (written by server.py on every move) once a second, remembers when each session last
changed, and serves a page with a mini board per session plus its spectator link. Bound to 127.0.0.1 only: the
games are visitors' data and never leave this PC. No server.py restart needed.
Usage: python live_dashboard.py   -> http://127.0.0.1:8770
"""
import base64
import json
import os
import urllib.request
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "live_games.json")
PORT = 8770
WATCH_BASE = "/watch#watch="          # 2026-10-16: spectator view proxied through this (password-protected) server
GAME = "http://127.0.0.1:8765"
_A = json.load(open(os.path.join(HERE, "admin_auth.json"), encoding="utf-8"))   # local only, never committed
AUTH = "Basic " + base64.b64encode(f"{_A['user']}:{_A['password']}".encode()).decode()
seen = {}            # sid -> {"n": move count, "changed": epoch seconds}
IGN = os.path.join(HERE, "dashboard_ignored.json")   # 2026-10-16 (user): sessions hidden from the dashboard


def ignored():
    try:
        return set(json.load(open(IGN, encoding="utf-8")))
    except Exception:
        return set()
lock = threading.Lock()


_cache = {"mtime": None, "games": {}}


def load_games():
    """2026-10-17: read live_games.json only when it changed, and close it at once -- on Windows an open
    handle makes server.py's os.replace fail ("live-game save failed: PermissionError")."""
    try:
        m = os.path.getmtime(LIVE)
        if m != _cache["mtime"]:
            with open(LIVE, "rb") as f:
                raw = f.read()
            _cache["games"], _cache["mtime"] = json.loads(raw.decode("utf-8")), m
    except (OSError, ValueError):
        pass
    return _cache["games"]


def scan():
    while True:
        try:
            games = load_games()
            now = time.time()
            with lock:
                for sid, g in games.items():
                    n = len(g.get("moves") or [])
                    s = seen.get(sid)
                    if s is None:
                        seen[sid] = {"n": n, "changed": None}      # unknown until it moves while we watch
                    elif s["n"] != n:
                        s["n"], s["changed"] = n, now
        except Exception:
            pass
        time.sleep(2)


def snapshot():
    games = load_games()
    now = time.time()
    out = []
    ign = ignored()
    with lock:
        for sid, g in games.items():
            ch = seen.get(sid, {}).get("changed")
            wr = g.get("winrate") or []
            out.append({
                "sid": sid, "board": g.get("board"), "moves": g.get("moves") or [], "human": g.get("human"),
                "sims": g.get("sims"), "model": (g.get("model_info") or {}).get("name") or g.get("model_key"),
                "over": g.get("over"), "result": g.get("result"), "started": g.get("started"),
                "ai_wr": wr[-1]["ai_winrate"] if wr else None, "idle": None if ch is None else now - ch,
                "watch": WATCH_BASE + sid, "ignored": sid in ign})
    return out


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>대국 현황판</title><style>
:root{--bg:#f2efe8;--panel:#fff;--ink:#23252a;--muted:#6c6f76;--line:#ddd6c8;--wood:#dcb478;--grid:#7a5a30;--live:#2f8a52;--idle:#b8860b;--done:#8a8f98;--acc:#2f6db5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--panel:#1f2227;--ink:#e6e7ea;--muted:#9aa0aa;--line:#32363d;--acc:#6aa0e6}}
:root[data-theme="dark"]{--bg:#16181c;--panel:#1f2227;--ink:#e6e7ea;--muted:#9aa0aa;--line:#32363d;--acc:#6aa0e6}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.5 "Malgun Gothic",system-ui,sans-serif}
header{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:20px;margin:0}.kpi{display:flex;gap:10px;flex-wrap:wrap}
.k{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:6px 14px}.k b{font-size:20px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:8px}
.card.live{border-color:var(--live);box-shadow:0 0 0 2px color-mix(in srgb,var(--live) 35%,transparent)}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.pill{font-size:11px;padding:2px 9px;border-radius:999px;color:#fff;white-space:nowrap}
.pill.live{background:var(--live)}.pill.idle{background:var(--idle)}.pill.done{background:var(--done)}
.meta{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
a.btn{display:block;text-align:center;padding:7px;border-radius:8px;background:var(--acc);color:#fff;text-decoration:none;font-weight:bold}
a.btn:focus-visible{outline:3px solid var(--ink)}
button.hide{font:inherit;font-size:12px;padding:5px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--muted);cursor:pointer}
svg{width:100%;height:auto;border-radius:8px}
.f{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:13px}
</style></head><body>
<header><h1>대국 현황판</h1><a href="/settings" style="color:var(--acc);font-weight:bold">⚙ 웹 대국 설정</a><div class="kpi" id="kpi"></div>
<label class="f"><input type="checkbox" id="onlylive" checked> 진행 중만 보기</label><label class="f"><input type="checkbox" id="showign"> 숨긴 판도 보기</label></header>
<div class="grid" id="grid"></div>
<script>
const N = 11;
function boardSVG(b, moves) {
  const P = 12, C = (200 - 2 * P) / (N - 1);
  let s = `<svg viewBox="0 0 200 200"><rect width="200" height="200" fill="var(--wood)"/>`;
  for (let i = 0; i < N; i++) { const p = P + i * C;
    s += `<line x1="${P}" y1="${p}" x2="${200 - P}" y2="${p}" stroke="var(--grid)" stroke-width=".6"/><line x1="${p}" y1="${P}" x2="${p}" y2="${200 - P}" stroke="var(--grid)" stroke-width=".6"/>`; }
  const last = moves.length ? moves[moves.length - 1] : null;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const v = b[r][c]; if (v === " ") continue;
    s += `<circle cx="${P + c * C}" cy="${P + r * C}" r="${C * .43}" fill="${v === "X" ? "#151515" : "#f6f4ee"}" stroke="${v === "X" ? "#000" : "#888"}" stroke-width=".6"/>`;
    if (last && last[1] === r && last[2] === c) s += `<circle cx="${P + c * C}" cy="${P + r * C}" r="${C * .18}" fill="#d0342c"/>`; }
  return s + "</svg>";
}
function ago(sec) { return sec < 60 ? `${Math.round(sec)}초 전` : sec < 3600 ? `${Math.round(sec / 60)}분 전` : `${Math.round(sec / 3600)}시간 전`; }
async function tick() {
  let d; try { d = await (await fetch("/data")).json(); } catch (e) { return; }
  const state = g => g.over ? "done" : (g.idle != null && g.idle < 300) ? "live" : "idle";
  d.forEach(g => g.st = state(g));
  const showIgn = document.getElementById("showign").checked;
  const nIgn = d.filter(g => g.ignored).length;
  d = d.filter(g => showIgn || !g.ignored);
  const nLive = d.filter(g => g.st === "live").length, nIdle = d.filter(g => g.st === "idle").length;
  document.getElementById("kpi").innerHTML = `<div class="k">진행 중 <b>${nLive}</b></div><div class="k">멈춤 <b>${nIdle}</b></div><div class="k">끝남 <b>${d.length - nLive - nIdle}</b></div><div class="k">숨김 <b>${nIgn}</b></div>`;
  const only = document.getElementById("onlylive").checked;
  const order = {live: 0, idle: 1, done: 2};
  const rows = d.filter(g => !only || g.st !== "done").sort((a, b) => order[a.st] - order[b.st] || (a.idle ?? 1e9) - (b.idle ?? 1e9));
  document.getElementById("grid").innerHTML = rows.map(g => {
    const label = g.st === "live" ? "진행 중" : g.st === "idle" ? "멈춤" : (g.result === "human" ? "사람 승" : g.result === "ai" ? "AI 승" : "무승부");
    const wr = g.ai_wr == null ? null : Math.round(g.ai_wr * 100);
    const when = g.idle == null ? "이 현황판을 연 뒤 아직 수 없음" : `마지막 수 ${ago(g.idle)}`;
    return `<div class="card ${g.st}"><div class="top"><b>${g.moves.length}수</b><span class="pill ${g.st}">${label}</span></div>
      ${boardSVG(g.board, g.moves)}
      <div class="meta">사람 ${g.human === "X" ? "흑" : "백"} · ${g.model || "?"} · 탐색 ${g.sims}</div>
      ${wr == null ? "" : `<div class="meta">AI 승률 ${wr}%</div><div class="bar"><i style="width:${wr}%"></i></div>`}
      <div class="meta">${when}</div>
      <a class="btn" href="${g.watch}" target="_blank" rel="noopener">관전하기</a>
      <button class="hide" onclick="ign('${g.sid}', ${!g.ignored})">${g.ignored ? "다시 보이기" : "숨기기"}</button></div>`;
  }).join("") || `<div class="meta">지금 진행 중인 판이 없어요.</div>`;
}
document.getElementById("onlylive").addEventListener("change", tick);
document.getElementById("showign").addEventListener("change", tick);
async function ign(sid, on) { await fetch("/ignore", {method: "POST", body: new URLSearchParams({sid, on: on ? "1" : "0"})}); tick(); }
tick(); setInterval(tick, 2000);
</script></body></html>"""


# ---- 2026-10-16 (user): admin settings for the public board, written into index.html's CFG block ----
import re as _re
from urllib.parse import parse_qs
INDEX = os.path.join(HERE, "index.html")
MODELS = [("v3_ep530", "NAPS v3 ep530 (정오, 축제)"), ("ep880", "NAPS-880 (지난 축전)"), ("dawn", "여명 (졸업 모델)"), ("ep890", "NAPS-890"),
          ("ep900", "NAPS-900"), ("ep905", "NAPS-905"), ("latest", "최신 best")]
CFG_RE = _re.compile(r"/\*CFG\*/const CFG = (\{.*?\});/\*/CFG\*/")


def read_cfg():
    return json.loads(CFG_RE.search(open(INDEX, encoding="utf-8").read()).group(1))


def write_cfg(cfg):
    html = open(INDEX, encoding="utf-8").read()
    html = CFG_RE.sub(lambda m: "/*CFG*/const CFG = " + json.dumps(cfg, ensure_ascii=False) + ";/*/CFG*/", html)
    tmp = INDEX + ".tmp"
    open(tmp, "w", encoding="utf-8").write(html)
    os.replace(tmp, INDEX)                 # server.py reads index.html per request: live without a restart
    with open(os.path.join(HERE, "admin_settings_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=cfg), ensure_ascii=False) + "\n")


def settings_page(msg=""):
    c = read_cfg()
    opt = lambda pairs, cur: "".join(f'<option value="{v}"{" selected" if str(v) == str(cur) else ""}>{t}</option>' for v, t in pairs)
    chk = lambda k: " checked" if c.get(k) else ""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>웹 대국 설정</title><style>
:root{{--bg:#f2efe8;--panel:#fff;--ink:#23252a;--muted:#6c6f76;--line:#ddd6c8;--acc:#2f6db5;--ok:#2f8a52}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181c;--panel:#1f2227;--ink:#e6e7ea;--muted:#9aa0aa;--line:#32363d;--acc:#6aa0e6}}}}
:root[data-theme="dark"]{{--bg:#16181c;--panel:#1f2227;--ink:#e6e7ea;--muted:#9aa0aa;--line:#32363d;--acc:#6aa0e6}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.6 "Malgun Gothic",system-ui,sans-serif}}
form{{max-width:520px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;display:grid;gap:14px}}
label{{display:grid;gap:4px}}select,input[type=number]{{font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}}
.c{{display:flex;gap:8px;align-items:center}}button{{font:inherit;padding:10px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-weight:bold}}
.msg{{color:var(--ok);font-weight:bold}}.sub{{color:var(--muted);font-size:13px}}a{{color:var(--acc)}}
</style></head><body><p><a href="/">← 대국 현황판</a></p><h1 style="font-size:20px">웹 대국 설정</h1>
<form method="post" action="/settings">
<div class="msg">{msg}</div>
<label>AI 모델<select name="model">{opt(MODELS, c["model"])}</select></label>
<label>AI 탐색 횟수 (50~2000, 많을수록 강하고 느림)<input type="number" name="sims" min="50" max="2000" step="50" value="{c["sims"]}"></label>
<label>사람 돌<select name="human">{opt([("X", "흑 고정 (사람 선공)"), ("O", "백 고정 (AI 선공)"), ("choose", "사람이 선택")], c["human"])}</select></label>
<label class="c"><input type="checkbox" name="levels"{chk("levels")}> 사람이 난이도 선택 (쉬움 30 · 보통 150 · 매우 어려움 2000). 끄면 위 탐색 횟수로 고정</label>
<label class="c"><input type="checkbox" name="undo"{chk("undo")}> 무르기 허용</label>
<label class="c"><input type="checkbox" name="analysis"{chk("analysis")}> 분석 정보 보이기 (후보 수·방문 수·예상 수순)</label><button type="submit">저장</button>
<div class="sub">저장하면 바로 적용돼요. 이미 열려 있는 화면은 새로고침해야 바뀌고, 진행 중인 판의 모델·탐색 수는 새 게임부터 바뀝니다.</div>
</form></body></html>"""


def save_settings(form):
    q = parse_qs(form)
    g = lambda k, d=None: (q.get(k) or [d])[0]
    model = g("model")
    if model not in dict(MODELS):
        raise ValueError("모델 값이 올바르지 않아요")
    sims = max(50, min(2000, int(g("sims", "300"))))
    human = g("human") if g("human") in ("X", "O", "choose") else "X"
    cfg = {"model": model, "sims": sims, "human": human, "undo": "undo" in q, "analysis": "analysis" in q, "levels": "levels" in q}
    write_cfg(cfg)
    return cfg


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Authorization") != AUTH:          # 2026-10-16 (user): admin-only
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="omok admin", charset="UTF-8"')
            self.end_headers()
            return
        if self.path == "/settings":
            body, ctype = settings_page().encode(), "text/html; charset=utf-8"
        elif self.path == "/watch":
            body, ctype = open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8"
        elif self.path == "/api/state":
            req = urllib.request.Request(GAME + "/api/state", headers={"X-Session": self.headers.get("X-Session", "")})
            body, ctype = urllib.request.urlopen(req, timeout=5).read(), "application/json"
        elif self.path.split("?")[0] in ("/gwa", "/gwa/"):    # 2026-10-26 (user): watch the ep530 vs GWA4 match here too
            body, ctype = open(os.path.join(HERE, "..", "gwa_live", "index.html"), "rb").read(), "text/html; charset=utf-8"
        elif self.path.startswith("/gwa/live.json"):
            body, ctype = open(os.path.join(HERE, "..", "gwa_live", "live.json"), "rb").read(), "application/json"
        elif self.path == "/data":
            body, ctype = json.dumps(snapshot(), ensure_ascii=False).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get("Authorization") != AUTH:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="omok admin", charset="UTF-8"')
            self.end_headers()
            return
        if self.path == "/ignore":
            q = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            sid = (q.get("sid") or [""])[0]
            ig = ignored()
            if sid:
                (ig.add if (q.get("on") or ["1"])[0] == "1" else ig.discard)(sid)
                json.dump(sorted(ig), open(IGN, "w", encoding="utf-8"))
            self.send_response(204); self.end_headers(); return
        if self.path != "/settings":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        try:
            cfg = save_settings(self.rfile.read(n).decode("utf-8"))
            msg = f"저장했어요 ({time.strftime('%H:%M:%S')}): {cfg}"
        except Exception as e:
            msg = f"저장 실패: {e}"
        body = settings_page(msg).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=scan, daemon=True).start()
    print(f"dashboard on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
