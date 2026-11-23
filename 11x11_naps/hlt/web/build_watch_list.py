"""Spectator link list for every session in live_games.json (2026-10-15, user).

The web board's spectator mode needs the session id (index.html#watch=<sid>); this writes watch_list.html with one
link per session. The session id ends with Date.now() in base 36, which gives the time the browser first connected.
Usage: python build_watch_list.py [tunnel_url]
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://cigarette-papers-coleman-clip.trycloudflare.com").rstrip("/")


def sid_time(sid):
    for k in (8, 9):
        try:
            d = datetime.datetime.fromtimestamp(int(sid[-k:], 36) / 1000)
        except (ValueError, OSError, OverflowError):
            continue
        if 2025 < d.year < 2028:
            return d
    return None


rows = []
for sid, g in json.load(open(os.path.join(HERE, "live_games.json"), encoding="utf-8")).items():
    t = sid_time(sid)
    m = g.get("model_info") or {}
    state = {"ai": "AI 승", "human": "사람 승", "draw": "무승부"}.get(g.get("result"), "?") if g.get("over") else "진행 중"
    rows.append((t or datetime.datetime.min, sid, m.get("name") or g.get("model_key"), g["human"], g["sims"],
                 len(g.get("moves") or []), state))
rows.sort(reverse=True)

items = "\n".join(
    f'<tr><td>{t.strftime("%m-%d %H:%M") if t != datetime.datetime.min else "?"}</td>'
    f'<td><span class="st {"live" if st == "진행 중" else "done"}">{st}</span></td><td>{name}</td>'
    f'<td>{"흑(선공)" if hu == "X" else "백(후공)"}</td><td class="n">{sims}</td><td class="n">{n}</td>'
    f'<td><a href="{BASE}/#watch={sid}" target="_blank" rel="noopener">관전</a> <code>{sid}</code></td></tr>'
    for t, sid, name, hu, sims, n, st in rows)
html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>관전 세션 목록</title><style>
:root{{--bg:#f3f1ec;--panel:#fff;--ink:#22252b;--muted:#6b6f78;--line:#ddd8cc;--acc:#2f6db5;--live:#2f8a52}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#17191d;--panel:#202328;--ink:#e6e7ea;--muted:#9aa0aa;--line:#33373e;--acc:#6aa0e6;--live:#5cc184}}}}
:root[data-theme="dark"]{{--bg:#17191d;--panel:#202328;--ink:#e6e7ea;--muted:#9aa0aa;--line:#33373e;--acc:#6aa0e6;--live:#5cc184}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.5 "Malgun Gothic",system-ui,sans-serif}}
h1{{font-size:19px;margin:0 0 4px}}p{{color:var(--muted);margin:0 0 12px;font-size:13px}}
.box{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;min-width:640px}}td,th{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{font-size:12px;color:var(--muted);font-weight:normal}}.n{{font-variant-numeric:tabular-nums;text-align:right}}
a{{color:var(--acc);font-weight:bold}}code{{color:var(--muted);font-size:11px}}
.st{{font-size:11px;padding:1px 8px;border-radius:999px;color:#fff;background:var(--muted)}}.st.live{{background:var(--live)}}
</style></head><body><h1>관전 세션 목록</h1>
<p>{datetime.datetime.now():%m-%d %H:%M} 기준 · 서버에 저장된 세션 {len(rows)}개 · "관전"을 누르면 그 판을 읽기 전용으로 실시간으로 따라갑니다</p>
<div class="box"><table><tr><th>처음 접속</th><th>상태</th><th>모델</th><th>사람</th><th class="n">탐색</th><th class="n">수</th><th>관전 링크</th></tr>
{items}</table></div></body></html>"""
open(os.path.join(HERE, "watch_list.html"), "w", encoding="utf-8").write(html)
for t, sid, name, hu, sims, n, st in rows:
    print(f"{t:%m-%d %H:%M} {st:5} {n:3}수 {BASE}/#watch={sid}")
