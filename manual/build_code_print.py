"""Printable core-source book for paper study (2026-10-09, user).
Collects the files that manual part 2 ("코드 해부") dissects, in data-flow order, renders them
with line numbers and syntax highlighting, and prints one A4 PDF (each file starts on a new page,
page numbers in the footer) via headless Chromium.
Run:  python build_code_print.py   -> manual/NAPS_핵심코드_원본.pdf
"""
import html
import os

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BatchLexer, CppLexer, HtmlLexer, PythonLexer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_HTML = os.path.join(HERE, "_code_print.html")
OUT_PDF = os.path.join(HERE, "NAPS_핵심코드_원본.pdf")

# (path relative to final/, role, manual part-2 chapter)
FILES = [
    ("10x10/_restart_train.py", "학습 실행 설정. 모든 하이퍼파라미터가 여기서 들어간다", "부록 6.3"),
    ("10x10/fast_board.py", "보드 크기, 승리 판정의 가장 밑바닥", "2부 1장"),
    ("10x10/alphago_lee.py", "판을 신경망 입력(숫자)으로 바꾸기, 초기 V1 신경망", "2부 2장"),
    ("10x10/alphago_lee_v2_seblock.py", "V2 신경망: SE 블록 + GroupNorm 잔차망", "2부 3장"),
    ("10x10/alphago_lee_v3_survival.py", "V3 신경망: V2 + 생존 헤드 (지금 쓰는 구조)", "2부 3장"),
    ("cpp_mcts/mcts_core_solver.cpp", "C++ MCTS 엔진: 노드, PUCT, Solver, 강제 탐색, 트리 재사용", "2부 4~7장"),
    ("10x10/alphazero_mcts_lrfix.py", "파이썬 MCTS 래퍼: 신경망 평가를 C++ 엔진에 연결", "2부 8장"),
    ("10x10/selfplay_mp_unified.py", "10프로세스 통합 대국 풀: 자가대국, 리그, 게이트를 한 풀에서", "2부 9장"),
    ("10x10/alphazero_mcts_make_lrfix.py", "학습 본체: 대국 생성, 버퍼, 학습, 게이트, 리그 선택", "2부 10~12장"),
    ("10x10/rapfi_match.py", "Rapfi와 대국하기 (pbrain 프로토콜)", "4부 6장"),
    ("10x10/rapfi_ladder.py", "Rapfi 사다리와 동등 노드 계산 (가중 직선 맞춤)", "4부 8~10장"),
    ("10x10/tactical_problem_check.py", "졸업 문제 채점 (3단계 검증)", "4부 13장"),
]

# 2026-10-09 (user): second group -- surrounding tools (web board, spectator, watchers, minimax baseline)
EXTRA_FILES = [
    ("10x10/determinisic_play.py", "결정론적 미니맥스 상대의 파이썬 입구 (C++로 위임)", "0부 0.7, 4부 4장"),
    ("cpp_mcts/deterministic_play_core_fast_deploy.cpp", "결정론적 미니맥스 C++ 본체 (배포본, 깊이 2)", "0부 0.7, 4부 4장"),
    ("cpp_mcts/build_mcts_core_solver.bat", "C++ 엔진 빌드 스크립트 (MSVC + pybind11)", "3부 2장"),
    ("10x10/watchdog_.py", "학습 프로세스 감시와 자동 재시작", "3부"),
    ("10x10/rapfi_ladder_watch.py", "20에폭마다 Rapfi 측정과 문제 채점을 자동 실행", "4부 8장"),
    ("10x10/rapfi_progress.py", "진행 중인 Rapfi 측정 보기", "-"),
    ("10x10/hourly_status.py", "한 시간마다 상태 기록", "-"),
    ("10x10/loss_trend_annotated.py", "손실 추세 그래프", "-"),
    ("10x10/value_head_reset.py", "가치망 머리만 초기화 (9/27, ep408)", "5부 8막"),
    ("10x10/mp_eval.py", "멀티프로세스 평가", "3부"),
    ("10x10/fork_status_check.py", "포크(쌍삼) 상태 점검", "-"),
    ("10x10/gomokuwithai_match.py", "Gomoku With AI 사이트와 자동 대국 (Playwright)", "5부 11막"),
    ("10x10/hlt/web/server.py", "웹 분석판 서버 (모델 선택, 배치 평가)", "3부"),
    ("10x10/hlt/web/index.html", "웹 분석판 화면", "-"),
    ("10x10/hlt/web/spectate.py", "관전석 서버 (모델끼리 대국)", "-"),
    ("10x10/hlt/web/spectate.html", "관전석 화면", "-"),
]

CSS = """
@page { size: A4; margin: 14mm 12mm 16mm 14mm; }
body { font-family: 'Malgun Gothic', sans-serif; color: #111; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 6mm; }
.cover p, .cover li { font-size: 10.5pt; line-height: 1.6; }
table.toc { border-collapse: collapse; width: 100%; font-size: 9.5pt; }
table.toc th, table.toc td { border-bottom: 1px solid #ccc; padding: 3px 6px; text-align: left; vertical-align: top; }
table.toc td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.file { page-break-before: always; }
.fhead { border-bottom: 2px solid #111; margin-bottom: 3mm; padding-bottom: 1.5mm; }
.fhead .path { font-family: Consolas, monospace; font-size: 12.5pt; font-weight: bold; }
.fhead .meta { font-size: 9pt; color: #444; margin-top: 1mm; }
.highlight pre, .highlighttable pre { margin: 0; }
table.highlighttable { border-collapse: collapse; width: 100%; table-layout: fixed; }
td.linenos { width: 9mm; padding-right: 2mm; vertical-align: top; }
td.linenos pre { color: #888; text-align: right; }
td.code { vertical-align: top; }
pre { font-family: Consolas, 'D2Coding', monospace; font-size: 7.6pt; line-height: 1.32;
      white-space: pre-wrap; word-break: break-all; }
td.linenos pre { white-space: pre; }
"""


# ---------------------------------------------------------------------------
# Appendix (2026-10-09, user): graphs and original records after the code.
# ---------------------------------------------------------------------------
import base64
import glob
import json
import re

TENX = os.path.join(ROOT, "10x10")
M = os.path.join(TENX, "model")

APPENDIX_CSS = """
.app { page-break-before: always; }
.app h2 { font-size: 15pt; border-bottom: 2px solid #111; padding-bottom: 1.5mm; margin: 0 0 3mm; }
.app h3 { font-size: 11.5pt; margin: 4mm 0 2mm; }
.app p, .app li { font-size: 9.5pt; line-height: 1.55; }
.app img.fig { width: 100%; }
.app .cap { font-size: 8.5pt; color: #444; margin-top: 1.5mm; }
.app table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 2mm 0; }
.app th, .app td { border: 1px solid #bbb; padding: 2px 5px; text-align: left; vertical-align: top; }
.app th { background: #eee; }
.app td.n { text-align: right; font-variant-numeric: tabular-nums; }
.app pre.txt { font-size: 8.6pt; line-height: 1.45; background: #f6f6f6; border: 1px solid #ddd; padding: 3mm; }
.app .md code { font-family: Consolas, monospace; font-size: 8.4pt; }
.app .md pre { font-size: 8pt; background: #f6f6f6; padding: 2mm; }
"""


def _img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def _read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def _promotions():
    promo = set()
    for f in glob.glob(os.path.join(TENX, "train_stdout_*.log")):
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.search(r"\[(2026-\d\d-\d\d \d\d:\d\d:\d\d)\] Epoch (\d+)/\d+.*PROMOTED", line)
            if m:
                promo.add((m.group(1), int(m.group(2))))
    return sorted(promo)


def _ladder_table():
    promo = _promotions()

    def best_at(t):
        b = None
        for ts, e in promo:
            if ts <= t:
                b = e
        return b
    rows = []
    for line in open(os.path.join(M, "rapfi_ladder_log.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        m = re.search(r"ep(\d+)", r["checkpoint"])
        ep, kind = (int(m.group(1)), "최신" if "latest" in r["checkpoint"] else "best") if m else (best_at(r["time"]), "best")
        lv = {x["nodes"]: x for x in r["levels"]}
        cells = "".join(f"<td class='n'>{lv[n]['win']}-{lv[n]['draw']}-{lv[n]['loss']}<br>{lv[n]['score']*100:.1f}%</td>"
                        if n in lv else "<td></td>" for n in (100, 300, 1000, 2000, 3000))
        eq = r.get("equal_nodes")
        eqs = f"{eq:,.0f}" if isinstance(eq, (int, float)) else (eq or "")
        rows.append(f"<tr><td>{r['time'][:16]}</td><td class='n'>{ep}</td><td>{kind}</td>{cells}<td class='n'>{eqs}</td></tr>")
    return ("<table><tr><th>측정 시각</th><th>에폭</th><th>종류</th><th>100</th><th>300</th><th>1000</th><th>2000</th>"
            "<th>3000</th><th>동등 노드</th></tr>" + "".join(rows) + "</table>"
            "<p class='cap'>각 칸은 승-무-패와 점수. 동등 노드는 측정 당시 기록된 값이다. 9/29 이전 기록은 옛 계산법(단일 교차점)으로 "
            "기록되어 있어 가중 직선 맞춤 값과 다르다(4부 9~10장, NAPS 계보 문서의 표 참고). 에폭은 체크포인트 이름 또는 그 시각의 "
            "최근 승격 에폭으로 추정했다.</p>")


def _problem_table():
    rows = []
    for line in open(os.path.join(M, "tactical_problem_log.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        res = {x["id"]: x for x in r["results"]}
        def cell(pid):
            x = res.get(pid)
            return "<td></td>" if not x else f"<td class='n'>{'O' if x['passed'] else 'X'} {x.get('trials_passed', '')}</td>"
        rows.append(f"<tr><td>{r['time'][:16]}</td><td>{html.escape(r['checkpoint'])}</td>"
                    + "".join(cell(p) for p in ("web58_p5_attack", "web58_p7_attack", "webB_quiet_three_defense",
                                                 "web58_d13_defense", "rapfi_cand3_diag_defense"))
                    + f"<td class='n'>{r.get('grad_passed', '')}/{r.get('grad_total', '')}</td></tr>")
    return ("<table><tr><th>채점 시각</th><th>체크포인트</th><th>5수 공격</th><th>7수 공격</th><th>열린3 방어</th>"
            "<th>13수 방어</th><th>대각선(관찰)</th><th>졸업</th></tr>" + "".join(rows) + "</table>"
            "<p class='cap'>O/X = 통과 여부, 숫자 = 3번 시도 중 통과 횟수(초기 기록은 1번 시도). 빈칸은 그 시점에 없던 문제.</p>")


def _board_svg(moves, n=11, cell=30, bad=None, good=None, numbers=True):
    pad = int(cell * 1.4)
    size = pad * 2 + cell * (n - 1)
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 {size} {size}'>",
         f"<rect width='{size}' height='{size}' fill='#e8c98f'/>"]
    for i in range(n):
        p = pad + i * cell
        s.append(f"<line x1='{pad}' y1='{p}' x2='{pad + cell * (n - 1)}' y2='{p}' stroke='#5a4420'/>")
        s.append(f"<line x1='{p}' y1='{pad}' x2='{p}' y2='{pad + cell * (n - 1)}' stroke='#5a4420'/>")
        s.append(f"<text x='{p}' y='{pad - cell * 0.75}' font-size='{cell * 0.36:.0f}' text-anchor='middle'>{i}</text>")
        s.append(f"<text x='{pad - cell * 0.8}' y='{p + 4}' font-size='{cell * 0.36:.0f}' text-anchor='middle'>{i}</text>")
    fs = cell * 0.33
    for k, (pl, r, c) in enumerate(moves, 1):
        x, y = pad + c * cell, pad + r * cell
        black = pl in ("B", "X")
        s.append(f"<circle cx='{x}' cy='{y}' r='{cell * 0.45}' fill='{'#111' if black else '#fafafa'}' stroke='#222'/>")
        if numbers:
            s.append(f"<text x='{x}' y='{y + fs * 0.36}' font-size='{fs:.1f}' text-anchor='middle' "
                     f"fill='{'#fff' if black else '#111'}'>{k}</text>")
    if good:
        x, y = pad + good[1] * cell, pad + good[0] * cell
        s.append(f"<circle cx='{x}' cy='{y}' r='{cell * 0.42}' fill='none' stroke='#1b8a3a' stroke-width='4'/>")
    if bad:
        x, y = pad + bad[1] * cell, pad + bad[0] * cell
        d = cell * 0.3
        s.append(f"<path d='M{x - d},{y - d} L{x + d},{y + d} M{x + d},{y - d} L{x - d},{y + d}' stroke='#d0021b' stroke-width='4'/>")
    s.append("</svg>")
    return "".join(s)


def _game_section():
    sgf = _read(os.path.join(M, "recovered_games", "dawn_white_vs_gomokuwithai_20260929.sgf"))
    note = _read(os.path.join(M, "recovered_games", "dawn_white_vs_gomokuwithai_20260929.txt"))
    moves = [(pl, ord(rc[1]) - 97, ord(rc[0]) - 97) for pl, rc in re.findall(r";([BW])\[([a-k]{2})\]", sgf)]
    return (f"<h3>여명(백) 대 Gomoku With AI (2026-09-29, {len(moves)}수, 오목 없이 중단된 판)</h3>"
            f"<pre class='txt'>{html.escape(note.strip())}</pre>"
            f"<div style='text-align:center'>{_board_svg(moves)}</div>"
            "<p class='cap'>흑 = 사이트 AI, 백 = 여명. 숫자는 수순. 좌표는 (행, 열), 0부터. 원본 SGF는 "
            "model/recovered_games/에 있다.</p>")


def _problem_book():
    probs = json.load(open(os.path.join(TENX, "tactical_problems", "web_game58_problems.json"), encoding="utf-8"))["problems"]
    stage_name = {0: "관찰용", 1: "1차(졸업)", 2: "2차(장학금)"}
    q, a = [], []
    for k, p in enumerate(probs, 1):
        moves = [tuple(m) for m in p["moves"]]
        side = "흑" if p["side_to_move"] == "X" else "백"
        kind = "공격: 이기는 수순을 찾아라" if p["type"] == "attack" else "방어: 지지 않는 수를 찾아라"
        tier = p.get("tier")
        label = stage_name.get(p.get("stage", 1), "") + (f" · {tier}단계 검증" if tier else "")
        q.append(f"<section class='app'><h2>문제 {k}. {html.escape(p['name'])}</h2>"
                 f"<p><b>{side} 차례</b> · {kind} · {label} · {len(moves)}수 진행</p>"
                 f"<div style='text-align:center'>{_board_svg(moves, cell=46)}</div>"
                 "<p class='cap'>숫자는 수순, 좌표는 (행, 열) 0부터. 정답은 뒤쪽 정답편에.</p></section>")
        good = None
        if p["type"] == "attack" and p.get("solution_rapfi"):
            good = p["solution_rapfi"][0][1:]
            ans = "Rapfi의 필승 수순: " + " → ".join(f"{'흑' if m[0] == 'X' else '백'}({m[1]},{m[2]})" for m in p["solution_rapfi"])
        else:
            m = re.search(r"plays \(?(\d+), ?(\d+)\)?", p.get("note", ""))
            good = [int(m.group(1)), int(m.group(2))] if m else None
            ans = ((f"오답(빨간 X): {tuple(p['bad_move'])}" if p.get("bad_move") else "")
                   + (f" · 살리는 수(초록 원): {tuple(good)}" if good else " · 살리는 수는 하나로 정해지지 않음"))
        a.append(f"<div style='break-inside:avoid;margin-bottom:5mm'><h3>문제 {k}. {html.escape(p['name'])}</h3>"
                 f"<p>{html.escape(ans)}</p>"
                 f"<div style='text-align:center'>{_board_svg(moves, cell=27, bad=p.get('bad_move'), good=good, numbers=False)}</div>"
                 f"<p class='cap'>{html.escape(p.get('note', ''))}</p></div>")
    intro = ("<section class='app'><h2>부록 H. 문제집</h2><p>졸업 문제와 2차 문제를 판 그림으로 모았다. "
             f"모두 {len(probs)}문제. 한 쪽에 한 문제씩, 정답과 해설은 모든 문제 뒤의 정답편에 모았다. "
             "정답은 Rapfi(한 수 3초)로 검증한 수다.</p></section>")
    return intro + "".join(q) + "<section class='app'><h2>문제집 정답편</h2>" + "".join(a) + "</section>"


def _game_pages(n_spectate):
    pages = []
    web = [json.loads(l) for l in open(os.path.join(TENX, "hlt", "web", "games.jsonl"), encoding="utf-8")]
    for k, g in enumerate(web, 1):
        mi = g.get("model") or {}
        mname = mi.get("name") or (f"epoch {mi['epoch']}" if mi.get("epoch") else mi.get("checkpoint", ""))
        res = {"ai": "AI 승", "human": "사람 승", "draw": "무승부"}.get(g["result"], g["result"])
        title = (f"웹 대국 {k} · {g['time'][:16]} · 사람({'흑' if g['human'] == 'X' else '백'}) 대 "
                 f"AI({'흑' if g['ai'] == 'X' else '백'}, {mname}, {g['sims']} sims) · {res} · {len(g['moves'])}수")
        pages.append((title, g["moves"]))
    for fn, tag in (("gomokuwithai_match_log.jsonl", "고정 1100 sims"), ("gomokuwithai_match_timematched_log.jsonl", "같은 시간 조건")):
        for k, g in enumerate([json.loads(l) for l in open(os.path.join(M, fn), encoding="utf-8")], 1):
            res = {"dawn": "여명 승", "site": "사이트 AI 승", "draw": "무승부"}.get(g["result"], g["result"])
            title = (f"여명 대 Gomoku With AI 어려움 ({tag}) {k} · {g['time'][:16]} · 여명 {'흑' if g['dawn'] == 'X' else '백'} · "
                     f"{res} · {g['plies']}수")
            pages.append((title, g["moves"]))
    sp = [json.loads(l) for l in open(os.path.join(M, "spectate_log.jsonl"), encoding="utf-8")]
    for k, g in enumerate(sp[:n_spectate], 1):
        winner = {"black": g["black_name"], "white": g["white_name"]}.get(g["result"], "")
        title = (f"관전석 {k} · {g['time'][:16]} · 흑 {g['black_name']} 대 백 {g['white_name']} · {g['sims']} sims · "
                 f"{'무승부' if g['result'] == 'draw' else winner + ' 승'} · {g['plies']}수")
        pages.append((title, g["moves"]))
    out = [f"<section class='app'><h2>부록 I. 대국 기보 모음</h2><p>웹 분석판에서 사람과 둔 {len(web)}판, "
           "Gomoku With AI 어려움 단계와 여명이 둔 70판, 관전석에서 epoch 820과 여명이 흑백을 바꿔 가며 둔 "
           f"{min(n_spectate, len(sp))}판(전체 {len(sp)}판 중 앞에서부터)을 한 쪽에 한 판씩 실었다. 흑 = X, 숫자 = 수순.</p></section>"]
    for title, moves in pages:
        out.append(f"<section class='app'><h3>{html.escape(title)}</h3>"
                   f"<div style='text-align:center'>{_board_svg([tuple(m) for m in moves], cell=46)}</div></section>")
    return "".join(out)


def appendix(n_spectate=60):
    import markdown
    out = []
    figs = [
        ("loss_trend_annotated.png", "손실 추세 전체 구간 (NAPS 2차 런, 파이프라인 변경 시점 표시, 보라 점선 = 여명 625의 손실)"),
        ("loss_trend_last50.png", "최근 50에폭 손실과 게이트 (검은 가로선 = 실제 판정 점수 승 + 무/2)"),
        ("rapfi_ladder_history.png", "Rapfi 동등 노드 추이 (가중 직선 맞춤으로 다시 계산, 2026-10-08 기준)"),
        ("gate_draw_promo_trend.png", "게이트 무승부 비율과 승격 빈도 추세 (2026-10-08, 805 에폭까지)"),
    ]
    out.append("<section class='app'><h2>부록 A. 그래프</h2><p>이 묶음을 만든 시점(2026-10-09)의 그래프다. "
               "학습이 계속되므로 최신 상태는 model/ 폴더의 같은 이름 파일을 본다.</p></section>")
    for f, cap in figs:
        p = os.path.join(M, f)
        if os.path.exists(p):
            out.append(f"<section class='app'><img class='fig' src='{_img(p)}'><div class='cap'>{html.escape(cap)} · model/{f}</div></section>")
    out.append("<section class='app'><h2>부록 B. 졸업 선언문</h2>"
               f"<pre class='txt'>{html.escape(_read(os.path.join(M, 'graduated_11x11', 'DECLARATION.txt')))}</pre>"
               f"<h3>졸업 모델 README</h3><pre class='txt'>{html.escape(_read(os.path.join(M, 'graduated_11x11', 'README.txt')))}</pre></section>")
    out.append("<section class='app'><h2>부록 C. NAPS 가문의 계보</h2><div class='md'>"
               + markdown.markdown(_read(os.path.join(TENX, "docs", "NAPS_lineage.md")), extensions=["tables", "fenced_code"])
               + "</div></section>")
    out.append("<section class='app'><h2>부록 D. 3대 레전드 질문</h2><div class='md'>"
               + markdown.markdown(_read(os.path.join(M, "graduated_11x11", "LEGEND_QUESTIONS.md")), extensions=["tables", "fenced_code"])
               + "</div></section>")
    out.append("<section class='app'><h2>부록 E. Rapfi 사다리 전체 기록</h2>" + _ladder_table() + "</section>")
    out.append("<section class='app'><h2>부록 F. 졸업 문제 채점 기록</h2>" + _problem_table() + "</section>")
    out.append("<section class='app'><h2>부록 G. 기보</h2>" + _game_section() + "</section>")
    out.append(_problem_book())
    out.append(_game_pages(n_spectate))
    return "".join(out)


def main(n_spectate=60):
    fmt = HtmlFormatter(linenos="table", style="default", nowrap=False)
    total = 0
    rows, bodies = [], []
    for i, (rel, role, chap) in enumerate(FILES, 1):
        path = os.path.join(ROOT, rel)
        src = open(path, encoding="utf-8", errors="replace").read()
        n = len(src.splitlines())
        total += n
        lexer = CppLexer() if rel.endswith(".cpp") else PythonLexer()
        rows.append(f"<tr><td class='n'>{i}</td><td><code>{html.escape(rel)}</code><br>{html.escape(role)}</td>"
                    f"<td>{chap}</td><td class='n'>{n:,}</td></tr>")
        bodies.append(f"<section class='file'><div class='fhead'><div class='path'>{i}. {html.escape(rel)}</div>"
                      f"<div class='meta'>{html.escape(role)} · 매뉴얼 {chap} · {n:,}줄</div></div>"
                      f"{highlight(src, lexer, fmt)}</section>")
    rows.append("<tr><td colspan='4'><b>주변 도구</b> (웹 분석판, 관전석, 감시 프로그램, 미니맥스 기준 상대)</td></tr>")
    for j, (rel, role, chap) in enumerate(EXTRA_FILES, len(FILES) + 1):
        path = os.path.join(ROOT, rel)
        src = open(path, encoding="utf-8", errors="replace").read()
        n = len(src.splitlines())
        total += n
        lexer = (CppLexer() if rel.endswith(".cpp") else PythonLexer() if rel.endswith(".py")
                 else HtmlLexer() if rel.endswith(".html") else BatchLexer())
        rows.append(f"<tr><td class='n'>{j}</td><td><code>{html.escape(rel)}</code><br>{html.escape(role)}</td>"
                    f"<td>{chap}</td><td class='n'>{n:,}</td></tr>")
        bodies.append(f"<section class='file'><div class='fhead'><div class='path'>{j}. {html.escape(rel)}</div>"
                      f"<div class='meta'>{html.escape(role)} · {n:,}줄</div></div>"
                      f"{highlight(src, lexer, fmt)}</section>")
    cover = f"""<section class='cover'><h1>NAPS 핵심 코드 원본</h1>
<p>NAPS-625 여명 오버홀 매뉴얼의 2부 「코드 해부」가 다루는 파일을 데이터가 흐르는 순서로 묶었다.
2026-10-09 기준의 실제 파일이며, 각 파일은 새 쪽에서 시작한다. 줄 번호는 원본 파일의 줄 번호와 같다.
코드 뒤에는 부록으로 그래프(A), 졸업 선언문(B), NAPS 계보(C), 레전드 질문(D), Rapfi 사다리 기록(E),
졸업 문제 채점 기록(F), 기보(G), 문제집(H), 대국 기보 모음(I)을 붙였다.</p>
<table class='toc'><tr><th>#</th><th>파일과 역할</th><th>매뉴얼</th><th>줄 수</th></tr>{''.join(rows)}
<tr><td></td><td><b>합계</b></td><td></td><td class='n'><b>{total:,}</b></td></tr></table>
<p>참고: 학습 중인 트레이너가 쓰는 엔진 바이너리(10x10/mcts_core.pyd)는 이 C++ 파일에서 웹 분석판용
principal_variation 함수만 빠진 빌드다(9/28 백업 mcts_core_solver.cpp.v3_before_pv와 같은 내용).</p></section>"""
    doc = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><style>{CSS}{APPENDIX_CSS}"
           f"{fmt.get_style_defs('.highlight')}</style></head><body>{cover}{''.join(bodies)}{appendix(n_spectate)}</body></html>")
    open(OUT_HTML, "w", encoding="utf-8").write(doc)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page()
        page.goto("file:///" + OUT_HTML.replace("\\", "/"))
        page.pdf(path=OUT_PDF, format="A4", print_background=True, display_header_footer=True,
                 header_template="<span></span>",
                 footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#666'>"
                                 "NAPS 핵심 코드 원본 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                 margin={"top": "14mm", "bottom": "16mm", "left": "14mm", "right": "12mm"})
        b.close()
    print("lines", total, "->", OUT_PDF)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
