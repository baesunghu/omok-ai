"""Static replay viewer for every web game (2026-10-15, user: "접속된 세션 판을 전부 보고 싶다").

Reads games.jsonl (finished games) + live_games.json (games still in progress) and writes games_viewer.html with
the data embedded. Local file only -- these are visitors' games, not for publishing.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "games_viewer.html")

games = []
for line in open(os.path.join(HERE, "games.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    m = r.get("model") or {}
    games.append(dict(time=r["time"], model=m.get("name") or m.get("key") or "?", human=r["human"], sims=r["sims"],
                      result=r["result"], moves=r["moves"], wr=r.get("ai_winrate") or [], undos=r.get("undos", 0),
                      live=False))
try:
    live = json.load(open(os.path.join(HERE, "live_games.json"), encoding="utf-8"))
except (OSError, ValueError):
    live = {}
for sid, g in live.items():
    if g.get("over") or not g.get("moves"):
        continue                                         # finished ones are already in games.jsonl
    m = g.get("model_info") or {}
    games.append(dict(time="진행 중", model=m.get("name") or g.get("model_key") or "?", human=g["human"],
                      sims=g["sims"], result=None, moves=g["moves"], wr=g.get("winrate") or [],
                      undos=g.get("undos", 0), live=True))

html = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>웹 대국 기록</title>
<style>
:root{--bg:#f3f1ec;--panel:#fff;--ink:#22252b;--muted:#6b6f78;--line:#ddd8cc;--wood:#d9b77a;--grid:#5a4628;
--acc:#2f6db5;--ai:#b5442f;--hu:#2f8a52;--dr:#7a7f8a;--sel:#e7eefa}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17191d;--panel:#202328;--ink:#e6e7ea;
--muted:#9aa0aa;--line:#33373e;--wood:#a3824f;--grid:#3b2d18;--acc:#6aa0e6;--ai:#e0775f;--hu:#5cc184;--dr:#9aa0aa;--sel:#26324a}}
:root[data-theme="dark"]{--bg:#17191d;--panel:#202328;--ink:#e6e7ea;--muted:#9aa0aa;--line:#33373e;--wood:#a3824f;
--grid:#3b2d18;--acc:#6aa0e6;--ai:#e0775f;--hu:#5cc184;--dr:#9aa0aa;--sel:#26324a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Malgun Gothic",system-ui,sans-serif}
header{padding:14px 16px 6px}h1{margin:0;font-size:19px}.sum{color:var(--muted);font-size:13px;margin-top:3px}
.wrap{display:grid;grid-template-columns:minmax(260px,360px) 1fr;gap:14px;padding:10px 16px 20px}
@media (max-width:760px){.wrap{grid-template-columns:1fr}}
.list{background:var(--panel);border:1px solid var(--line);border-radius:8px;max-height:82vh;overflow:auto}
.filters{display:flex;gap:6px;flex-wrap:wrap;padding:8px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
.filters button{border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:999px;padding:3px 10px;cursor:pointer;font:inherit;font-size:12px}
.filters button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.item{display:grid;grid-template-columns:auto 1fr auto;gap:2px 8px;padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.item:hover,.item.sel{background:var(--sel)}.item .n{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px}
.item .m{font-size:12px;color:var(--muted);grid-column:2/4}
.pill{font-size:11px;padding:1px 7px;border-radius:999px;color:#fff;white-space:nowrap}
.p-ai{background:var(--ai)}.p-human{background:var(--hu)}.p-draw{background:var(--dr)}.p-live{background:var(--acc)}
.view{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:10px;min-width:0}
.meta{font-size:13px;color:var(--muted)}.meta b{color:var(--ink)}
canvas{width:100%;max-width:560px;aspect-ratio:1;display:block;margin:0 auto}
.ctl{display:flex;gap:6px;align-items:center;justify-content:center;flex-wrap:wrap}
.ctl button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer}
.ctl input[type=range]{width:min(320px,60vw)}.ctl .pos{font-variant-numeric:tabular-nums;min-width:70px;text-align:center}
button:focus-visible,.item:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.wr{width:100%;max-width:560px;height:90px;margin:0 auto;display:block}
label.tg{font-size:12px;color:var(--muted);display:flex;gap:4px;align-items:center}
</style></head><body>
<header><h1>웹 대국 기록</h1><div class="sum" id="sum"></div></header>
<div class="wrap">
<div class="list"><div class="filters" id="filters"></div><div id="items"></div></div>
<div class="view">
<div class="meta" id="meta"></div>
<canvas id="bd" width="560" height="560"></canvas>
<div class="ctl"><button id="first">처음</button><button id="prev">◀</button>
<input type="range" id="rng" min="0" value="0"><button id="next">▶</button><button id="last">끝</button>
<span class="pos" id="pos"></span><label class="tg"><input type="checkbox" id="nums" checked>수 번호</label></div>
<svg class="wr" id="wr" viewBox="0 0 560 90" preserveAspectRatio="none"></svg>
<div class="meta">AI 승률 (AI가 수를 둘 때 기록) · ← → 키로 한 수씩</div>
</div></div>
<script>
const G=__DATA__;const N=11;let cur=-1,step=0,filt='all';
const RES={ai:['AI 승','p-ai'],human:['사람 승','p-human'],draw:['무승부','p-draw']};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function tag(g){return g.live?['진행 중','p-live']:RES[g.result]||['?','p-draw']}
function side(g){return g.human==='X'?'사람 흑(선공)':'사람 백(후공)'}
const tot={ai:0,human:0,draw:0};G.forEach(g=>{if(!g.live)tot[g.result]=(tot[g.result]||0)+1});
document.getElementById('sum').textContent=`끝난 판 ${G.filter(g=>!g.live).length} (AI 승 ${tot.ai} · 사람 승 ${tot.human} · 무승부 ${tot.draw})`+
 (G.some(g=>g.live)?` · 진행 중 ${G.filter(g=>g.live).length}`:'');
const F=[['all','전체'],['human','사람 승'],['ai','AI 승'],['draw','무승부'],['live','진행 중']];
const fe=document.getElementById('filters');
F.forEach(([k,t])=>{const b=document.createElement('button');b.textContent=t;b.onclick=()=>{filt=k;render()};b.dataset.k=k;fe.appendChild(b)});
function render(){fe.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.k===filt));
 const it=document.getElementById('items');it.innerHTML='';
 G.map((g,i)=>[g,i]).reverse().filter(([g])=>filt==='all'||(filt==='live'?g.live:(!g.live&&g.result===filt))).forEach(([g,i])=>{
  const d=document.createElement('div');d.className='item'+(i===cur?' sel':'');d.tabIndex=0;
  const [t,c]=tag(g);
  d.innerHTML=`<span class="n">#${i+1}</span><span>${g.time.slice(0,16)}</span><span class="pill ${c}">${t}</span>`+
   `<span class="m">${g.model} · ${side(g)} · ${g.sims}회 탐색 · ${g.moves.length}수${g.undos>0?' · 무르기 '+g.undos:''}</span>`;
  d.onclick=()=>pick(i);d.onkeydown=e=>{if(e.key==='Enter')pick(i)};it.appendChild(d)})}
function pick(i){cur=i;step=G[i].moves.length;const r=document.getElementById('rng');r.max=G[i].moves.length;render();draw()}
function draw(){const cv=document.getElementById('bd'),x=cv.getContext('2d'),S=cv.width,m=30,c=(S-2*m)/(N-1);
 x.fillStyle=css('--wood');x.fillRect(0,0,S,S);x.strokeStyle=css('--grid');x.lineWidth=1;
 x.fillStyle=css('--grid');x.font='12px sans-serif';x.textAlign='center';x.textBaseline='middle';
 for(let k=0;k<N;k++){x.beginPath();x.moveTo(m,m+k*c);x.lineTo(S-m,m+k*c);x.stroke();x.beginPath();x.moveTo(m+k*c,m);x.lineTo(m+k*c,S-m);x.stroke();
  x.fillText(String.fromCharCode(65+k),m+k*c,12);x.fillText(k+1,12,m+k*c)}
 [[5,5],[2,2],[2,8],[8,2],[8,8]].forEach(([r,q])=>{x.beginPath();x.arc(m+q*c,m+r*c,3,0,7);x.fill()});
 if(cur<0)return;const g=G[cur],mv=g.moves.slice(0,step),show=document.getElementById('nums').checked;
 mv.forEach(([p,r,q],k)=>{const X=m+q*c,Y=m+r*c;x.beginPath();x.arc(X,Y,c*.44,0,7);
  x.fillStyle=p==='X'?'#111':'#f6f6f2';x.fill();x.lineWidth=1.2;x.strokeStyle=p==='X'?'#a0a0a8':'#3a3a3a';x.stroke();
  if(show||k===mv.length-1){x.fillStyle=p==='X'?'#fff':'#111';x.font=(k===mv.length-1?'bold ':'')+Math.round(c*.36)+'px sans-serif';x.fillText(k+1,X,Y+1)}
  if(k===mv.length-1){x.strokeStyle='#e04a2f';x.lineWidth=2.4;x.beginPath();x.arc(X,Y,c*.48,0,7);x.stroke()}});
 const [t]=tag(g);document.getElementById('meta').innerHTML=`<b>#${cur+1}</b> ${g.time} · <b>${t}</b> · ${g.model} · ${side(g)} · ${g.sims}회 탐색`;
 document.getElementById('pos').textContent=`${step} / ${g.moves.length}`;document.getElementById('rng').value=step;
 const w=document.getElementById('wr'),L=Math.max(1,g.moves.length);let s=`<line x1="0" y1="45" x2="560" y2="45" stroke="${css('--line')}" stroke-dasharray="4 4"/>`;
 const pts=g.wr.map(o=>[o.move_no/L*560,90-o.ai_winrate*90]);
 if(pts.length)s+=`<polyline fill="none" stroke="${css('--ai')}" stroke-width="2" points="${pts.map(p=>p.join(',')).join(' ')}"/>`;
 s+=`<line x1="${step/L*560}" y1="0" x2="${step/L*560}" y2="90" stroke="${css('--acc')}" stroke-width="1.5"/>`;w.innerHTML=s}
const go=d=>{if(cur<0)return;step=Math.max(0,Math.min(G[cur].moves.length,step+d));draw()};
document.getElementById('prev').onclick=()=>go(-1);document.getElementById('next').onclick=()=>go(1);
document.getElementById('first').onclick=()=>go(-999);document.getElementById('last').onclick=()=>go(999);
document.getElementById('rng').oninput=e=>{step=+e.target.value;draw()};document.getElementById('nums').onchange=draw;
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)});
render();draw();if(G.length)pick(G.length-1);
</script></body></html>"""
open(OUT, "w", encoding="utf-8").write(html.replace("__DATA__", json.dumps(games, ensure_ascii=False)))
print(f"{len(games)} games ({sum(g['live'] for g in games)} in progress) -> {OUT}")
