"""Text the user's phone when a gate PROMOTES a new best (2026-10-11, user request).

Watches model/training_report_11x11.txt every 60 s. When a new "-> PROMOTED" gate line appears, it sends ONE SMS
through Twilio's REST API (no extra packages). Only promotions are sent -- nothing else.

Credentials are NOT in this repo. Put them in %USERPROFILE%\\.naps_twilio.json :
    {"account_sid": "AC...", "auth_token": "...", "from": "+1...", "to": "+8210..."}

Usage:
    python promotion_sms.py --dry-run    print what would be sent for the latest promotion, send nothing
    python promotion_sms.py --test       send one test message now (to check delivery)
    python promotion_sms.py              watch forever (starts from the current report, no backlog)
"""
import base64, json, os, re, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "model", "training_report_11x11.txt")
CONF = os.path.join(os.path.expanduser("~"), ".naps_twilio.json")
STATE = os.path.join(HERE, "model", "promotion_sms_state.json")
LOG = os.path.join(HERE, "promotion_sms.log")
LADDER = os.path.join(HERE, "model", "rapfi_ladder_log.jsonl")
LADDER_ALERT = 10000   # second tier LADDER_ALERT2 = 20000 (정오 bar), same logic, added 2026-10-12 # 2026-10-12 (user): also text when a ladder run's equal point is >= 10,000 nodes
GATE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d):\d\d\] Epoch (\d+)/\d+ \| Policy Loss: ([\d.]+) \| Value Loss: ([\d.]+).*?"
                  r"Gate vs best \((\d+) games\): Win (\d+)% / Draw (\d+)% / Loss (\d+)% -> PROMOTED")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


LADDER_PROMO = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d):\d\d\] Epoch (\d+) \| Ladder promotion: (\S+) -> best \(eq (\d+) vs best (\d+), P ([\d.]+)\) -> PROMOTED \(ladder\)")


def promotions():
    out = []
    with open(REPORT, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GATE.search(line) or LADDER_PROMO.search(line)
            if m:
                out.append(m)
    return out


def text_for(m):
    if m.re is LADDER_PROMO:   # 2026-10-12: ladder promotion path
        when, ep, name, eq, beq, p = m.groups()
        return f"NAPS {name} 사다리 승격! 대등 {int(eq):,} vs {int(beq):,} ({when[11:]})"
    when, ep, pol, val, n, w, d, l = m.groups()
    n = int(n); W, D, L = (round(int(x) * n / 100) for x in (w, d, l))
    return f"NAPS ep{ep} 승격! {W}승{D}무{L}패 ({when[11:]})"   # short: one segment even with the trial prefix


def send(body):
    c = json.load(open(CONF, encoding="utf-8-sig"))  # Notepad may add a BOM
    url = f"https://api.twilio.com/2010-04-01/Accounts/{c['account_sid']}/Messages.json"
    data = urllib.parse.urlencode({"From": c["from"], "To": c["to"], "Body": body}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{c['account_sid']}:{c['auth_token']}".encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode())
    return res.get("sid"), res.get("status")


def ladder_hits():
    """(line_no, text) for ladder runs whose equal point is >= LADDER_ALERT (or above the top level)."""
    out = []
    if not os.path.exists(LADDER):
        return out
    with open(LADDER, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            eq = r.get("equal_nodes")
            ep = re.search(r"ep(\d+)", r.get("checkpoint", ""))
            ep = ep.group(1) if ep else "?"
            top = max((L.get("nodes", 0) for L in r.get("levels", [])), default=0)
            # 2026-10-12: a short ladder (e.g. 1500/3000/5000) can extrapolate far past its top level;
            # only claim a value the ladder actually reached, otherwise say "above the top level"
            if eq == "above max level" or (isinstance(eq, (int, float)) and eq > top):
                if top >= LADDER_ALERT:
                    tag = " 2만 돌파" if top >= 20000 else ""
                    out.append((i, f"NAPS ep{ep} 대등국면 {top:,}노드 초과!{tag} ({r['time'][11:16]})"))
                elif isinstance(eq, (int, float)) and eq >= LADDER_ALERT:
                    out.append((i, f"NAPS ep{ep} 최고단계 {top:,} 초과, 추정 {eq:,.0f}노드 ({r['time'][11:16]})"))
            elif isinstance(eq, (int, float)) and eq >= 20000:
                out.append((i, f"NAPS ep{ep} 대등국면 {eq:,.0f}노드! 2만 돌파 ({r['time'][11:16]})"))
            elif isinstance(eq, (int, float)) and eq >= LADDER_ALERT:
                out.append((i, f"NAPS ep{ep} 대등국면 {eq:,.0f}노드! ({r['time'][11:16]})"))
    return out


def main():
    if "--dry-run" in sys.argv:
        p = promotions()
        print(text_for(p[-1]) if p else "(no promotion in the report yet)")
        return
    if "--test" in sys.argv:
        print(send("NAPS 알림 테스트: 승격되면 이렇게 문자가 와요."))
        return
    last = promotions()
    seen = int(last[-1].group(2)) if last else -1
    if os.path.exists(STATE):
        seen = max(seen, json.load(open(STATE)).get("last_epoch", -1))
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    with open(LADDER, encoding="utf-8", errors="ignore") as f:
        ladder_seen = st.get("ladder_line", sum(1 for _ in f) - 1)
    log(f"watching for promotions after ep{seen}, ladder runs after line {ladder_seen} (alert >= {LADDER_ALERT:,})")
    while True:
        try:
            for m in promotions():
                ep = int(m.group(2))
                if ep > seen:
                    body = text_for(m)
                    sid, status = send(body)
                    log(f"sent '{body}' sid={sid} status={status}")
                    seen = ep
                    json.dump({"last_epoch": seen, "ladder_line": ladder_seen}, open(STATE, "w"))
            for i, body in ladder_hits():
                if i > ladder_seen:
                    sid, status = send(body)
                    log(f"sent '{body}' sid={sid} status={status}")
            n = sum(1 for _ in open(LADDER, encoding="utf-8", errors="ignore")) - 1
            if n != ladder_seen:
                ladder_seen = n
                json.dump({"last_epoch": seen, "ladder_line": ladder_seen}, open(STATE, "w"))
        except Exception as e:   # network or Twilio error: keep watching, retry next minute
            log(f"error: {e!r}")
        time.sleep(60)


if __name__ == "__main__":
    main()
