"""Text the user "epXXX 정오 달성!" when a v3 checkpoint meets the NOON bar (2026-10-24, user request).

NOON (user, 10/23-24; all-or-nothing, no "quasi"):
  every tier-1 problem passed (all 5, incl. the observation one) + every tier-2 problem (5/5) + tier 3 >= 15/18.
Watches model/tactical_problem_log.jsonl every 60 s; one SMS per checkpoint, only for full official runs.
SMS goes through 10x10/promotion_sms.py send() (Twilio credentials in %USERPROFILE%\\.naps_twilio.json).
Usage: python v3/noon_sms_watch.py [--dry-run]     (run from 11x11_v2)
"""
import importlib.util
import json
import os
import re
import sys
import time

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TLOG = os.path.join(V2, "model", "tactical_problem_log.jsonl")
PROBS = os.path.join(V2, "tactical_problems", "web_game58_problems.json")
STATE = os.path.join(V2, "v3", "noon_sms_state.json")
spec = importlib.util.spec_from_file_location("psms", os.path.join(os.path.dirname(V2), "10x10", "promotion_sms.py"))
psms = importlib.util.module_from_spec(spec); spec.loader.exec_module(psms)
tier = {p["id"]: p.get("tier", 1) for p in json.load(open(PROBS, encoding="utf-8"))["problems"]}
DRY = "--dry-run" in sys.argv


def noon(rec):
    res = {x["id"]: x["passed"] for x in rec.get("results", [])}
    by = {t: [res[i] for i in res if tier.get(i) == t] for t in (1, 2, 3)}
    if len(by[1]) < 5 or len(by[2]) < 5 or len(by[3]) < 18:   # only full official runs (all three tiers graded)
        return False
    return all(by[1]) and all(by[2]) and sum(by[3]) >= 15


sent = set(json.load(open(STATE))) if os.path.exists(STATE) else set()
while True:
    for line in open(TLOG, encoding="utf-8"):
        rec = json.loads(line)
        m = re.search(r"v3_ep(\d+)\.pth$", str(rec.get("checkpoint", "")))
        if not m or m.group(1) in sent or not noon(rec):
            continue
        body = f"ep{m.group(1)} 정오 달성!"
        if DRY:
            print("would send:", body)
        else:
            sid, st = psms.send(body)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "sent:", body, sid, st, flush=True)
        sent.add(m.group(1))
        json.dump(sorted(sent), open(STATE, "w"))
    if DRY:
        break
    time.sleep(60)
