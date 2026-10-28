"""Policy probability of the Rapfi answer on failed tier-2/3 problems, per checkpoint (2026-10-23, user).
Usage: python v3/policy_probe.py name=path [name=path ...]   (run from 11x11_v2). Raw net outputs, no search."""
import json, re, sys, numpy as np, torch
sys.path.insert(1, "v3")
from net_v3 import NetV3
from alphago_lee import board_to_input_array_from_numeric
IDS = ["gwa_loss_0929_2341_p35", "gwa_loss_0929_2351_p50", "gwa_loss_0930_0106_p16", "gwa_loss_0930_0109_p27", "gwa_loss_0930_0220_p4",
       "gwa_loss_0930_0257_p13", "gwa_loss_0930_0259_p22", "gwa_loss_0930_0303_p6", "gwa_loss_0930_1048_p14", "gwa_loss_0930_0004_p43"]
ps = {p["id"]: p for p in json.load(open("tactical_problems/web_game58_problems.json", encoding="utf-8"))["problems"]}
ms = {}
for a in sys.argv[1:]:
    k, p = a.split("=", 1); m = NetV3(); sd = torch.load(p, map_location="cpu"); m.load_state_dict(sd.get("model", sd)); m.eval(); ms[k] = m
print("problem      answer  " + "  ".join(f"{k:>18}" for k in ms))
for i in IDS:
    p = ps[i]; mm = re.findall(r"plays \((\d+), (\d+)\)", p.get("note", ""))
    ans = tuple(map(int, mm[-1])) if mm else tuple(p.get("answer", (0, 0)))
    b = np.zeros((11, 11), np.int8)
    for c, r, cc in p["moves"]: b[r][cc] = 1 if c == "X" else 2
    x = torch.from_numpy(board_to_input_array_from_numeric(b, 1 if p["side_to_move"] == "X" else 2)[None]).float()
    row = []
    for m in ms.values():
        with torch.no_grad(): lp, v = m(x)
        pr = torch.exp(lp)[0].numpy(); j = ans[0] * 11 + ans[1]
        row.append(f"{pr[j]*100:5.1f}% #{int((pr > pr[j]).sum()) + 1:<3} v{float(v):+.2f}")
    print(f"{i[9:]:<13}{str(ans):<8}" + "  ".join(f"{r:>18}" for r in row))
