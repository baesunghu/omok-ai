# smoke test for train_v3.py: a few self-play games with a random NetV3, rows, one training pass
import importlib.util
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("t", "train_v3.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
import torch

t.GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
m = t.NetV3().to(t.DEV)
print("start", time.strftime("%H:%M:%S"), flush=True)
raw, st = t.selfplay(m)
print(st, flush=True)
rows, w, l = t.build_rows(raw)
print("X/O/D", w, "avg len", sum(l) / len(l), {k: v.shape for k, v in rows.items()}, flush=True)
print("policy sums", rows["p"].sum(1)[:4], "own classes", [int((rows["w"] == c).sum()) for c in range(3)], flush=True)
buf = t.Buffer()
buf.add(rows)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
print("train", t.train(m, opt, buf, len(rows["v"])), flush=True)
