"""Export a v3 checkpoint to ONNX for the in-browser demo (2026-11-01, user: free hosting, model runs in the browser).
Input: (B, 5, 11, 11) float32 planes [curr, opp, empty, fork_self, fork_opp] (alphago_lee.board_to_input_array_from_numeric).
Outputs: policy (B, 121) log-probabilities, value (B, 1) in [-1, 1] for the side to move.
Checks onnxruntime against PyTorch on random real-looking boards.
Usage: python web/export_onnx.py [checkpoint] [out.onnx]   (run from 11x11_v2)
"""
import os
import sys

import numpy as np
import torch

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V2); sys.path.insert(1, os.path.join(V2, "v3"))
os.chdir(V2)
from alphago_lee import board_to_input_array_from_numeric
from net_v3 import NetV3

ck = sys.argv[1] if len(sys.argv) > 1 else "v3/model/v3_latest.pth"
out = sys.argv[2] if len(sys.argv) > 2 else "web/site/naps_v3.onnx"
sd = torch.load(ck, map_location="cpu")
if "model" in sd:
    print("epoch", sd.get("epoch")); sd = sd["model"]
m = NetV3(); m.load_state_dict(sd); m.eval()
os.makedirs(os.path.dirname(out), exist_ok=True)
x = torch.zeros(1, 5, 11, 11)
torch.onnx.export(m, x, out, input_names=["planes"], output_names=["policy", "value"],
                  dynamic_axes={"planes": {0: "b"}, "policy": {0: "b"}, "value": {0: "b"}}, opset_version=17)
print("wrote", out, os.path.getsize(out) // 1024, "KB")

import onnxruntime as ort
rng = np.random.default_rng(0)
boards = []
for _ in range(16):
    b = np.zeros((11, 11), np.int8)
    cells = rng.choice(121, rng.integers(2, 40), replace=False)
    for i, c in enumerate(cells):
        b.flat[c] = 1 + i % 2
    boards.append(board_to_input_array_from_numeric(b, "X" if len(cells) % 2 == 0 else "O"))
xb = np.stack(boards).astype(np.float32)
with torch.no_grad():
    pt, vt = m(torch.from_numpy(xb))
po, vo = ort.InferenceSession(out).run(None, {"planes": xb})
print("max |policy diff|", float(np.abs(pt.numpy().reshape(16, -1) - po.reshape(16, -1)).max()),
      "max |value diff|", float(np.abs(vt.numpy().reshape(-1) - vo.reshape(-1)).max()))
np.savez("web/site_check.npz", x=xb, p=po, v=vo)   # reference values for checking the JS feature encoder later
