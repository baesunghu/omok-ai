"""2026-11-11 (user: re-verify CATFISH at the original recipe, game-noise ~1 %): extra head-to-head games on the SAME checkpoints
but with openings that head2head.py's 100-opening set (H2H_ALL) never used. head2head is noiseless, so replaying old openings would
only repeat the same games.
Openings: first stone in rows/cols 3..7, second within +-2 of it (both on the board), minus the 100 of H2H_ALL, shuffled with the fixed
seed 20261111; env H2H_EXT_FROM / H2H_EXT_TO pick the slice [from, to) (openings, 2 games each), H2H_SHARD "k/n" still works.
Usage: python head2head_ext.py A.pth B.pth [sims]   (run from 11x11_v2/v3)"""
import os
import random

os.environ["H2H_ALL"] = "1"
_shard = os.environ.pop("H2H_SHARD", None)   # head2head would shard the OLD set at import; we shard the new one below
import head2head as H   # noqa: E402  (module-level OPENINGS = the 100 used before)
if _shard:
    os.environ["H2H_SHARD"] = _shard

old = set(H.OPENINGS)
pool = [((r, c), (r + dr, c + dc)) for r in range(3, 8) for c in range(3, 8)
        for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]
pool = [o for o in pool if o not in old]
random.Random(20261111).shuffle(pool)
lo = int(os.environ.get("H2H_EXT_FROM", "0"))
hi = int(os.environ.get("H2H_EXT_TO", "150"))
ops = pool[lo:hi]
if os.environ.get("H2H_SHARD"):
    k, n = map(int, os.environ["H2H_SHARD"].split("/"))
    ops = ops[k::n]
H.OPENINGS = ops
print(f"extra openings: pool {len(pool)}, slice [{lo},{hi}) -> {len(ops)} openings = {2 * len(ops)} games", flush=True)
H.main()
