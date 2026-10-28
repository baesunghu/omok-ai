"""
Rapfi match with a pooled GPU evaluator (2026-10-10, EXPERIMENTAL, measurement only).

Same games as rapfi_match.py (same openings, colours, sims, greedy, no noise), but organised like the
training self-play pool (selfplay_mp_unified.py):
  * W worker processes each play games (own Rapfi process + own C++ MCTSEngine). Workers hold NO
    network; they encode leaf boards into input planes (CPU work, in parallel) and send them to
  * one coordinator process that owns the network on the GPU, gathers requests from all workers until
    TARGET boards are pending or MAX_WAIT seconds pass, runs ONE forward pass, and sends each worker
    its slice back.
Batch composition changes the cuDNN kernel, so outputs can differ by ~1e-5 from rapfi_match.py:
results are statistically, not bit-for-bit, the same.

Usage: python rapfi_match_pooled.py games sims n<nodes> checkpoint [workers]
Env:   RAPFI_MATCH_LOG (default model/rapfi_match_log.jsonl)
"""
import json, os, sys, time
import multiprocessing as mp
from multiprocessing.connection import wait as conn_wait

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET, MAX_WAIT = 48, 0.004


def coordinator(ckpt, conns, ready, stats_q):
    sys.path.insert(0, HERE); os.chdir(HERE)
    import numpy as np, torch
    from alphago_lee_v3_survival import AlphaZeroTicTacToeV3
    dev = torch.device("cuda")
    m = AlphaZeroTicTacToeV3(); m.load_state_dict(torch.load(ckpt, map_location=dev)); m.to(dev).eval()
    ready.set()
    live = list(conns); flushes = boards = 0
    while live:
        pending = []
        for c in conn_wait(live):            # block until at least one request (or a close)
            try:
                pending.append((c, c.recv()))
            except EOFError:
                live.remove(c)
        if not pending:
            continue
        deadline = time.time() + MAX_WAIT
        while sum(len(p[1]) for p in pending) < TARGET and live:
            left = deadline - time.time()
            if left <= 0:
                break
            got = conn_wait([c for c in live if c not in [p[0] for p in pending]], timeout=left)
            if not got:
                break
            for c in got:
                try:
                    pending.append((c, c.recv()))
                except EOFError:
                    live.remove(c)
        planes = np.concatenate([p[1] for p in pending], axis=0)
        with torch.inference_mode():
            pl, v = m(torch.from_numpy(planes).float().to(dev))
        probs = torch.exp(pl).cpu().numpy().astype(np.float64)
        vals = v.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
        k = 0
        for c, x in pending:
            n = len(x)
            c.send((probs[k:k + n].reshape(n, -1), vals[k:k + n]))
            k += n
        flushes += 1; boards += len(planes)
    stats_q.put((flushes, boards))


def worker(wid, nw, games, sims, nodes, ckpt_name, conn, log_path, out_q):
    sys.path.insert(0, HERE); os.chdir(HERE)
    import numpy as np
    import mcts_core
    from alphago_lee import board_to_input_array_from_numeric
    from fast_board import fast_check_win
    from rapfi_match import Rapfi, OPENINGS, N, WIN_LENGTH

    def ev(bs, ps):
        conn.send(np.stack([board_to_input_array_from_numeric(b, p) for b, p in zip(bs, ps)]).astype(np.float32))
        return conn.recv()

    rapfi = Rapfi(30000, nodes)
    try:
        for g in range(games):
            if g % nw != wid:
                continue
            ai = 'X' if g % 2 == 0 else 'O'
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            board = [[' '] * N for _ in range(N)]
            moves = [('X', opening[0], opening[1])]; board[opening[0]][opening[1]] = 'X'
            turn, result, t0 = 'O', None, time.time()
            engine = mcts_core.MCTSEngine()
            while True:
                if len(moves) >= N * N:
                    result = "draw"; break
                if turn == ai:
                    _, _, cands = engine.search_batched(board, turn, ev, sims, 1.414, False, 0.3, 0.25, N, WIN_LENGTH, 16)
                    r, c = max(cands, key=lambda x: x['visit_count'])['move']
                else:
                    r, c = rapfi.move(moves, turn)
                board[r][c] = turn; moves.append((turn, r, c))
                if fast_check_win(board, turn):
                    result = "ai_win" if turn == ai else "rapfi_win"; break
                turn = 'O' if turn == 'X' else 'X'
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ai": ai, "opening": opening, "result": result,
                   "plies": len(moves), "sims": sims, "rapfi_ms": None, "rapfi_nodes": nodes,
                   "checkpoint": ckpt_name, "game": g, "moves": moves, "pooled": True}
            out_q.put(row)
            print(f"game {g + 1}/{games}: AI={ai} -> {result} in {len(moves)} plies ({time.time() - t0:.0f}s)", flush=True)
    finally:
        rapfi.close(); conn.close()


def main():
    games, sims, lim, ckpt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    nodes = int(lim[1:])
    log_path = os.environ.get("RAPFI_MATCH_LOG", os.path.join(HERE, "model", "rapfi_match_log.jsonl"))
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(nw)]
    ready, stats_q, out_q = ctx.Event(), ctx.Queue(), ctx.Queue()
    co = ctx.Process(target=coordinator, args=(ckpt, [a for a, _ in pairs], ready, stats_q)); co.start()
    ready.wait(120)
    ws = [ctx.Process(target=worker, args=(i, nw, games, sims, nodes, os.path.basename(ckpt), pairs[i][1], log_path, out_q))
          for i in range(nw)]
    t0 = time.time()
    for w in ws: w.start()
    for _, b in pairs: b.close()           # parent keeps no worker ends
    rows = []
    while len(rows) < games:
        rows.append(out_q.get())
    for w in ws: w.join()
    for a, _ in pairs: a.close()
    co.join(30)
    fl, bd = stats_q.get(timeout=30)
    with open(log_path, "a", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["game"]):
            f.write(json.dumps(r) + "\n")
    print(f"\n{games} games in {time.time() - t0:.0f}s with {nw} workers | GPU flushes {fl}, avg {bd / max(fl, 1):.1f} boards/flush")


if __name__ == "__main__":
    main()
