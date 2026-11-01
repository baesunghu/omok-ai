// NAPS 11x11 in the browser (2026-11-01). The v3 network runs locally via onnxruntime-web; the search is a
// plain PUCT MCTS (c = 1.414, no noise) like v3/head2head.py, plus a quick win/block check in place of the
// C++ VCF solver. Freestyle: five or more in a row wins.
const N = 11, NC = N * N, WIN = 5;
const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];
let session = null;

export async function loadModel(url) {
  ort.env.wasm.numThreads = 1;   // static hosts (GitHub Pages) are not cross-origin isolated, so no wasm threads
  for (const ep of navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) {   // GPU in the browser when there is one
    try { session = await ort.InferenceSession.create(url, { executionProviders: [ep] }); backend = ep; return; }
    catch (e) { console.warn(ep, "unavailable:", e); }
  }
  throw new Error("no ONNX backend");
}
export let backend = "";

// board: Int8Array(121), 0 empty, 1 = X (black), 2 = O (white)
export function isWin(b, cell, who) {
  const r = (cell / N) | 0, c = cell % N;
  for (const [dr, dc] of DIRS) {
    let k = 1;
    for (const s of [1, -1]) {
      let rr = r + s * dr, cc = c + s * dc;
      while (rr >= 0 && rr < N && cc >= 0 && cc < N && b[rr * N + cc] === who) { k++; rr += s * dr; cc += s * dc; }
    }
    if (k >= WIN) return true;
  }
  return false;
}

function winningCells(b, who) {           // empty cells where `who` completes five now
  const out = [];
  for (let i = 0; i < NC; i++) if (b[i] === 0) { b[i] = who; if (isWin(b, i, who)) out.push(i); b[i] = 0; }
  return out;
}

// all 5-cell windows on the board (a five-or-more line always contains one)
const WINDOWS = [];
for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) for (const [dr, dc] of DIRS) {
  const er = r + 4 * dr, ec = c + 4 * dc;
  if (er >= 0 && er < N && ec >= 0 && ec < N) WINDOWS.push([0, 1, 2, 3, 4].map(k => (r + k * dr) * N + c + k * dc));
}

// fork plane: empty cells where placing `who` leaves >= 2 distinct immediate winning cells
// (same definition as alphago_lee._fork_mask). Window counting: a window with 4 `who` + 1 empty makes that empty
// a winning cell; placing at `a` removes `a` from that set and turns every window with 3 `who` + 2 empties
// (one of them `a`) into a new winning cell at the other empty.
function forkPlane(b, who, out, off) {
  const base = new Set(), threats = new Map();
  for (const w of WINDOWS) {
    let mine = 0, e1 = -1, e2 = -1, other = false;
    for (const i of w) {
      if (b[i] === who) mine++;
      else if (b[i] === 0) { if (e1 < 0) e1 = i; else if (e2 < 0) e2 = i; else { other = true; break; } }
      else { other = true; break; }
    }
    if (other) continue;
    if (mine === 4) base.add(e1);
    else if (mine === 3 && e2 >= 0) {
      if (!threats.has(e1)) threats.set(e1, new Set()); threats.get(e1).add(e2);
      if (!threats.has(e2)) threats.set(e2, new Set()); threats.get(e2).add(e1);
    }
  }
  for (let a = 0; a < NC; a++) {
    out[off + a] = 0;
    if (b[a] !== 0) continue;
    const lit = new Set(base); lit.delete(a);
    const t = threats.get(a);
    if (t) for (const x of t) lit.add(x);
    if (lit.size >= 2) out[off + a] = 1;
  }
}

export function planes(b, toMove) {        // (5, 11, 11): curr, opp, empty, fork_self, fork_opp
  const x = new Float32Array(5 * NC), opp = 3 - toMove;
  for (let i = 0; i < NC; i++) {
    x[i] = b[i] === toMove ? 1 : 0; x[NC + i] = b[i] === opp ? 1 : 0; x[2 * NC + i] = b[i] === 0 ? 1 : 0;
  }
  forkPlane(b, toMove, x, 3 * NC); forkPlane(b, opp, x, 4 * NC);
  return x;
}

async function evaluate(b, toMove) {
  const res = await session.run({ planes: new ort.Tensor("float32", planes(b, toMove), [1, 5, N, N]) });
  const lp = res.policy.data, v = res.value.data[0];
  const p = new Float32Array(NC); let s = 0;
  for (let i = 0; i < NC; i++) if (b[i] === 0) { p[i] = Math.exp(lp[i]); s += p[i]; }
  for (let i = 0; i < NC; i++) p[i] = s > 0 ? p[i] / s : (b[i] === 0 ? 1 : 0);
  return { p, v };
}

class Node {
  constructor(prior) { this.P = prior; this.N = 0; this.W = 0; this.kids = null; this.terminal = 0; }
}

// returns {move, visits: Float32Array, value} for `toMove` (1 or 2); onProgress(done, total)
export async function think(board, toMove, sims, onProgress) {
  const b = Int8Array.from(board);
  const mine = winningCells(b, toMove);
  if (mine.length) return { move: mine[0], value: 1, visits: null };
  const theirs = winningCells(b, 3 - toMove);
  if (theirs.length) return { move: theirs[0], value: -0.9, visits: null };
  const root = new Node(1);
  for (let s = 0; s < sims; s++) {
    const path = [root], bb = Int8Array.from(b);
    let node = root, who = toMove, value;
    while (node.kids && !node.terminal) {
      let best = -1, bestU = -Infinity;
      const sq = Math.sqrt(node.N);
      for (const [mv, ch] of node.kids) {
        const q = ch.N ? -ch.W / ch.N : 0;            // child W is from the child's mover view
        const u = q + 1.414 * ch.P * sq / (1 + ch.N);
        if (u > bestU) { bestU = u; best = mv; }
      }
      bb[best] = who;
      node = node.kids.get(best);
      path.push(node);
      if (isWin(bb, best, who)) node.terminal = 1;
      else if (!bb.includes(0)) node.terminal = 2;
      who = 3 - who;
    }
    if (node.terminal === 1) value = -1;          // side to move here has just lost
    else if (node.terminal === 2) value = 0;
    else {
      const { p, v } = await evaluate(bb, who);
      node.kids = new Map();
      for (let i = 0; i < NC; i++) if (bb[i] === 0) node.kids.set(i, new Node(p[i]));
      value = v;
    }
    for (let i = path.length - 1; i >= 0; i--) { path[i].N++; path[i].W += value; value = -value; }
    if (onProgress && (s % 8 === 7)) { onProgress(s + 1, sims); await new Promise(r => setTimeout(r)); }
  }
  let move = -1, most = -1;
  const visits = new Float32Array(NC);
  for (const [mv, ch] of root.kids) { visits[mv] = ch.N; if (ch.N > most) { most = ch.N; move = mv; } }
  const ch = root.kids.get(move);
  return { move, visits, value: ch.N ? -ch.W / ch.N : 0 };
}
