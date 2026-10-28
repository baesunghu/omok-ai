#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <cstring>

namespace py = pybind11;

struct Node {
    Node* parent = nullptr;
    std::map<int, std::unique_ptr<Node>> children;
    int n_visits = 0;
    double total_value = 0.0;
    double mean_value = 0.0;
    double prior_prob = 1.0;
    int player_turn = 1;  // 1=X, 2=O
    bool is_expanded = false;
    // MCTS-Solver (mcts_core_solver.cpp, 2026-09-27): game-theoretic result
    // for the player to move at THIS node, once proven: +1 = that player
    // wins, -1 = that player loses, 0 = unknown. proof_dist = plies from this
    // node to the deciding five under the proof (fastest win / slowest loss).
    int proven = 0;
    int proof_dist = 0;

    void update(double leaf_value) {
        n_visits++;
        total_value += leaf_value;
        mean_value = total_value / n_visits;
    }
};

static bool check_win(const std::vector<int8_t>& b, int player, int BOARD_SIZE, int WIN_LENGTH) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (b[r * BOARD_SIZE + c] != player) continue;
            if (c <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++)
                    if (b[r * BOARD_SIZE + c + k] != player) { ok = false; break; }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++)
                    if (b[(r + k) * BOARD_SIZE + c] != player) { ok = false; break; }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH && c <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++)
                    if (b[(r + k) * BOARD_SIZE + (c + k)] != player) { ok = false; break; }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH && c >= WIN_LENGTH - 1) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++)
                    if (b[(r + k) * BOARD_SIZE + (c - k)] != player) { ok = false; break; }
                if (ok) return true;
            }
        }
    }
    return false;
}

static bool check_draw(const std::vector<int8_t>& b) {
    for (auto v : b) if (v == 0) return false;
    return true;
}

// ============================================================
// Neighbor-radius move pruning -- identical to mcts_core_pruned_deploy.cpp.
// See that file's comment for the full rule (empty-board center-2x2 seed,
// Chebyshev-radius neighborhood, all-empty-cell safety net).
// ============================================================
static const int NEIGHBOR_RADIUS = 2;

static std::vector<int> relevant_cells(const std::vector<int8_t>& b, int BOARD_SIZE) {
    std::vector<int> occupied;
    occupied.reserve(b.size());
    for (int i = 0; i < (int)b.size(); i++) {
        if (b[i] != 0) occupied.push_back(i);
    }

    if (occupied.empty()) {
        int mid = BOARD_SIZE / 2;
        std::vector<int> center;
        center.push_back((mid - 1) * BOARD_SIZE + (mid - 1));
        center.push_back((mid - 1) * BOARD_SIZE + mid);
        center.push_back(mid * BOARD_SIZE + (mid - 1));
        center.push_back(mid * BOARD_SIZE + mid);
        return center;
    }

    std::vector<char> seen(b.size(), 0);
    std::vector<int> relevant;
    for (int idx : occupied) {
        int r0 = idx / BOARD_SIZE, c0 = idx % BOARD_SIZE;
        int rlo = std::max(0, r0 - NEIGHBOR_RADIUS), rhi = std::min(BOARD_SIZE - 1, r0 + NEIGHBOR_RADIUS);
        int clo = std::max(0, c0 - NEIGHBOR_RADIUS), chi = std::min(BOARD_SIZE - 1, c0 + NEIGHBOR_RADIUS);
        for (int r = rlo; r <= rhi; r++) {
            for (int c = clo; c <= chi; c++) {
                int j = r * BOARD_SIZE + c;
                if (b[j] == 0 && !seen[j]) {
                    seen[j] = 1;
                    relevant.push_back(j);
                }
            }
        }
    }

    if (relevant.empty()) {
        std::vector<int> all_empty;
        for (int i = 0; i < (int)b.size(); i++) if (b[i] == 0) all_empty.push_back(i);
        return all_empty;
    }
    return relevant;
}

// Mate-distance terminal values (mcts_core_mate_distance.cpp, 2026-09-25).
// In 11x11 free-style gomoku the first player wins nearly every game, so
// from the defender's side almost every line ends at -1 and the search had
// no reason to prefer a move that holds out longer. Scaling a terminal
// result by how many plies below the root it was reached makes a later
// loss slightly less bad than an earlier one (and a faster win slightly
// better than a slower one) -- the same idea as a chess engine's
// mate-in-N scoring. The floor keeps deep forced results clearly decisive.
// Depth is measured from the current root; after tree-reuse promotion,
// values already stored in a reused subtree were scaled against the old
// root (1-2 plies deeper), a small uniform offset accepted for simplicity.
static const double MATE_DISTANCE_ALPHA = 0.005;
static const double MATE_DISTANCE_FLOOR = 0.5;

static double mate_scaled(double v, int depth) {
    double mag = 1.0 - MATE_DISTANCE_ALPHA * depth;
    if (mag < MATE_DISTANCE_FLOOR) mag = MATE_DISTANCE_FLOOR;
    return v * mag;
}

// ============================================================
// MCTS-Solver (Winands, Bjornsson & Saito, CG 2008), 2026-09-27.
//
// Plain MCTS averages. A move after which the opponent has an immediate
// five kept a Q of only ~-0.86 (an average over the opponent's replies,
// not all of which were the winning one), so when the value net ALSO
// misjudged the blocking move (measured at ep470: "after blocking, still
// lost" at -0.9), the search could not tell block from non-block and left
// immediate fours unblocked in 3/20 fixed test positions.
//
// Proven results are backed up minimax-style instead:
//   * at expansion, every child move that makes five for the mover is a
//     terminal: that child is a proven LOSS for its own mover, and the
//     expanded node is a proven WIN (proof_dist 1);
//   * a node is a proven WIN if ANY child is a proven loss (for the
//     child's mover), and a proven LOSS if ALL children are proven wins;
//   * selection never enters a child that is a proven win for its mover
//     (a provably losing move) while any unproven alternative exists, and
//     stops at proven nodes, backing up their exact (mate-scaled) value;
//   * the returned visit distribution drops provably losing moves, and a
//     proven-win root reports the fastest winning move only.
// This uses only the game rules (terminal detection), no gomoku tactics.
// "ALL children" means all children in the neighbor-radius candidate set,
// which always contains every cell adjacent to a stone -- the only cells
// that can complete a five or block one.
// ============================================================
static bool wins_at(const std::vector<int8_t>& b, int idx, int player, int BOARD_SIZE, int WIN_LENGTH) {
    // True if `player` placing a stone on the (empty) cell idx makes
    // WIN_LENGTH or more in a row -- same freestyle rule as check_win.
    const int r0 = idx / BOARD_SIZE, c0 = idx % BOARD_SIZE;
    static const int DR[4] = {0, 1, 1, 1};
    static const int DC[4] = {1, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int count = 1;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            int r = r0 + sgn * DR[d], c = c0 + sgn * DC[d];
            while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && b[r * BOARD_SIZE + c] == player) {
                count++;
                r += sgn * DR[d];
                c += sgn * DC[d];
            }
        }
        if (count >= WIN_LENGTH) return true;
    }
    return false;
}

// ============================================================
// VCF (victory by continuous fours), 11x11_v2, 2026-10-21 (user).
// The side to move wins if it can keep making fours -- each four leaves the
// defender exactly one blocking cell -- until it gets two winning cells at
// once (double four / open four) or completes five. Pure game-rule search:
// no learned or hand-tuned gomoku knowledge. Bounded by attacker depth and a
// node budget so it can run at every node expansion.
//   vcf_first_move(b, A, &plies) -> first attacking cell or -1;
//   plies = length of the forced line up to the deciding five.
// A block that itself makes five for the defender refutes the line, and if
// the defender already has a five-point on the attacker's turn (and the
// attacker has no immediate five) the line fails: the attacker must block.
// ============================================================
static int g_vcf_depth = 0;          // max attacker fours per line; 0 = VCF off
static long g_vcf_budget = 3000;     // node budget per vcf_first_move call

// All WIN_LENGTH-cell windows of the board as cell-index lists (built once
// per board geometry). Speed-up 2026-10-21: under freestyle rules an empty
// cell wins for `player` iff some window through it holds WIN_LENGTH-1 of
// that player's stones, and a move can only make a four if some window
// through it holds WIN_LENGTH-2 of the mover's stones and no defender stone.
// Both scans give exactly the cells the old full-board wins_at loops found.
static const std::vector<std::vector<int>>& board_windows(int BOARD_SIZE, int WIN_LENGTH) {
    static thread_local std::vector<std::vector<int>> W;   // per thread: searches run GIL-free in parallel
    static thread_local int wb = -1, ww = -1;
    if (wb != BOARD_SIZE || ww != WIN_LENGTH) {
        W.clear();
        static const int DR[4] = {0, 1, 1, 1};
        static const int DC[4] = {1, 0, 1, -1};
        for (int r = 0; r < BOARD_SIZE; r++)
            for (int c = 0; c < BOARD_SIZE; c++)
                for (int d = 0; d < 4; d++) {
                    int re = r + (WIN_LENGTH - 1) * DR[d], ce = c + (WIN_LENGTH - 1) * DC[d];
                    if (re < 0 || re >= BOARD_SIZE || ce < 0 || ce >= BOARD_SIZE) continue;
                    std::vector<int> w;
                    for (int k = 0; k < WIN_LENGTH; k++) w.push_back((r + k * DR[d]) * BOARD_SIZE + c + k * DC[d]);
                    W.push_back(w);
                }
        wb = BOARD_SIZE; ww = WIN_LENGTH;
    }
    return W;
}

// For each cell, the indices (into board_windows) of the windows through it.
static const std::vector<std::vector<int>>& cell_windows(int BOARD_SIZE, int WIN_LENGTH) {
    static thread_local std::vector<std::vector<int>> CW;
    static thread_local int cb = -1, cw = -1;
    if (cb != BOARD_SIZE || cw != WIN_LENGTH) {
        const auto& W = board_windows(BOARD_SIZE, WIN_LENGTH);
        CW.assign(BOARD_SIZE * BOARD_SIZE, {});
        for (int i = 0; i < (int)W.size(); i++)
            for (int j : W[i]) CW[j].push_back(i);
        cb = BOARD_SIZE; cw = WIN_LENGTH;
    }
    return CW;
}

static int find_win_cell(const std::vector<int8_t>& b, int player, int BOARD_SIZE, int WIN_LENGTH) {
    int best = -1;
    for (const auto& w : board_windows(BOARD_SIZE, WIN_LENGTH)) {
        int own = 0, empty = -1, other = 0;
        for (int j : w) {
            if (b[j] == player) own++;
            else if (b[j] == 0) empty = j;
            else other++;
        }
        if (own == WIN_LENGTH - 1 && other == 0 && empty >= 0 && (best < 0 || empty < best)) best = empty;
    }
    return best;
}

// can[m] = 1 if placing A on m could make a four (see board_windows).
static void four_candidates(const std::vector<int8_t>& b, int A, int BOARD_SIZE, int WIN_LENGTH,
                            std::vector<int8_t>& can) {
    can.assign(b.size(), 0);
    for (const auto& w : board_windows(BOARD_SIZE, WIN_LENGTH)) {
        int own = 0, other = 0;
        for (int j : w) {
            if (b[j] == A) own++;
            else if (b[j] != 0) other++;
        }
        if (own == WIN_LENGTH - 2 && other == 0)
            for (int j : w) if (b[j] == 0) can[j] = 1;
    }
}

// Number of winning cells for `player` that use the stone on `idx`; *first =
// one of them. Only called when `player` had no winning cell before idx was
// placed, so every winning cell's five runs through idx (same set the old
// per-line wins_at scan found).
static int win_cells_through(const std::vector<int8_t>& b, int idx, int player, int BOARD_SIZE, int WIN_LENGTH,
                             int* first) {
    const auto& W = board_windows(BOARD_SIZE, WIN_LENGTH);
    int n = 0, seen[32];
    *first = -1;
    for (int wi : cell_windows(BOARD_SIZE, WIN_LENGTH)[idx]) {
        int own = 0, other = 0, empty = -1;
        for (int j : W[wi]) {
            if (b[j] == player) own++;
            else if (b[j] == 0) empty = j;
            else other++;
        }
        if (own != WIN_LENGTH - 1 || other != 0 || empty < 0) continue;
        bool dup = false;
        for (int s2 = 0; s2 < n; s2++) if (seen[s2] == empty) { dup = true; break; }
        if (dup) continue;
        if (n < 32) seen[n] = empty;
        if (n == 0) *first = empty;
        n++;
    }
    return n;
}

static bool vcf_rec(std::vector<int8_t>& b, int A, int D, int depth, long& budget, int BOARD_SIZE, int WIN_LENGTH,
                    int* first_move, int* plies) {
    if (--budget < 0) return false;
    int w = find_win_cell(b, A, BOARD_SIZE, WIN_LENGTH);
    if (w >= 0) { *first_move = w; *plies = 1; return true; }
    if (depth <= 0) return false;
    if (find_win_cell(b, D, BOARD_SIZE, WIN_LENGTH) >= 0) return false;
    std::vector<int8_t> can;
    four_candidates(b, A, BOARD_SIZE, WIN_LENGTH, can);
    for (int m = 0; m < (int)b.size(); m++) {
        if (b[m] != 0 || !can[m]) continue;
        b[m] = (int8_t)A;
        int block;
        int nw = win_cells_through(b, m, A, BOARD_SIZE, WIN_LENGTH, &block);
        if (nw >= 2) { b[m] = 0; *first_move = m; *plies = 3; return true; }
        if (nw == 1 && !wins_at(b, block, D, BOARD_SIZE, WIN_LENGTH)) {
            b[block] = (int8_t)D;
            int fm, pl;
            bool ok = vcf_rec(b, A, D, depth - 1, budget, BOARD_SIZE, WIN_LENGTH, &fm, &pl);
            b[block] = 0;
            if (ok) { b[m] = 0; *first_move = m; *plies = pl + 2; return true; }
        }
        b[m] = 0;
        if (budget < 0) return false;
    }
    return false;
}

// ============================================================
// VCT (victory by continuous threats), 11x11_v2, 2026-10-22 (user). TEST ONLY
// (problem grading); not called by the search. The attacker keeps making a
// four or a three -- a three here = a move after which the attacker has at
// least one cell that would give two winning cells at once (open four /
// double four). After a four the defender has one reply (the block); after
// a three every defence that could matter is tried: every empty cell of
// every window through a double-threat cell, plus every defender move that
// makes a four (tempo). The attacker wins only if ALL of them fail, so a
// reported win is sound; anything unclear (defender gets a four, budget or
// depth runs out) counts as no win.
// ============================================================
// cells x where the attacker playing x gets >= 2 winning cells (needs: attacker has no winning cell now)
static void vct_double_cells(std::vector<int8_t>& b, int A, int N, int WL, std::vector<int>& out) {
    out.clear();
    std::vector<int8_t> can;
    four_candidates(b, A, N, WL, can);
    for (int x = 0; x < (int)b.size(); x++) {
        if (!can[x] || b[x] != 0) continue;
        b[x] = (int8_t)A;
        int f;
        int n = win_cells_through(b, x, A, N, WL, &f);
        b[x] = 0;
        if (n >= 2) out.push_back(x);
    }
}

static bool vct_rec(std::vector<int8_t>& b, int A, int D, int depth, long& budget, int N, int WL, int* first) {
    if (--budget < 0) return false;
    int w = find_win_cell(b, A, N, WL);
    if (w >= 0) { *first = w; return true; }
    if (depth <= 0) return false;
    // Defender has a four: the attacker's move must be the block, and the attack
    // goes on only if that block is itself a four or a three (tried in the loop
    // below as the only candidate). Two defender five-points: no win.
    int forced = find_win_cell(b, D, N, WL);
    if (forced >= 0) {
        b[forced] = (int8_t)A;
        bool second = find_win_cell(b, D, N, WL) >= 0;
        b[forced] = 0;
        if (second) return false;
    }
    std::vector<int> dbl;
    if (forced < 0) {
        vct_double_cells(b, A, N, WL, dbl);
        if (!dbl.empty()) { *first = dbl[0]; return true; }  // open four / double four now
    }
    // attacker candidates: cells in windows holding WL-3 or WL-2 attacker stones and no defender stone
    std::vector<int8_t> cand(b.size(), 0);
    for (const auto& win : board_windows(N, WL)) {
        int own = 0, other = 0;
        for (int j : win) { if (b[j] == A) own++; else if (b[j] != 0) other++; }
        if (other == 0 && own >= WL - 3 && own <= WL - 2)
            for (int j : win) if (b[j] == 0) cand[j] = 1;
    }
    std::vector<int> T, defs;
    std::vector<int8_t> mark(b.size());
    for (int m = 0; m < (int)b.size(); m++) {
        if (forced >= 0 ? m != forced : (!cand[m] || b[m] != 0)) continue;
        b[m] = (int8_t)A;
        int blk;
        int nw = win_cells_through(b, m, A, N, WL, &blk);
        bool won = false;
        if (nw >= 2) {
            won = true;
        } else if (nw == 1) {                                   // four: one forced block
            if (!wins_at(b, blk, D, N, WL)) {
                b[blk] = (int8_t)D;
                int f;
                won = vct_rec(b, A, D, depth - 1, budget, N, WL, &f);
                b[blk] = 0;
            }
        } else {
            vct_double_cells(b, A, N, WL, T);
            if (!T.empty()) {                                   // three: try every relevant defence
                std::fill(mark.begin(), mark.end(), 0);
                for (int t : T) {
                    mark[t] = 1;
                    for (int wi : cell_windows(N, WL)[t])
                        for (int j : board_windows(N, WL)[wi]) if (b[j] == 0) mark[j] = 1;
                }
                std::vector<int8_t> dcan;
                four_candidates(b, D, N, WL, dcan);             // defender counter-fours (tempo)
                for (int j = 0; j < (int)b.size(); j++) if (dcan[j] && b[j] == 0) mark[j] = 1;
                won = true;
                for (int d = 0; d < (int)b.size() && won; d++) {
                    if (!mark[d] || b[d] != 0) continue;
                    b[d] = (int8_t)D;
                    int f;
                    bool ok = vct_rec(b, A, D, depth - 1, budget, N, WL, &f);
                    b[d] = 0;
                    if (!ok) won = false;
                    if (budget < 0) won = false;
                }
            }
        }
        b[m] = 0;
        if (won) { *first = m; return true; }
        if (budget < 0) return false;
    }
    return false;
}

static int vcf_first_move(const std::vector<int8_t>& b0, int A, int BOARD_SIZE, int WIN_LENGTH, int* plies) {
    if (g_vcf_depth <= 0) return -1;
    std::vector<int8_t> b = b0;
    long budget = g_vcf_budget;
    int fm = -1, pl = 0;
    if (vcf_rec(b, A, A == 1 ? 2 : 1, g_vcf_depth, budget, BOARD_SIZE, WIN_LENGTH, &fm, &pl)) { *plies = pl; return fm; }
    return -1;
}

// Recompute `node`'s proof from its children. Proofs only ever go from
// unknown to proven (never revoked). Returns true if it changed.
static bool update_proof(Node* node) {
    if (node->proven != 0 || !node->is_expanded || node->children.empty()) return false;
    int best_win = -1;
    bool all_lose = true;
    int worst_loss = 0;
    for (auto& kv : node->children) {
        Node* ch = kv.second.get();
        if (ch->proven == -1) {
            int d = ch->proof_dist + 1;
            if (best_win < 0 || d < best_win) best_win = d;
        }
        if (ch->proven != 1) {
            all_lose = false;
        } else {
            int d = ch->proof_dist + 1;
            if (d > worst_loss) worst_loss = d;
        }
    }
    if (best_win >= 0) { node->proven = 1; node->proof_dist = best_win; return true; }
    if (all_lose) { node->proven = -1; node->proof_dist = worst_loss; return true; }
    return false;
}

// After `node`'s proof changed, walk up re-deriving ancestors until one
// does not change.
static void propagate_proof(Node* node) {
    Node* p = node->parent;
    while (p != nullptr && update_proof(p)) p = p->parent;
}

// Mark immediate wins among a freshly expanded node's children (the node's
// mover completes five), then derive/propagate the node's own proof.
static void mark_immediate_wins(Node* node, const std::vector<int8_t>& b, int player, int BOARD_SIZE, int WIN_LENGTH) {
    bool any = false;
    for (auto& kv : node->children) {
        if (wins_at(b, kv.first, player, BOARD_SIZE, WIN_LENGTH)) {
            kv.second->proven = -1;      // the child's mover (opponent) has lost
            kv.second->proof_dist = 0;
            any = true;
        }
    }
    // VCF (11x11_v2): the mover wins by continuous fours -> its first four is a
    // proven win, exactly like an immediate five but proof_dist plies away.
    if (!any && g_vcf_depth > 0) {
        int plies = 0;
        int m = vcf_first_move(b, player, BOARD_SIZE, WIN_LENGTH, &plies);
        if (m >= 0) {
            auto it = node->children.find(m);
            if (it != node->children.end()) {
                it->second->proven = -1;
                it->second->proof_dist = std::max(0, plies - 1);
            }
        }
    }
    if (update_proof(node)) propagate_proof(node);
}

// Exact value, for the player to move at `node`, of a proven node reached
// `depth` plies below the root.
static double proven_value(const Node* node, int depth) {
    return mate_scaled((double)node->proven, depth + node->proof_dist);
}

// PUCT selection that never enters a provably losing move (a child proven
// to be a WIN for its own mover) while an unproven alternative exists.
// Callers only select at unproven nodes, which always have one.
static int select_child(Node* node, double c_puct) {
    int best_action = -1, fallback = -1;
    double best_score = -1e18, fallback_score = -1e18;
    double sqrt_pv = std::sqrt((double)node->n_visits);
    for (auto& kv : node->children) {
        Node* ch = kv.second.get();
        double pb_c = c_puct * sqrt_pv / (1 + ch->n_visits);
        double score = -ch->mean_value + pb_c * ch->prior_prob;
        if (ch->proven == 1) {
            if (score > fallback_score) { fallback_score = score; fallback = kv.first; }
            continue;
        }
        if (score > best_score) { best_score = score; best_action = kv.first; }
    }
    return best_action >= 0 ? best_action : fallback;
}

// Visits at the root that still inform the result: those spent in children
// not proven to lose. After tree reuse, or after a late proof, visits spent
// in a refuted main line would otherwise use up the budget.
static long counted_root_visits(const Node& root) {
    if (!root.is_expanded || root.children.empty()) return root.n_visits;
    long n = 0;
    for (auto& kv : root.children) if (kv.second->proven != 1) n += kv.second->n_visits;
    return n;
}

// ============================================================
// Forced playouts + policy target pruning (KataGo, Wu 2019 Sec. 3.2),
// 2026-09-27. Root only, and only when forced_k > 0 (the Python side passes
// k=2 for noisy training searches; evaluation/gate/relabel searches run
// without noise and get forced_k = 0, i.e. plain PUCT).
// A root child c with prior P(c) that has fewer than
//   n_forced(c) = sqrt(k * P(c) * N_root)
// visits is selected first, so a move the net rates highly but whose first
// evaluation is bad (typical of a defensive or fork point the value head
// misreads) still gets enough visits to be refuted or confirmed. Before the
// visits are returned as the training target, those forced visits are
// pruned again from every child except the most-visited one, as far as the
// child's PUCT value would not exceed the best child's, and children left
// with a single visit are dropped -- so the policy target is not polluted.
// ============================================================
static int select_root_forced(Node* node, double c_puct, double forced_k) {
    double n_total = (double)node->n_visits;
    int best_forced = -1;
    double best_forced_score = -1e18;
    double sqrt_pv = std::sqrt(n_total);
    for (auto& kv : node->children) {
        Node* ch = kv.second.get();
        if (ch->proven == 1 || ch->prior_prob <= 0.0) continue;
        double n_forced = std::sqrt(forced_k * ch->prior_prob * n_total);
        if ((double)ch->n_visits < n_forced) {
            double score = -ch->mean_value + c_puct * sqrt_pv / (1 + ch->n_visits) * ch->prior_prob;
            if (score > best_forced_score) { best_forced_score = score; best_forced = kv.first; }
        }
    }
    return best_forced >= 0 ? best_forced : select_child(node, c_puct);
}

static double puct_score(double c_puct, double sqrt_pv, Node* child) {
    double pb_c = c_puct * sqrt_pv / (1 + child->n_visits);
    return -child->mean_value + pb_c * child->prior_prob;
}

struct GroupEntry {
    Node* node;
    std::vector<int8_t> board;
    int player;
    std::vector<Node*> path;      // includes root at index 0
    bool has_terminal = false;
    double terminal_value = 0.0;
    std::string board_key;        // set only if it needs eval
    bool needs_eval = false;
};

// ============================================================
// MCTSEngine: stateful wrapper that persists the search tree across
// consecutive search()/search_batched() calls on the SAME instance.
//
// prepare_root() decides, on every call, whether the incoming board can
// reuse a subtree from the PREVIOUS call:
//   - exactly one cell differs from the previously-seen root board, AND
//   - that cell was empty before and is now occupied by whichever player
//     was to move at the previous root (i.e. this is literally "the move
//     the previous root's owner just played"), AND
//   - the previous root already has a child at that cell (it was at least
//     considered as a candidate during the previous call's expansion)
//   -> that child is promoted (moved, not copied) to be the new root, with
//      parent set to nullptr so a later backprop never walks past it into
//      freed memory.
// Any other case (first call, board-size change, 2+ differing cells, or no
// matching child) discards the old tree and starts a brand-new root -- this
// is exactly today's unconditional-reset behavior, so gate/league/vs-det
// matches (where two independent engines alternate, making the board 2
// cells different from either engine's own last root) are completely
// unaffected: they always take this fresh-start path.
//
// A node can be "promoted to root" at most once in its lifetime: once moved
// out of its parent's children map it can never be found by a later
// prepare_root lookup, so Dirichlet noise (applied only to the CURRENT
// root's direct children, once per call where add_noise=true) can never be
// applied twice to the same node's priors.
// ============================================================
// Search result with proofs applied to the reported visit counts:
//   * proven-win root: only the fastest winning move is reported (visits
//     max(its visits, 1)); every other move 0;
//   * unproven root: provably losing moves report 0 visits; if that leaves
//     no visits at all, the highest-prior surviving move gets 1 so callers
//     that normalize visits never divide by zero;
//   * proven-loss root (every move loses): unchanged (plain visits).
// root_value is the exact proven value when the root is proven.
// ============================================================
// MEGA (Minimum Exploration Guarantee Algorithm), 11x11_v2, 2026-10-21 (user).
// Stage 1: before the normal search, every root candidate is topped up to a
// minimum number of visits by its Chebyshev distance to the nearest stone
// (distance 1 -> mega_d1, distance 2 -> mega_d2); each forced visit goes on
// down the tree with normal PUCT below the root. Stage 2: the usual
// num_simulations. Stage-1 statistics stay in the tree (stage 2 simply
// continues from them). The stage-1 visits are subtracted again before the
// visit counts are returned as the policy target (user decision), so the
// target reflects only what stage 2 chose to look at; the value side keeps
// everything stage 1 learned. Python enables it only on full (noisy,
// training) searches.
// ============================================================
static int nearest_stone_dist(const std::vector<int8_t>& b, int idx, int BOARD_SIZE) {
    const int r0 = idx / BOARD_SIZE, c0 = idx % BOARD_SIZE;
    int best = 1 << 20;
    for (int j = 0; j < (int)b.size(); j++) {
        if (b[j] == 0) continue;
        int d = std::max(std::abs(j / BOARD_SIZE - r0), std::abs(j % BOARD_SIZE - c0));
        if (d < best) best = d;
    }
    return best == (1 << 20) ? 1 : best;
}

static long mega_plan(Node& root, const std::vector<int8_t>& b, int BOARD_SIZE, int d1, int d2,
                      std::map<int, long>& left) {
    long total = 0;
    left.clear();
    if (d1 <= 0 && d2 <= 0) return 0;
    for (auto& kv : root.children) {
        if (kv.second->proven == 1) continue;   // provably losing move: never worth a visit
        int want = nearest_stone_dist(b, kv.first, BOARD_SIZE) <= 1 ? d1 : d2;
        long need = std::max<long>(0, want - kv.second->n_visits);
        if (need > 0) { left[kv.first] = need; total += need; }
    }
    return total;
}

// Root pick during stage 1: the candidate with the most forced visits still
// owed (ties: higher prior). Returns -1 when stage 1 is done.
static int mega_pick(Node& root, std::map<int, long>& left, std::map<int, long>& given) {
    int best = -1;
    long best_left = 0;
    double best_prior = -1.0;
    for (auto& kv : left) {
        if (kv.second <= 0) continue;
        Node* ch = root.children[kv.first].get();
        if (ch->proven == 1) continue;
        if (kv.second > best_left || (kv.second == best_left && ch->prior_prob > best_prior)) {
            best = kv.first; best_left = kv.second; best_prior = ch->prior_prob;
        }
    }
    if (best >= 0) { left[best]--; given[best]++; }
    return best;
}

// Search result without Python objects (2026-10-21, C++ self-play): root
// children in ascending action order with their reported (policy-target)
// visit counts, q values and proofs, plus the reported root value.
struct SearchOut {
    std::vector<int> acts;
    std::vector<long> visits;
    std::vector<double> q;
    std::vector<int> proven;
    double root_value = 0.0;
    // Gumbel root (11x11_v3): improved-policy target aligned with acts, and the move the
    // Gumbel/sequential-halving search chose (-1 when Gumbel was not used).
    std::vector<double> target;
    int chosen = -1;
};

static SearchOut build_result_c(Node& root, int NUM_CELLS, int BOARD_SIZE,
                                 double c_puct = 0.0, double forced_k = 0.0,
                                 const std::map<int, long>* mega_given = nullptr) {
    std::map<int, long> shown;
    for (auto& kv : root.children) shown[kv.first] = kv.second->n_visits;
    double root_value = root.mean_value;

    if (root.proven == 1) {
        int best = -1, best_d = 1 << 30;
        for (auto& kv : root.children) {
            Node* ch = kv.second.get();
            if (ch->proven == -1 && ch->proof_dist + 1 < best_d) { best_d = ch->proof_dist + 1; best = kv.first; }
        }
        for (auto& kv : shown) kv.second = 0;
        if (best >= 0) {
            shown[best] = std::max<long>(1, root.children[best]->n_visits);
            root_value = mate_scaled(1.0, best_d);  // from the move actually reported
        } else {
            root_value = proven_value(&root, 0);
        }
    } else if (root.proven == -1) {
        // Every move loses: prefer the SLOWEST loss (largest proof_dist);
        // among equally slow losses, report prior-shaped pseudo-visits so
        // the pick is the net's favourite (usually a block) and the policy
        // target is not a flat/frozen distribution over all candidates.
        int slow_d = -1;
        for (auto& kv : root.children) slow_d = std::max(slow_d, kv.second->proof_dist + 1);
        for (auto& kv : root.children) {
            Node* ch = kv.second.get();
            shown[kv.first] = (ch->proof_dist + 1 == slow_d)
                ? std::max<long>(1, (long)std::llround(ch->prior_prob * 1000.0)) : 0;
        }
        root_value = mate_scaled(-1.0, std::max(0, slow_d));
    } else {
        long kept = 0;
        int best_prior_act = -1;
        double best_prior = -1.0;
        double val_sum = 0.0;
        for (auto& kv : root.children) {
            Node* ch = kv.second.get();
            if (ch->proven == 1) { shown[kv.first] = 0; continue; }
            kept += ch->n_visits;
            val_sum += -ch->mean_value * ch->n_visits;  // child value from the root mover's side
            if (ch->prior_prob > best_prior) { best_prior = ch->prior_prob; best_prior_act = kv.first; }
        }
        if (kept == 0 && best_prior_act >= 0) shown[best_prior_act] = 1;

        // Policy target pruning (see the forced-playout comment above).
        if (forced_k > 0.0 && kept > 0) {
            int star = -1;
            long star_n = -1;
            for (auto& kv : root.children) {
                if (kv.second->proven == 1) continue;
                if (kv.second->n_visits > star_n) { star_n = kv.second->n_visits; star = kv.first; }
            }
            double N = (double)kept;
            double sqrt_n = std::sqrt(N);
            Node* cs = root.children[star].get();
            double puct_star = -cs->mean_value + c_puct * cs->prior_prob * sqrt_n / (1.0 + cs->n_visits);
            for (auto& kv : root.children) {
                Node* ch = kv.second.get();
                if (kv.first == star || ch->proven == 1 || ch->n_visits <= 0) continue;
                long n = ch->n_visits;
                long n_forced = (long)std::floor(std::sqrt(forced_k * ch->prior_prob * N));
                double q = -ch->mean_value;
                long n_min = n;  // cannot prune if its value already beats the best child's PUCT
                if (puct_star > q) {
                    double need = c_puct * ch->prior_prob * sqrt_n / (puct_star - q) - 1.0;
                    n_min = std::max<long>(0, (long)std::ceil(need));
                }
                long pruned = std::max(n - n_forced, n_min);
                if (pruned > n) pruned = n;
                if (pruned <= 1) pruned = 0;
                shown[kv.first] = pruned;
            }
        }
        // MEGA: take the stage-1 forced visits back out of the policy target.
        if (mega_given != nullptr && !mega_given->empty()) {
            long left_total = 0;
            for (auto& kv : root.children) {
                auto g = mega_given->find(kv.first);
                if (g != mega_given->end()) {
                    long cap = std::max<long>(0, kv.second->n_visits - g->second);
                    if (shown[kv.first] > cap) shown[kv.first] = cap;
                }
                left_total += shown[kv.first];
            }
            if (left_total == 0) {          // stage 2 added nothing anywhere: fall back to the most-visited move
                int star = -1; long sv = -1;
                for (auto& kv : root.children)
                    if (kv.second->proven != 1 && kv.second->n_visits > sv) { sv = kv.second->n_visits; star = kv.first; }
                if (star >= 0) shown[star] = 1;
            }
        }
        // Root value over the moves that still count: backups from moves
        // already proven to lose would otherwise drag the reported value
        // (used as the q in the z/q value-target mix) pessimistic.
        if (kept > 0) root_value = val_sum / (double)kept;
    }

    SearchOut out;
    out.root_value = root_value;
    for (auto& kv : root.children) {
        out.acts.push_back(kv.first);
        out.visits.push_back(shown[kv.first]);
        out.q.push_back(kv.second->mean_value);
        out.proven.push_back(kv.second->proven);
    }
    return out;
}

static py::tuple result_to_py(const SearchOut& out, int NUM_CELLS, int BOARD_SIZE) {
    long total_visits = 0;
    for (long v : out.visits) total_visits += v;

    py::array_t<double> target_policy(NUM_CELLS);
    auto tp = target_policy.mutable_unchecked<1>();
    for (int i = 0; i < NUM_CELLS; i++) tp(i) = 0.0;

    py::list candidates;
    for (size_t k = 0; k < out.acts.size(); k++) {
        int act = out.acts[k];
        long v = out.visits[k];
        double prob = total_visits > 0 ? (double)v / total_visits : 0.0;
        tp(act) = prob;
        py::dict cand;
        cand["action_idx"] = act;
        cand["move"] = py::make_tuple(act / BOARD_SIZE, act % BOARD_SIZE);
        cand["visit_count"] = v;
        cand["prob"] = prob;
        cand["q_value"] = out.q[k];
        cand["proven"] = out.proven[k];  // for the child's own mover
        candidates.append(cand);
    }
    return py::make_tuple(target_policy, out.root_value, candidates);
}

static py::tuple build_result(Node& root, int NUM_CELLS, int BOARD_SIZE,
                               double c_puct = 0.0, double forced_k = 0.0,
                               const std::map<int, long>* mega_given = nullptr) {
    return result_to_py(build_result_c(root, NUM_CELLS, BOARD_SIZE, c_puct, forced_k, mega_given), NUM_CELLS, BOARD_SIZE);
}

class MCTSEngine {
public:
    py::tuple search(py::list board, std::string current_player, py::function eval_fn,
                      int num_simulations, double c_puct, bool add_noise,
                      double dirichlet_alpha, double noise_weight,
                      int BOARD_SIZE, int WIN_LENGTH, double forced_k = 0.0,
                      int mega_d1 = 0, int mega_d2 = 0);

    // Batched search core without Python (eval_batch(boards, players, probs_out, values_out));
    // search_batched below and the C++ self-play runner both call it.
    template <class F>
    SearchOut search_core(const std::vector<int8_t>& new_board, int new_player, F&& eval_batch,
                          int num_simulations, double c_puct, bool add_noise,
                          double dirichlet_alpha, double noise_weight,
                          int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                          double forced_k, int mega_d1, int mega_d2, int gumbel_m = 0);

    py::tuple search_batched(py::list board, std::string current_player, py::function eval_batch_fn,
                              int num_simulations, double c_puct, bool add_noise,
                              double dirichlet_alpha, double noise_weight,
                              int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                              double forced_k = 0.0, int mega_d1 = 0, int mega_d2 = 0);

    // Principal variation (analysis only, 2026-09-28): starting with root
    // child `first_action`, follow the most-visited child at each level of
    // the tree kept from the last search. Returns [(action, visits), ...].
    // Read-only: does not change the tree or any search behavior.
    py::list principal_variation(int first_action, int max_len) {
        py::list out;
        if (!root_) return out;
        auto it = root_->children.find(first_action);
        if (it == root_->children.end() || !it->second) return out;
        Node* n = it->second.get();
        out.append(py::make_tuple(first_action, n->n_visits));
        for (int d = 1; d < max_len && n->is_expanded && !n->children.empty(); d++) {
            int best = -1;
            long bv = 0;
            for (auto& kv : n->children) {
                if (kv.second->n_visits > bv) { bv = kv.second->n_visits; best = kv.first; }
            }
            if (best < 0) break;
            out.append(py::make_tuple(best, bv));
            n = n->children[best].get();
        }
        return out;
    }

    void reset() {
        root_.reset();
        root_board_.clear();
        root_player_ = 0;
    }

private:
    std::unique_ptr<Node> root_;
    std::vector<int8_t> root_board_;
    int root_player_ = 0;

    // Returns true if root_ was reused (promoted from a previous child),
    // false if it was (re)constructed fresh. On return, root_/root_board_/
    // root_player_ always describe the incoming position.
    bool prepare_root(const std::vector<int8_t>& new_board, int new_player, int BOARD_SIZE) {
        int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;

        if (!root_ || (int)root_board_.size() != NUM_CELLS) {
            root_ = std::make_unique<Node>();
            root_->player_turn = new_player;
            root_board_ = new_board;
            root_player_ = new_player;
            return false;
        }

        int diff_idx = -1;
        int diff_count = 0;
        for (int i = 0; i < NUM_CELLS; i++) {
            if (root_board_[i] != new_board[i]) {
                diff_count++;
                if (diff_count > 1) break;
                diff_idx = i;
            }
        }

        // Same position, same mover (2026-09-28): keep the tree, so a caller
        // can run one search in chunks (the analysis board streams every 50
        // sims). Training never re-searches an identical position.
        if (diff_count == 0 && new_player == root_player_) {
            return true;
        }

        if (diff_count == 1 &&
            root_board_[diff_idx] == 0 &&
            new_board[diff_idx] == root_player_) {
            auto it = root_->children.find(diff_idx);
            if (it != root_->children.end() && it->second) {
                std::unique_ptr<Node> promoted = std::move(it->second);
                promoted->parent = nullptr;
                root_ = std::move(promoted);
                root_board_ = new_board;
                root_player_ = new_player;
                return true;
            }
        }

        root_ = std::make_unique<Node>();
        root_->player_turn = new_player;
        root_board_ = new_board;
        root_player_ = new_player;
        return false;
    }
};

py::tuple MCTSEngine::search(py::list board, std::string current_player, py::function eval_fn,
                              int num_simulations, double c_puct, bool add_noise,
                              double dirichlet_alpha, double noise_weight,
                              int BOARD_SIZE, int WIN_LENGTH, double forced_k,
                              int mega_d1, int mega_d2) {
    int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;

    std::vector<int8_t> new_board(NUM_CELLS, 0);
    for (int r = 0; r < BOARD_SIZE; r++) {
        py::list row = board[r];
        for (int c = 0; c < BOARD_SIZE; c++) {
            std::string cell = py::str(row[c]);
            new_board[r * BOARD_SIZE + c] = (cell == "X") ? 1 : (cell == "O" ? 2 : 0);
        }
    }
    int new_player = (current_player == "X") ? 1 : 2;

    prepare_root(new_board, new_player, BOARD_SIZE);
    std::vector<int8_t>& root_board = root_board_;
    int root_player = root_player_;
    Node& root = *root_;

    std::unordered_map<std::string, std::pair<std::vector<double>, double>> eval_cache;

    auto board_to_pyarray = [&](const std::vector<int8_t>& b) {
        return py::array_t<int8_t>({BOARD_SIZE, BOARD_SIZE}, b.data());
    };

    auto cache_key = [&](const std::vector<int8_t>& b, int player) {
        std::string s;
        s.reserve(NUM_CELLS + 1);
        for (auto v : b) s.push_back('0' + v);
        s.push_back('0' + player);
        return s;
    };

    auto expand_node = [&](Node* node, const std::vector<int8_t>& b, int player) -> double {
        std::string key = cache_key(b, player);
        std::vector<double> probs;
        double leaf_value;
        auto it = eval_cache.find(key);
        if (it != eval_cache.end()) {
            probs = it->second.first;
            leaf_value = it->second.second;
        } else {
            probs.resize(NUM_CELLS);
            py::gil_scoped_acquire acquire;
            py::object result = eval_fn(board_to_pyarray(b), player == 1 ? std::string("X") : std::string("O"));
            py::tuple rt = result.cast<py::tuple>();
            py::array_t<double> probs_arr = rt[0].cast<py::array_t<double>>();
            leaf_value = rt[1].cast<double>();
            auto buf = probs_arr.unchecked<1>();
            for (int i = 0; i < NUM_CELLS; i++) probs[i] = buf(i);
            eval_cache[key] = {probs, leaf_value};
        }
        node->is_expanded = true;
        int next_player = (player == 1) ? 2 : 1;
        for (int i : relevant_cells(b, BOARD_SIZE)) {
            auto child = std::make_unique<Node>();
            child->parent = node;
            child->prior_prob = probs[i];
            child->player_turn = next_player;
            node->children[i] = std::move(child);
        }
        mark_immediate_wins(node, b, player, BOARD_SIZE, WIN_LENGTH);
        return leaf_value;
    };

    std::map<int, long> mega_left, mega_given_out;   // MEGA stage-1 bookkeeping (see mega_plan)
    {
        py::gil_scoped_release release;

        if (!root.is_expanded) {
            expand_node(&root, root_board, root_player);
        }

        if (add_noise && !root.children.empty()) {
            std::vector<int> actions;
            for (auto& kv : root.children) actions.push_back(kv.first);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
            std::vector<double> noise(actions.size());
            double sum = 0;
            for (auto& v : noise) { v = gamma(gen); sum += v; }
            if (sum <= 0) sum = 1.0;
            for (auto& v : noise) v /= sum;
            for (size_t idx = 0; idx < actions.size(); idx++) {
                auto& child = root.children[actions[idx]];
                child->prior_prob = (1 - noise_weight) * child->prior_prob + noise_weight * noise[idx];
            }
        }

        // Budget: top up to num_simulations visits that still count (see
        // counted_root_visits); hard cap so the loop always terminates.
        long mega_total = mega_plan(root, root_board, BOARD_SIZE, mega_d1, mega_d2, mega_left);
        long target = (long)num_simulations + mega_total;
        long hard_cap = 3L * std::max<long>(1, target);
        for (long sim = 0; sim < hard_cap; sim++) {
            if (root.proven != 0) break;  // result already proven: nothing left to search
            if (counted_root_visits(root) >= target) break;
            Node* node = &root;
            std::vector<int8_t> sim_board = root_board;
            int sim_player = root_player;
            int depth = 0;

            while (node->is_expanded && !node->children.empty() && node->proven == 0) {
                int best_action = (node == &root) ? mega_pick(root, mega_left, mega_given_out) : -1;
                if (best_action < 0)
                    best_action = (node == &root && forced_k > 0.0)
                        ? select_root_forced(node, c_puct, forced_k) : select_child(node, c_puct);
                int r = best_action / BOARD_SIZE, c = best_action % BOARD_SIZE;
                sim_board[r * BOARD_SIZE + c] = sim_player;
                node = node->children[best_action].get();
                sim_player = (sim_player == 1) ? 2 : 1;
                depth++;
            }

            double leaf_value;
            if (node->proven != 0) {
                leaf_value = proven_value(node, depth);
            } else {
                bool xwin = check_win(sim_board, 1, BOARD_SIZE, WIN_LENGTH);
                bool owin = check_win(sim_board, 2, BOARD_SIZE, WIN_LENGTH);
                if (xwin) leaf_value = mate_scaled((sim_player == 1) ? 1.0 : -1.0, depth);
                else if (owin) leaf_value = mate_scaled((sim_player == 2) ? 1.0 : -1.0, depth);
                else if (check_draw(sim_board)) leaf_value = 0.0;
                else {
                    leaf_value = expand_node(node, sim_board, sim_player);
                    if (node->proven != 0) leaf_value = proven_value(node, depth);
                }
            }

            Node* curr = node;
            double curr_val = leaf_value;
            while (curr != nullptr) {
                if (curr->player_turn == sim_player) curr->update(curr_val);
                else curr->update(-curr_val);
                curr = curr->parent;
            }
        }
    }

    return build_result(root, NUM_CELLS, BOARD_SIZE, c_puct, forced_k, &mega_given_out);
}

template <class F>
SearchOut MCTSEngine::search_core(const std::vector<int8_t>& new_board, int new_player, F&& eval_batch,
                                  int num_simulations, double c_puct, bool add_noise,
                                  double dirichlet_alpha, double noise_weight,
                                  int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                                  double forced_k, int mega_d1, int mega_d2, int gumbel_m) {
    const double VIRTUAL_LOSS = 1.0;
    int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
    eval_batch_size = std::max(1, eval_batch_size);


    prepare_root(new_board, new_player, BOARD_SIZE);
    std::vector<int8_t>& root_board = root_board_;
    int root_player = root_player_;
    Node& root = *root_;

    std::unordered_map<std::string, std::pair<std::vector<double>, double>> eval_cache;


    auto cache_key = [&](const std::vector<int8_t>& b, int player) {
        std::string s;
        s.reserve(NUM_CELLS + 1);
        for (auto v : b) s.push_back('0' + v);
        s.push_back('0' + player);
        return s;
    };

    auto eval_one = [&](const std::vector<int8_t>& b, int player) -> std::pair<std::vector<double>, double> {
        std::vector<const std::vector<int8_t>*> bs{&b};
        std::vector<int> ps{player};
        std::vector<std::vector<double>> pr;
        std::vector<double> vs;
        eval_batch(bs, ps, pr, vs);
        return {std::move(pr[0]), vs[0]};
    };

    auto expand_with_probs = [&](Node* node, const std::vector<int8_t>& b, int player, const std::vector<double>& probs) {
        node->is_expanded = true;
        int next_player = (player == 1) ? 2 : 1;
        for (int i : relevant_cells(b, BOARD_SIZE)) {
            auto child = std::make_unique<Node>();
            child->parent = node;
            child->prior_prob = probs[i];
            child->player_turn = next_player;
            node->children[i] = std::move(child);
        }
        mark_immediate_wins(node, b, player, BOARD_SIZE, WIN_LENGTH);
    };

    auto backprop = [&](Node* leaf, double leaf_value, int sim_player) {
        Node* curr = leaf;
        double curr_val = leaf_value;
        while (curr != nullptr) {
            if (curr->player_turn == sim_player) curr->update(curr_val);
            else curr->update(-curr_val);
            curr = curr->parent;
        }
    };

    std::map<int, long> mega_left, mega_given_out;   // MEGA stage-1 bookkeeping (see mega_plan)
    {
        if (!root.is_expanded) {
            auto [probs, leaf_value] = eval_one(root_board, root_player);
            (void)leaf_value;
            expand_with_probs(&root, root_board, root_player, probs);
        }

        const bool gumbel = gumbel_m > 0 && !root.children.empty();
        if (add_noise && !gumbel && !root.children.empty()) {
            std::vector<int> actions;
            for (auto& kv : root.children) actions.push_back(kv.first);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
            std::vector<double> noise(actions.size());
            double sum = 0;
            for (auto& v : noise) { v = gamma(gen); sum += v; }
            if (sum <= 0) sum = 1.0;
            for (auto& v : noise) v /= sum;
            for (size_t idx = 0; idx < actions.size(); idx++) {
                auto& child = root.children[actions[idx]];
                child->prior_prob = (1 - noise_weight) * child->prior_prob + noise_weight * noise[idx];
            }
        }

        // ============================================================
        // Gumbel AlphaZero root (Danihelka et al., ICLR 2022), 11x11_v3, 2026-10-22 (user).
        // Root only: sample m candidates by Gumbel-top-k on the prior logits, share the
        // budget among them by sequential halving (each phase every surviving candidate
        // is topped up to an equal visit target, then the worse half by
        // g + logit + sigma(q) is dropped). Below the root the search is plain PUCT.
        // The training target is softmax(logit + sigma(completed q)) (built after the
        // search), so it improves on the prior even with few simulations.
        // ============================================================
        std::map<int, double> g_noise, g_logit;
        std::vector<int> g_rem;
        std::map<int, long> g_tgt;
        if (gumbel) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::extreme_value_distribution<double> G(0.0, 1.0);
            std::vector<std::pair<double, int>> order;
            for (auto& kv : root.children) {
                if (kv.second->proven == 1) continue;
                g_logit[kv.first] = std::log(std::max(kv.second->prior_prob, 1e-12));
                g_noise[kv.first] = add_noise ? G(gen) : 0.0;
            }
            // 2026-10-23 (user): Dirichlet on top of Gumbel. Candidate sampling uses the noised prior
            // (log((1-w)p + w*eta)); the sequential-halving score and the training target keep the clean logit.
            std::map<int, double> sample_logit = g_logit;
            if (add_noise && noise_weight > 0 && !g_logit.empty()) {
                std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
                std::vector<int> acts; std::vector<double> eta; double es = 0;
                for (auto& kv : g_logit) { acts.push_back(kv.first); eta.push_back(gamma(gen)); es += eta.back(); }
                if (es <= 0) es = 1.0;
                for (size_t k = 0; k < acts.size(); k++) {
                    double p = std::exp(g_logit[acts[k]]);
                    sample_logit[acts[k]] = std::log(std::max((1 - noise_weight) * p + noise_weight * eta[k] / es, 1e-12));
                }
            }
            for (auto& kv : g_logit) order.push_back({g_noise[kv.first] + sample_logit[kv.first], kv.first});
            std::sort(order.begin(), order.end(), std::greater<>());
            int m = std::min<int>(gumbel_m, (int)order.size());
            for (int k = 0; k < m; k++) g_rem.push_back(order[k].second);
        }
        auto g_sigma = [&](double q) {           // q from the root mover's side, in [-1, 1]
            long maxn = 0;
            for (auto& kv : root.children) maxn = std::max<long>(maxn, kv.second->n_visits);
            return (50.0 + (double)maxn) * 1.0 * (0.5 * (q + 1.0));
        };
        auto g_score = [&](int a) {
            Node* ch = root.children[a].get();
            return g_noise[a] + g_logit[a] + (ch->n_visits > 0 ? g_sigma(-ch->mean_value) : 0.0);
        };
        auto g_pick = [&]() -> int {             // candidate furthest below its phase target
            int best = -1;
            long need = 0;
            for (int a : g_rem) {
                long d = g_tgt[a] - root.children[a]->n_visits;
                if (d > need) { need = d; best = a; }
            }
            return best;
        };
        int g_phases = 1, g_phase = 0;
        if (gumbel && g_rem.size() > 1) g_phases = (int)std::ceil(std::log2((double)g_rem.size()));
        auto g_start_phase = [&]() {
            long per = std::max<long>(1, (long)num_simulations / ((long)g_phases * (long)g_rem.size()));
            for (int a : g_rem) g_tgt[a] = root.children[a]->n_visits + per;
        };
        if (gumbel && !g_rem.empty()) g_start_phase();

        // Budget: top up to num_simulations visits that still count (see
        // counted_root_visits), re-checked every group; hard cap so the loop
        // always terminates.
        long mega_total = gumbel ? 0 : mega_plan(root, root_board, BOARD_SIZE, mega_d1, mega_d2, mega_left);
        long target = (long)num_simulations + mega_total;
        long hard_cap = 3L * std::max<long>(1, target);
        long spent = 0;
        while (spent < hard_cap) {
            if (root.proven != 0) break;  // result already proven: nothing left to search
            int group_size;
            if (gumbel) {
                if (g_rem.empty()) break;
                long deficit = 0;
                for (int a : g_rem) deficit += std::max<long>(0, g_tgt[a] - root.children[a]->n_visits);
                if (deficit <= 0) {                       // phase done: halve, next phase
                    if (g_rem.size() <= 1 || ++g_phase >= g_phases) break;
                    std::sort(g_rem.begin(), g_rem.end(), [&](int x, int y) { return g_score(x) > g_score(y); });
                    g_rem.resize((g_rem.size() + 1) / 2);
                    g_start_phase();
                    continue;
                }
                group_size = (int)std::min<long>(eval_batch_size, deficit);
            } else {
                long remaining = target - counted_root_visits(root);
                if (remaining <= 0) break;
                group_size = (int)std::min<long>(eval_batch_size, remaining);
            }
            spent += group_size;

            std::vector<GroupEntry> group;
            group.reserve(group_size);

            for (int g = 0; g < group_size; g++) {
                Node* node = &root;
                std::vector<int8_t> sim_board = root_board;
                int sim_player = root_player;
                std::vector<Node*> path = {node};

                while (node->is_expanded && !node->children.empty() && node->proven == 0) {
                    int best_action = -1;
                    if (node == &root) {
                        if (gumbel) {
                            best_action = g_pick();
                            if (best_action < 0) best_action = g_rem[0];
                        } else {
                            best_action = mega_pick(root, mega_left, mega_given_out);
                        }
                    }
                    if (best_action < 0)
                        best_action = (node == &root && forced_k > 0.0)
                            ? select_root_forced(node, c_puct, forced_k) : select_child(node, c_puct);
                    int r = best_action / BOARD_SIZE, c = best_action % BOARD_SIZE;
                    sim_board[r * BOARD_SIZE + c] = sim_player;
                    node = node->children[best_action].get();
                    sim_player = (sim_player == 1) ? 2 : 1;
                    path.push_back(node);
                    node->n_visits += 1;
                    // Virtual loss must make this node LESS attractive to
                    // its parent. A node's mean_value is from ITS OWN
                    // mover's perspective and the parent selects on
                    // -child->mean_value, so the pending visit has to push
                    // mean_value UP (+VIRTUAL_LOSS). Every earlier build
                    // used -VIRTUAL_LOSS, which pulled the parent back to
                    // the same leaf: all eval_batch_size selections in a
                    // batch landed on one leaf, so 1100 sims at batch 16
                    // were only ~69 distinct evaluations (found 2026-09-25:
                    // on a must-block position batch 1 put 72% of visits on
                    // the block, batch 16 put 1.5%).
                    node->total_value += VIRTUAL_LOSS;
                    node->mean_value = node->total_value / node->n_visits;
                }

                GroupEntry entry;
                entry.node = node;
                entry.board = sim_board;
                entry.player = sim_player;
                entry.path = path;

                int depth = (int)path.size() - 1;
                if (node->proven != 0) {
                    entry.has_terminal = true;
                    entry.terminal_value = proven_value(node, depth);
                } else {
                    bool xwin = check_win(sim_board, 1, BOARD_SIZE, WIN_LENGTH);
                    bool owin = check_win(sim_board, 2, BOARD_SIZE, WIN_LENGTH);
                    if (xwin) { entry.has_terminal = true; entry.terminal_value = mate_scaled((sim_player == 1) ? 1.0 : -1.0, depth); }
                    else if (owin) { entry.has_terminal = true; entry.terminal_value = mate_scaled((sim_player == 2) ? 1.0 : -1.0, depth); }
                    else if (check_draw(sim_board)) { entry.has_terminal = true; entry.terminal_value = 0.0; }
                }

                group.push_back(std::move(entry));
            }

            for (auto& entry : group) {
                for (size_t i = 1; i < entry.path.size(); i++) {
                    Node* n = entry.path[i];
                    n->n_visits -= 1;
                    n->total_value -= VIRTUAL_LOSS;
                    n->mean_value = (n->n_visits > 0) ? (n->total_value / n->n_visits) : 0.0;
                }
            }

            std::vector<std::string> batch_keys;
            std::vector<const std::vector<int8_t>*> pending_boards;
            std::vector<int> pending_players;
            std::unordered_map<std::string, int> key_to_idx;

            for (auto& entry : group) {
                if (entry.has_terminal) continue;
                entry.board_key = cache_key(entry.board, entry.player);
                if (eval_cache.find(entry.board_key) != eval_cache.end()) continue;
                entry.needs_eval = true;
                if (key_to_idx.find(entry.board_key) != key_to_idx.end()) continue;
                key_to_idx[entry.board_key] = (int)batch_keys.size();
                batch_keys.push_back(entry.board_key);
                pending_boards.push_back(&entry.board);
                pending_players.push_back(entry.player);
            }

            if (!batch_keys.empty()) {
                std::vector<std::vector<double>> pr;
                std::vector<double> vs;
                eval_batch(pending_boards, pending_players, pr, vs);
                for (size_t idx = 0; idx < batch_keys.size(); idx++)
                    eval_cache[batch_keys[idx]] = {std::move(pr[idx]), vs[idx]};
            }

            for (auto& entry : group) {
                if (entry.has_terminal) {
                    backprop(entry.node, entry.terminal_value, entry.player);
                    continue;
                }
                auto& cached = eval_cache[entry.board_key];
                if (!entry.node->is_expanded) {
                    expand_with_probs(entry.node, entry.board, entry.player, cached.first);
                }
                if (entry.node->proven != 0) {
                    // expansion just proved this leaf (e.g. its mover has an
                    // immediate five): back up the exact value, not the net's.
                    backprop(entry.node, proven_value(entry.node, (int)entry.path.size() - 1), entry.player);
                } else {
                    backprop(entry.node, cached.second, entry.player);
                }
            }
        }

    SearchOut out = build_result_c(root, NUM_CELLS, BOARD_SIZE, c_puct, gumbel ? 0.0 : forced_k, &mega_given_out);
    if (gumbel && root.proven == 0 && !g_logit.empty()) {
        // completed q: visited children keep their q; unvisited ones get the
        // prior-weighted mean q of the visited ones (a simplified v_mix).
        double pq = 0.0, ps = 0.0;
        for (auto& kv : g_logit) {
            Node* ch = root.children[kv.first].get();
            if (ch->n_visits > 0) { pq += ch->prior_prob * -ch->mean_value; ps += ch->prior_prob; }
        }
        double vmix = ps > 0 ? pq / ps : root.mean_value;
        std::vector<double> z(out.acts.size(), 0.0);
        std::vector<char> ok(out.acts.size(), 0);
        double zmax = -1e300;
        for (size_t k = 0; k < out.acts.size(); k++) {
            auto it = g_logit.find(out.acts[k]);
            if (it == g_logit.end()) continue;      // provably losing move: target 0
            Node* ch = root.children[out.acts[k]].get();
            z[k] = it->second + g_sigma(ch->n_visits > 0 ? -ch->mean_value : vmix);
            ok[k] = 1;
            zmax = std::max(zmax, z[k]);
        }
        double zs = 0.0;
        out.target.assign(out.acts.size(), 0.0);
        for (size_t k = 0; k < out.acts.size(); k++)
            if (ok[k]) { out.target[k] = std::exp(z[k] - zmax); zs += out.target[k]; }
        if (zs > 0) for (auto& v : out.target) v /= zs;
        int best = g_rem.empty() ? -1 : g_rem[0];
        for (int a : g_rem) if (g_score(a) > g_score(best)) best = a;
        out.chosen = best;
    }
    return out;
    }
}

py::tuple MCTSEngine::search_batched(py::list board, std::string current_player, py::function eval_batch_fn,
                                      int num_simulations, double c_puct, bool add_noise,
                                      double dirichlet_alpha, double noise_weight,
                                      int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                                      double forced_k, int mega_d1, int mega_d2) {
    int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
    std::vector<int8_t> new_board(NUM_CELLS, 0);
    for (int r = 0; r < BOARD_SIZE; r++) {
        py::list row = board[r];
        for (int c = 0; c < BOARD_SIZE; c++) {
            std::string cell = py::str(row[c]);
            new_board[r * BOARD_SIZE + c] = (cell == "X") ? 1 : (cell == "O" ? 2 : 0);
        }
    }
    int new_player = (current_player == "X") ? 1 : 2;

    auto py_eval = [&](const std::vector<const std::vector<int8_t>*>& boards, const std::vector<int>& players,
                       std::vector<std::vector<double>>& probs_out, std::vector<double>& vals_out) {
        py::gil_scoped_acquire acquire;
        py::list boards_arg;
        py::list players_arg;
        for (size_t i = 0; i < boards.size(); i++) {
            boards_arg.append(py::array_t<int8_t>({BOARD_SIZE, BOARD_SIZE}, boards[i]->data()));
            players_arg.append(players[i] == 1 ? std::string("X") : std::string("O"));
        }
        py::object result = eval_batch_fn(boards_arg, players_arg);
        py::tuple rt = result.cast<py::tuple>();
        py::array_t<double> probs_arr = rt[0].cast<py::array_t<double>>();
        py::array_t<double> values_arr = rt[1].cast<py::array_t<double>>();
        auto pbuf = probs_arr.unchecked<2>();
        auto vbuf = values_arr.unchecked<1>();
        probs_out.assign(boards.size(), std::vector<double>(NUM_CELLS));
        vals_out.assign(boards.size(), 0.0);
        for (size_t idx = 0; idx < boards.size(); idx++) {
            for (int i = 0; i < NUM_CELLS; i++) probs_out[idx][i] = pbuf((long)idx, i);
            vals_out[idx] = vbuf((long)idx);
        }
    };

    SearchOut out;
    {
        py::gil_scoped_release release;
        out = search_core(new_board, new_player, py_eval, num_simulations, c_puct, add_noise,
                          dirichlet_alpha, noise_weight, BOARD_SIZE, WIN_LENGTH, eval_batch_size,
                          forced_k, mega_d1, mega_d2);
    }
    return result_to_py(out, NUM_CELLS, BOARD_SIZE);
}

// mcts_core_tree_reuse.cpp: mcts_core_pruned_deploy.cpp (the currently
// deployed 10x10\mcts_core.pyd source) restructured around a stateful
// MCTSEngine class so that consecutive search()/search_batched() calls on
// the SAME instance can promote an already-explored child to be the new
// root instead of discarding the whole tree every move. See the big
// comment above the MCTSEngine class for the exact reuse/reset rule and
// why gate/league/vs-deterministic matches (2 engines alternating moves)
// are provably unaffected -- they always see a 2-cell diff from either
// engine's own perspective and take the fresh-start path, identical to
// today's unconditional-reset behavior.
//
// The module macro's first argument is "mcts_core_tree_reuse" (not
// "mcts_core"), same reasoning as mcts_core_v2/v3/pruned_deploy: lets this
// .pyd be imported in the SAME process as the still-live production
// "mcts_core" module for side-by-side correctness/consistency testing
// without a name collision. When/if tree reuse is promoted to replace the
// production engine, this macro's first argument should be changed back to
// mcts_core -- at that point only one of the two will ever be loaded in a
// given process, so no collision applies.
// Module name set at build time (build_mcts_core_mate_distance.bat passes
// /DMCTS_MODULE_NAME=<name>): mcts_core_mate_distance for side-by-side
// testing, mcts_core for the deploy build.
#ifndef MCTS_MODULE_NAME
#define MCTS_MODULE_NAME mcts_core_solver
#endif
// ============================================================
// C++ self-play (11x11_v2, 2026-10-21, user). Plays a whole epoch's
// self-play and vs-League games in C++ threads inside the trainer process
// and gathers every game's leaf evaluations into large GPU batches; the
// network itself still runs in PyTorch through one Python callback
// eval_fn(tag, planes) -> (probs, values). Game logic mirrors
// alphazero_mcts_make_lrfix._play_one_mcts_game and
// _play_one_mcts_vs_league_game (PCR, sims_for_state boost, temperature,
// forced playouts, MEGA, max_plies). Recorded plies come back as raw
// boards; Python builds their input planes with board_to_input_array.
// ============================================================

// Network input planes for one board (curr, opp, empty, fork(curr),
// fork(opp)), float32, same values as
// alphazero_mcts_lrfix._board_to_input_array_batch_from_numeric.
static void fork_mask_planes(const int8_t* b, int who, int BOARD_SIZE, int WIN_LENGTH, float* out) {
    const int NC = BOARD_SIZE * BOARD_SIZE;
    static thread_local std::vector<uint8_t> lit, adj;
    lit.assign(NC, 0);
    adj.assign((size_t)NC * NC, 0);
    for (const auto& w : board_windows(BOARD_SIZE, WIN_LENGTH)) {
        int cc = 0, ec = 0;
        for (int j : w) { cc += b[j] == who; ec += b[j] == 0; }
        if (cc == WIN_LENGTH - 1 && ec == 1) {
            for (int j : w) if (b[j] == 0) { lit[j] = 1; break; }
        } else if (cc == WIN_LENGTH - 2 && ec == 2) {
            int x = -1, y = -1;
            for (int j : w) if (b[j] == 0) { if (x < 0) x = j; else { y = j; break; } }
            adj[(size_t)x * NC + y] = 1;
            adj[(size_t)y * NC + x] = 1;
        }
    }
    int base = 0;
    for (int i = 0; i < NC; i++) base += lit[i];
    for (int a = 0; a < NC; a++) {
        if (b[a] != 0) { out[a] = 0.0f; continue; }
        const uint8_t* row = &adj[(size_t)a * NC];
        int tc = 0, ov = 0;
        for (int k = 0; k < NC; k++) if (row[k]) { tc++; if (lit[k]) ov++; }
        int total = base - (lit[a] ? 1 : 0) + tc - ov;
        out[a] = total >= 2 ? 1.0f : 0.0f;
    }
}

static void encode_planes(const std::vector<int8_t>& b, int player, int BOARD_SIZE, int WIN_LENGTH, float* out) {
    const int NC = BOARD_SIZE * BOARD_SIZE;
    const int opp = player == 1 ? 2 : 1;
    for (int i = 0; i < NC; i++) {
        out[i] = b[i] == player ? 1.0f : 0.0f;
        out[NC + i] = b[i] == opp ? 1.0f : 0.0f;
        out[2 * NC + i] = b[i] == 0 ? 1.0f : 0.0f;
    }
    fork_mask_planes(b.data(), player, BOARD_SIZE, WIN_LENGTH, out + 3 * NC);
    fork_mask_planes(b.data(), opp, BOARD_SIZE, WIN_LENGTH, out + 4 * NC);
}

struct SPConfig {
    int num_simulations = 1100, boosted_sims = 0, boost_threshold = 0;
    bool boost_when_narrow = false;
    int temperature_moves = 6;
    double dirichlet_alpha = 0.3, noise_weight = 0.25, c_puct = 1.414, full_search_prob = 1.0;
    int fast_sims = 0, max_plies = 0;
};

struct SPTask {
    int kind = 0;                 // 0 = self-play, 1 = vs-League
    bool has_start = false;
    std::vector<int8_t> start_board;
    int start_player = 1;
    bool mcts_plays_x = true;     // league
    int opening = -1;             // league: X's pre-placed first stone (-1 none)
    int league_tag = 1;           // league: eval tag of the opponent
    SPConfig cfg;
};

struct SPRecord {
    std::vector<int8_t> board;
    int player;
    std::vector<float> policy;    // NUM_CELLS, visit share
    bool has_q;
    double q;
};

struct SPResult {
    std::vector<SPRecord> hist;
    int outcome = 0;
    std::vector<int> ply_idx;
    int total_plies = 0;
    std::vector<int> moves;       // every move of the game in order (v3 auxiliary targets)
    bool truncated = false;
    int mcts_result = 0;          // league: 1 mcts, -1 league, 0 draw
};

static int sims_for_state_c(int num_empty, int NUM_CELLS, const SPConfig& c) {
    if (!c.boosted_sims) return c.num_simulations;
    int sims = c.boost_when_narrow ? (num_empty <= c.boost_threshold ? c.boosted_sims : c.num_simulations)
                                   : (num_empty > c.boost_threshold ? c.boosted_sims : c.num_simulations);
    if (num_empty > NUM_CELLS - 10) sims = std::max(sims, c.boosted_sims);
    return sims;
}

struct EvalReq {
    int tag = 0;
    int n = 0;
    std::vector<float> input;
    std::vector<std::vector<double>>* probs = nullptr;
    std::vector<double>* vals = nullptr;
    bool done = false;
};

class SelfPlayRunner {
public:
    int BOARD_SIZE = 11, WIN_LENGTH = 5, eval_batch = 32;
    double forced_k = 2.0;
    int mega_d1 = 30, mega_d2 = 15, mega_min_sims = 1000;
    bool record_league_side = false;
    int gumbel_m_full = 0, gumbel_m_fast = 0;   // 11x11_v3: Gumbel root candidates (0 = PUCT)
    std::vector<SPTask> tasks;
    // stats
    long n_batches = 0, n_rows = 0, max_rows = 0;
    std::atomic<long> cache_hits{0};
    long cache_cap = 400000;

    void add_task(int kind, py::object start_board, std::string start_player, bool mcts_plays_x,
                  int opening, int league_tag, py::dict cfg) {
        SPTask t;
        t.kind = kind;
        const int NC = BOARD_SIZE * BOARD_SIZE;
        if (!start_board.is_none()) {
            t.has_start = true;
            t.start_board.assign(NC, 0);
            py::list bl = start_board.cast<py::list>();
            for (int r = 0; r < BOARD_SIZE; r++) {
                py::list row = bl[r];
                for (int c = 0; c < BOARD_SIZE; c++) {
                    std::string x = py::str(row[c]);
                    t.start_board[r * BOARD_SIZE + c] = x == "X" ? 1 : (x == "O" ? 2 : 0);
                }
            }
            t.start_player = start_player == "O" ? 2 : 1;
        }
        t.mcts_plays_x = mcts_plays_x;
        t.opening = opening;
        t.league_tag = league_tag;
        auto gi = [&](const char* k, int d) { return (cfg.contains(k) && !cfg[k].is_none()) ? cfg[k].cast<int>() : d; };
        auto gd = [&](const char* k, double d) { return (cfg.contains(k) && !cfg[k].is_none()) ? cfg[k].cast<double>() : d; };
        t.cfg.num_simulations = gi("num_simulations", 1100);
        t.cfg.boosted_sims = gi("boosted_sims", 0);
        t.cfg.boost_threshold = gi("boost_threshold", 0);
        t.cfg.boost_when_narrow = cfg.contains("boost_when_narrow") && !cfg["boost_when_narrow"].is_none()
                                  && cfg["boost_when_narrow"].cast<bool>();
        t.cfg.temperature_moves = gi("temperature_moves", 6);
        t.cfg.dirichlet_alpha = gd("dirichlet_alpha", 0.3);
        t.cfg.noise_weight = gd("noise_weight", 0.25);
        t.cfg.c_puct = gd("c_puct", 1.414);
        t.cfg.full_search_prob = gd("full_search_prob", 1.0);
        t.cfg.fast_sims = gi("fast_sims", 0);
        t.cfg.max_plies = gi("max_plies", 0);
        tasks.push_back(std::move(t));
    }

    py::list run(py::function eval_fn, int n_threads, int target_rows, double max_wait_ms) {
        results_.assign(tasks.size(), SPResult());
        next_task_ = 0;
        error_ = false;
        err_msg_.clear();
        n_threads = std::max(1, std::min<int>(n_threads, (int)tasks.size()));
        live_ = n_threads;
        waiting_ = 0;
        pending_rows_ = 0;
        std::vector<std::thread> th;
        {
            py::gil_scoped_release rel;
            for (int t = 0; t < n_threads; t++) th.emplace_back(&SelfPlayRunner::thread_main, this, (unsigned)t);
        }
        const int NC = BOARD_SIZE * BOARD_SIZE;
        const int PL = 5 * NC;
        for (;;) {
            std::vector<EvalReq*> batch;
            bool finished = false;
            {
                py::gil_scoped_release rel;
                std::unique_lock<std::mutex> lk(m_);
                auto ready = [&] {
                    return error_ || live_ == 0 ||
                           (!queue_.empty() && (pending_rows_ >= target_rows || waiting_ >= live_));
                };
                if (!ready()) cv_main_.wait_for(lk, std::chrono::microseconds((long)(max_wait_ms * 1000)), ready);
                if (error_ || (live_ == 0 && queue_.empty())) finished = true;
                else { batch.assign(queue_.begin(), queue_.end()); queue_.clear(); pending_rows_ = 0; }
            }
            if (finished) break;
            if (batch.empty()) continue;
            try {
                std::map<int, std::vector<EvalReq*>> by_tag;
                for (auto* r : batch) by_tag[r->tag].push_back(r);
                for (auto& kv : by_tag) {
                    long rows = 0;
                    for (auto* r : kv.second) rows += r->n;
                    py::array_t<float> arr({(py::ssize_t)rows, (py::ssize_t)5, (py::ssize_t)BOARD_SIZE, (py::ssize_t)BOARD_SIZE});
                    float* dst = static_cast<float*>(arr.request().ptr);
                    for (auto* r : kv.second) { std::memcpy(dst, r->input.data(), sizeof(float) * r->n * PL); dst += (size_t)r->n * PL; }
                    py::tuple rt = eval_fn(kv.first, arr).cast<py::tuple>();
                    py::array_t<double, py::array::c_style | py::array::forcecast> pa = rt[0].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                    py::array_t<double, py::array::c_style | py::array::forcecast> va = rt[1].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                    auto pb = pa.unchecked<2>();
                    auto vb = va.unchecked<1>();
                    long row = 0;
                    for (auto* r : kv.second) {
                        r->probs->assign(r->n, std::vector<double>(NC));
                        r->vals->assign(r->n, 0.0);
                        for (int i = 0; i < r->n; i++, row++) {
                            for (int k = 0; k < NC; k++) (*r->probs)[i][k] = pb(row, k);
                            (*r->vals)[i] = vb(row);
                        }
                    }
                    n_batches++;
                    n_rows += rows;
                    max_rows = std::max(max_rows, rows);
                }
            } catch (std::exception& e) {
                std::lock_guard<std::mutex> lk(m_);
                error_ = true;
                err_msg_ = std::string("eval_fn failed: ") + e.what();
                cv_worker_.notify_all();
                break;
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                for (auto* r : batch) r->done = true;
                waiting_ -= (int)batch.size();
            }
            cv_worker_.notify_all();
        }
        {
            py::gil_scoped_release rel;
            {
                std::lock_guard<std::mutex> lk(m_);
                if (error_) cv_worker_.notify_all();
            }
            for (auto& t : th) t.join();
        }
        if (error_) throw std::runtime_error(err_msg_);
        return results_to_py();
    }

private:
    std::vector<SPResult> results_;
    std::atomic<int> next_task_{0};
    std::mutex m_;
    std::condition_variable cv_main_, cv_worker_;
    std::deque<EvalReq*> queue_;
    long pending_rows_ = 0;
    int live_ = 0, waiting_ = 0;
    bool error_ = false;
    std::string err_msg_;

    // Epoch-wide evaluation cache shared by all games (per model tag): the
    // weights are frozen for the whole run, so a position evaluated once is
    // never sent to the GPU again (openings and transpositions repeat across
    // games). Values are stored as float -- they come from float32 network
    // outputs, so nothing is lost. Cleared when it reaches cache_cap entries.
    struct CacheVal { std::vector<float> p; float v; };
    std::unordered_map<std::string, CacheVal> cache_;
    std::mutex cache_m_;

    std::string ckey(int tag, const std::vector<int8_t>& b, int player) {
        std::string k;
        k.reserve(b.size() + 2);
        k.push_back((char)('A' + tag));
        for (auto v : b) k.push_back((char)('0' + v));
        k.push_back((char)('0' + player));
        return k;
    }

    void eval(int tag, const std::vector<const std::vector<int8_t>*>& boards, const std::vector<int>& players,
              std::vector<std::vector<double>>& probs_out, std::vector<double>& vals_out) {
        const int NC = BOARD_SIZE * BOARD_SIZE;
        const int PL = 5 * NC;
        const int n = (int)boards.size();
        probs_out.assign(n, std::vector<double>());
        vals_out.assign(n, 0.0);
        std::vector<std::string> keys(n);
        std::vector<int> miss;
        {
            std::lock_guard<std::mutex> lk(cache_m_);
            for (int i = 0; i < n; i++) {
                keys[i] = ckey(tag, *boards[i], players[i]);
                auto it = cache_.find(keys[i]);
                if (it == cache_.end()) { miss.push_back(i); continue; }
                probs_out[i].assign(it->second.p.begin(), it->second.p.end());
                vals_out[i] = it->second.v;
                cache_hits++;
            }
        }
        if (miss.empty()) return;
        std::vector<std::vector<double>> mp;
        std::vector<double> mv;
        EvalReq r;
        r.tag = tag;
        r.n = (int)miss.size();
        r.input.resize((size_t)r.n * PL);
        for (int j = 0; j < r.n; j++)
            encode_planes(*boards[miss[j]], players[miss[j]], BOARD_SIZE, WIN_LENGTH, r.input.data() + (size_t)j * PL);
        r.probs = &mp;
        r.vals = &mv;
        {
            std::unique_lock<std::mutex> lk(m_);
            if (error_) throw std::runtime_error("self-play aborted");
            queue_.push_back(&r);
            pending_rows_ += r.n;
            waiting_++;
            cv_main_.notify_one();
            cv_worker_.wait(lk, [&] { return r.done || error_; });
            if (!r.done) throw std::runtime_error("self-play aborted");
        }
        std::lock_guard<std::mutex> lk(cache_m_);
        if ((long)cache_.size() + r.n > cache_cap) cache_.clear();
        for (int j = 0; j < r.n; j++) {
            int i = miss[j];
            CacheVal cv;
            cv.p.assign(mp[j].begin(), mp[j].end());
            cv.v = (float)mv[j];
            cache_[keys[i]] = std::move(cv);
            probs_out[i] = std::move(mp[j]);
            vals_out[i] = mv[j];
        }
    }

    void thread_main(unsigned salt) {
        std::random_device rd;
        std::mt19937 rng(rd() ^ (salt * 0x9E3779B9u));
        try {
            for (;;) {
                int i = next_task_++;
                if (i >= (int)tasks.size()) break;
                results_[i] = tasks[i].kind == 0 ? play_self(tasks[i], rng) : play_league(tasks[i], rng);
            }
        } catch (std::exception& e) {
            std::lock_guard<std::mutex> lk(m_);
            if (!error_) { error_ = true; err_msg_ = e.what(); }
            cv_worker_.notify_all();
        }
        std::lock_guard<std::mutex> lk(m_);
        live_--;
        cv_main_.notify_one();
    }

    SearchOut do_search(MCTSEngine& eng, int tag, const std::vector<int8_t>& b, int player, int sims, bool noisy,
                        const SPConfig& c) {
        auto ev = [&](const std::vector<const std::vector<int8_t>*>& bs, const std::vector<int>& ps,
                      std::vector<std::vector<double>>& po, std::vector<double>& vo) { eval(tag, bs, ps, po, vo); };
        int gm = tag != 0 ? 0 : (noisy ? gumbel_m_full : gumbel_m_fast);
        bool mega = gm == 0 && noisy && sims >= mega_min_sims;
        return eng.search_core(b, player, ev, sims, c.c_puct, noisy || gm > 0, c.dirichlet_alpha, c.noise_weight,
                               BOARD_SIZE, WIN_LENGTH, eval_batch, (noisy && gm == 0) ? forced_k : 0.0,
                               mega ? mega_d1 : 0, mega ? mega_d2 : 0, gm);
    }

    std::vector<float> policy_of(const SearchOut& o) {
        std::vector<float> p(BOARD_SIZE * BOARD_SIZE, 0.0f);
        if (!o.target.empty()) {
            for (size_t k = 0; k < o.acts.size(); k++) p[o.acts[k]] = (float)o.target[k];
            return p;
        }
        long tot = 0;
        for (long v : o.visits) tot += v;
        for (size_t k = 0; k < o.acts.size(); k++) p[o.acts[k]] = tot > 0 ? (float)((double)o.visits[k] / tot) : 0.0f;
        return p;
    }

    // np.random.choice(actions, p=visits/sum) / argmax (first max, ascending action order)
    int pick(const SearchOut& o, bool sample, std::mt19937& rng) {
        // 2026-10-23 (user): early-move temperature also with Gumbel -- sample from the improved policy target
        if (o.chosen >= 0 && sample && !o.target.empty()) {
            double tot = 0;
            for (double v : o.target) tot += v;
            if (tot > 0) {
                std::uniform_real_distribution<double> U(0.0, tot);
                double x = U(rng), acc = 0;
                for (size_t k = 0; k < o.acts.size(); k++) { acc += o.target[k]; if (x < acc) return o.acts[k]; }
            }
        }
        if (o.chosen >= 0) return o.chosen;     // Gumbel: the sampled candidates already randomize
        if (sample) {
            double tot = 0;
            for (long v : o.visits) tot += (double)v;
            if (tot > 0) {
                std::uniform_real_distribution<double> U(0.0, tot);
                double x = U(rng), acc = 0;
                for (size_t k = 0; k < o.acts.size(); k++) { acc += (double)o.visits[k]; if (x < acc) return o.acts[k]; }
                for (size_t k = o.acts.size(); k-- > 0;) if (o.visits[k] > 0) return o.acts[k];
            }
        }
        size_t best = 0;
        for (size_t k = 1; k < o.acts.size(); k++) if (o.visits[k] > o.visits[best]) best = k;
        return o.acts[best];
    }

    SPResult play_self(const SPTask& t, std::mt19937& rng) {
        const int NC = BOARD_SIZE * BOARD_SIZE;
        const SPConfig& c = t.cfg;
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        MCTSEngine eng;
        std::vector<int8_t> b(NC, 0);
        int player;
        if (t.has_start) { b = t.start_board; player = t.start_player; }
        else player = (rng() & 1) ? 1 : 2;
        SPResult res;
        int ply = 0;
        for (;;) {
            if (c.max_plies && ply >= c.max_plies) { res.outcome = 0; res.truncated = true; break; }
            if (check_win(b, 1, BOARD_SIZE, WIN_LENGTH)) { res.outcome = 1; break; }
            if (check_win(b, 2, BOARD_SIZE, WIN_LENGTH)) { res.outcome = -1; break; }
            if (check_draw(b)) { res.outcome = 0; break; }
            int num_empty = 0;
            for (int i = 0; i < NC; i++) num_empty += b[i] == 0;
            bool full = c.full_search_prob >= 1.0 || U01(rng) < c.full_search_prob;
            int sims = full ? sims_for_state_c(num_empty, NC, c)
                            : (c.fast_sims ? c.fast_sims : std::max(1, c.num_simulations / 6));
            SearchOut o = do_search(eng, 0, b, player, sims, full, c);
            if (o.acts.empty()) break;
            int a = pick(o, ply < c.temperature_moves, rng);
            if (full) {
                res.hist.push_back({b, player, policy_of(o), true, o.root_value});
                res.ply_idx.push_back(ply);
            }
            b[a] = (int8_t)player;
            res.moves.push_back(a);
            player = player == 1 ? 2 : 1;
            ply++;
        }
        res.total_plies = ply;
        return res;
    }

    SPResult play_league(const SPTask& t, std::mt19937& rng) {
        const int NC = BOARD_SIZE * BOARD_SIZE;
        const SPConfig& c = t.cfg;
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        MCTSEngine eng, leng;
        const int mcts_player = t.mcts_plays_x ? 1 : 2;
        std::vector<int8_t> b(NC, 0);
        int player = 1;
        if (t.opening >= 0) { b[t.opening] = 1; player = 2; }
        SPResult res;
        int decisions = 0;
        for (;;) {
            int stones = 0;
            for (int i = 0; i < NC; i++) stones += b[i] != 0;
            if (c.max_plies && stones >= c.max_plies) { res.outcome = 0; break; }
            if (check_win(b, 1, BOARD_SIZE, WIN_LENGTH)) { res.outcome = 1; break; }
            if (check_win(b, 2, BOARD_SIZE, WIN_LENGTH)) { res.outcome = -1; break; }
            if (check_draw(b)) { res.outcome = 0; break; }
            int num_empty = NC - stones;
            bool full = c.full_search_prob >= 1.0 || U01(rng) < c.full_search_prob;
            int sims = full ? sims_for_state_c(num_empty, NC, c)
                            : (c.fast_sims ? c.fast_sims : std::max(1, c.num_simulations / 6));
            int a;
            if (player == mcts_player) {
                SearchOut o = do_search(eng, 0, b, player, sims, full, c);
                if (o.acts.empty()) { res.outcome = 0; break; }
                a = pick(o, decisions < c.temperature_moves, rng);
                decisions++;
                if (full) { res.hist.push_back({b, player, policy_of(o), true, o.root_value}); res.ply_idx.push_back(stones); }
            } else {
                SearchOut o = do_search(leng, t.league_tag, b, player, sims, false, c);
                if (o.acts.empty()) { res.outcome = 0; break; }
                a = pick(o, false, rng);
                if (full && record_league_side) res.hist.push_back({b, player, policy_of(o), false, 0.0});
                decisions++;
            }
            b[a] = (int8_t)player;
            res.moves.push_back(a);
            player = player == 1 ? 2 : 1;
        }
        res.total_plies = (int)res.moves.size();
        if (res.outcome == 1) res.mcts_result = mcts_player == 1 ? 1 : -1;
        else if (res.outcome == -1) res.mcts_result = mcts_player == 2 ? 1 : -1;
        else res.mcts_result = 0;
        return res;
    }

    py::list results_to_py() {
        py::list out;
        for (size_t i = 0; i < results_.size(); i++) {
            const SPResult& r = results_[i];
            py::list hist;
            for (const auto& h : r.hist) {
                py::array_t<int8_t> bd({BOARD_SIZE, BOARD_SIZE}, h.board.data());
                py::array_t<float> pol({BOARD_SIZE, BOARD_SIZE}, h.policy.data());
                hist.append(py::make_tuple(bd, h.player == 1 ? "X" : "O", pol,
                                           h.has_q ? py::object(py::float_(h.q)) : py::object(py::none())));
            }
            py::list plies;
            for (int p : r.ply_idx) plies.append(p);
            py::list mv;
            for (int a : r.moves) mv.append(a);
            out.append(py::make_tuple(hist, r.outcome, plies,
                                      r.truncated ? py::object(py::none()) : py::object(py::int_(r.total_plies)),
                                      r.mcts_result, mv));
        }
        return out;
    }
};

PYBIND11_MODULE(MCTS_MODULE_NAME, m) {
    // module_local: lets this build be imported side by side with another
    // mcts_core build (same C++ type name) in one process for A/B testing.
    // 11x11_v2: VCF switch for the whole module (0 = off).
    m.def("set_vcf", [](int depth, long budget) { g_vcf_depth = depth; g_vcf_budget = budget; },
          py::arg("depth"), py::arg("budget") = 3000);
    m.def("vct_move", [](py::list board, std::string player, int BOARD_SIZE, int WIN_LENGTH, int depth, long budget) {
              std::vector<int8_t> b(BOARD_SIZE * BOARD_SIZE, 0);
              for (int r = 0; r < BOARD_SIZE; r++) { py::list row = board[r];
                  for (int c = 0; c < BOARD_SIZE; c++) { std::string x = py::str(row[c]); b[r * BOARD_SIZE + c] = x == "X" ? 1 : (x == "O" ? 2 : 0); } }
              int A = player == "X" ? 1 : 2, first = -1;
              long left = budget;
              bool ok = vct_rec(b, A, 3 - A, depth, left, BOARD_SIZE, WIN_LENGTH, &first);
              return py::make_tuple(ok ? first : -1, budget - left); },
          "First move of a VCT (fours and threes) for `player` or -1, and nodes used (test only; not used by the search).",
          py::arg("board"), py::arg("player"), py::arg("BOARD_SIZE"), py::arg("WIN_LENGTH"), py::arg("depth") = 6, py::arg("budget") = 20000);
    m.def("vcf_move", [](py::list board, std::string player, int BOARD_SIZE, int WIN_LENGTH) {
              std::vector<int8_t> b(BOARD_SIZE * BOARD_SIZE, 0);
              for (int r = 0; r < BOARD_SIZE; r++) { py::list row = board[r];
                  for (int c = 0; c < BOARD_SIZE; c++) { std::string x = py::str(row[c]); b[r * BOARD_SIZE + c] = x == "X" ? 1 : (x == "O" ? 2 : 0); } }
              int plies = 0; int mv = vcf_first_move(b, player == "X" ? 1 : 2, BOARD_SIZE, WIN_LENGTH, &plies);
              return py::make_tuple(mv, plies); },
          "First move of a VCF for `player` (or -1) and the line length in plies (test/analysis).");
    m.def("encode_planes", [](py::array_t<int8_t, py::array::c_style | py::array::forcecast> board, std::string player,
                              int BOARD_SIZE, int WIN_LENGTH) {
              std::vector<int8_t> b(board.data(), board.data() + BOARD_SIZE * BOARD_SIZE);
              py::array_t<float> out({5, BOARD_SIZE, BOARD_SIZE});
              encode_planes(b, player == "X" ? 1 : 2, BOARD_SIZE, WIN_LENGTH, static_cast<float*>(out.request().ptr));
              return out; },
          "Network input planes (curr, opp, empty, fork curr, fork opp) of a numeric board (test).");
    py::class_<SelfPlayRunner>(m, "SelfPlayRunner", py::module_local())
        .def(py::init<>())
        .def_readwrite("board_size", &SelfPlayRunner::BOARD_SIZE)
        .def_readwrite("win_length", &SelfPlayRunner::WIN_LENGTH)
        .def_readwrite("eval_batch", &SelfPlayRunner::eval_batch)
        .def_readwrite("forced_k", &SelfPlayRunner::forced_k)
        .def_readwrite("mega_d1", &SelfPlayRunner::mega_d1)
        .def_readwrite("mega_d2", &SelfPlayRunner::mega_d2)
        .def_readwrite("mega_min_sims", &SelfPlayRunner::mega_min_sims)
        .def_readwrite("record_league_side", &SelfPlayRunner::record_league_side)
        .def_readwrite("gumbel_m_full", &SelfPlayRunner::gumbel_m_full)
        .def_readwrite("gumbel_m_fast", &SelfPlayRunner::gumbel_m_fast)
        .def_readonly("n_batches", &SelfPlayRunner::n_batches)
        .def_readonly("n_rows", &SelfPlayRunner::n_rows)
        .def_readonly("max_rows", &SelfPlayRunner::max_rows)
        .def_readwrite("cache_cap", &SelfPlayRunner::cache_cap)
        .def_property_readonly("cache_hits", [](SelfPlayRunner& r) { return r.cache_hits.load(); })
        .def("add_task", &SelfPlayRunner::add_task,
             py::arg("kind"), py::arg("start_board"), py::arg("start_player"), py::arg("mcts_plays_x"),
             py::arg("opening"), py::arg("league_tag"), py::arg("cfg"))
        .def("run", &SelfPlayRunner::run,
             "Play every added task; eval_fn(tag, planes float32 (n,5,N,N)) -> (probs (n,N*N), values (n,)). "
             "Returns [(history, outcome, ply_indices, total_plies|None, mcts_result), ...] in task order.",
             py::arg("eval_fn"), py::arg("n_threads"), py::arg("target_rows"), py::arg("max_wait_ms"));
    py::class_<MCTSEngine>(m, "MCTSEngine", py::module_local())
        .def(py::init<>())
        .def("search", &MCTSEngine::search,
             "Stateful MCTS search (PUCT) with tree reuse across consecutive same-mover diff-1 calls on this instance",
             py::arg("board"), py::arg("current_player"), py::arg("eval_fn"), py::arg("num_simulations"),
             py::arg("c_puct"), py::arg("add_noise"), py::arg("dirichlet_alpha"), py::arg("noise_weight"),
             py::arg("BOARD_SIZE"), py::arg("WIN_LENGTH"), py::arg("forced_k") = 0.0,
             py::arg("mega_d1") = 0, py::arg("mega_d2") = 0)
        .def("search_batched", &MCTSEngine::search_batched,
             "Stateful batched MCTS search (virtual loss) with tree reuse across consecutive same-mover diff-1 calls on this instance",
             py::arg("board"), py::arg("current_player"), py::arg("eval_batch_fn"), py::arg("num_simulations"),
             py::arg("c_puct"), py::arg("add_noise"), py::arg("dirichlet_alpha"), py::arg("noise_weight"),
             py::arg("BOARD_SIZE"), py::arg("WIN_LENGTH"), py::arg("eval_batch_size"), py::arg("forced_k") = 0.0,
             py::arg("mega_d1") = 0, py::arg("mega_d2") = 0)
        .def("principal_variation", &MCTSEngine::principal_variation,
             "Most-visited line below root child first_action from the last search: [(action, visits), ...]",
             py::arg("first_action"), py::arg("max_len") = 10)
        .def("reset", &MCTSEngine::reset,
             "Discard the persisted tree; the next search()/search_batched() call starts fresh, matching today's per-call behavior");
}
