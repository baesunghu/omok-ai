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
    for (auto& kv : node->children) {
        if (wins_at(b, kv.first, player, BOARD_SIZE, WIN_LENGTH)) {
            kv.second->proven = -1;      // the child's mover (opponent) has lost
            kv.second->proof_dist = 0;
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
static py::tuple build_result(Node& root, int NUM_CELLS, int BOARD_SIZE,
                               double c_puct = 0.0, double forced_k = 0.0) {
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
        // Root value over the moves that still count: backups from moves
        // already proven to lose would otherwise drag the reported value
        // (used as the q in the z/q value-target mix) pessimistic.
        if (kept > 0) root_value = val_sum / (double)kept;
    }

    long total_visits = 0;
    for (auto& kv : shown) total_visits += kv.second;

    py::array_t<double> target_policy(NUM_CELLS);
    auto tp = target_policy.mutable_unchecked<1>();
    for (int i = 0; i < NUM_CELLS; i++) tp(i) = 0.0;

    py::list candidates;
    for (auto& kv : root.children) {
        int act = kv.first;
        Node* child = kv.second.get();
        long v = shown[act];
        double prob = total_visits > 0 ? (double)v / total_visits : 0.0;
        tp(act) = prob;
        py::dict cand;
        cand["action_idx"] = act;
        cand["move"] = py::make_tuple(act / BOARD_SIZE, act % BOARD_SIZE);
        cand["visit_count"] = v;
        cand["prob"] = prob;
        cand["q_value"] = child->mean_value;
        cand["proven"] = child->proven;  // for the child's own mover
        candidates.append(cand);
    }
    return py::make_tuple(target_policy, root_value, candidates);
}

class MCTSEngine {
public:
    py::tuple search(py::list board, std::string current_player, py::function eval_fn,
                      int num_simulations, double c_puct, bool add_noise,
                      double dirichlet_alpha, double noise_weight,
                      int BOARD_SIZE, int WIN_LENGTH, double forced_k = 0.0);

    py::tuple search_batched(py::list board, std::string current_player, py::function eval_batch_fn,
                              int num_simulations, double c_puct, bool add_noise,
                              double dirichlet_alpha, double noise_weight,
                              int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                              double forced_k = 0.0);

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
                              int BOARD_SIZE, int WIN_LENGTH, double forced_k) {
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
        long hard_cap = 3L * std::max(1, num_simulations);
        for (long sim = 0; sim < hard_cap; sim++) {
            if (root.proven != 0) break;  // result already proven: nothing left to search
            if (counted_root_visits(root) >= num_simulations) break;
            Node* node = &root;
            std::vector<int8_t> sim_board = root_board;
            int sim_player = root_player;
            int depth = 0;

            while (node->is_expanded && !node->children.empty() && node->proven == 0) {
                int best_action = (node == &root && forced_k > 0.0)
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

    return build_result(root, NUM_CELLS, BOARD_SIZE, c_puct, forced_k);
}

py::tuple MCTSEngine::search_batched(py::list board, std::string current_player, py::function eval_batch_fn,
                                      int num_simulations, double c_puct, bool add_noise,
                                      double dirichlet_alpha, double noise_weight,
                                      int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                                      double forced_k) {
    const double VIRTUAL_LOSS = 1.0;
    int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
    eval_batch_size = std::max(1, eval_batch_size);

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

    auto eval_one = [&](const std::vector<int8_t>& b, int player) -> std::pair<std::vector<double>, double> {
        py::gil_scoped_acquire acquire;
        py::list boards_arg;
        boards_arg.append(board_to_pyarray(b));
        py::list players_arg;
        players_arg.append(player == 1 ? std::string("X") : std::string("O"));
        py::object result = eval_batch_fn(boards_arg, players_arg);
        py::tuple rt = result.cast<py::tuple>();
        py::array_t<double> probs_arr = rt[0].cast<py::array_t<double>>();
        py::array_t<double> values_arr = rt[1].cast<py::array_t<double>>();
        auto pbuf = probs_arr.unchecked<2>();
        auto vbuf = values_arr.unchecked<1>();
        std::vector<double> probs(NUM_CELLS);
        for (int i = 0; i < NUM_CELLS; i++) probs[i] = pbuf(0, i);
        return {probs, vbuf(0)};
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

    {
        py::gil_scoped_release release;

        if (!root.is_expanded) {
            auto [probs, leaf_value] = eval_one(root_board, root_player);
            (void)leaf_value;
            expand_with_probs(&root, root_board, root_player, probs);
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
        // counted_root_visits), re-checked every group; hard cap so the loop
        // always terminates.
        long hard_cap = 3L * std::max(1, num_simulations);
        long spent = 0;
        while (spent < hard_cap) {
            if (root.proven != 0) break;  // result already proven: nothing left to search
            long remaining = (long)num_simulations - counted_root_visits(root);
            if (remaining <= 0) break;
            int group_size = (int)std::min<long>(eval_batch_size, remaining);
            spent += group_size;

            std::vector<GroupEntry> group;
            group.reserve(group_size);

            for (int g = 0; g < group_size; g++) {
                Node* node = &root;
                std::vector<int8_t> sim_board = root_board;
                int sim_player = root_player;
                std::vector<Node*> path = {node};

                while (node->is_expanded && !node->children.empty() && node->proven == 0) {
                    int best_action = (node == &root && forced_k > 0.0)
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
                py::gil_scoped_acquire acquire;
                py::list boards_arg;
                py::list players_arg;
                for (size_t i = 0; i < pending_boards.size(); i++) {
                    boards_arg.append(board_to_pyarray(*pending_boards[i]));
                    players_arg.append(pending_players[i] == 1 ? std::string("X") : std::string("O"));
                }
                py::object result = eval_batch_fn(boards_arg, players_arg);
                py::tuple rt = result.cast<py::tuple>();
                py::array_t<double> probs_arr = rt[0].cast<py::array_t<double>>();
                py::array_t<double> values_arr = rt[1].cast<py::array_t<double>>();
                auto pbuf = probs_arr.unchecked<2>();
                auto vbuf = values_arr.unchecked<1>();
                for (size_t idx = 0; idx < batch_keys.size(); idx++) {
                    std::vector<double> probs(NUM_CELLS);
                    for (int i = 0; i < NUM_CELLS; i++) probs[i] = pbuf((long)idx, i);
                    eval_cache[batch_keys[idx]] = {probs, vbuf((long)idx)};
                }
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
    }

    return build_result(root, NUM_CELLS, BOARD_SIZE, c_puct, forced_k);
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
PYBIND11_MODULE(MCTS_MODULE_NAME, m) {
    // module_local: lets this build be imported side by side with another
    // mcts_core build (same C++ type name) in one process for A/B testing.
    py::class_<MCTSEngine>(m, "MCTSEngine", py::module_local())
        .def(py::init<>())
        .def("search", &MCTSEngine::search,
             "Stateful MCTS search (PUCT) with tree reuse across consecutive same-mover diff-1 calls on this instance",
             py::arg("board"), py::arg("current_player"), py::arg("eval_fn"), py::arg("num_simulations"),
             py::arg("c_puct"), py::arg("add_noise"), py::arg("dirichlet_alpha"), py::arg("noise_weight"),
             py::arg("BOARD_SIZE"), py::arg("WIN_LENGTH"), py::arg("forced_k") = 0.0)
        .def("search_batched", &MCTSEngine::search_batched,
             "Stateful batched MCTS search (virtual loss) with tree reuse across consecutive same-mover diff-1 calls on this instance",
             py::arg("board"), py::arg("current_player"), py::arg("eval_batch_fn"), py::arg("num_simulations"),
             py::arg("c_puct"), py::arg("add_noise"), py::arg("dirichlet_alpha"), py::arg("noise_weight"),
             py::arg("BOARD_SIZE"), py::arg("WIN_LENGTH"), py::arg("eval_batch_size"), py::arg("forced_k") = 0.0)
        .def("principal_variation", &MCTSEngine::principal_variation,
             "Most-visited line below root child first_action from the last search: [(action, visits), ...]",
             py::arg("first_action"), py::arg("max_len") = 10)
        .def("reset", &MCTSEngine::reset,
             "Discard the persisted tree; the next search()/search_batched() call starts fresh, matching today's per-call behavior");
}
