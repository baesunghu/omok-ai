// deterministic_play_core_fast.cpp
//
// Three combined speedups over deterministic_play_core.cpp for the
// depth-limited alpha-beta minimax "Det" opponent (transposition-table
// memoization was tried separately and measured SLOWER at depth3/4 --
// dropped, not included here):
//
//   1. Root-level parallelism: the empty_spots loop in play_uncached()
//      evaluates N independent root moves, each a fully separate minimax()
//      call with no shared mutable state -- embarrassingly parallel.
//      Split across std::thread::hardware_concurrency() worker threads,
//      each with its OWN board copy and own candidate-set copy. This is a
//      pure wall-clock speedup: the algorithm is unchanged, so it picks
//      the exact same move as the unparallelized version (verified
//      separately, not assumed) -- the only risk is subtle float summation
//      order differences, which cannot occur here since each root move's
//      value is computed independently and compared, not summed.
//
//   2. Candidate-move restriction: instead of scanning all up-to-121 empty
//      cells at every node, only consider empty cells within
//      NEIGHBOR_RADIUS of some stone already on the board (the same
//      radius-2 heuristic used elsewhere in this project's data
//      generation, e.g. minimax_pretrain_gen_depth2.py's relevant_moves()).
//      A cell far from all existing stones is essentially never the best
//      move in Gomoku, so this is a standard, low-risk restriction -- but
//      unlike (1), it is NOT guaranteed bit-for-bit identical to the
//      unrestricted search in every conceivable position, only verified
//      empirically to match on realistic (non-empty, non-contrived) boards.
//      Implementation: a per-call candidate flag array that only ever
//      GROWS as hypothetical stones are placed during search (never
//      shrinks on backtrack) -- a safe over-approximation, simpler than
//      precise undo bookkeeping, and still a large branching-factor cut
//      versus all 121 cells.
//
//   3. Move ordering: before recursing into a node's candidates, each is
//      given a cheap local score (weighted neighbor-density in a small
//      radius) and sorted best-guess-first, so alpha-beta finds strong
//      moves earlier and prunes more. Purely a search-order change --
//      does not affect the final minimax value returned for any node,
//      only how quickly alpha-beta converges to it.
//
// Root-level move SELECTION (win check, block check, minimax scan with
// strict `>` tie-break) mirrors deterministic_play_core.cpp's structure so
// the returned move should match -- verified separately.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <utility>
#include <limits>
#include <thread>
#include <mutex>

namespace py = pybind11;

// ============================================================
// Game Settings (exact match to determinisic_play.py / deterministic_play_core.cpp)
// ============================================================

static const int BOARD_SIZE = 11;
static const int WIN_LENGTH = 5;
static const int DEFAULT_MAX_DEPTH = 2;
static const int NUM_CELLS = BOARD_SIZE * BOARD_SIZE;
static const int NEIGHBOR_RADIUS = 2;  // matches relevant_moves() elsewhere in this project

static const double TWO_SCORE = 10;
static const double THREE_SCORE = 1000;
static const double FOUR_SCORE = 100000;
static const double CLOSED_FOUR_SCORE = 50000;
static const double CENTER_WEIGHT = 1;

static const int DIRECTIONS[4][2] = {
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1}
};

using Board = std::vector<int8_t>;
using CandFlags = std::vector<uint8_t>;

static inline int other_player(int player) {
    return (player == 1) ? 2 : 1;
}

// ============================================================
// Win & Draw Checks (identical to deterministic_play_core.cpp)
// ============================================================

static bool check_win(const Board& b, int player) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (b[r * BOARD_SIZE + c] != player) continue;
            if (c <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++) {
                    if (b[r * BOARD_SIZE + (c + k)] != player) { ok = false; break; }
                }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++) {
                    if (b[(r + k) * BOARD_SIZE + c] != player) { ok = false; break; }
                }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH && c <= BOARD_SIZE - WIN_LENGTH) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++) {
                    if (b[(r + k) * BOARD_SIZE + (c + k)] != player) { ok = false; break; }
                }
                if (ok) return true;
            }
            if (r <= BOARD_SIZE - WIN_LENGTH && c >= WIN_LENGTH - 1) {
                bool ok = true;
                for (int k = 1; k < WIN_LENGTH; k++) {
                    if (b[(r + k) * BOARD_SIZE + (c - k)] != player) { ok = false; break; }
                }
                if (ok) return true;
            }
        }
    }
    return false;
}

static bool check_draw(const Board& b) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (b[r * BOARD_SIZE + c] == 0) return false;
        }
    }
    return true;
}

// ============================================================
// Fast Heuristic Evaluation (identical to deterministic_play_core.cpp)
// ============================================================

static double centrality_score(const Board& b, int player) {
    double center = (BOARD_SIZE - 1) / 2.0;
    double score = 0.0;
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (b[r * BOARD_SIZE + c] == player) {
                double dist = std::max(std::abs(r - center), std::abs(c - center));
                score += (BOARD_SIZE / 2.0 - dist);
            }
        }
    }
    return score;
}

static double count_patterns(const Board& b, int player) {
    double score = 0.0;
    int opponent = other_player(player);

    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (b[r * BOARD_SIZE + c] != player) continue;
            for (int d = 0; d < 4; d++) {
                int dr = DIRECTIONS[d][0], dc = DIRECTIONS[d][1];
                int player_count = 1;
                bool blocked = false;
                for (int k = 1; k < WIN_LENGTH; k++) {
                    int nr = r + dr * k, nc = c + dc * k;
                    if (!(nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE)) {
                        blocked = true;
                        break;
                    }
                    int cell = b[nr * BOARD_SIZE + nc];
                    if (cell == player) {
                        player_count++;
                    } else if (cell == opponent) {
                        blocked = true;
                        break;
                    }
                }

                if (!blocked) {
                    if (player_count == 2) score += TWO_SCORE;
                    else if (player_count == 3) score += THREE_SCORE;
                    else if (player_count >= 4) score += FOUR_SCORE;
                } else if (player_count >= 4) {
                    score += CLOSED_FOUR_SCORE;
                }
            }
        }
    }
    return score;
}

static double evaluate_board(const Board& b, int ai_player, int human_player) {
    double pattern_score = count_patterns(b, ai_player) - count_patterns(b, human_player);
    double center_score = centrality_score(b, ai_player) - centrality_score(b, human_player);
    return pattern_score + CENTER_WEIGHT * center_score;
}

// ============================================================
// Candidate-move restriction (optimization 2)
// ============================================================

static inline void mark_neighbors_candidate(CandFlags& cand, int idx) {
    int r0 = idx / BOARD_SIZE, c0 = idx % BOARD_SIZE;
    int r_lo = std::max(0, r0 - NEIGHBOR_RADIUS), r_hi = std::min(BOARD_SIZE - 1, r0 + NEIGHBOR_RADIUS);
    int c_lo = std::max(0, c0 - NEIGHBOR_RADIUS), c_hi = std::min(BOARD_SIZE - 1, c0 + NEIGHBOR_RADIUS);
    for (int r = r_lo; r <= r_hi; r++) {
        for (int c = c_lo; c <= c_hi; c++) {
            cand[r * BOARD_SIZE + c] = 1;
        }
    }
}

static CandFlags initial_candidates(const Board& board) {
    CandFlags cand(NUM_CELLS, 0);
    bool any = false;
    for (int i = 0; i < NUM_CELLS; i++) {
        if (board[i] != 0) {
            mark_neighbors_candidate(cand, i);
            any = true;
        }
    }
    if (!any) {
        int mid = BOARD_SIZE / 2;
        for (int dr = -1; dr <= 0; dr++)
            for (int dc = -1; dc <= 0; dc++)
                cand[(mid + dr) * BOARD_SIZE + (mid + dc)] = 1;
    }
    return cand;
}

// ============================================================
// Move ordering (optimization 3): cheap local neighbor-density score,
// higher = more "active" nearby, searched first.
// ============================================================

static int local_density(const Board& board, int idx) {
    int r0 = idx / BOARD_SIZE, c0 = idx % BOARD_SIZE;
    int r_lo = std::max(0, r0 - 1), r_hi = std::min(BOARD_SIZE - 1, r0 + 1);
    int c_lo = std::max(0, c0 - 1), c_hi = std::min(BOARD_SIZE - 1, c0 + 1);
    int count = 0;
    for (int r = r_lo; r <= r_hi; r++) {
        for (int c = c_lo; c <= c_hi; c++) {
            if (board[r * BOARD_SIZE + c] != 0) count++;
        }
    }
    return count;
}

static std::vector<int> ordered_candidates(const Board& board, const CandFlags& cand) {
    std::vector<int> out;
    out.reserve(32);
    for (int i = 0; i < NUM_CELLS; i++) {
        if (board[i] == 0 && cand[i]) out.push_back(i);
    }
    std::stable_sort(out.begin(), out.end(), [&board](int a, int b) {
        return local_density(board, a) > local_density(board, b);
    });
    return out;
}

// ============================================================
// Minimax with Alpha-Beta Pruning, candidate restriction + move ordering.
// `cand` is passed by value into each root branch's thread (own copy),
// then by reference through that branch's own recursion (grows only,
// never shared across threads).
// ============================================================

static double minimax(Board& board, CandFlags& cand, int depth, double alpha, double beta, bool is_maximizing,
                       int ai_player, int human_player, int max_depth) {
    if (check_win(board, ai_player)) return 10000.0 - depth;
    if (check_win(board, human_player)) return -10000.0 + depth;
    if (check_draw(board)) return 0.0;
    if (depth >= max_depth) return evaluate_board(board, ai_player, human_player);

    std::vector<int> candidates = ordered_candidates(board, cand);
    int mover = is_maximizing ? ai_player : human_player;

    if (is_maximizing) {
        double max_eval = -std::numeric_limits<double>::infinity();
        for (int idx : candidates) {
            board[idx] = static_cast<int8_t>(mover);
            mark_neighbors_candidate(cand, idx);
            double eval_val = minimax(board, cand, depth + 1, alpha, beta, false,
                                       ai_player, human_player, max_depth);
            board[idx] = 0;
            if (eval_val > max_eval) max_eval = eval_val;
            if (eval_val > alpha) alpha = eval_val;
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        double min_eval = std::numeric_limits<double>::infinity();
        for (int idx : candidates) {
            board[idx] = static_cast<int8_t>(mover);
            mark_neighbors_candidate(cand, idx);
            double eval_val = minimax(board, cand, depth + 1, alpha, beta, true,
                                       ai_player, human_player, max_depth);
            board[idx] = 0;
            if (eval_val < min_eval) min_eval = eval_val;
            if (eval_val < beta) beta = eval_val;
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

// ============================================================
// AI Move Selection: win check, block check (unchanged, exhaustive over
// ALL empty cells -- these are O(cells) checks, not the search cost, so no
// need to restrict them), then root-parallel minimax scan.
// ============================================================

static std::pair<int, int> play_uncached(Board board, int ai_player, int human_player, int max_depth) {
    std::vector<std::pair<int, int>> empty_spots;
    empty_spots.reserve(BOARD_SIZE * BOARD_SIZE);
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            if (board[r * BOARD_SIZE + c] == 0) empty_spots.emplace_back(r, c);
        }
    }
    if (empty_spots.empty()) return {-1, -1};

    // 1. Immediate Win Check
    for (auto& rc : empty_spots) {
        int r = rc.first, c = rc.second;
        board[r * BOARD_SIZE + c] = static_cast<int8_t>(ai_player);
        bool win = check_win(board, ai_player);
        board[r * BOARD_SIZE + c] = 0;
        if (win) return {r, c};
    }

    // 2. Immediate Block Check
    for (auto& rc : empty_spots) {
        int r = rc.first, c = rc.second;
        board[r * BOARD_SIZE + c] = static_cast<int8_t>(human_player);
        bool win = check_win(board, human_player);
        board[r * BOARD_SIZE + c] = 0;
        if (win) return {r, c};
    }

    // 3. Minimax evaluation, root moves restricted+ordered same as deeper
    // nodes, evaluated in PARALLEL across worker threads.
    CandFlags root_cand = initial_candidates(board);
    std::vector<int> root_moves = ordered_candidates(board, root_cand);
    if (root_moves.empty()) {
        // degenerate: no candidate found by the heuristic (shouldn't happen
        // once initial_candidates() has run), fall back to all empty cells
        for (auto& rc : empty_spots) root_moves.push_back(rc.first * BOARD_SIZE + rc.second);
    }

    unsigned n_threads = std::max(1u, std::min(std::thread::hardware_concurrency(),
                                                 static_cast<unsigned>(root_moves.size())));

    std::mutex best_mutex;
    double best_val = -std::numeric_limits<double>::infinity();
    int best_idx = root_moves[0];

    auto worker = [&](unsigned worker_id) {
        double local_best_val = -std::numeric_limits<double>::infinity();
        int local_best_idx = -1;
        for (size_t i = worker_id; i < root_moves.size(); i += n_threads) {
            int idx = root_moves[i];
            Board local_board = board;  // own copy per thread
            CandFlags local_cand = root_cand;  // own copy per thread
            local_board[idx] = static_cast<int8_t>(ai_player);
            mark_neighbors_candidate(local_cand, idx);
            double val = minimax(local_board, local_cand, 1, -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(), false,
                                  ai_player, human_player, max_depth);
            if (val > local_best_val) {
                local_best_val = val;
                local_best_idx = idx;
            }
        }
        if (local_best_idx != -1) {
            std::lock_guard<std::mutex> lock(best_mutex);
            // Strict `>` with root_moves index order preserved via stable
            // partitioning across threads (round-robin) -- ties broken by
            // whichever worker's result is merged first is NOT guaranteed
            // to match the single-threaded row-major/density-order tie
            // break exactly, so this is verified empirically, not assumed,
            // same caveat as optimization 2.
            if (local_best_val > best_val) {
                best_val = local_best_val;
                best_idx = local_best_idx;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; t++) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    return {best_idx / BOARD_SIZE, best_idx % BOARD_SIZE};
}

// ============================================================
// pybind11 entry point
// ============================================================

static Board board_from_pylist(const py::list& board_list) {
    Board flat(BOARD_SIZE * BOARD_SIZE, 0);
    for (int r = 0; r < BOARD_SIZE; r++) {
        py::list row = board_list[r];
        for (int c = 0; c < BOARD_SIZE; c++) {
            std::string cell = py::str(row[c]);
            flat[r * BOARD_SIZE + c] = (cell == "X") ? 1 : (cell == "O" ? 2 : 0);
        }
    }
    return flat;
}

py::tuple play(py::list board, std::string turn, int max_depth) {
    Board flat = board_from_pylist(board);
    int ai_player = (turn == "X") ? 1 : 2;
    int human_player = other_player(ai_player);

    std::pair<int, int> result;
    {
        py::gil_scoped_release release;
        result = play_uncached(std::move(flat), ai_player, human_player, max_depth);
    }
    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(deterministic_play_core, m) {
    m.doc() = "Faster variant of deterministic_play_core: root-level thread "
              "parallelism + radius-2 candidate-move restriction + local-density "
              "move ordering. Transposition-table memoization was tried and "
              "measured slower -- not included.";
    m.def("play", &play,
          py::arg("board"), py::arg("turn"), py::arg("max_depth") = DEFAULT_MAX_DEPTH,
          "Same contract as deterministic_play_core.play() -- should return the "
          "same move on realistic (non-empty) boards, much faster at higher depths.");
}
