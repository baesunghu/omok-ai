# one-off patch: adds the Gumbel root to the v3 copy of mcts_core_solver.cpp (2026-10-22)
p = "mcts_core_solver.cpp"
s = open(p, encoding="utf-8").read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""    std::vector<int> proven;
    double root_value = 0.0;
};""", """    std::vector<int> proven;
    double root_value = 0.0;
    // Gumbel root (11x11_v3): improved-policy target aligned with acts, and the move the
    // Gumbel/sequential-halving search chose (-1 when Gumbel was not used).
    std::vector<double> target;
    int chosen = -1;
};""")
rep("""                          int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                          double forced_k, int mega_d1, int mega_d2);""", """                          int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                          double forced_k, int mega_d1, int mega_d2, int gumbel_m = 0);""")
rep("""                                  int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                                  double forced_k, int mega_d1, int mega_d2) {
    const double VIRTUAL_LOSS = 1.0;""", """                                  int BOARD_SIZE, int WIN_LENGTH, int eval_batch_size,
                                  double forced_k, int mega_d1, int mega_d2, int gumbel_m) {
    const double VIRTUAL_LOSS = 1.0;""")
rep("""        if (add_noise && !root.children.empty()) {
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
        long mega_total = mega_plan(root, root_board, BOARD_SIZE, mega_d1, mega_d2, mega_left);
        long target = (long)num_simulations + mega_total;
        long hard_cap = 3L * std::max<long>(1, target);
        long spent = 0;
        while (spent < hard_cap) {
            if (root.proven != 0) break;  // result already proven: nothing left to search
            long remaining = target - counted_root_visits(root);
            if (remaining <= 0) break;
            int group_size = (int)std::min<long>(eval_batch_size, remaining);
            spent += group_size;
""", """        const bool gumbel = gumbel_m > 0 && !root.children.empty();
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
                order.push_back({g_noise[kv.first] + g_logit[kv.first], kv.first});
            }
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
""")
rep("""                while (node->is_expanded && !node->children.empty() && node->proven == 0) {
                    int best_action = (node == &root) ? mega_pick(root, mega_left, mega_given_out) : -1;
                    if (best_action < 0)
                        best_action = (node == &root && forced_k > 0.0)
                            ? select_root_forced(node, c_puct, forced_k) : select_child(node, c_puct);
                    int r = best_action / BOARD_SIZE, c = best_action % BOARD_SIZE;
                    sim_board[r * BOARD_SIZE + c] = sim_player;
                    node = node->children[best_action].get();
                    sim_player = (sim_player == 1) ? 2 : 1;
                    path.push_back(node);
                    node->n_visits += 1;""", """                while (node->is_expanded && !node->children.empty() && node->proven == 0) {
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
                    node->n_visits += 1;""")
rep("""    return build_result_c(root, NUM_CELLS, BOARD_SIZE, c_puct, forced_k, &mega_given_out);
}

py::tuple MCTSEngine::search_batched(""", """    SearchOut out = build_result_c(root, NUM_CELLS, BOARD_SIZE, c_puct, gumbel ? 0.0 : forced_k, &mega_given_out);
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

py::tuple MCTSEngine::search_batched(""")

# --- self-play runner: Gumbel switches, target/chosen use, move list for aux targets ---
rep("""    int mega_d1 = 30, mega_d2 = 15, mega_min_sims = 1000;
    bool record_league_side = false;""", """    int mega_d1 = 30, mega_d2 = 15, mega_min_sims = 1000;
    bool record_league_side = false;
    int gumbel_m_full = 0, gumbel_m_fast = 0;   // 11x11_v3: Gumbel root candidates (0 = PUCT)""")
rep("""        bool mega = noisy && sims >= mega_min_sims;
        return eng.search_core(b, player, ev, sims, c.c_puct, noisy, c.dirichlet_alpha, c.noise_weight,
                               BOARD_SIZE, WIN_LENGTH, eval_batch, noisy ? forced_k : 0.0,
                               mega ? mega_d1 : 0, mega ? mega_d2 : 0);""", """        int gm = tag != 0 ? 0 : (noisy ? gumbel_m_full : gumbel_m_fast);
        bool mega = gm == 0 && noisy && sims >= mega_min_sims;
        return eng.search_core(b, player, ev, sims, c.c_puct, noisy || gm > 0, c.dirichlet_alpha, c.noise_weight,
                               BOARD_SIZE, WIN_LENGTH, eval_batch, (noisy && gm == 0) ? forced_k : 0.0,
                               mega ? mega_d1 : 0, mega ? mega_d2 : 0, gm);""")
rep("""        std::vector<float> p(BOARD_SIZE * BOARD_SIZE, 0.0f);
        long tot = 0;""", """        std::vector<float> p(BOARD_SIZE * BOARD_SIZE, 0.0f);
        if (!o.target.empty()) {
            for (size_t k = 0; k < o.acts.size(); k++) p[o.acts[k]] = (float)o.target[k];
            return p;
        }
        long tot = 0;""")
rep("""    int pick(const SearchOut& o, bool sample, std::mt19937& rng) {
        if (sample) {""", """    int pick(const SearchOut& o, bool sample, std::mt19937& rng) {
        if (o.chosen >= 0) return o.chosen;     // Gumbel: the sampled candidates already randomize
        if (sample) {""")
rep("""    int total_plies = 0;
    bool truncated = false;""", """    int total_plies = 0;
    std::vector<int> moves;       // every move of the game in order (v3 auxiliary targets)
    bool truncated = false;""")
rep("""            b[a] = (int8_t)player;
            player = player == 1 ? 2 : 1;
            ply++;""", """            b[a] = (int8_t)player;
            res.moves.push_back(a);
            player = player == 1 ? 2 : 1;
            ply++;""")
rep("""            out.append(py::make_tuple(hist, r.outcome, plies,
                                      r.truncated ? py::object(py::none()) : py::object(py::int_(r.total_plies)),
                                      r.mcts_result));""", """            py::list mv;
            for (int a : r.moves) mv.append(a);
            out.append(py::make_tuple(hist, r.outcome, plies,
                                      r.truncated ? py::object(py::none()) : py::object(py::int_(r.total_plies)),
                                      r.mcts_result, mv));""")
open(p, "w", encoding="utf-8").write(s)
print("patched")
