import math
import queue
import threading
import numpy as np
import torch
import os

from alphago_lee import (AlphaZeroTicTacToe, board_to_input_tensor, board_to_input_array,
                          boards_to_input_tensor, board_to_input_array_from_numeric)
from fast_board import BOARD_SIZE, WIN_LENGTH, fast_check_win, fast_check_draw

NUM_CELLS = BOARD_SIZE * BOARD_SIZE

try:
    import mcts_core as _mcts_core
except ImportError:
    _mcts_core = None

# Gomoku-specific move-generation pruning: a stone this far from every
# existing stone can't join a WIN_LENGTH-in-a-row with any of them soon, so
# restricting expansion to empty cells within NEIGHBOR_RADIUS (Chebyshev
# distance, i.e. a square neighborhood -- diagonal 5-in-a-rows matter just as
# much as orthogonal ones) of some stone cuts the branching factor on this
# 100-cell board without excluding any move that could plausibly matter soon.
# This only affects the pure-Python search path below (10x10 has no compiled
# mcts_core.pyd, unlike 5x5, so this is the path actually used here) --
# standard technique in real Gomoku engines, not a hardcoded answer to any
# specific position.
NEIGHBOR_RADIUS = 2


def _relevant_moves(board):
    """Empty cells worth considering as the next move: within NEIGHBOR_RADIUS
    of some existing stone. Falls back to the true center cells on an empty
    board (nothing to anchor a neighborhood to yet), and to every empty cell
    in the -- should never happen -- case the neighborhood is somehow empty
    despite empty cells existing, so a pruning bug can never masquerade as
    "no legal moves" (which callers treat as a draw)."""
    occupied = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] != ' ']
    if not occupied:
        mid = BOARD_SIZE // 2
        return [(mid - 1, mid - 1), (mid - 1, mid), (mid, mid - 1), (mid, mid)]

    relevant = set()
    for (r0, c0) in occupied:
        for dr in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            for dc in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == ' ':
                    relevant.add((r, c))

    if not relevant:
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == ' ']
    return list(relevant)


class PooledEvaluator:
    """
    Combines eval_batch_fn calls from multiple concurrent self-play worker
    threads into fewer, larger GPU forward passes. Measured on this machine
    (GTX 1050): 6 threads each issuing independent batch=8 calls only hit
    2.17x aggregate throughput (concurrent GPU access partly serializes,
    unlike CPU cores) -- pooling those into shared batch~48 calls amortizes
    per-call overhead (kernel launch, .cpu() sync) across more samples.

    One dedicated background thread owns every model() call, so there is no
    lock contention or race on who runs a batch. Worker threads call
    submit() and block on a per-request threading.Event; the background
    thread accumulates requests from the queue until either target_batch
    samples are pending or max_wait seconds pass since the last flush, then
    runs one batched forward pass and wakes every waiting thread with its
    slice of the result.

    Safe to construct once per training run (not per-epoch): it holds a
    reference to the model object, not a snapshot of its weights, so it
    keeps seeing the current weights across epochs as optimizer.step()
    mutates the same nn.Module in place. Call stop() only at process exit.
    """
    def __init__(self, model, device, target_batch=48, max_wait=0.004):
        self.model = model
        self.device = device
        self.target_batch = target_batch
        self.max_wait = max_wait
        self._queue = queue.Queue()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # 2026-09-11 instrumentation: tracks real achieved batch sizes so a
        # target_batch change (e.g. 48->128, see forward_pass_benchmark.py's
        # finding that bs=128 gives ~17% more throughput in a synthetic
        # single-shot benchmark) can be verified against ACTUAL concurrency
        # in live self-play traffic instead of assumed. Printed periodically
        # from _flush, not exposed as a public API -- purely observational.
        self._flush_count = 0
        self._total_samples = 0
        self._max_flush = 0
        self._at_target_count = 0  # flushes that hit target_batch (vs timed out short)

    def stats_summary(self):
        if self._flush_count == 0:
            return "no flushes yet"
        avg = self._total_samples / self._flush_count
        return (f"{self._flush_count} flushes, avg batch {avg:.1f}, max {self._max_flush}, "
                f"target_batch={self.target_batch}, {self._at_target_count}/{self._flush_count} "
                f"({100*self._at_target_count/self._flush_count:.0f}%) hit target vs timed out short")

    def submit(self, input_arrays):
        """input_arrays: list of (C,H,W) CPU numpy arrays (board_to_input_array
        output, no device transfer yet). Returns (probs_arr, values_arr)
        aligned to input order, matching a direct model() call's output
        shape/dtype. Deferring the host->device transfer to _flush lets many
        callers' arrays combine into one np.stack + one .to(device) call
        instead of each call paying its own transfer."""
        n = len(input_arrays)
        if n == 0:
            return np.zeros((0, NUM_CELLS), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        event = threading.Event()
        req = {'inputs': input_arrays, 'event': event, 'result': None}
        self._queue.put(req)
        event.wait()
        return req['result']

    def stop(self):
        self._stop = True
        self._queue.put(None)
        self._thread.join(timeout=1.0)

    def _run(self):
        pending = []
        pending_count = 0
        while True:
            timeout = self.max_wait if pending else None
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = 'FLUSH'
            if item is None:
                if pending:
                    self._flush(pending)
                break
            if self._stop and item == 'FLUSH':
                break
            if item != 'FLUSH':
                pending.append(item)
                pending_count += len(item['inputs'])
            if pending and (pending_count >= self.target_batch or item == 'FLUSH'):
                self._flush(pending)
                pending = []
                pending_count = 0

    def _flush(self, pending):
        all_inputs = []
        offsets = []
        for r in pending:
            offsets.append((len(all_inputs), len(r['inputs'])))
            all_inputs.extend(r['inputs'])
        batch = boards_to_input_tensor(all_inputs, self.device)
        with torch.inference_mode():
            policy_logits, value_tensor = self.model(batch)
        probs_batch = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
        if probs_batch.ndim == 1:
            probs_batch = probs_batch.reshape(1, -1)
        values_batch = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
        for (start, count), r in zip(offsets, pending):
            r['result'] = (probs_batch[start:start + count], values_batch[start:start + count])
            r['event'].set()

        n = len(all_inputs)
        self._flush_count += 1
        self._total_samples += n
        if n > self._max_flush:
            self._max_flush = n
        if n >= self.target_batch:
            self._at_target_count += 1
        if self._flush_count % 500 == 0:
            print(f"  [PooledEvaluator stats] {self.stats_summary()}", flush=True)


class MCTSNode:
    """
    Monte Carlo Tree Search Node for AlphaZero.
    Tracks visit count N, total value W, mean value Q, prior probability P.
    """
    def __init__(self, parent=None, prior_prob=1.0, player_turn='X'):
        self.parent = parent
        self.children = {}  # action_index (0..24) -> MCTSNode
        self.n_visits = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        self.prior_prob = prior_prob
        self.player_turn = player_turn
        self.is_expanded = False

    def value(self):
        return self.mean_value

    def update(self, leaf_value):
        self.n_visits += 1
        self.total_value += leaf_value
        self.mean_value = self.total_value / self.n_visits


class MCTS:
    """
    High-Performance AlphaZero Monte Carlo Tree Search Engine with Evaluation Caching.
    Uses PUCT formula: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
    """
    def __init__(self, c_puct=1.414, eval_batch_size=1, shared_cache=None, pooled_evaluator=None):
        """
        eval_batch_size=1 (default) runs the original one-leaf-per-simulation
        search, unchanged. eval_batch_size>1 groups that many simulations'
        selection phases together (via standard virtual loss) and sends their
        leaves through the model in one batched forward call -- measured
        ~1.5-2x faster self-play throughput at batch_size=4-8. This trades a
        small amount of search-order fidelity for speed (a simulation's path
        selection can see virtual-loss-inflated visit counts from other
        in-flight simulations in the same group), so it is only wired into
        the training self-play/vs-Det generation paths, not the formal
        evaluate_vs_deterministic metric or the human-play API, which stay on
        the exact eval_batch_size=1 sequential path.

        shared_cache=None (default) keeps the original behavior: a private
        cache cleared at the start of every search() call. Pass a dict to
        share it across many search() calls instead (e.g. one dict per
        training epoch, reused across every game and every move generated in
        that epoch) -- valid because (board, player) -> (probs, value) is a
        pure function of the model's weights, which are frozen for the whole
        self-play/vs-Det generation phase each epoch. The caller is
        responsible for building a fresh dict every epoch, since
        optimizer.step() changes the weights and would otherwise make the
        cache stale.
        """
        self.c_puct = c_puct
        self.eval_batch_size = max(1, eval_batch_size)
        self._eval_cache = shared_cache if shared_cache is not None else {}
        self._owns_cache = shared_cache is None
        self._pooled_evaluator = pooled_evaluator
        # Persistent C++ search-tree handle (2026-09-11, tree reuse): one
        # MCTSEngine per MCTS Python object, matching this object's own
        # lifetime -- every game-playing call site already constructs a
        # fresh MCTS() per game (or per side), and calls .search() once per
        # ply on that same object, so this maps 1:1 onto "one persistent
        # tree per game/side" with no new object-lifecycle management
        # needed anywhere else. See MCTSEngine's own class docstring in
        # cpp_mcts/mcts_core_tree_reuse.cpp for the full reuse design.
        self._cpp_engine = _mcts_core.MCTSEngine() if _mcts_core is not None else None

    def get_puct_score(self, sqrt_parent_visits, child_node):
        # child_node.value() is stored from the perspective of whoever moves
        # AT the child (the opponent, relative to the parent choosing among
        # its children) -- negate it so PUCT selects by the parent's own
        # perspective, not the opponent's.
        pb_c = self.c_puct * sqrt_parent_visits / (1 + child_node.n_visits)
        return -child_node.value() + pb_c * child_node.prior_prob

    def _cpp_eval_fn(self, model, device):
        """Single-board eval callback for _mcts_core.mcts_search. Routes
        through self._eval_cache first so eval_batch_size==1 callers keep the
        exact same per-call-cleared-cache semantics as the pure-Python path
        (no eval_batch_size==1 call site ever passes a shared_cache, so this
        cache is always freshly cleared for this search() call anyway).

        mcts_core.pyd (currently built from cpp_mcts\\mcts_core_tree_reuse.cpp
        -- a stateful MCTSEngine with persistent-tree reuse, itself stacked
        on cpp_mcts\\mcts_core_pruned_deploy.cpp) hands the leaf board to
        Python as a (BOARD_SIZE, BOARD_SIZE) numpy int8 array (0=empty, 1=X, 2=O)
        instead of a nested list of "X"/"O"/" " strings, so this routes
        through board_to_input_array_from_numeric (integer comparisons)
        instead of board_to_input_tensor (numpy string-equality), and the
        cache key is board_arr.tobytes() instead of a tuple-of-tuples of
        Python strings -- much cheaper to build/hash for this hot per-leaf
        path (measured ~2.8x faster leaf-conversion in isolation)."""
        def eval_fn(board_arr, player):
            key = (board_arr.tobytes(), player)
            cached = self._eval_cache.get(key)
            if cached is not None:
                return cached
            planes = board_to_input_array_from_numeric(board_arr, player)
            input_tensor = torch.from_numpy(planes).float().to(device)
            input_tensor = input_tensor.unsqueeze(0).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                policy_logits, value_tensor = model(input_tensor)
            probs = torch.exp(policy_logits).squeeze().cpu().numpy().astype(np.float64)
            leaf_value = float(value_tensor.item())
            self._eval_cache[key] = (probs, leaf_value)
            return probs, leaf_value
        return eval_fn

    def _cpp_eval_batch_fn(self, model, device):
        """Batched eval callback for _mcts_core.mcts_search_batched. Splits
        the incoming group into cache hits/misses against self._eval_cache
        (which may be a shared_cache spanning many search() calls across a
        whole training epoch) so that optimization -- deduping NN forward
        passes across games/moves within an epoch -- is preserved exactly as
        in the pure-Python _search_batched path, even though the C++ engine's
        own internal eval_cache only dedupes within a single search() call.

        mcts_core.pyd hands each leaf board as a (BOARD_SIZE, BOARD_SIZE)
        numpy int8 array (see _cpp_eval_fn above for the full rationale),
        so this routes through board_to_input_array_from_numeric instead of
        board_to_input_array, with cache keys built from
        board_arr.tobytes() instead of a tuple-of-tuples of strings."""
        def eval_batch_fn(boards, players):
            n = len(boards)
            probs_out = [None] * n
            values_out = [None] * n
            miss_idx = []
            miss_keys = []
            miss_inputs = []
            for i in range(n):
                key = (boards[i].tobytes(), players[i])
                cached = self._eval_cache.get(key)
                if cached is not None:
                    probs_out[i], values_out[i] = cached
                else:
                    miss_idx.append(i)
                    miss_keys.append(key)
                    miss_inputs.append(board_to_input_array_from_numeric(boards[i], players[i]))

            if miss_inputs:
                if self._pooled_evaluator is not None:
                    probs_batch, values_batch = self._pooled_evaluator.submit(miss_inputs)
                else:
                    batch = boards_to_input_tensor(miss_inputs, device)
                    with torch.inference_mode():
                        policy_logits, value_tensor = model(batch)
                    probs_batch = torch.exp(policy_logits).cpu().numpy().astype(np.float64)
                    if probs_batch.ndim == 1:
                        probs_batch = probs_batch.reshape(1, -1)
                    values_batch = value_tensor.squeeze(-1).cpu().numpy().astype(np.float64).reshape(-1)
                for j, key in enumerate(miss_keys):
                    p = probs_batch[j].copy()
                    v = float(values_batch[j])
                    self._eval_cache[key] = (p, v)
                    idx = miss_idx[j]
                    probs_out[idx] = p
                    values_out[idx] = v

            probs_arr = np.stack(probs_out).astype(np.float64)
            values_arr = np.array(values_out, dtype=np.float64)
            return probs_arr, values_arr
        return eval_batch_fn

    def _search_cpp(self, root_board, current_player, model, device, num_simulations,
                     add_noise, dirichlet_alpha, noise_weight):
        """Delegates the whole search to the persistent C++ engine
        (self._cpp_engine, one MCTSEngine per MCTS object -- see __init__),
        then reconstructs a plain MCTSNode tree (root + direct children only
        -- the only depth every call site actually reads) so every caller
        of MCTS.search() keeps working unchanged.

        Calling search()/search_batched() on the SAME self._cpp_engine every
        ply (rather than a stateless module-level function) is what makes
        tree reuse work: the engine itself decides, from a board diff
        against what it last saw, whether to promote an already-explored
        subtree or start fresh -- transparent to this method and to every
        caller. See MCTSEngine's class docstring in
        cpp_mcts/mcts_core_tree_reuse.cpp for the full design."""
        if self.eval_batch_size > 1:
            eval_batch_fn = self._cpp_eval_batch_fn(model, device)
            target_policy, root_value, candidates = self._cpp_engine.search_batched(
                root_board, current_player, eval_batch_fn, num_simulations, self.c_puct,
                add_noise, dirichlet_alpha, noise_weight, BOARD_SIZE, WIN_LENGTH, self.eval_batch_size
            )
        else:
            eval_fn = self._cpp_eval_fn(model, device)
            target_policy, root_value, candidates = self._cpp_engine.search(
                root_board, current_player, eval_fn, num_simulations, self.c_puct,
                add_noise, dirichlet_alpha, noise_weight, BOARD_SIZE, WIN_LENGTH
            )

        next_player = 'O' if current_player == 'X' else 'X'
        root = MCTSNode(parent=None, prior_prob=1.0, player_turn=current_player)
        root.mean_value = root_value
        root.total_value = root_value
        root.n_visits = 1
        root.is_expanded = True
        for cand in candidates:
            child = MCTSNode(parent=root, prior_prob=0.0, player_turn=next_player)
            child.n_visits = cand['visit_count']
            child.mean_value = cand['q_value']
            child.total_value = cand['q_value'] * cand['visit_count']
            child.is_expanded = True
            root.children[cand['action_idx']] = child
        return root

    def search(self, root_board, current_player, model, device, num_simulations=40, add_noise=False,
               dirichlet_alpha=0.3, noise_weight=0.25):
        if self._owns_cache:
            self._eval_cache.clear()  # Clear cache per move search

        if _mcts_core is not None:
            return self._search_cpp(root_board, current_player, model, device, num_simulations,
                                     add_noise, dirichlet_alpha, noise_weight)

        root = MCTSNode(parent=None, prior_prob=1.0, player_turn=current_player)

        # First expansion for root
        self._expand_node(root, root_board, current_player, model, device)

        if add_noise and len(root.children) > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
            for idx, act in enumerate(actions):
                root.children[act].prior_prob = (1 - noise_weight) * root.children[act].prior_prob + noise_weight * noise[idx]

        if self.eval_batch_size > 1:
            self._search_batched(root, root_board, current_player, model, device, num_simulations)
            return root

        for _ in range(num_simulations):
            node = root
            sim_board = [row[:] for row in root_board]
            sim_player = current_player

            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                best_action = None
                best_score = -math.inf
                sqrt_pv = math.sqrt(node.n_visits)
                for action, child in node.children.items():
                    score = self.get_puct_score(sqrt_pv, child)
                    if score > best_score:
                        best_score = score
                        best_action = action

                r, c = divmod(best_action, BOARD_SIZE)
                sim_board[r][c] = sim_player
                node = node.children[best_action]
                sim_player = 'O' if sim_player == 'X' else 'X'

            # 2. Evaluation & Expansion
            if fast_check_win(sim_board, 'X'):
                leaf_value = 1.0 if sim_player == 'X' else -1.0
            elif fast_check_win(sim_board, 'O'):
                leaf_value = 1.0 if sim_player == 'O' else -1.0
            elif fast_check_draw(sim_board):
                leaf_value = 0.0
            else:
                leaf_value = self._expand_node(node, sim_board, sim_player, model, device)

            # 3. Backpropagation
            curr_node = node
            curr_val = leaf_value
            while curr_node is not None:
                if curr_node.player_turn == sim_player:
                    curr_node.update(curr_val)
                else:
                    curr_node.update(-curr_val)
                curr_node = curr_node.parent

        return root

    def _expand_node(self, node, board, current_player, model, device):
        board_key = (tuple(tuple(r) for r in board), current_player)

        if board_key in self._eval_cache:
            probs, leaf_value = self._eval_cache[board_key]
        else:
            model_input = board_to_input_tensor(board, current_player, device)
            with torch.inference_mode():
                policy_logits, value_tensor = model(model_input)

            probs = torch.exp(policy_logits).squeeze().cpu().numpy()
            leaf_value = value_tensor.item()
            self._eval_cache[board_key] = (probs, leaf_value)

        node.is_expanded = True
        next_player = 'O' if current_player == 'X' else 'X'

        for (r, c) in _relevant_moves(board):
            i = r * BOARD_SIZE + c
            node.children[i] = MCTSNode(parent=node, prior_prob=probs[i], player_turn=next_player)

        return leaf_value

    def _select_best_action(self, node):
        best_action = None
        best_score = -math.inf
        sqrt_pv = math.sqrt(node.n_visits)
        for action, child in node.children.items():
            score = self.get_puct_score(sqrt_pv, child)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expand_with_probs(self, node, board, current_player, probs):
        node.is_expanded = True
        next_player = 'O' if current_player == 'X' else 'X'
        for (r, c) in _relevant_moves(board):
            i = r * BOARD_SIZE + c
            node.children[i] = MCTSNode(parent=node, prior_prob=probs[i], player_turn=next_player)

    @staticmethod
    def _backprop(leaf_node, leaf_value, sim_player):
        curr_node = leaf_node
        curr_val = leaf_value
        while curr_node is not None:
            if curr_node.player_turn == sim_player:
                curr_node.update(curr_val)
            else:
                curr_node.update(-curr_val)
            curr_node = curr_node.parent

    def _search_batched(self, root, root_board, current_player, model, device, num_simulations):
        """
        eval_batch_size>1 path: groups selections via virtual loss so a batch
        of leaves can be sent through the model in one forward call instead
        of one-at-a-time. See __init__ docstring for the search-order caveat.
        """
        VIRTUAL_LOSS = 1.0
        remaining = num_simulations

        while remaining > 0:
            group_size = min(self.eval_batch_size, remaining)
            remaining -= group_size

            # Phase 1: run `group_size` selections, applying virtual loss along
            # each path so successive selections in this group diverge instead
            # of all landing on the same leaf.
            group_entries = []
            for _ in range(group_size):
                node = root
                sim_board = [row[:] for row in root_board]
                sim_player = current_player
                path = [node]

                while node.is_expanded and len(node.children) > 0:
                    best_action = self._select_best_action(node)
                    r, c = divmod(best_action, BOARD_SIZE)
                    sim_board[r][c] = sim_player
                    node = node.children[best_action]
                    sim_player = 'O' if sim_player == 'X' else 'X'
                    path.append(node)
                    node.n_visits += 1
                    node.total_value += -VIRTUAL_LOSS
                    node.mean_value = node.total_value / node.n_visits

                terminal_value = None
                if fast_check_win(sim_board, 'X'):
                    terminal_value = 1.0 if sim_player == 'X' else -1.0
                elif fast_check_win(sim_board, 'O'):
                    terminal_value = 1.0 if sim_player == 'O' else -1.0
                elif fast_check_draw(sim_board):
                    terminal_value = 0.0

                group_entries.append({
                    'node': node, 'board': sim_board, 'player': sim_player,
                    'path': path, 'terminal_value': terminal_value,
                })

            # Phase 2: revert virtual loss for every touched node before
            # applying real backprop values.
            for entry in group_entries:
                for n in entry['path'][1:]:
                    n.n_visits -= 1
                    n.total_value -= -VIRTUAL_LOSS
                    n.mean_value = (n.total_value / n.n_visits) if n.n_visits > 0 else 0.0

            # Phase 3: batched NN evaluation for every non-terminal leaf in
            # this group, deduplicated by board state.
            need_eval = [e for e in group_entries if e['terminal_value'] is None]
            unique_keys = {}
            batch_inputs = []
            batch_meta = []

            for e in need_eval:
                board_key = (tuple(tuple(r) for r in e['board']), e['player'])
                e['board_key'] = board_key
                if board_key in self._eval_cache:
                    continue
                if board_key not in unique_keys:
                    unique_keys[board_key] = len(batch_inputs)
                    batch_inputs.append(board_to_input_tensor(e['board'], e['player'], device))
                    batch_meta.append(board_key)

            if batch_inputs:
                batch_tensor = torch.cat(batch_inputs, dim=0)
                with torch.inference_mode():
                    policy_logits, value_tensor = model(batch_tensor)
                probs_batch = torch.exp(policy_logits).cpu().numpy()
                if probs_batch.ndim == 1:
                    probs_batch = probs_batch.reshape(1, -1)
                values_batch = value_tensor.squeeze(-1).cpu().numpy().reshape(-1)
                for idx, board_key in enumerate(batch_meta):
                    # .copy(): probs_batch[idx] is a view into the shared batch
                    # array: with a long-lived shared_cache, every entry would
                    # otherwise keep the whole batch array (and every other
                    # row in it) alive for as long as the cache exists.
                    self._eval_cache[board_key] = (probs_batch[idx].copy(), float(values_batch[idx]))

            # Phase 4: expand + backprop every entry in the group.
            for e in group_entries:
                node = e['node']
                sim_player = e['player']
                if e['terminal_value'] is not None:
                    self._backprop(node, e['terminal_value'], sim_player)
                    continue
                probs, leaf_value = self._eval_cache[e['board_key']]
                if not node.is_expanded:
                    self._expand_with_probs(node, e['board'], sim_player, probs)
                self._backprop(node, leaf_value, sim_player)


# ============================================================
# Global Model Caching & Public API
# ============================================================

_current_mcts_model = None
_current_mcts_device = None
_current_mcts_model_path = None
_mcts_engine = MCTS(c_puct=1.414)


def get_mcts_model(model_name="alphazero_mcts.pth"):
    global _current_mcts_model, _current_mcts_device, _current_mcts_model_path

    if _current_mcts_model is None or _current_mcts_device is None:
        _current_mcts_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _current_mcts_model = AlphaZeroTicTacToe()
        _current_mcts_model.to(_current_mcts_device)

    model_path = os.path.join('model', model_name)
    if _current_mcts_model_path != model_path and os.path.exists(model_path):
        try:
            _current_mcts_model.load_state_dict(torch.load(model_path, map_location=_current_mcts_device))
            _current_mcts_model.eval()
            _current_mcts_model_path = model_path
        except Exception as e:
            print(f"MCTS Model Notice: Initialization ({e})")

    return _current_mcts_model, _current_mcts_device


def play_mcts(board, turn, model_name="alphazero_mcts.pth", num_simulations=30, add_noise=False):
    """
    Public MCTS Inference API.
    Executes MCTS tree search and returns (chosen_move, confidence_val, candidates_list, target_policy_vec).
    """
    model, device = get_mcts_model(model_name)
    root = _mcts_engine.search(board, turn, model, device, num_simulations=num_simulations, add_noise=add_noise)

    if not root.children:
        return None, 0.0, [], np.zeros(NUM_CELLS, dtype=np.float32)

    target_policy_vec = np.zeros(NUM_CELLS, dtype=np.float32)
    candidates = []

    total_visits = sum(child.n_visits for child in root.children.values())

    for act, child in root.children.items():
        r, c = divmod(act, BOARD_SIZE)
        prob = child.n_visits / total_visits if total_visits > 0 else 0.0
        target_policy_vec[act] = prob
        candidates.append({
            'move': [r, c],
            'action_idx': act,
            'visit_count': child.n_visits,
            'prob': round(float(prob), 4),
            'q_value': round(float(child.value()), 4)
        })

    candidates.sort(key=lambda x: x['visit_count'], reverse=True)

    best_cand = candidates[0]
    best_move = (best_cand['move'][0], best_cand['move'][1])
    root_confidence = round(float(root.value()), 4)

    return best_move, root_confidence, candidates, target_policy_vec
