import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import random
import os

import alphago_lee
from alphago_lee import (
    BOARD_SIZE, WIN_LENGTH, NUM_CELLS, INPUT_SIZE,
    AlphaZeroTicTacToe, board_to_input_tensor, check_win, check_draw
)
import determinisic_play

# Keep 3x3 and other board sizes writing to separate checkpoint/log files so
# concurrent runs (e.g. a 3x3 job still training while a 5x5 job starts)
# never clobber each other's state on disk. 3x3 keeps its original unsuffixed
# paths for continuity with earlier runs.
_SUFFIX = "" if BOARD_SIZE == 3 else f"_{BOARD_SIZE}x{BOARD_SIZE}"

# ============================================================
# Report Logging (for research write-up: loss / entropy / win-rate vs. Deterministic)
# ============================================================
REPORT_LOG_PATH = f"model/lee_training_report{_SUFFIX}.txt"


def report_logger(msg):
    os.makedirs(os.path.dirname(REPORT_LOG_PATH), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(REPORT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


# ============================================================
# Model / Device Setup
# ============================================================
# Same split as alphazero_mcts_make.py: self-play here does one single-sample
# (batch=1) forward pass per move (no tree search), and single-sample calls
# are faster on CPU than on this machine's GPU due to CUDA kernel-launch
# overhead outweighing the tiny model's compute. Training is batched
# forward+backward, which is faster on GPU. So the model moves between two
# devices at each phase boundary instead of living on one.

def get_model():
    return AlphaZeroTicTacToe()


def get_devices():
    self_play_device = torch.device("cpu")
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Self-play device: {self_play_device} | Train device: {train_device}")
    return self_play_device, train_device


# ============================================================
# Board Data Augmentation (Rotations & Flips)
# ============================================================

def augment_sample(board_tensor_3d, policy_2d):
    """
    Applies 8 symmetries (4 rotations x 2 flips) to a 3D board tensor
    (channels, BOARD_SIZE, BOARD_SIZE) and 2D target policy map
    (BOARD_SIZE, BOARD_SIZE). Channel-count-agnostic: rotation/flip only
    ever touch axes 1 and 2 (the spatial dims), so this keeps working
    unchanged regardless of how many planes board_tensor_3d has (e.g. if
    board_to_input_array's channel count changes).
    """
    augmented = []
    for k in range(4):
        # Rotate 90 * k degrees
        rot_b = np.rot90(board_tensor_3d, k, axes=(1, 2))
        rot_p = np.rot90(policy_2d, k, axes=(0, 1))

        # Original rotation
        augmented.append((rot_b.copy(), rot_p.flatten().copy()))

        # Horizontally flipped rotation
        flip_b = np.flip(rot_b, axis=2)
        flip_p = np.flip(rot_p, axis=1)
        augmented.append((flip_b.copy(), flip_p.flatten().copy()))

    return augmented


# ============================================================
# Fast Self-Play Data Generation
# ============================================================

def _play_one_lee_game(model, device, use_mcts_self_play):
    """
    Plays a single self-play game to completion and returns (history, outcome).
    Model is only read from (eval mode, no_grad, no state mutated), so this is
    safe to call concurrently from multiple worker threads.
    """
    current_board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = random.choice(['X', 'O'])
    history = []
    outcome = 0

    while True:
        if check_win(current_board, 'X'):
            outcome = 1
            break
        if check_win(current_board, 'O'):
            outcome = -1
            break
        if check_draw(current_board):
            outcome = 0
            break

        board_snapshot = [row[:] for row in current_board]
        empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if current_board[r][c] == ' ']
        if not empty_spots:
            break

        target_policy_map = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        if use_mcts_self_play or current_player == 'X':
            # Neural Network Move with Softmax / Dirichlet exploration
            model_input = board_to_input_tensor(current_board, current_player, device)
            with torch.no_grad():
                policy_logits, _ = model(model_input)

            move_probs = torch.exp(policy_logits).squeeze().cpu().numpy()

            valid_indices = [r * BOARD_SIZE + c for (r, c) in empty_spots]
            valid_probs = np.array([move_probs[idx] for idx in valid_indices])
            prob_sum = valid_probs.sum()
            if prob_sum > 0:
                valid_probs /= prob_sum
            else:
                valid_probs = np.ones(len(valid_indices)) / len(valid_indices)

            # Add noise for exploration during training
            if random.random() < 0.2:
                chosen_idx = random.choice(valid_indices)
            else:
                chosen_idx = np.random.choice(valid_indices, p=valid_probs)

            r, c = divmod(chosen_idx, BOARD_SIZE)
            target_policy_map[r, c] = 1.0
        else:
            # Fast Deterministic Heuristic AI
            r, c = determinisic_play.play(current_board, current_player)
            if r == -1:
                break
            target_policy_map[r, c] = 1.0

        # Store current board state as network input planes via the shared
        # canonical builder (alphago_lee.board_to_input_array) instead of
        # building the planes inline here, so this file automatically
        # tracks alphago_lee.py's input format (e.g. channel count) with no
        # separate edit needed if that builder changes.
        board_3d = alphago_lee.board_to_input_array(board_snapshot, current_player)

        history.append((board_3d, current_player, target_policy_map))
        current_board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'

    return history, outcome


def generate_self_play_data(model, device, num_games=20, use_mcts_self_play=True, max_workers=None):
    """
    Generates self-play training games. Games are played concurrently across
    a thread pool -- each neural-net forward pass releases the GIL inside
    PyTorch's C++ kernels, so multiple games really do make progress on
    different CPU cores at once (same reasoning as alphazero_mcts_make.py).
    """
    max_workers = max_workers or max(1, min(num_games, (os.cpu_count() or 4) - 2))
    print(f"Generating {num_games} self-play games (workers: {max_workers})...")
    raw_samples = []
    outcome_counts = {'x_win': 0, 'o_win': 0, 'draw': 0}

    model.eval()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_play_one_lee_game, model, device, use_mcts_self_play) for _ in range(num_games)]
        for future in futures:
            history, outcome = future.result()
            if outcome == 1:
                outcome_counts['x_win'] += 1
            elif outcome == -1:
                outcome_counts['o_win'] += 1
            else:
                outcome_counts['draw'] += 1

            for (board_3d, player_turn, target_policy_map) in history:
                final_val = outcome if player_turn == 'X' else -outcome
                aug_samples = augment_sample(board_3d, target_policy_map)
                for (aug_b, aug_p) in aug_samples:
                    raw_samples.append((aug_b, aug_p, final_val))

    print(f"Generated {len(raw_samples)} augmented training samples from {num_games} games. "
          f"Outcomes: X {outcome_counts['x_win']} / O {outcome_counts['o_win']} / Draw {outcome_counts['draw']}")
    return raw_samples, outcome_counts


# ============================================================
# Evaluation vs. Deterministic (direct metric for the research goal)
# ============================================================

def _lee_pick_move(model, device, board, turn):
    """Greedy (no exploration) move choice from the policy head, for evaluation."""
    empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == ' ']
    if not empty_spots:
        return -1, -1
    model_input = board_to_input_tensor(board, turn, device)
    with torch.no_grad():
        policy_logits, _ = model(model_input)
    move_probs = torch.exp(policy_logits).squeeze().cpu().numpy()
    valid_indices = [r * BOARD_SIZE + c for (r, c) in empty_spots]
    best_idx = max(valid_indices, key=lambda idx: move_probs[idx])
    return divmod(best_idx, BOARD_SIZE)


def _play_vs_deterministic(model, device, lee_plays_x):
    lee_player = 'X' if lee_plays_x else 'O'
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 'X'

    while True:
        if check_win(board, lee_player):
            return 'lee'
        if check_win(board, 'O' if lee_player == 'X' else 'X'):
            return 'det'
        if check_draw(board):
            return 'draw'

        if current_player == lee_player:
            r, c = _lee_pick_move(model, device, board, current_player)
        else:
            r, c = determinisic_play.play(board, current_player)
        if r == -1:
            return 'draw'

        board[r][c] = current_player
        current_player = 'O' if current_player == 'X' else 'X'


def evaluate_vs_deterministic(model, device, num_games=20):
    """
    Plays num_games against determinisic_play (minimax/alpha-beta), alternating
    who moves first, and returns (win_rate, draw_rate, loss_rate) from the
    Lee policy net's perspective.
    """
    model.eval()
    wins = draws = losses = 0
    for i in range(num_games):
        lee_plays_x = (i % 2 == 0)
        result = _play_vs_deterministic(model, device, lee_plays_x)
        if result == 'lee':
            wins += 1
        elif result == 'draw':
            draws += 1
        else:
            losses += 1
    n = max(1, num_games)
    return wins / n, draws / n, losses / n


# ============================================================
# Logger
# ============================================================

def logger(message):
    with open("training_log.txt", "a") as log_file:
        log_file.write(message + "\n")


# ============================================================
# Fast PyTorch Mini-Batch Training
# ============================================================

def train_alpha_zero_model(
    model,
    num_epochs=5,
    learning_rate=0.001,
    num_self_play_games=30,
    batch_size=64,
    self_play_workers=None,
    eval_interval=10,
    eval_games=20,
):
    self_play_device, train_device = get_devices()
    self_play_workers = self_play_workers or max(1, (os.cpu_count() or 4) - 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()

    checkpoint_path = f"model/alphago_lee{_SUFFIX}.pth"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"Loaded existing Lee checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Checkpoint load notice ({e})")
    elif _SUFFIX == "" and os.path.exists("model/model_ep70.pth"):
        try:
            model.load_state_dict(torch.load("model/model_ep70.pth", map_location="cpu"))
            print("Seeded from model/model_ep70.pth (latest prior Lee checkpoint)")
        except Exception as e:
            print(f"Seed checkpoint load notice ({e})")

    print(f"\n--- Starting AlphaZero Model Training ({num_epochs} Epochs, Batch Size {batch_size}) ---")

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            save_model(model, f"model/model_ep{epoch}{_SUFFIX}.pth")

        # Self-play phase: CPU, one intra-op thread per call so the
        # ThreadPoolExecutor workers supply the parallelism instead.
        model.to(self_play_device)
        torch.set_num_threads(1)
        raw_samples, outcome_counts = generate_self_play_data(
            model, self_play_device, num_games=num_self_play_games, max_workers=self_play_workers
        )
        if not raw_samples:
            continue

        inputs_np = np.array([s[0] for s in raw_samples], dtype=np.float32)
        policies_np = np.array([s[1] for s in raw_samples], dtype=np.float32)
        values_np = np.array([[s[2]] for s in raw_samples], dtype=np.float32)

        dataset = TensorDataset(
            torch.from_numpy(inputs_np),
            torch.from_numpy(policies_np),
            torch.from_numpy(values_np)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training phase: GPU (batched forward+backward is faster there).
        model.to(train_device)
        if train_device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for b_inputs, b_policies, b_values in loader:
            b_inputs = b_inputs.to(train_device)
            b_policies = b_policies.to(train_device)
            b_values = b_values.to(train_device)

            optimizer.zero_grad()
            pred_policy_logits, pred_value = model(b_inputs)

            policy_loss = policy_criterion(pred_policy_logits, b_policies)
            value_loss = value_criterion(pred_value, b_values)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            # Target policy is one-hot (entropy 0 by construction), so the
            # informative entropy here is of the *predicted* distribution --
            # it tracks how sharp/confident the policy head has become.
            with torch.no_grad():
                pred_probs = torch.exp(pred_policy_logits)
                batch_entropy = -(pred_probs * pred_policy_logits).sum(dim=1).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += batch_entropy
            num_batches += 1

        avg_policy_loss = total_policy_loss / max(1, num_batches)
        avg_value_loss = total_value_loss / max(1, num_batches)
        avg_entropy = total_entropy / max(1, num_batches)

        log_msg = f"Epoch {epoch + 1}: Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}"
        logger(log_msg)
        print(f"  {log_msg}")

        report_msg = (f"Epoch {epoch + 1}/{num_epochs} | Policy Loss: {avg_policy_loss:.4f} | "
                      f"Value Loss: {avg_value_loss:.4f} | Predicted-Policy Entropy: {avg_entropy:.4f} | "
                      f"Self-Play (this epoch, {num_self_play_games} games): "
                      f"X-win {outcome_counts['x_win']} / O-win {outcome_counts['o_win']} / "
                      f"Draw {outcome_counts['draw']}")
        line = report_logger(report_msg)
        print(f"  {line}")

        # Save after every trained epoch so a crash or early stop never loses
        # more than one epoch's worth of training.
        torch.save(model.state_dict(), checkpoint_path)

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.to(self_play_device)
            torch.set_num_threads(1)
            win_rate, draw_rate, loss_rate = evaluate_vs_deterministic(
                model, self_play_device, num_games=eval_games
            )
            eval_msg = (f"Epoch {epoch + 1}/{num_epochs} | Eval vs Deterministic "
                        f"({eval_games} games): Win {win_rate:.0%} | Draw {draw_rate:.0%} | Loss {loss_rate:.0%}")
            line = report_logger(eval_msg)
            print(f"  {line}")

    torch.save(model.state_dict(), checkpoint_path)
    save_model(model, f"model/model_ep{num_epochs}_final{_SUFFIX}.pth")
    print("--- Training complete ---")


# ============================================================
# Model Checkpointing
# ============================================================

def save_model(model, path="model/model_ep2.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="model/model_ep2.pth"):
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            print(f"Loaded model checkpoint from {path}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Starting fresh.")
    else:
        print(f"No checkpoint found at {path}. Starting fresh.")


# ============================================================
# Main Execution Entrypoint
# ============================================================

if __name__ == '__main__':
    model = get_model()

    print(f"Board Size : {BOARD_SIZE} x {BOARD_SIZE}")
    print(f"Win Length : {WIN_LENGTH}")

    train_alpha_zero_model(
        model,
        num_epochs=100,
        learning_rate=0.001,
        num_self_play_games=30,
        batch_size=64,
    )
