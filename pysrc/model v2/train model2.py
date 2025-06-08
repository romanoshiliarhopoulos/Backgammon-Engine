#!/usr/bin/env python3
"""
Full training loop for Backgammon self-play with MCTS and PyTorch.
Assumes you have:
  - backgammon_env (C++ bindings via pybind11)
  - model2.py with SeqBackgammonNet
  - encode_state.py with encode_state, build_sequence_mask
"""
import os
import sys

# ensure C++ extension, script dir, and pysrc bindings 
root = os.path.dirname(__file__)
parent = os.path.abspath(os.path.join(root, '..', '..'))
sys.path[:0] = [
    root,
    os.path.join(parent, 'build'),
    os.path.join(parent, 'pysrc')
]

import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from backgammon_env import Game, Player, PlayerType
from model2 import SeqBackgammonNet
from encode_state import encode_state, build_sequence_mask

# ==== Hyperparameters =====
NUM_ITERATIONS     = 100
NUM_SELFPLAY_GAMES = 10
MCTS_SIMS          = 0   # dummy, not used 
BATCH_SIZE         = 64
EPOCHS_PER_ITER    = 1
LR                 = 3e-4
WEIGHT_POLICY      = 1.0
WEIGHT_VALUE       = 1.0
WEIGHT_DECAY       = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== MCTS fallback: use network prior directly ==== 
def run_mcts_and_get_root_pi(model, game, player, state_tensor, mask_tensor, sims, device):
    """
    Fallback policy target: just use a single network forward pass.
    Returns the softmax over masked logits as the "visit-count" distribution.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(state_tensor.to(device), masks=mask_tensor.to(device))
        big_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask_tensor.to(device), big_neg)
        pi = F.softmax(masked, dim=-1)
    return pi.cpu()

# ==== Utility: Training on one mini-batch ====
def train_on_batch(model, optimizer, states, masks, pis, zs):
    model.train()
    optimizer.zero_grad()
    logits, raw_values = model(states.to(DEVICE), masks=masks.to(DEVICE))
    pred_values = torch.tanh(raw_values)
    big_neg = torch.finfo(logits.dtype).min
    illegal = ~masks.to(DEVICE)
    masked_logits = logits.masked_fill(illegal, big_neg)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -torch.sum(pis.to(DEVICE) * log_probs) / pis.size(0)
    value_loss  = F.mse_loss(pred_values, zs.to(DEVICE))
    loss = WEIGHT_POLICY * policy_loss + WEIGHT_VALUE * value_loss
    loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item(), loss.item()

# ==== Self-play generation (network-based policy) ====
def self_play_game(model):
    # Initialize game and players
    print("game")
    game = Game(0)
    p1 = Player("P1", PlayerType.PLAYER1)
    p2 = Player("P2", PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    game.roll_dice()

    trace = []
    while True:
        is_over, winner = game.is_game_over()
        if is_over:
            break

        current_idx = game.getTurn()  # 0 or 1
        cur_player = p1 if current_idx == 0 else p2
        board  = game.getGameBoard()
        jailed = game.getJailedCount(current_idx)
        offcnt = game.getBornOffCount(current_idx)
        die1, die2 = game.get_last_dice()

        # Encode state planes
        base_planes = encode_state(board, jailed, offcnt, current_idx)
        die1_p = torch.full((1,24), float(die1)/6.0)
        die2_p = torch.full((1,24), float(die2)/6.0)
        turn_p = torch.full((1,24), float(current_idx==0))
        planes = torch.cat([base_planes, die1_p, die2_p, turn_p], dim=0)
        state = planes.unsqueeze(0)

        # Build legal-move mask (pass Player instance)
        mask, seqs, orders, all_t, all_flat, valid = build_sequence_mask(
            game, cur_player, batch_size=1, device='cpu', max_steps=4
        )

        # Get policy target
        pi = run_mcts_and_get_root_pi(model, game, current_idx, state, mask, MCTS_SIMS, DEVICE)
        trace.append((state.squeeze(0), mask.squeeze(0), pi.squeeze(0), current_idx))

                # Pick best action using only the first sub-step distribution
        # pi has shape [1, max_steps, N]; take step=0
        first_step_probs = pi.squeeze(0)[0]  # shape [N]
        # Mask illegal for step0
        step0_mask = mask.squeeze(0)[0]      # shape [N]
        big_neg = torch.finfo(first_step_probs.dtype).min
        masked_probs = first_step_probs.masked_fill(~step0_mask, big_neg)
        idx = torch.argmax(masked_probs).item()
        origin = idx // 26
        dest   = idx % 26

                # Execute move using the first die always as fallback
        game.tryMove(cur_player, die1, origin, dest)

        # Next turn
        game.roll_dice()
        game.setTurn(1 - current_idx)

    # Build training examples
    examples = []
    for s,m,pi,p in trace:
        z = 1.0 if p == winner else -1.0
        examples.append((s,m,pi,z))
    return examples

# ==== Main training loop with stats ====
def main():
    model = SeqBackgammonNet(n_channels=9, hidden_dim=128, max_steps=4, num_res_blocks=10).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    policy_hist, value_hist, total_hist = [], [], []

    for it in range(1, NUM_ITERATIONS+1):
        print("romans")
        data = []
        for _ in range(NUM_SELFPLAY_GAMES):
            data.extend(self_play_game(model))

        random.shuffle(data)
        states = torch.stack([d[0] for d in data])
        masks  = torch.stack([d[1] for d in data])
        pis    = torch.stack([d[2] for d in data])
        zs     = torch.tensor([d[3] for d in data], dtype=torch.float32)

        ds = DataLoader(TensorDataset(states, masks, pis, zs), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        sum_p = sum_v = sum_t = 0.0
        cnt = 0
        for st, mk, pi, z in ds:
            p_l, v_l, t_l = train_on_batch(model, optimizer, st, mk, pi, z)
            sum_p += p_l; sum_v += v_l; sum_t += t_l; cnt += 1

        avg_p = sum_p / cnt
        avg_v = sum_v / cnt
        avg_t = sum_t / cnt
        policy_hist.append(avg_p)
        value_hist.append(avg_v)
        total_hist.append(avg_t)

        print(f"Iter {it}: Policy={avg_p:.4f}, Value={avg_v:.4f}, Total={avg_t:.4f}")
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/model_{it}.pth')

    # Loss summary
    print("\nLoss summary:")
    print("Iter | Policy | Value | Total")
    print("-----------------------------")
    for i, (p, v, t) in enumerate(zip(policy_hist, value_hist, total_hist), 1):
        print(f"{i:4d} | {p:6.4f} | {v:6.4f} | {t:6.4f}")

if __name__=='__main__':
    main()
