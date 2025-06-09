#!/usr/bin/env python3
"""
Self-play training loop for SeqBackgammonNet using the C++ Backgammon environment.
"""
import random
import torch
import os
import sys
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

# Ensure C++ extension is on path
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

import backgammon_env as bg
from encode_state import encode_state, build_sequence_mask
from model2 import SeqBackgammonNet 

# Hyperparameters
gamma = 0.99
learning_rate = 1e-4
episodes_per_update = 10
max_steps_per_episode = 500

# Initialize network and optimizer
device = torch.device("mps" if torch.mps.is_available() else "cpu")
net = SeqBackgammonNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


class Memory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()


def select_action(state_tensor, mask_tensor, memory):
    logits, value = net(state_tensor, masks=mask_tensor)

    # Flatten policy head for current step 0
    logits = logits[:, 0, :]  # assume single-step sampling; for doubles repeat
    log_probs = F.log_softmax(logits, dim=-1)
    dist = Categorical(logits=log_probs)
    action = dist.sample()

    memory.log_probs.append(dist.log_prob(action))
    memory.values.append(value)
    return action.item()


def play_one_episode():
    memory = Memory()
    # Initialize game and players
    game = bg.Game(0) 
    p1 = bg.Player("p1", bg.PlayerType.PLAYER1)
    p2 = bg.Player("p2", bg.PlayerType.PLAYER2)
    game.setPlayers([p1, p2])
    game.reset()

    for step in range(max_steps_per_episode):
        # Encode state
        board       = game.getGameBoard()            # list of ints length 24
        jailed      = game.getJailedCount()          # number of checkers on the bar
        borne_off   = game.getBornOffCount()         # number of checkers borne off
        turn        = game.getTurn()                 # current player index
        # UPDATED: new encode_state signature
        state_tensor = encode_state(board, jailed, borne_off, turn)    # [C,24]
        state_tensor = state_tensor.unsqueeze(0).to(device)            # [1,C,24]

        # Roll dice and build mask
        dice = game.roll_dice()   # e.g. [die1, die2]
         # new build_sequence_mask returns 6 things; we only need the mask
        mask_on_device, seqs, dice_orders, all_t, all_flat, valid_mask = \
            build_sequence_mask(game, curr_player=turn, batch_size=1, device=device, max_steps=4)
        mask_tensor = mask_on_device   # already shape [1, max_steps, 26*26]

        # Select and apply action
        action_idx = select_action(state_tensor, mask_tensor, memory)
        origin = action_idx // 26
        dest = action_idx % 26
        success, err = game.tryMove(p1, dice, origin, dest)
        if not success:
            # invalid move: penalize heavily and end episode
            memory.rewards.append(-1.0)
            break

        # Check game over
        over, winner = game.is_game_over()
        if over:
            reward = 1.0 if winner == p1.getNum() else -1.0
            memory.rewards.append(reward)
            break

        # intermediate reward = 0
        memory.rewards.append(0.0)

        # swap players to self-play both sides
        game.setTurn(game.getTurn() % 2 + 1)
        p1, p2 = p2, p1

    # At end, if not ended by win/loss, assign zero reward
    if len(memory.rewards) < len(memory.log_probs):
        memory.rewards.extend([0.0] * (len(memory.log_probs) - len(memory.rewards)))

    return memory


def update_parameters(memory):
    # Compute discounted returns
    R = 0
    returns = []
    for r in reversed(memory.rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    values = torch.cat(memory.values)
    log_probs = torch.stack(memory.log_probs)

    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Actor-Critic loss
    advantage = returns - values.squeeze(-1)
    policy_loss = -(log_probs * advantage.detach()).mean()
    value_loss = advantage.pow(2).mean()
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train(num_iterations=1000):
    for iteration in range(1, num_iterations+1):
        total_loss = 0.0
        for _ in range(episodes_per_update):
            mem = play_one_episode()
            total_loss += update_parameters(mem)

        avg_loss = total_loss / episodes_per_update
        print(f"Iteration {iteration:>4d} | Avg Loss: {avg_loss:.4f}")

        # Optional: save checkpoint
        if iteration % 50 == 0:
            torch.save(net.state_dict(), f"checkpoint_{iteration}.pth")


if __name__ == "__main__":
    train(500)
