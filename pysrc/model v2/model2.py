#!/usr/bin/env python3
import random
import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure C++ extension is on path (if needed)
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

from encode_state import encode_state
from encode_state import build_sequence_mask
from torch.distributions import Categorical


class ResidualBlock1D(nn.Module):
    """
    A 1D residual block with two convolutional layers.
    Each conv layer uses kernel_size=3, padding=1, and 64 channels.
    The block applies ReLU -> Conv1d -> ReLU -> Conv1d, with a skip connection.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class SeqBackgammonNet(nn.Module):
    """
    Dual-head neural network for Backgammon:
    - Convolutional front end with 10 residual blocks (64 filters each).
    - Policy head: outputs per-step logits for up to `max_steps` sub-moves.
    - Value head: outputs a single scalar estimating board evaluation.

    Input tensor `x` is expected to have shape [batch_size, n_channels, 24],
    where n_channels = 9 (board, bar, borne-off, dice, to-move flags).

    Policy output has shape [batch_size, max_steps, N], where
    N = S * S and S = 26 (origin indices 0–25 and destination indices 0–25).
    For each sub-step, the network predicts logits over 26×26 possible origin-destination
    pairs. The ordering allows for up to 4 sub-steps in the case of doubles.
    """

    def __init__(self,
                 n_channels: int = 9,
                 hidden_dim: int = 128,
                 max_steps: int = 4,
                 num_res_blocks: int = 10):
        super().__init__()
        # Initial convolution to map input channels to 64 feature maps
        self.conv_input = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1, bias=False)
        self.norm_input = nn.GroupNorm(1, 64)

        # Create `num_res_blocks` residual blocks, each preserving 64 channels
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(channels=64) for _ in range(num_res_blocks)
        ])

        # After convolutions, flatten and project to hidden dimension
        # Input width is 24 (points on the board)
        conv_output_size = 64 * 24
        self.fc_feat = nn.Linear(conv_output_size, hidden_dim)

        # Policy head parameters
        self.max_steps = max_steps
        self.S = 26
        self.N = self.S * self.S  # 26 origins × 26 destinations = 676

        # For each sub-step, predict logits over N possibilities
        self.policy_logits = nn.Linear(hidden_dim, max_steps * self.N)

        # Value head: raw scalar output (apply tanh externally if desired)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self,
                x: torch.Tensor,
                masks: torch.Tensor = None
                ) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass.

        Args:
            x: Tensor of shape [batch_size, n_channels, 24]
            masks: Boolean tensor of shape [batch_size, max_steps, N] indicating
                   which logits are valid (True) vs. invalid (False). Invalid logits
                   will be set to -inf before softmax.

        Returns:
            logits: Tensor of shape [batch_size, max_steps, N]
            values: Tensor of shape [batch_size], raw scalar predictions
        """
        bsz = x.size(0)

        # Initial conv + BN + ReLU
        out = self.conv_input(x)       # [batch_size, 64, 24]
        out = self.norm_input(self.conv_input(x))
        out = F.relu(out)

        # Pass through residual blocks
        for block in self.res_blocks:
            out = block(out)          # still [batch_size, 64, 24]

        # Flatten and project to hidden_dim
        out = out.view(bsz, -1)        # [batch_size, 64*24]
        feat = F.relu(self.fc_feat(out))  # [batch_size, hidden_dim]

        # Policy head
        logits = self.policy_logits(feat)  # [batch_size, max_steps * N]
        logits = logits.view(bsz, self.max_steps, self.N)  # [batch_size, 4, 676]
        if masks is not None:
            # masks: expected shape [batch_size, max_steps, N]; True for valid, False for invalid
            NEG_INF = -1e9
            logits = logits.masked_fill(~masks, NEG_INF)

        # Value head (output raw scalar)
        values = self.value_head(feat).squeeze(-1)  # [batch_size]

        return logits, values
    
    def select_action(self, game, player, temperature=1.0):
        """Select action using the neural network with proper move validation"""
        # Get dice values
        dice1, dice2 = game.get_last_dice()
        
        # Encode current state
        board = game.getGameBoard()
        pieces = game.getPieces()
        jailed = pieces.numJailed(player.getNum())
        borne_off = pieces.numFreed(player.getNum())
        turn = game.getTurn()
        
        # Get opponent info for proper encoding
        opponent_num = 1 - player.getNum()
        opp_jailed = pieces.numJailed(opponent_num)
        opp_borne_off = pieces.numFreed(opponent_num)
        
        state = encode_state(board, pieces, turn, dice1, dice2).unsqueeze(0).to("cpu")
        
        # Get legal moves and build mask
        mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
            game, player, batch_size=1, device="cpu"
        )
        
        if not seqs:  # No legal moves
            return None, None, None, None, None, None, None, None
        
        # Forward pass through network
        logits, value = self.forward(state, mask)

        _S = 26
        seq_logits = []
        """
        seq_logits[i] is the sum of the network's scores across every individual 
        move in sequence i. The network's confidence in an entire sequence is the 
        sum of its confidences in each step of that sequence.

        """
        seq_log_probs = []
        for seq in seqs:
            log_prob_sum = torch.tensor(0.0, device=logits.device)  # Start with a tensor, not float
            for t, (origin, dest) in enumerate(seq):
                pos = origin * _S + dest
                step_logits = logits[0, t, :]  # All logits for this step
                step_probs = F.softmax(step_logits, dim=0)
                log_prob_sum = log_prob_sum + torch.log(step_probs[pos] + 1e-8)
            seq_log_probs.append(log_prob_sum)

        seq_log_probs = torch.stack(seq_log_probs)
        probs = F.softmax(seq_log_probs / temperature, dim=0)


        m = Categorical(probs)           # distribution over sequences
        idx_t    = m.sample()            # tensor([i])
        log_prob = m.log_prob(idx_t)     # scalar tensor: log π(a|s)
        entropy  = m.entropy()           # scalar tensor: entropy bonus

        idx = idx_t.item()
        selected_sequence = seqs[idx]
        dice_order       = dice_orders[idx]

        # now return both log_prob & entropy
        mask = mask.to(torch.bool)
        return selected_sequence, dice_order, log_prob, entropy, mask.squeeze(0), seqs, dice_orders, idx_t



# Example usage sanity check. 
if __name__ == "__main__":
    dummy_x = torch.randn(2, 9, 24)

    dummy_masks = torch.ones(2, 4, 26 * 26, dtype=torch.bool)

    model = SeqBackgammonNet(n_channels=9, hidden_dim=128, max_steps=4, num_res_blocks=10)
    logits, values = model(dummy_x, masks=dummy_masks)

    print("Logits shape:", logits.shape)  # Expect [2, 4, 676]
    print("Values shape:", values.shape)  # Expect [2]
