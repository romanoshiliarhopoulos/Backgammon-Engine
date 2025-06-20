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
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        self.bn_input = nn.BatchNorm1d(64)

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
        out = self.bn_input(out)
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
            logits = logits.masked_fill(~masks, float('-inf'))

        # Value head (output raw scalar)
        values = self.value_head(feat).squeeze(-1)  # [batch_size]

        return logits, values


# Example usage sanity check. 
if __name__ == "__main__":
    dummy_x = torch.randn(2, 9, 24)

    dummy_masks = torch.ones(2, 4, 26 * 26, dtype=torch.bool)

    model = SeqBackgammonNet(n_channels=9, hidden_dim=128, max_steps=4, num_res_blocks=10)
    logits, values = model(dummy_x, masks=dummy_masks)

    print("Logits shape:", logits.shape)  # Expect [2, 4, 676]
    print("Values shape:", values.shape)  # Expect [2]
