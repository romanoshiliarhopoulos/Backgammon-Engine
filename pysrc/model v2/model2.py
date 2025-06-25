#!/usr/bin/env python3
#model2.py
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

def init_weights(m):
    """
    Conservative weight initialization to prevent -inf logits
    """
    if isinstance(m, nn.Linear):
        # Use Xavier/Glorot initialization instead of Kaiming
        nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain to start conservative
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)  # Small positive bias
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

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
                 hidden_dim: int = 128, # Adjusted from 64 to 128 for more capacity based on common practice
                 max_steps: int = 4,
                 num_res_blocks: int = 10):
        super().__init__()
        
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.num_res_blocks = num_res_blocks
        
        self.S = 26 # Number of possible points (including bar/borne-off for indexing)
        self.output_dim_per_step = self.S * self.S # 26 * 26 = 676 possible (origin, dest) pairs

        # --- Convolutional Front End (Common Encoder) ---
        # Initial convolutional layer to map input channels to hidden_dim
        self.initial_conv = nn.Conv1d(n_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.initial_norm = nn.GroupNorm(1, hidden_dim) # Normalize after initial conv

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Global average pooling to flatten spatial dimensions for heads
        # This will transform [batch_size, hidden_dim, 24] to [batch_size, hidden_dim]
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)


        # --- Policy Head ---
        # Takes the flattened feature vector from the encoder
        # Outputs logits for each possible (origin, dest) pair for each sub-step
        # Policy head needs to output `max_steps * output_dim_per_step` raw values
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_steps * self.output_dim_per_step)
        )

        # --- Value Head ---
        # Takes the flattened feature vector from the encoder
        # Outputs a single scalar value prediction
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),      # Added LayerNorm here
            nn.LeakyReLU(0.01), # Changed from ReLU to LeakyReLU
            nn.Linear(hidden_dim, 1) # Output a single value
            # NO ACTIVATION HERE for value head (e.g., Sigmoid) unless you're explicitly normalizing target returns
        )

        # Apply custom weight initialization to all layers in the network
        self.apply(init_weights)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass through the network.
        x: Input state tensor [batch_size, n_channels, 24]
        mask: Mask tensor [batch_size, max_steps, N] for legal moves
        """
        batch_size = x.size(0)

        # --- Convolutional Front End ---
        # Initial convolution and activation
        x = F.relu(self.initial_norm(self.initial_conv(x)))

        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x) # x shape remains [batch_size, hidden_dim, 24]

        # Global average pooling to get a fixed-size feature vector per batch item
        # pooled_features shape: [batch_size, hidden_dim, 1] -> squeeze to [batch_size, hidden_dim]
        pooled_features = self.global_avg_pool(x).squeeze(-1)


        # --- Policy Head ---
        # Pass pooled features through the policy head to get raw logits
        # raw_policy_logits shape: [batch_size, max_steps * output_dim_per_step]
        raw_policy_logits = self.policy_head(pooled_features)
        
        # Reshape policy logits to [batch_size, max_steps, output_dim_per_step]
        logits = raw_policy_logits.view(batch_size, self.max_steps, self.output_dim_per_step)
        
        # Apply the mask to logits: set illegal moves to a very low value (-inf effectively)
        # This is CRITICAL. The mask ensures that illegal moves have negligible probability.
        # Ensure mask is broadcastable or of correct shape.
        # It's better if mask matches logits shape: [batch_size, max_steps, output_dim_per_step]
        # Or if mask is [batch_size, N] for each step, repeat it for max_steps.
        # Assuming mask is already [batch_size, max_steps, output_dim_per_step]
        logits = logits.masked_fill(~mask, float('-inf')) # Invert the mask for masked_fill
        
        # --- NO CLAMPING ON LOGITS HERE! Let the gradients flow freely.
        # The `clip_grad_norm_` in the trainer will handle exploding gradients.

        # --- Value Head ---
        # Pass pooled features through the value head to get the value prediction
        values = self.value_head(pooled_features)

        return logits, values
    
    def select_action(self, game, player, temperature=1.0):
        """Select action using the neural network with proper move validation"""

        self.eval() # Set model to evaluation mode

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
        with torch.no_grad(): # Disable gradient calculation
            logits, value = self.forward(state, mask)

        _S = 26
        seq_logits = []
        """
        seq_logits[i] is the sum of the network's scores across every individual 
        move in sequence i. The network's confidence in an entire sequence is the 
        sum of its confidences in each step of that sequence.

        """
        seq_scores = []
        for seq in seqs:
            score = torch.tensor(0.0, device="cpu")
            for t, (origin, dest) in enumerate(seq):
                pos = origin * self.S + dest
                score += logits[0, t, pos] # Summing raw logits
            seq_scores.append(score)

        seq_scores_t = torch.stack(seq_scores)
        probs = F.softmax(seq_scores_t / temperature, dim=0)


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

    print("Logits shape:", logits.shape)  # 
    print("Values shape:", values.shape)  # 
