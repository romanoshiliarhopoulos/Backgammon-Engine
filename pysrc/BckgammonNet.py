import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgammonNet(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=128):
        super().__init__()
        # feature extractor: 1D convolutions over the 24-point board
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc_feat = nn.Linear(64 * 24, hidden_dim)

        # policy head: include bar (0) and bear-off (25) for both origin and destination -> 26x26 actions
        self.S = 26
        self.policy_logits = nn.Linear(hidden_dim, self.S * self.S)

        # value head: scalar
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, legal_mask=None):
        """
        x: Tensor of shape [batch, 6, 24]
        legal_mask: optional BoolTensor of shape [batch, 26*26] indicating allowed (origin, dest) pairs

        returns:
            probs: Tensor of shape [batch, 26*26] - action probabilities
            value: Tensor of shape [batch] - state value estimate
        """
        b = x.size(0)
        h = F.relu(self.conv1(x))        # [b, 32, 24]
        h = F.relu(self.conv2(h))        # [b, 64, 24]
        h = h.view(b, -1)                # [b, 64*24]
        h = F.relu(self.fc_feat(h))      # [b, hidden_dim]

        logits = self.policy_logits(h)   # [b, 676]
        if legal_mask is not None:
            # mask out illegal moves
            logits = logits.masked_fill(~legal_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1) # [b, 676]

        value = torch.tanh(self.value_head(h)).squeeze(-1)  # [b]
        return probs, value
