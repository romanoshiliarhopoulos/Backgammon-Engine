import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgammonNet(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=128):
        super().__init__()
        # feature extractor: 1D convolutions over the 24 points
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc_feat = nn.Linear(64 * 24, hidden_dim)
        
        # policy head: we’ll output a logit for **every possible** (origin, dest) pair:
        self.policy_logits = nn.Linear(hidden_dim, 24*24)
        
        # value head: scalar
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, legal_mask=None):
        """
        x: [batch, 6, 24]
        legal_mask: [batch, 24*24] boolean mask of which (o,d) pairs are legal
        returns: (probs over 24×24 moves, value estimate)
        """
        b = x.shape[0]
        h = F.relu(self.conv1(x))    # [b,32,24]
        h = F.relu(self.conv2(h))    # [b,64,24]
        h = h.view(b, -1)            # [b,64*24]
        h = F.relu(self.fc_feat(h))  # [b,hidden_dim]
        
        logits = self.policy_logits(h)  # [b,576]
        if legal_mask is not None:
            # mask out illegal moves
            logits = logits.masked_fill(~legal_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)  # [b,576]
        
        value = torch.tanh(self.value_head(h)).squeeze(-1)  # [b]
        return probs, value
