import torch
import torch.nn as nn
import torch.nn.functional as F

class BackgammonNet(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=128, max_steps=4):
        super().__init__()
        self.conv1    = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2    = nn.Conv1d(32,        64, kernel_size=3, padding=1)
        self.fc_feat  = nn.Linear(64 * 24, hidden_dim)
        self.max_steps = max_steps
        self.S   = 26
        self.N   = self.S * self.S

        # now predict T * (origin→dest) logits in one shot
        self.policy_logits = nn.Linear(hidden_dim, max_steps * self.N)
        self.value_head    = nn.Linear(hidden_dim, 1)

    def forward(self, x, masks=None):
        """
        x:     [batch, 6, 24]
        masks: optional BoolTensor [batch, T, 676]

        returns:
          probs: Tensor [batch, T, 676]
          value: Tensor [batch]
        """
        b = x.size(0)
        h = F.relu(self.conv1(x))          # [b,32,24]
        h = F.relu(self.conv2(h))          # [b,64,24]
        h = h.view(b, -1)                  # [b,64*24]
        h = F.relu(self.fc_feat(h))        # [b,hidden_dim]

        logits = self.policy_logits(h)     # [b, T*676]
        logits = logits.view(b, self.max_steps, self.N)

        if masks is not None:
            # mask out illegal moves at each step
            logits = logits.masked_fill(~masks, float("-inf"))

        probs = F.softmax(logits, dim=-1)  # [b, T, 676]
        value = torch.tanh(self.value_head(h)).squeeze(-1)  # [b]
        return probs, value
