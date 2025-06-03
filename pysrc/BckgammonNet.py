#!/usr/bin/env python3
import random
import sys, os
import time
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

import torch
import matplotlib.pyplot as plt # type: ignore
from encode_state import encode_state
from encode_state    import build_sequence_mask
import torch.nn.functional as F
from torch.distributions import Categorical

#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # type: ignore
from torch.distributions import Categorical


class SeqBackgammonNet(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=128, max_steps=4):
        super().__init__()
        self.conv1       = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2       = nn.Conv1d(32,       64, kernel_size=3, padding=1)
        self.fc_feat     = nn.Linear(64*24, hidden_dim)
        self.max_steps   = max_steps
        self.S           = 26
        self.N           = self.S * self.S
        # predict all T * (origin→dest) logits
        self.policy_logits = nn.Linear(hidden_dim, max_steps*self.N)
        self.value_head    = nn.Linear(hidden_dim, 1)

    def forward(self, x, masks=None):
        b = x.size(0)
        h = F.relu(self.conv1(x))       # [b,32,24]
        h = F.relu(self.conv2(h))       # [b,64,24]
        h = h.view(b, -1)               # [b,64*24]
        h = F.relu(self.fc_feat(h))     # [b,hidden_dim]

        logits = self.policy_logits(h)  # [b, T*676]
        logits = logits.view(b, self.max_steps, self.N)
        if masks is not None:
            logits = logits.masked_fill(~masks, float('-inf'))
        return logits, torch.tanh(self.value_head(h)).squeeze(-1)

    