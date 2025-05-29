#!/usr/bin/env python3
# train.py (project root)
import sys, os
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

import torch
from backgammon_env import Game, Player, PlayerType

# instead of your_rl_model, import directly:
from BckgammonNet     import BackgammonNet
from encode_state     import encode_state, build_legal_mask

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net = BackgammonNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    num_episodes = 1000
    gamma        = 0.99
    c1           = 0.5

    for ep in range(1, num_episodes+1):
        # … (paste the training‐loop from above here) …
        print(f"Episode {ep} completed")
    
if __name__ == "__main__":
    main()
