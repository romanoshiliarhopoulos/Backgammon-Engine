import random
import os, sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

_search_paths = [
    os.path.join(_here, "..", "build"),          
    os.path.join(_here, "..", "build", "Release"), 
    os.path.join(_here, "..", "build", "Debug"),   
]

for p in _search_paths:
    p = os.path.normpath(p)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        break
# ──────────────────────────────────────────────────────────────────────────────────
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg # type: ignore

class TDLGammonModel(nn.Module):
    def __init__(self, input_size=198, hidden_size=128):
        super().__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.learning_rate = 0.1

        self.initialize_weights()

        self.eligibility_traces = {}
        self.initialize_traces()

        self.lambda_decay = 0.7

    def initialize_traces(self):
        """Initialize eligibility traces for all parameters"""
        self.eligibility_traces = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.eligibility_traces[name] = torch.zeros_like(param.data)

    def initialize_weights(self):
        """Randomly initializes model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = torch.sigmoid(self.fc1(x))  
        x = torch.sigmoid(self.fc2(x))
        return x

    def update_learning_params(self, episode):
        """Updates learning rate and lambda decay over time"""

        self.learning_rate = max(0.01, 0.1 * (0.96 ** (episode // 40000)))
        self.lambda_decay = max(0.7, 0.9 * (0.96 ** (episode // 30000)))

    def encode_state(self, game):
        """ Encodes the game state. 
        Args:
            - self
            - game: Game object. Instance of game class from backgammon_env
        Returns:
            - a 1D tensor of 198 entries (corresponding to model's input nodes)

        Tensor: 24 possible board locations taking up 8 entries each (24*8 = 192) and 2 inputs representing
        the player turn, next two pieces that are on the bar (each input being one half the number of checkers
        on the bar for corresponding player) and next two inputs are number of pieces borne-off by each player

        For each of the 24 board position we have 8 entries (4 and 4 for each player respectively)
        Entries 0-4:
            -0: Is on if there is one piece from p1 on the specific position
            -1: Is on if there are two pieces from p1 on the specific position
            -2: Is on if there are three pieces from p1 on the specific position
            -3: Represents half the number of checkers beyond three in that position. (eg if peices = N, then you have (N-3)/2)

            4-7: are the same, but for p2

            (note: if it has 8 pieces [1,1,1,2.5]
        """

        return torch.from_numpy(self.encode_state_np(game))

    def encode_state_np(self, game):
        """NumPy float32[198] encoding of the live game (see encode_state docstring)."""
        row = np.empty((1, 28), dtype=np.int64)
        row[0, :24] = game.getGameBoard()
        row[0, 24] = game.getJailedCount(bg.PlayerType.PLAYER1)
        row[0, 25] = game.getJailedCount(bg.PlayerType.PLAYER2)
        row[0, 26] = game.getBornOffCount(bg.PlayerType.PLAYER1)
        row[0, 27] = game.getBornOffCount(bg.PlayerType.PLAYER2)
        return self._encode_states_np(row, game.getTurn())[0]

    def _encode_states_np(self, states, turn):
        """Vectorized encoder (global PLAYER1/PLAYER2 frame, matches the original
        scheme so existing checkpoints stay compatible). `states` is an (N, 28)
        int array (board[24] + jailed_p1, jailed_p2, freed_p1, freed_p2); `turn`
        is the current player. Slots 0-3 of each point are PLAYER1's checkers,
        4-7 PLAYER2's. Returns an (N, 198) float32 array, identical per row to the
        original encode_state."""
        states = np.asarray(states)
        N = states.shape[0]
        board = states[:, :24]
        X = np.zeros((N, 198), dtype=np.float32)

        rows = np.arange(N)
        for i in range(24):
            col = board[:, i]
            n = np.abs(col)
            base = 8 * i + np.where(col > 0, 0, 4)  # offset 0 for p1, 4 for p2
            m = n >= 1
            X[rows[m], base[m] + 0] = 1.0
            m = n >= 2
            X[rows[m], base[m] + 1] = 1.0
            m = n >= 3
            X[rows[m], base[m] + 2] = 1.0
            m = n >= 4
            X[rows[m], base[m] + 3] = (n[m] - 3) / 2

        p1_turn = (turn == bg.PlayerType.PLAYER1)
        X[:, 192] = 1.0 if p1_turn else 0.0
        X[:, 193] = 0.0 if p1_turn else 1.0
        X[:, 194] = states[:, 24] / 2
        X[:, 195] = states[:, 25] / 2
        X[:, 196] = states[:, 26] / 15.0
        X[:, 197] = states[:, 27] / 15.0
        return X

    def select_best_action(self, game, actions):
        """Evaluates all possible action sequences using the neural
        net and returns the one leading to the most favorable action.

        Kept for compatibility; the fast path is make_move(), which uses the
        fused C++ evaluateTurnSequences + a single batched forward pass."""

        device = next(self.parameters()).device
        values = [] #holds the values of each action
        for seq in actions:
            sim = game.clone()
            if not self._simulate_sequence(sim, seq):
                # assign worst value if sequence illegal in clone
                values.append(float('-inf') if game.getTurn()==bg.PlayerType.PLAYER1 else float('inf'))
                continue
            rep = self.encode_state(sim).to(device)
            with torch.no_grad():
                values.append(self(rep.unsqueeze(0)).item())

        idx = max(range(len(values)), key=values.__getitem__) \
            if game.getTurn()==bg.PlayerType.PLAYER1 else \
            min(range(len(values)), key=values.__getitem__)

        return actions[idx]

    def _simulate_sequence(self, sim_game, seq):
        for o, dst in seq:
            die = abs(o - dst)
            player = sim_game.getPlayers(sim_game.getTurn())
            ok, _ = sim_game.tryMove(player, int(die), o, dst)
            if not ok:
                return False
        return True

    def make_move(self, game, game_idx:int = 1, epsilon: float = 0.0):
        """Selects and applies a move. Candidate afterstates are scored with V (≈
        P(PLAYER1 wins)); PLAYER1 maximizes it, PLAYER2 minimizes it. `epsilon` > 0
        enables ε-greedy exploration (training only; evaluation passes epsilon=0)."""
        # evaluation mode (turns off dropout)
        self.eval()

        # Reconstruct Python-side Player objects
        p1 = game.getPlayers(bg.PlayerType.PLAYER1)
        p2 = game.getPlayers(bg.PlayerType.PLAYER2)
        turn_player = {
            bg.PlayerType.PLAYER1: p1,
            bg.PlayerType.PLAYER2: p2
        }

        # Get the dice that were rolled this turn
        d = game.get_last_dice()
        turn = game.getTurn()

        # Enumerate every legal sequence AND its resulting state in one C++ call,
        # then score them all with a single batched forward pass.
        actions, states = game.evaluateTurnSequences(turn, d[0], d[1])
        if not actions:
            return []

        if epsilon > 0.0 and random.random() < epsilon:
            idx = random.randrange(len(actions))
        else:
            device = next(self.parameters()).device
            X = torch.from_numpy(self._encode_states_np(states, turn)).to(device)
            with torch.inference_mode():
                values = self(X).squeeze(1)
            idx = int(torch.argmax(values) if turn == bg.PlayerType.PLAYER1
                      else torch.argmin(values))
        best_seq = actions[idx]

        # Apply to game
        for o, dst in best_seq:
            die = abs(o - dst)
            player = turn_player[turn]
            game.tryMove(player, int(die), o, dst)

        return best_seq