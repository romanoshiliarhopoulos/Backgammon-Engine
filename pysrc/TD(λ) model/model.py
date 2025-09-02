import random
import os, sys

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

class TDGammonModel(nn.Module):
    def __init__(self, input_size=198, hidden_size=128):
        super().__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        self.learning_rate = 0.1

        self._initialize_weights()


    def _initialize_weights(self):
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

        encoded_state = [0.0] * 198 
        raw_gameboard = game.getGameBoard()

        #populate positions 0-191 through gamebaord
        for i in range(len(raw_gameboard)):
            if raw_gameboard[i] != 0:
                offset = 0 if raw_gameboard[i]>0 else 4
                num_pieces = abs(raw_gameboard[i])
                encoded_state[8*i + 0 +offset] = 1 
                encoded_state[8*i + 1 +offset] = 1 if num_pieces >=2 else 0
                encoded_state[8*i + 2 +offset] = 1 if num_pieces >=3 else 0
                encoded_state[8*i + 3 +offset] = (num_pieces-3)/2 if num_pieces >=4 else 0
        
        #populate 192-197
        encoded_state[192] = 1 if game.getTurn() == bg.PlayerType.PLAYER1 else 0
        encoded_state[193] = 1 if game.getTurn() == bg.PlayerType.PLAYER2 else 0

        encoded_state[194] = game.getJailedCount(bg.PlayerType.PLAYER1) / 2
        encoded_state[195] = game.getJailedCount(bg.PlayerType.PLAYER2) / 2

        encoded_state[196] = game.getBornOffCount(bg.PlayerType.PLAYER1) / 15.0
        encoded_state[197] = game.getBornOffCount(bg.PlayerType.PLAYER2) / 15.0

        return torch.Tensor(encoded_state)
    
    def select_best_action(self, game, actions):
        """Evaluates all possible action sequences using the neural 
        net and returns the one leading to the most favorable action"""
        device = next(self.parameters()).device
        values = []
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

    def make_move(self, game, game_idx:int = 1):
        """makes a moves based on a given state"""
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
        d1 = d[0]
        d2 = d[1]

        # Enumerate all legal sequences
        actions = game.legalTurnSequences(game.getTurn(), d1, d2)
        if not actions:
            return []

        best_seq = self.select_best_action(game, actions)

        # Apply to game
        for o, dst in best_seq:
            die = abs(o - dst)
            player = turn_player[game.getTurn()]
            game.tryMove(player, int(die), o, dst)

        return best_seq