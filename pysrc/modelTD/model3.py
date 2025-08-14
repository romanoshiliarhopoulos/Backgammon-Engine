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
    def __init__(self, input_size=198, hidden_size_1=128, hidden_size_2=64, dropout_rate=0.2):
        super().__init__()
        
        # Input to a larger hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.ln1 = nn.LayerNorm(hidden_size_1)
        
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.ln2 = nn.LayerNorm(hidden_size_2)
        
        # Output Layer: win probability
        self.fc3 = nn.Linear(hidden_size_2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #input layer pass
        x = self.fc1(x)
        x = self.ln1(x) 
        x = F.relu(x) 
        x = self.dropout(x) 

        #hidden layer pass
        x = self.fc2(x)
        x = self.ln2(x) 
        x = F.relu(x) 
        x = self.dropout(x) 

        #output layer pass

        x = self.fc3(x)
        # Apply sigmoid only at the very end, with proper scaling
        #x = torch.sigmoid(x * 0.5)  # Scale down to prevent saturation
        return x


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

        encoded_state[196] = float(game.getBornOffCount(bg.PlayerType.PLAYER1))
        encoded_state[197] = float(game.getBornOffCount(bg.PlayerType.PLAYER2))

        return torch.Tensor(encoded_state)
    
    def _greedy_sequence(self, game, actions):
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

        # Score each candidate by cloning + network evaluation
        device = next(self.parameters()).device
        values = []
        for seq in actions:
            sim = game.clone()
            ok = True
            for o, dst in seq:
                die = abs(o - dst)
                player = turn_player[sim.getTurn()]
                success, _ = sim.tryMove(player, int(die), o, dst)
                if not success:
                    ok = False
                    break
            if not ok:
                # illegal clone move → worst possible
                if game.getTurn() == bg.PlayerType.PLAYER1:
                    values.append(float('-inf'))
                else:
                    values.append(float('inf'))
            else:
                # encode & forward
                rep = self.encode_state(sim).to(device)
                with torch.no_grad():
                    values.append(self(rep.unsqueeze(0)).item())

        # Pick best or worst depending on side
        if game.getTurn() == bg.PlayerType.PLAYER1:
            idx = max(range(len(values)), key=lambda i: values[i])
        else:
            idx = min(range(len(values)), key=lambda i: values[i])
        best_seq = actions[idx]

        # -------- ε-greedy exploration block --------
        ε = max(0.2 * (1 - game_idx / 5000), 0.00)   # linear decay
        if random.random() < ε:
            best_seq = random.choice(actions)        # explore
        else:
            best_seq = self._greedy_sequence(game, actions)  # exploit

        # Apply to game
        for o, dst in best_seq:
            die = abs(o - dst)
            player = turn_player[game.getTurn()]
            game.tryMove(player, int(die), o, dst)

        

        return best_seq