#!/usr/bin/env python3

from model3 import TDGammonModel
import torch
import random
import torch
import os
import sys
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim
import numpy as np
from collections import deque
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from tqdm import trange

import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg


def random_move(game):
    """makes a random move based on a give game state"""
    d1, d2 = game.get_last_dice()
    # Enumerate all legal sequences
    actions = game.legalTurnSequences(game.getTurn(), d1, d2)
    
    if not actions:
        return []
    
    #pick a random choice
    selected_index =  random.randint(0, len(actions) - 1)

    chosen_sequence = actions[selected_index]
    for o, dst in chosen_sequence:
        die = abs(o - dst)
        player = bg.PlayerType.PLAYER2
        game.tryMove(player, int(die), o, dst)

    return chosen_sequence


def play_random(model, num_games):
    """benchmarks against a random player"""
    num_wins = 0

    for i in trange(num_games, desc="Games"):
        game = bg.Game(i%2) #50/50 on who starts first
        p1 = bg.Player("RL agent", bg.PlayerType.PLAYER1)
        p2 = bg.Player("Random player", bg.PlayerType.PLAYER2)
        game.setPlayers(p1, p2)

        #main game loop
        while True:
            dice = game.roll_dice()
            turn = game.getTurn()
            
            if turn % 2 == bg.PlayerType.PLAYER1:
                """RL agents move"""
                model.make_move(game)
            else:
                """Random agent's move"""
                random_move(game=game)
            
            # Check for game end
            over, winner = game.is_game_over()
            if over:
                #print(f"Total moves: {total_moves}")
                num_wins = num_wins + 1 if winner == bg.PlayerType.PLAYER1 else num_wins

            # Switch turn
            next_turn = bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1 else bg.PlayerType.PLAYER1
            game.setTurn(next_turn)
            
    return num_wins/num_games

def main():
    print("Starting Benchmarking tests:")
    
    model = TDGammonModel()
    # load raw state dict
    state_dict = torch.load("tdgammon_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    num_games = 500
    win_rate_random = play_random(model=model, num_games=num_games)
    print(f"Against random bot: {win_rate_random*100:.1f}%")

if __name__ == "__main__":
    main()