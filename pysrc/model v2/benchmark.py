#!/usr/bin/env python3
"""
Benchmarking latest model against a variety of backgammon players
"""
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

import datetime
import matplotlib.pyplot as plt


# Ensure C++ extension is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg
from encode_state import encode_state, build_sequence_mask
from model2 import SeqBackgammonNet

logger = logging.getLogger(__name__)



def play_random(model):
    """
    Benchmarks the latest model against a model playing a random legal move each time:
    We need to benchmark against 100 games

    returns a scalar of the win rate: (number of wins) / (number of games)

    """
    total_games = 100
    num_wins = 0

    
    for i in range(total_games):
        game = bg.Game(0)
        model_agent = bg.Player("RL model", bg.PlayerType.PLAYER1)
        random_agent = bg.Player("Random agent", bg.PlayerType.PLAYER2)
        game.setPlayers(model_agent, random_agent)

        #game loop
        turn_count = 0 
        winner = None
        while True and turn_count < 500:
            if turn_count > 498:
                logger.info("more than 500 turns")

            # Check if game is over
            is_over, winner = game.is_game_over()
            if is_over and winner == model_agent:
                num_wins += 1
                break # go to the next game

            #roll dice and get current player
            dice = game.roll_dice()
            current_player = model_agent if game.getTurn() == 0 else random_agent

            if current_player == random_agent:
                #select a random legal move and execute it
                # Get legal moves and build mask
                mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game, random_agent, batch_size=1, device="cpu"
                )
                # Find valid moves from the mask
                valid_indices = torch.where(valid_mask[0])[0]  # Get indices where mask is True
                
                if len(valid_indices) > 0:
                    # Select random valid move
                    random_idx = torch.randint(0, len(valid_indices), (1,)).item()
                    selected_move_idx = valid_indices[random_idx].item()
                    
                    # Execute the selected move
                    selected_sequence = seqs[0][selected_move_idx]
                    game.make_move(selected_sequence, dice_orders[0][selected_move_idx])
                else:
                    # No legal moves available - pass turn
                    pass
            else:
                """It is the backgammonNet's turn ,make move according to model."""
                mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game, model_agent, batch_size=1, device="cpu"
                )
                
                if torch.any(valid_mask[0]):
                    # Get model prediction
                    with torch.no_grad():
                        state_tensor = encode_state(game, dice)
                        logits = model(state_tensor.unsqueeze(0))
                        
                        # Apply mask to logits to only consider legal moves
                        masked_logits = logits[0].clone()
                        masked_logits[~valid_mask[0]] = float('-inf')
                        
                        # Select best legal move
                        best_move_idx = torch.argmax(masked_logits).item()
                        
                        # Execute the move
                        selected_sequence = seqs[0][best_move_idx]
                        game.make_move(selected_sequence, dice_orders[0][best_move_idx])
                else:
                    # No legal moves available - pass turn
                    pass

            turn_count += 1

    win_rate = num_wins / total_games
    return win_rate
    

def main():
    logger.info("Starting Benchmarking tests:")
    model = SeqBackgammonNet()
    model.load_state_dict(torch.load("model_checkpoint_4500.pt", map_location="cpu"))
    model.eval()
    win_rate_random = play_random(model=model)
    logger.info(f"Against random bot: {win_rate_random}")
    logger.info("Romanos")




if __name__ == "__main__":
    main()