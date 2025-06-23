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

def execute_move(sequence, dice_order, action_probs, current_player, game):
    """Helper function to complete moves for agents"""
    if sequence is None:
        # No legal moves, switch turns
        game.setTurn(1 - game.getTurn())
        return False, 1
            
    # Execute the selected sequence
    success = True
    for i, (origin, dest) in enumerate(sequence):
        if i < len(dice_order):
            dice_val = dice_order[i]
            move_success, error = game.tryMove(current_player, dice_val, origin, dest)
            if not move_success:
                logger.warning(f"Move failed: {error}")
                success = False
                return False, 2
    return True, 1
    
def select_action(model, game, player, temperature=1.0):
        """Select action using the neural network with proper move validation"""
        # Get dice values
        dice1, dice2 = game.get_last_dice()
        
        # Encode current state
        board = game.getGameBoard()
        pieces = game.getPieces()
        jailed = pieces.numJailed(player.getNum())
        borne_off = pieces.numFreed(player.getNum())
        turn = game.getTurn()
        
        # Get opponent info for proper encoding
        opponent_num = 1 - player.getNum()
        opp_jailed = pieces.numJailed(opponent_num)
        opp_borne_off = pieces.numFreed(opponent_num)
        
        state = encode_state(board, pieces, turn, dice1, dice2).unsqueeze(0).to("cpu")
        
        # Get legal moves and build mask
        mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
            game, player, batch_size=1, device="cpu"
        )
        
        if not seqs:  # No legal moves
            return None, None, None
        
        # Forward pass through network
        with torch.no_grad():
            logits, value = model(state, mask)

        _S = 26
        seq_logits = []
        """
        seq_logits[i] is the sum of the network's scores across every individual 
        move in sequence i. The network's confidence in an entire sequence is the 
        sum of its confidences in each step of that sequence.

        """
        for seq in seqs:
            # sum the network’s raw logits at each (step, origin→dest) in this sequence
            s = logits.new_zeros(())
            for t, (origin, dest) in enumerate(seq):
                pos = origin * _S + dest
                s = s + logits[0, t, pos]
            seq_logits.append(s)

        seq_logits = torch.stack(seq_logits)         #turn scores into a prob distribution
        probs = F.softmax(seq_logits / max(temperature, 1e-6), dim=0)

        m = Categorical(probs)  #draws a sample from the probs distribution
        idx = m.sample().item()                     

        selected_sequence = seqs[idx]
        dice_order       = dice_orders[idx]

        return selected_sequence, dice_order, probs

from tqdm import trange

def play_random(model):
    """
    Benchmarks the latest model against a model playing a random legal move each time:
    We need to benchmark against 100 games

    returns a scalar of the win rate: (number of wins) / (number of games)

    """

    total_games = 500
    num_wins = 0
    
    for i in trange(total_games, desc="Playing games"):
        #print(f"Game {i} started...")
        game = bg.Game(0)
        model_agent = bg.Player("RL model", bg.PlayerType.PLAYER1)
        random_agent = bg.Player("Random agent", bg.PlayerType.PLAYER2)
        game.setPlayers(model_agent, random_agent)

        #game loop
        turn_count = 0 
        winner = None
        while True and turn_count < 500:
            #game.printGameBoard()
            if turn_count > 498:
                logger.info("more than 500 turns")
            # Check if game is over
            is_over, winner = game.is_game_over()
            if is_over:                
                if winner == model_agent.getNum():
                    num_wins += 1
                break # go to the next game

            #roll dice and get current player
            dice = game.roll_dice()
            current_player = model_agent if game.getTurn() == 0 else random_agent

            if current_player == random_agent:
                """It is the random agents's turn."""

                #select a random legal move and execute it
                # Get dice values
                dice1, dice2 = game.get_last_dice()
                
                # Encode current state
                board = game.getGameBoard()
                pieces = game.getPieces()
                turn = game.getTurn()
                
                state = encode_state(board, pieces, turn, dice1, dice2).unsqueeze(0).to("cpu")
                
                # Get legal moves and build mask
                mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game, random_agent, batch_size=1, device="cpu"
                )
                
                if not seqs:  # No legal moves
                    #print(f"No legal moves for the Random bot, turn: {turn_count}")
                    #print(f"Dice: {game.get_last_dice()}")
                    #game.printGameBoard()
                    continue #pass turn
                
                selected_idx = random.randint(0, len(seqs) - 1)
                
                selected_sequence = seqs[selected_idx]
                dice_order = dice_orders[selected_idx]

                # Return uniform probabilities for 
                sequence_probs = torch.ones(len(seqs)) / len(seqs)
                #executes move based on randomly selected index. 
                status, num = execute_move(selected_sequence, dice_order, sequence_probs, current_player, game)
                
                if not status and num == 1 :
                    turn_count +=1
                    continue
                elif not status and num == 2:
                    break
                turn_count +=1
                game.setTurn(0)
            else:
                """It is the backgammonNet's turn ,make move according to model."""

                full_mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game, model_agent, batch_size=1, device="cpu")
                
                sequence, dice_order, action_probs = select_action(model, game, current_player, 1.0)
                
                status, num = execute_move(sequence, dice_order, action_probs, current_player, game)
                if not status and num == 1 :
                    turn_count +=1
                    continue #pass this turn
                elif not status and num == 2:
                    break #something catastrophical went wrong
                turn_count +=1
                game.setTurn(1)


    win_rate = num_wins / total_games
    return win_rate
    

def main():
    print("Starting Benchmarking tests:")
    
    model = SeqBackgammonNet()
    checkpoint = torch.load("model_checkpoint_4500.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    win_rate_random = play_random(model=model)
    
    print(f"Against random bot: {win_rate_random* 100}%")




if __name__ == "__main__":
    main()