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
import os.path
import random

import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg

def printBanner():
        print("""
·············································································
: ____    _    ____ _  ______    _    __  __ __  __  ___  _   _             :
:| __ )  / \  / ___| |/ / ___|  / \  |  \/  |  \/  |/ _ \| \ | |            :
:|  _ \ / _ \| |   | ' | |  _  / _ \ | |\/| | |\/| | | | |  \| |            :
:| |_) / ___ | |___| . | |_| |/ ___ \| |  | | |  | | |_| | |\  |  _   _   _ :
:|____/_/   \_\____|_|\_\____/_/   \_|_|  |_|_|  |_|\___/|_| \_| (_) (_) (_):
·············································································
""")
def prompt_die_choice(d1, d2):
    """
    Prompts the user to choose between two die values and ensures a valid choice.
    """
    while True:
        try:
            c = int(input(f"Choose which die to play first ({d1} or {d2}): "))
            if c == d1 or c == d2:
                return c
            else:
                print(f"Invalid. Enter {d1} or {d2}: ")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prompt_pair(prompt):
    """
    Prompts the user to enter two integers (origin and destination) and returns them as a tuple.
    """
    while True:
        try:
            user_input = input(f"{prompt} (origin dest): ")
            parts = user_input.split()
            if len(parts) == 2:
                o = int(parts[0])
                d = int(parts[1])
                return (o, d)
            else:
                print("Invalid input. Please enter two numbers separated by a space (e.g., '1 5').")
        except ValueError:
            print("Invalid input. Please ensure both values are integers.")

def roll_dice():
    return random.randint(1, 6)

def play_model(player_name, model_name, model):
    #rolling dice to determine who goes first
    button_pressed = input("Press r to roll Dice ")
    while( button_pressed !="r"):
         button_pressed = input("Press r! ")

    agent_roll = roll_dice()
    player_roll = roll_dice()

    while(agent_roll == player_roll):
        agent_roll = roll_dice()
        player_roll = roll_dice()
    
    print(f"You rolled a {player_roll} and the agent rolled a {agent_roll}")
    turn = 0 if player_roll>agent_roll else 1
    game = bg.Game(turn)
    p1 = bg.Player(player_name, bg.PlayerType.PLAYER1)
    p2 = bg.Player("CPU", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    game.printGameBoard()

    game_over = False
    winner = -1

    while not game_over:
        turn = game.getTurn()
        dice = game.roll_dice()
        current_player = game.getPlayers(turn)

        if turn == 1:
             #model's turn
             model.make_move(game)
        else:
             #player move
            print(f"You rolled a {dice[0]}, {dice[1]}")
            if dice[0] != dice[1]:
                first = prompt_die_choice(dice[0], dice[1])
                second = dice[1] if first==dice[0] else dice[0]

                o1, dest1 = prompt_pair(f"Move for die {first}")
                success_first_move, err = game.try_move(current_player, first, o1, dest1)
                if not success_first_move:
                    print(f"Error: {err}")

                game.printGameBoard()
                # second move
                o2, dest2 = prompt_pair(f"Move for die {second}")
                success_second_move, err = game.try_move(current_player, second, o2, dest2)
                if not success_second_move:
                    print(f"Error: {err}")
            
            else:
                #case where you rolled a double
                pass
        game_over, winner =game.is_game_over()

    print(f"GAME OVER! winner: {winner}")

def main():

    printBanner()
    
    player_name = input("Enter player name: ")
    model_name = input("Enter model name: ")

    path = '/Users/romanos/Backgammon_Engine/'
    while not os.path.isfile(path+model_name+'.pth'):
         model_name = input("Enter a correct model name, file does not exist: ")

    model = TDGammonModel()
    # load raw state dict
    state_dict = torch.load(path+model_name+'.pth', map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
         play_model(player_name, model_name, model)
    
if __name__ == "__main__":
    main()