#!/usr/bin/env python3

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'modelTD')))
from model3 import TDGammonModel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),  'TD(λ) model')))
from model import TDLGammonModel

import backgammon_env as bg

def main():
    game = bg.Game(0)
    game.setGameBoard([-5,-4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5])
    game.printGameBoard()
    print(game.legalTurnSequences(1, 6, 5))
    assert 1==1



if __name__ == "__main__":
    main()