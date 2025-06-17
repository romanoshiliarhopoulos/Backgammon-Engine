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



def main():
    print("Romanos")




if __name__ == "__main__":
    main()