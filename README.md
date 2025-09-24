# TD-Gammon: Backgammon Reinforcement Learning Engine

A  backgammon game engine with reinforcement learning capabilities, featuring a C++ game implementation with Python bindings and a neural network trained using TD(λ) learning algorithms.

## 🎯 Project Overview

This project implements a complete backgammon game engine with an AI player that learns through self-play using temporal difference learning with eligibility traces. The system combines a high-performance C++ game engine with a Python-based neural network for strategic decision making.

### Key Features

- **Complete Backgammon Implementation**: Full game rules including piece movement, capturing, jail mechanics, and bearing off
- **TD(λ) Reinforcement Learning**: Neural network trained using temporal difference learning with eligibility traces
- **High-Performance C++ Engine**: Fast game state management and move generation
- **Python Integration**: Seamless C++/Python interface using pybind11
- **Self-Play Training**: AI improves through thousands of games against itself
- **Comprehensive Testing**: Extensive unit tests for game logic validation

## 🏗️ Architecture

### Core Components

#### C++ Game Engine
- **Game Class** (`game.cpp`, `game.hpp`): Core game state management, move validation, and rule enforcement
- **Player Class** (`player.cpp`, `player.hpp`): Player representation and management
- **Pieces Class** (`Pieces.cpp`, `Pieces.hpp`): Handles jailed and freed pieces tracking
- **Python Bindings** (`backgammon_bindings.cpp`): pybind11 interface for Python integration

#### Python AI System
- **Neural Network** (`model.py`): TD-Gammon inspired architecture with eligibility traces
- **Training Pipeline** (`train.py`): Self-play training loop with learning metrics
- **Game Interface** (`main.cpp`): Human-playable console interface

## 🧠 Neural Network Architecture

### Model Design
- **Input Layer**: 198 features encoding complete game state
  - 192 features for board positions (8 per point × 24 points)
  - 6 features for player turn, jail counts, and borne-off pieces
- **Hidden Layer**: 128 neurons with sigmoid activation
- **Output Layer**: Single neuron outputting position evaluation (0-1)

### State Encoding
Each board position uses 8 features (4 per player):
- Feature 0: Has 1 piece
- Feature 1: Has 2+ pieces  
- Feature 2: Has 3+ pieces
- Feature 3: Excess pieces beyond 3 (normalized)

Additional features encode turn state, jail counts, and progress toward victory.

### Learning Algorithm
- **TD(λ) Learning**: Temporal difference learning with eligibility traces
- **Self-Play Training**: AI plays against itself to discover optimal strategies
- **Adaptive Parameters**: Learning rate and λ decay adjust during training
- **Move Selection**: Evaluates all legal moves and selects highest-value sequence
