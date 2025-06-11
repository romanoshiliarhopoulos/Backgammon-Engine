#!/usr/bin/env python3
"""
Self-play training loop for SeqBackgammonNet using the C++ Backgammon environment.
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

# Ensure C++ extension is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg
from encode_state import encode_state, build_sequence_mask
from model2 import SeqBackgammonNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameExperience:
    """Store experience from a single game step"""
    state: torch.Tensor
    action_probs: torch.Tensor
    value: float
    reward: float
    turn: int
    
class RewardCalculator:
    """Calculate rewards for different game events"""
    
    # Reward constants
    WIN_REWARD = 1.0
    LOSS_REWARD = -1.0
    HIT_OPPONENT_REWARD = 0.1
    BEAR_OFF_REWARD = 0.05
    ESCAPE_JAR_REWARD = 0.02
    PROGRESS_REWARD = 0.001
    
    @staticmethod
    def calculate_intermediate_rewards(game_before, game_after, player_num):
        """Calculate intermediate rewards based on game state changes"""
        reward = 0.0
        
        # Get pieces info before and after
        pieces_before = game_before.getPieces()
        pieces_after = game_after.getPieces()
        
        # Reward for bearing off pieces
        borne_off_before = pieces_before.numFreed(player_num)
        borne_off_after = pieces_after.numFreed(player_num)
        if borne_off_after > borne_off_before:
            reward += RewardCalculator.BEAR_OFF_REWARD * (borne_off_after - borne_off_before)
        
        # Reward for hitting opponent pieces (jailing them)
        opponent_num = 1 - player_num
        opponent_jailed_before = pieces_before.numJailed(opponent_num)
        opponent_jailed_after = pieces_after.numJailed(opponent_num)
        if opponent_jailed_after > opponent_jailed_before:
            reward += RewardCalculator.HIT_OPPONENT_REWARD * (opponent_jailed_after - opponent_jailed_before)
        
        # Reward for escaping from jail
        player_jailed_before = pieces_before.numJailed(player_num)
        player_jailed_after = pieces_after.numJailed(player_num)
        if player_jailed_after < player_jailed_before:
            reward += RewardCalculator.ESCAPE_JAR_REWARD * (player_jailed_before - player_jailed_after)
        
        # Small progress reward for moving pieces forward
        board_before = game_before.getGameBoard()
        board_after = game_after.getGameBoard()
        progress = RewardCalculator._calculate_progress_change(board_before, board_after, player_num)
        reward += RewardCalculator.PROGRESS_REWARD * progress
        
        return reward
    
    @staticmethod
    def _calculate_progress_change(board_before, board_after, player_num):
        """Calculate how much progress was made moving pieces toward home"""
        progress = 0.0
        multiplier = 1 if player_num == 0 else -1
        
        for i in range(24):
            pieces_before = board_before[i] * multiplier if board_before[i] * multiplier > 0 else 0
            pieces_after = board_after[i] * multiplier if board_after[i] * multiplier > 0 else 0
            
            if player_num == 0:  # Player 1 moves from low to high indices
                position_value = i + 1
            else:  # Player 2 moves from high to low indices
                position_value = 24 - i
            
            progress += (pieces_after - pieces_before) * position_value
        
        return progress

class SelfPlayTrainer:
    """Main training class for self-play learning"""
    
    def __init__(self, 
                 model: SeqBackgammonNet,
                 device: str = 'mps',
                 lr: float = 1e-3,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 value_loss_weight: float = 1.0,
                 policy_loss_weight: float = 1.0,
                 entropy_weight: float = 0.01):
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Training hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.entropy_weight = entropy_weight
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.games_played = 0
        self.training_steps = 0
        self.win_rates = deque(maxlen=100)
        # Add this to your SelfPlayTrainer.__init__

        self.cache_clear_interval = 5  # Clear cache every 10 games

        

    
    
    def select_action(self, game, player, temperature=1.0):
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
        
        state = encode_state(board, pieces, turn, dice1, dice2).unsqueeze(0).to(self.device)
        
        # Get legal moves and build mask - this is crucial
        mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
            game, player, batch_size=1, device=self.device
        )
        
        if not seqs:  # No legal moves
            return None, None, None
        
        # Forward pass through network
        with torch.no_grad():
            logits, value = self.model(state, mask)
        
        # Instead of calculating probabilities manually, use the legal sequences directly
        # Sample from the available legal sequences with equal probability initially
        if temperature > 0:
            # For now, sample uniformly from legal sequences
            # You can later implement proper probability calculation
            selected_idx = random.randint(0, len(seqs) - 1)
        else:
            selected_idx = 0  # Take first legal sequence
        
        selected_sequence = seqs[selected_idx]
        dice_order = dice_orders[selected_idx]
        
        # Return uniform probabilities for 
        sequence_probs = torch.ones(len(seqs)) / len(seqs)
        
        return selected_sequence, dice_order, sequence_probs

    
    def play_game(self, temperature=1.0):
        """Play a complete self-play game and collect experiences"""
        # Initialize game and players
        game = bg.Game(0)  # Start with player 1
        player1 = bg.Player("Player1", bg.PlayerType.PLAYER1)
        player2 = bg.Player("Player2", bg.PlayerType.PLAYER2)
        game.setPlayers(player1, player2)
        
        experiences = []
        game_rewards = {0: [], 1: []}  # Track rewards for each player
        turn_count = 0
        while True and turn_count< 500:
            if turn_count > 499:
                print("more than 500 turns")
            # Check if game is over
            is_over, winner = game.is_game_over()
            if is_over:
                # Assign final rewards
                final_reward_winner = RewardCalculator.WIN_REWARD
                final_reward_loser = RewardCalculator.LOSS_REWARD
                
                # Update all experiences with final rewards
                for exp in experiences:
                    if exp.turn == winner:
                        exp.reward += final_reward_winner
                    else:
                        exp.reward += final_reward_loser
                
                self.win_rates.append(1 if winner == 0 else 0)
                break
            
            # Roll dice
            dice = game.roll_dice()
            current_player = player1 if game.getTurn() == 0 else player2
            
            # Get current state for experience recording
            board = game.getGameBoard()
            pieces = game.getPieces()
            jailed = pieces.numJailed(current_player.getNum())
            borne_off = pieces.numFreed(current_player.getNum())
            turn = game.getTurn()
            state = encode_state(board, pieces, turn, dice[0], dice[1])

            
            # Select and execute action
            game_before = game.clone()
            sequence, dice_order, action_probs = self.select_action(game, current_player, temperature)
            
            if sequence is None:
                # No legal moves, switch turns
                game.setTurn(1 - game.getTurn())
                turn_count +=1
                continue
            
            # Execute the selected sequence
            success = True
            for i, (origin, dest) in enumerate(sequence):
                if i < len(dice_order):
                    dice_val = dice_order[i]
                    move_success, error = game.tryMove(current_player, dice_val, origin, dest)
                    if not move_success:
                        logger.warning(f"Move failed: {error}")
                        success = False
                        break
            
            if success:
                # Calculate intermediate rewards
                intermediate_reward = RewardCalculator.calculate_intermediate_rewards(
                    game_before, game, current_player.getNum()
                )
                
                # Get value prediction
                state_tensor = state.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.model(state_tensor)
                    value = value.item()
                
                # Store experience
                experience = GameExperience(
                    state=state,
                    action_probs=action_probs,
                    value=value,
                    reward=intermediate_reward,
                    turn=current_player.getNum()
                )
                experiences.append(experience)
                game_rewards[current_player.getNum()].append(intermediate_reward)
            
            # Switch turns
            game.setTurn(1 - game.getTurn())
            turn_count +=1
        
        return experiences
    
    def compute_returns(self, experiences, gamma=0.99):
        """Compute discounted returns for experiences"""
        returns = []
        running_return = 0
        
        # Process experiences in reverse order
        for exp in reversed(experiences):
            running_return = exp.reward + gamma * running_return
            returns.append(running_return)
        
        returns.reverse()
        return returns
    
    def train_step(self):
        """Perform one training step on a batch of experiences"""
        if len(self.experience_buffer) < self.batch_size:
            return None
        
        # Sample batch from experience buffer
        batch_experiences = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        states = torch.stack([exp.state for exp in batch_experiences]).to(self.device)
        returns = torch.tensor([exp.reward for exp in batch_experiences], dtype=torch.float32).to(self.device)
        values = torch.tensor([exp.value for exp in batch_experiences], dtype=torch.float32).to(self.device)
        
        # Forward pass
        logits, predicted_values = self.model(states)
        
        # Compute losses
        value_loss = F.mse_loss(predicted_values, returns)
        
        # Policy loss (simplified - using value advantage as proxy)
        advantages = returns - predicted_values.detach()
        policy_loss = -torch.mean(advantages * torch.log(torch.clamp(values, min=1e-8)))
        
        # Entropy loss for exploration
        entropy_loss = 0.0  # Simplified for now
        
        # Total loss
        total_loss = (self.value_loss_weight * value_loss + 
                     self.policy_loss_weight * policy_loss + 
                     self.entropy_weight * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.training_steps += 1
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss
        }
    
    def train(self, num_games=1000, games_per_update=10, save_interval=100):
        """Main training loop"""
        logger.info(f"Starting training for {num_games} games")
        
        for game_idx in range(num_games):
            # Play game with temperature annealing
            temperature = max(0.1, 1.0 - game_idx / (num_games * 0.8))
            experiences = self.play_game(temperature=temperature)
            
            # Also clear PyTorch cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Compute returns and add to buffer
            returns = self.compute_returns(experiences)
            for exp, ret in zip(experiences, returns):
                exp.reward = ret
                self.experience_buffer.append(exp)
            
            self.games_played += 1
            
            # Training step
            if game_idx % games_per_update == 0 and len(self.experience_buffer) >= self.batch_size:
                loss_info = self.train_step()
                
                win_rate = np.mean(self.win_rates) if self.win_rates else 0.0
                logger.info(f"Game {game_idx}: Loss={loss_info['total_loss']:.4f}, "
                          f"Win Rate={win_rate:.3f}, Temp={temperature:.3f}")
            if game_idx % self.cache_clear_interval == 0:
                logger.info(f"Clearing sequence cache at game {game_idx}")
                from encode_state import clear_sequence_cache, get_cache_stats
                stats = get_cache_stats()
                logger.info(f"Cache stats before clear: {stats}")
                clear_sequence_cache()
            # Save model
            if game_idx % save_interval == 0 and game_idx > 0:
                self.save_model(f"model_checkpoint_{game_idx}.pt")
                logger.info(f"Model saved at game {game_idx}")
        
        logger.info("Training completed!")
    
    def save_model(self, filepath):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
            'training_steps': self.training_steps,
        }, filepath)
    
    def load_model(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games_played = checkpoint.get('games_played', 0)
        self.training_steps = checkpoint.get('training_steps', 0)

def main():
    """Main training function"""
    # Set device
    device = 'cpu' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SeqBackgammonNet(
        n_channels=9,
        hidden_dim=128,
        max_steps=4,
        num_res_blocks=10
    )
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        model=model,
        device=device,
        lr=1e-3,
        buffer_size=10000,
        batch_size=32,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        entropy_weight=0.01
    )
    
    # Start training
    trainer.train(
        num_games=5000,
        games_per_update=5,
        save_interval=500
    )
    
    # Save final model
    trainer.save_model("final_model.pt")

if __name__ == "__main__":
    main()
