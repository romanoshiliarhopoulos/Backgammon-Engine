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
import matplotlib as plt

# Ensure C++ extension is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg
from encode_state import encode_state, build_sequence_mask
from model2 import SeqBackgammonNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def plot_all_metrics(win_history, episode_rewards, episode_turns, episode_losses, num_completed):
    """
    Create and save plots for training metrics. If no episodes completed, skip plotting.
    """
    if num_completed == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("figures", exist_ok=True)

    wins = torch.tensor(win_history, dtype=torch.float32).cpu().numpy()
    rewards = torch.tensor(episode_rewards, dtype=torch.float32).cpu().numpy()
    turns = torch.tensor(episode_turns, dtype=torch.float32).cpu().numpy()
    losses = torch.tensor(episode_losses, dtype=torch.float32).cpu().numpy()
    episodes = torch.arange(1, len(wins) + 1).cpu().numpy()

    window_size = max(5, len(wins) // 10)

    def moving_average(data, window):
        return torch.from_numpy(data).to(torch.float32).unfold(0, window, 1).mean(dim=1).cpu().numpy()

    # 1. Win Rate Analysis
    plt.figure(figsize=(12, 8))

    # Subplot 1: Cumulative Win Rate
    plt.subplot(2, 2, 1)
    cum_wins = torch.cumsum(torch.from_numpy(wins), dim=0).cpu().numpy()
    cum_win_rate = cum_wins / episodes
    plt.plot(episodes, cum_win_rate, 'b-', alpha=0.7, label='Cumulative Win Rate')

    if len(cum_win_rate) >= window_size:
        ma_win_rate = moving_average(cum_win_rate, window_size)
        ma_episodes = episodes[window_size - 1:]
        plt.plot(ma_episodes, ma_win_rate, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Cumulative Win Rate')
    plt.title('Win Rate Over Time')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()

    # Subplot 2: Episode-by-Episode Win Rate (windowed)
    plt.subplot(2, 2, 2)
    if len(wins) >= window_size:
        windowed_win_rate = moving_average(wins, window_size)
        windowed_episodes = episodes[window_size - 1:]
        plt.plot(windowed_episodes, windowed_win_rate, 'g-', linewidth=2)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random Performance')
        plt.legend()

    plt.xlabel('Episode')
    plt.ylabel(f'Win Rate (Last {window_size} Episodes)')
    plt.title('Recent Win Rate Trend')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)

    # 2. Cumulative Reward Analysis
    plt.subplot(2, 2, 3)
    cumulative_rewards = torch.cumsum(torch.from_numpy(rewards), dim=0).cpu().numpy()
    plt.plot(episodes, cumulative_rewards, 'purple', alpha=0.7, label='Cumulative Reward')

    if len(rewards) >= window_size:
        ma_rewards = moving_average(cumulative_rewards, window_size)
        ma_episodes = episodes[window_size - 1:]
        plt.plot(ma_episodes, ma_rewards, 'orange', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward Accumulation')
    plt.grid(alpha=0.3)
    plt.legend()

    # 3. Sample Efficiency (Average Reward per Episode)
    plt.subplot(2, 2, 4)
    avg_reward_per_episode = cumulative_rewards / episodes
    plt.plot(episodes, avg_reward_per_episode, 'brown', alpha=0.7, label='Avg Reward/Episode')

    if len(avg_reward_per_episode) >= window_size:
        ma_avg_rewards = moving_average(avg_reward_per_episode, window_size)
        ma_episodes = episodes[window_size - 1:]
        plt.plot(ma_episodes, ma_avg_rewards, 'red', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Episode')
    plt.title('Sample Efficiency')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"figures/training_metrics_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close('all')

    # 4. Game Length and Learning Stability
    plt.figure(figsize=(12, 6))

    # Subplot 1: Turns per Episode
    plt.subplot(1, 2, 1)
    plt.plot(episodes, turns, 'b-', alpha=0.6, label='Turns per Episode')

    if len(turns) >= window_size:
        ma_turns = moving_average(turns, window_size)
        ma_episodes = episodes[window_size - 1:]
        plt.plot(ma_episodes, ma_turns, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Turns per Game')
    plt.title('Game Length Trend')
    plt.grid(alpha=0.3)
    plt.legend()

    # Subplot 2: Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(episodes, losses, 'green', alpha=0.6, label='Training Loss')

    if len(losses) >= window_size:
        ma_losses = moving_average(losses, window_size)
        ma_episodes = episodes[window_size - 1:]
        plt.plot(ma_episodes, ma_losses, 'darkgreen', linewidth=2, label=f'Moving Avg ({window_size})')

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"figures/game_analysis_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close('all')

    # 5. Performance Summary Statistics
    plt.figure(figsize=(10, 6))

    # Calculate milestones
    episodes_to_50_percent = None
    episodes_to_60_percent = None
    episodes_to_70_percent = None
    for i, rate in enumerate(cum_win_rate):
        if episodes_to_50_percent is None and rate >= 0.5:
            episodes_to_50_percent = i + 1
        if episodes_to_60_percent is None and rate >= 0.6:
            episodes_to_60_percent = i + 1
        if episodes_to_70_percent is None and rate >= 0.7:
            episodes_to_70_percent = i + 1

    final_win_rate = cum_win_rate[-1]
    avg_turns = turns.mean()
    total_reward = rewards.sum()
    final_loss = losses[-1]

    plt.subplot(1, 2, 1)
    performance_data = [
        final_win_rate,
        avg_turns / 100.0,
        (total_reward / num_completed),
        (1.0 - final_loss),
    ]
    performance_labels = [
        'Final Win Rate',
        'Avg Turns/100',
        'Avg Reward',
        'Learning Stability'
    ]
    colors = ['green', 'blue', 'purple', 'orange']

    bars = plt.bar(performance_labels, performance_data, color=colors, alpha=0.7)
    plt.title('Final Performance Summary')
    plt.ylabel('Normalized Values')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, performance_data):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    milestones = []
    milestone_episodes = []
    if episodes_to_50_percent:
        milestones.append('50% Win Rate')
        milestone_episodes.append(episodes_to_50_percent)
    if episodes_to_60_percent:
        milestones.append('60% Win Rate')
        milestone_episodes.append(episodes_to_60_percent)
    if episodes_to_70_percent:
        milestones.append('70% Win Rate')
        milestone_episodes.append(episodes_to_70_percent)

    if milestones:
        plt.barh(milestones, milestone_episodes, color='skyblue', alpha=0.7)
        plt.xlabel('Episodes to Achieve')
        plt.title('Learning Milestones')
        for i, ep_count in enumerate(milestone_episodes):
            plt.text(ep_count + max(milestone_episodes) * 0.01, i,
                     f'{ep_count}', va='center', ha='left')
    else:
        plt.text(0.5, 0.5, 'No milestones\nachieved yet',
                 ha='center', va='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.title('Learning Milestones')

    plt.tight_layout()
    plt.savefig(f"figures/performance_summary_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close('all')

    # Print summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Total Episodes: {num_completed}")
    print(f"Final Win Rate: {final_win_rate:.3f}")
    print(f"Average Game Length: {avg_turns:.1f} turns")
    print(f"Total Reward Earned: {total_reward:.2f}")
    print(f"Average Reward per Episode: {(total_reward/ num_completed):.3f}")
    print(f"Final Training Loss: {final_loss:.4f}")
    if episodes_to_50_percent:
        print(f"Episodes to 50% win rate: {episodes_to_50_percent}")
    if episodes_to_60_percent:
        print(f"Episodes to 60% win rate: {episodes_to_60_percent}")
    if episodes_to_70_percent:
        print(f"Episodes to 70% win rate: {episodes_to_70_percent}")

    return timestamp
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
        self.win_history = []
        
        self.episode_rewards = []
        self.episode_turns  = []
        self.episode_losses = []

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
