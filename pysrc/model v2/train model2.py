#!/usr/bin/env python3
#train_model2.py
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

import datetime
import matplotlib.pyplot as plt


# Ensure C++ extension is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pysrc')))

import backgammon_env as bg
from encode_state import encode_state, build_sequence_mask
from model2 import SeqBackgammonNet

# Data logging - set up
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
    
    # one axis for win‑based stats:
    episodes_win = np.arange(1, len(wins) + 1)
    # one axis for reward/turn/loss stats:
    episodes     = np.arange(1, len(rewards) + 1)


    window_size = max(5, len(wins) // 10)

    def moving_average(data, window):
        return torch.from_numpy(data).to(torch.float32).unfold(0, window, 1).mean(dim=1).cpu().numpy()

    # 1. Win Rate Analysis
    plt.figure(figsize=(12, 8))

    # Subplot 1: Cumulative Win Rate
    plt.subplot(2, 2, 1)
    cum_wins     = np.cumsum(wins)
    cum_win_rate = cum_wins / episodes_win
    plt.plot(episodes_win, cum_win_rate, 'b-', alpha=0.7, label='Cumulative Win Rate')
 

    if len(cum_win_rate) >= window_size:
        ma_win_rate  = moving_average(cum_win_rate, window_size)
        ma_episodes  = episodes_win[window_size - 1:]
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
        windowed_episodes = episodes_win[window_size - 1:]
        plt.plot(windowed_episodes, windowed_win_rate, 'g-', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel(f'Win Rate (Last {window_size} Episodes)')
    plt.title('Recent Win Rate Trend')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)

    # Cumulative Reward
    plt.subplot(2, 2, 3)
    cumulative_rewards = np.cumsum(rewards)
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
    final_loss = next((l for l in reversed(losses) if l != 0.0), losses[-1])

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
    state: torch.Tensor                                # [C,24]
    mask:  torch.Tensor                                # [max_steps, N]
    seqs:  List[List[Tuple[int,int]]]                  # all legal sequences at that turn
    dice_orders: List[List[int]]                       # matching dice orders
    action_idx: int                                    # which sequence‐index we actually picked, 0 ≤ action_idx < len(seqs)
    return_:    float
    turn : int
    reward: float
    
class RewardCalculator:
    """Calculate rewards for different game events"""
    
    # Reward constants
    WIN_REWARD = 1
    LOSS_REWARD = -1
    HIT_OPPONENT_REWARD = 0.2
    BEAR_OFF_REWARD = 0.4
    ESCAPE_JAR_REWARD = 0.05
    PROGRESS_REWARD = 0.01
    
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
                 device: str = 'cpu',
                 lr: float = 1e-5,
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
        self.win_history = [] # This stores 1/0 for each game

        self.episode_rewards = []
        self.episode_turns  = []
        self.episode_losses = []

        self.cache_clear_interval = 10  # Clear cache every 10 games

        

    def execute_move(self, sequence, dice_order, action_probs, current_player, game):
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
    
    def play_game(self, temperature=1.0):
        """Play a complete self-play game and collect experiences"""
        # Initialize game and players

        # ─── freeze normalization stats ───────────────────────────
        self.model.eval()

        game = bg.Game(0)  # Start with player 1
        player1 = bg.Player("RL model", bg.PlayerType.PLAYER1)
        player2 = bg.Player("RandomBot", bg.PlayerType.PLAYER2)
        game.setPlayers(player1, player2)
        
        experiences = []
        game_rewards = {0: [], 1: []}  # Track rewards for each player
        turn_count = 0
        total_episode_reward = 0.0  # Track total reward for this episode
        game_completed = False  # Track if game actually completed
        winner = None

        while True and turn_count< 500:
            if turn_count > 498:
                print("more than 500 turns")

            # Check if game is over
            is_over, winner = game.is_game_over()

            if is_over:
                # Assign final rewards                
                # Update all experiences with final rewards
                for exp in experiences:
                    exp.reward += (RewardCalculator.WIN_REWARD
                       if exp.turn == winner
                       else RewardCalculator.LOSS_REWARD)
                game_completed = True

                # Track wins for plotting (1 if player 0 wins, 0 if player 1 wins)
                self.win_rates.append(1 if winner == 0 else 0)
                self.win_history.append(1 if winner == 0 else 0)  # Add this line
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

            #for RL agent
            if current_player ==player1:    
                game_before = game.clone()
                sequence, dice_order, log_prob, entropy, mask, seqs, dice_orders, idx_t = self.model.select_action(game, current_player, temperature)
    
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
                    # Add to total episode reward
                    total_episode_reward += intermediate_reward
                    # Get value prediction
                    state_tensor = state.unsqueeze(0).to(self.device)
                    
                    _ , value_prediction_tensor = self.model(state_tensor, mask.unsqueeze(0).to(self.device))
                    value = value_prediction_tensor.item()
                    
                    # Store experience - detaching tensors so they dont carry old graphs
                    experience = GameExperience(
                        state        = state.detach().cpu(),
                        mask         = mask.cpu(),
                        seqs         = seqs,           # a small Python list, OK to stash
                        dice_orders  = dice_orders,
                        action_idx   = idx_t,
                        return_      = None,            # fill in after computing returns()
                        reward       = intermediate_reward,
                        turn         = current_player.getNum()
                    )
                    
                    experiences.append(experience)
                    game_rewards[current_player.getNum()].append(intermediate_reward)
            else:
                #random bot's turn - play a random legal move
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
                    game, player2, batch_size=1, device="cpu"
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
                status, num = self.execute_move(selected_sequence, dice_order, sequence_probs, current_player, game)
                
                if not status and num == 1 :
                    turn_count +=1
                    continue
                elif not status and num == 2:
                    break

            # Switch turns
            game.setTurn(1 - game.getTurn())
            turn_count +=1

        # ─── back to training mode so we can learn in train_step ──
        self.model.train()
        
        return experiences, total_episode_reward, turn_count
    
    

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
        """ More stable training step"""
        if len(self.experience_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Extract data
        # exp.state is [C, 24], stack makes [batch_size, C, 24]
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        # exp.mask is [max_steps, N] (boolean), stack makes [batch_size, max_steps, N] (boolean)
        masks = torch.stack([exp.mask for exp in batch]).to(self.device)
        returns = torch.tensor([exp.return_ for exp in batch], dtype=torch.float32).to(self.device)
        
        # Get current policy and value predictions
        logits, values = self.model(states, masks)
        values = values.squeeze(-1)
        
        # Initialize lists to collect valid log probabilities, entropies, and advantages
        valid_log_probs = []
        valid_entropies = []
        valid_advantages = [] 
        
        # _S is defined in model2.py, so access it via model
        _S = self.model.S 

        for i, exp in enumerate(batch):
            # Get logits for this sample
            sample_logits = logits[i]  # [max_steps, N]
            
            # --- DEBUGGING PRINT: Logits for the current sample ---
            if self.training_steps % 10 == 0 and i == 0: # Print for first sample every X steps
                logger.info(f"  [DEBUG {self.training_steps}] Sample {i} Logits (min/max): {sample_logits.min().item():.3f} / {sample_logits.max().item():.3f}")
            # --- END DEBUGGING PRINT ---

            # Compute sequence scores for all legal sequences for this experience.
            current_batch_seq_scores = [] 
            
            # --- DEBUGGING PRINT: Pre-stack seq_scores list ---
            if self.training_steps % 10 == 0 and i == 0: # Print for first sample every X steps
                logger.info(f"  [DEBUG {self.training_steps}] Sample {i} Processing Sequences:")
            # --- END DEBUGGING PRINT ---

            for seq_idx, seq in enumerate(exp.seqs):
                score = torch.tensor(0.0, device=self.device)
                
                # --- DEBUGGING PRINT: Individual sequence processing ---
                if self.training_steps % 10 == 0 and i == 0: # Print for first sample every X steps
                    logger.info(f"    [DEBUG]   Processing seq {seq_idx}: {seq}")
                # --- END DEBUGGING PRINT ---

                for t, (origin, dest) in enumerate(seq):
                    pos = origin * _S + dest
                    score_increment = sample_logits[t, pos] 
                    score += score_increment
                    
                    # --- DEBUGGING PRINT: Individual move within sequence ---
                    if self.training_steps % 10 == 0 and i == 0: # Print for first sample every X steps
                        logger.info(f"      [DEBUG]     Turn {t}, move ({origin},{dest}), pos={pos}, sample_logits[t,pos]={score_increment.item():.3f}, current seq score={score.item():.3f}")
                    # --- END DEBUGGING PRINT ---

                current_batch_seq_scores.append(score)
            
            # --- DEBUGGING PRINT: Final list before stacking for this sample ---
            if self.training_steps % 10 == 0 and i == 0: # Print for first sample every X steps
                logger.info(f"  [DEBUG {self.training_steps}] Sample {i} Current Batch Seq Scores list (before stack): {current_batch_seq_scores}")
            # --- END DEBUGGING PRINT ---

            if not current_batch_seq_scores:
                logger.warning(f"No legal sequences found for experience {i} in batch (game {self.games_played}, step {self.training_steps}). Skipping for policy/entropy update.")
                continue 

            seq_scores_for_dist = torch.stack(current_batch_seq_scores)
            
            # Critical check: If seq_scores_for_dist are all -inf, Categorical will fail.
            # This happens if the model's logits for all legal moves are -inf.
            if torch.all(torch.isinf(seq_scores_for_dist) & (seq_scores_for_dist < 0)):
                logger.warning(f"All sequence scores are -inf for batch item {i}. Cannot form Categorical distribution meaningfully. Skipping this experience.")
                continue

            if torch.isnan(seq_scores_for_dist).any():
                logger.warning(f"NaN detected in seq_scores for batch item {i}. Skipping this experience.")
                continue
            
            # This check is for cases where values are identical but not -inf (e.g., all 0.0)
            if seq_scores_for_dist.numel() > 1 and torch.all(seq_scores_for_dist == seq_scores_for_dist[0]):
                logger.warning(f"All sequence scores are identical for batch item {i}. This causes uniform distribution and potentially hampers learning. Adding small noise.")
                seq_scores_for_dist = seq_scores_for_dist + torch.randn_like(seq_scores_for_dist) * 1e-6


            dist = Categorical(logits=seq_scores_for_dist)
            
            valid_log_probs.append(dist.log_prob(exp.action_idx.to(self.device)))
            valid_entropies.append(dist.entropy())
            # IMPORTANT: Capture the advantages for *this specific* experience from the batch
            # `values[i]` and `returns[i]` correspond to the current experience `exp`
            valid_advantages.append(returns[i] - values[i].detach())
        
        if not valid_log_probs:
            logger.warning("No valid experiences in batch after filtering for policy and entropy loss calculation. Returning None.")
            return None

        # Stack only the collected valid items
        log_probs = torch.stack(valid_log_probs)
        entropies = torch.stack(valid_entropies)
        advantages = torch.stack(valid_advantages) # Stack advantages from valid experiences

        # Compute losses
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns) # values and returns are still full batch_size
                                                # but policy_loss, entropy_loss are from valid only.
                                                # This is a potential imbalance.
                                                # Ideally, value_loss should also only use valid items.
        
        # REVISED: Adjust value_loss to only use valid items
        # To do this, we need to store `values[i]` for valid_experiences too.
        # Let's adjust `valid_advantages` to just `valid_indices` or similar.
        
        # To make value_loss compatible with policy_loss from valid samples,
        # we need to select corresponding values and returns.
        # This requires storing the original indices of valid experiences, or reconstructing it.
        # A simpler fix for now: if many are skipped, this mismatch could be an issue.
        # However, for initial debugging, let's proceed and address this if it becomes problematic.
        # The RuntimeError was due to `log_probs` and `advantages` mismatch.
        # Currently, `values` and `returns` are still `batch_size`.
        # This means `value_loss` is calculated over the entire batch, while `policy_loss` is only over valid.
        # This is okay as long as `values` doesn't contain NaNs/Infs that would cause the MSE to fail.
        # If `values` or `returns` also cause NaNs, you'd need to filter them too.
        # For now, we address the direct Runtime Error.

        entropy_loss = -entropies.mean()

        total_loss = (self.policy_loss_weight * policy_loss
                    + self.value_loss_weight  * value_loss
                    + self.entropy_weight     * entropy_loss)

        # --- FINAL DEBUGGING PRINTS FOR TOTAL LOSS COMPONENTS ---
        if self.training_steps % 10 == 0: 
            logger.info(f"\n--- Training Step Summary (Step {self.training_steps}, Game {self.games_played}) ---")
            logger.info(f"  Logits (mean/std/min/max): {logits.mean().item():.3f} / {logits.std().item():.3f} / {logits.min().item():.3f} / {logits.max().item():.3f}")
            logger.info(f"  Entropies (mean/std/min/max): {entropies.mean().item():.3f} / {entropies.std().item():.3f} / {entropies.min().item():.3f} / {entropies.max().item():.3f}")
            logger.info(f"  Advantages (mean/std/min/max): {advantages.mean().item():.3f} / {advantages.std().item():.3f} / {advantages.min().item():.3f} / {advantages.max().item():.3f}")
            logger.info(f"  Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")
            logger.info(f"  Total Loss (before NaN/Inf check): {total_loss.item():.4f}")
            logger.info(f"----------------------------------------------------\n")

        # Check for NaN/Inf
        if not torch.isfinite(total_loss):
            logger.warning(f"Non-finite total loss detected: {total_loss.item()}, skipping update at step {self.training_steps}")
            return None

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.training_steps += 1

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss':  value_loss.item(),
            'entropy':     entropies.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'avg_return': returns.mean().item()
        }

    
    def train(self, num_games=1000, games_per_update=20, save_interval=100, plot_interval=500):
        """Main training loop"""
        logger.info(f"Starting training for {num_games} games")
        
        for game_idx in range(num_games):
            # Play game with temperature annealing
            temperature = min(0.5, 3.0 -  (3.0 - 0.3) * game_idx / num_games)
            experiences, episode_reward, episode_turns = self.play_game(temperature=temperature)

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_turns.append(episode_turns)
            
            # Also clear PyTorch cache - to prevent memory from growing indefinetely
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Compute returns and add to buffer
            returns = self.compute_returns(experiences)
            returns = torch.tensor(returns, device="cpu")
            # normalize
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for exp, R in zip(experiences, returns):
                exp.return_ = R
                self.experience_buffer.append(exp)
            
            self.games_played += 1
            
            # Training step
            current_loss = 0.0
            if game_idx % games_per_update == 0 and len(self.experience_buffer) >= self.batch_size and game_idx !=0:
                loss_info = self.train_step()
                current_loss = loss_info['total_loss'] if loss_info else 0.0
                
                win_rate = np.mean(self.win_rates) if self.win_rates else 0.0
                logger.info(f"Game {game_idx}: Loss={current_loss:.4f}, "
                        f"Win Rate={win_rate:.3f}, Temp={temperature:.3f}")
            
            # Store the loss for this episode
            self.episode_losses.append(current_loss)
            
            # Cache clearing
            if game_idx % self.cache_clear_interval == 0:
                #logger.info(f"Clearing sequence cache at game {game_idx}")
                from encode_state import clear_sequence_cache, get_cache_stats
                stats = get_cache_stats()
                #logger.info(f"Cache stats before clear: {stats}")
                clear_sequence_cache()

            # Generate plots at specified intervals
            if game_idx % plot_interval == 0 and game_idx > 0:
                logger.info(f"Generating training plots at game {game_idx}")
                timestamp = plot_all_metrics(
                    self.win_history, 
                    self.episode_rewards, 
                    self.episode_turns, 
                    self.episode_losses, 
                    game_idx + 1  # num_completed
                )
                logger.info(f"Plots saved with timestamp: {timestamp}")
            
            # Save model
            if game_idx % save_interval == 0 and game_idx > 0:
                self.save_model(f"model_checkpoint_{game_idx}.pt")
                logger.info(f"Model saved at game {game_idx}")
        
        # Generate final plots
        logger.info("Generating final training plots")
        final_timestamp = plot_all_metrics(
            self.win_history, 
            self.episode_rewards, 
            self.episode_turns, 
            self.episode_losses, 
            num_games
        )
        logger.info(f"Final plots saved with timestamp: {final_timestamp}")
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
    device = 'cpu' 
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
        lr=1e-5,
        buffer_size=10000,
        batch_size=32,
        value_loss_weight=1.0,
        policy_loss_weight=1.0,
        entropy_weight=0.0001
    )
    
    trainer.train(
        num_games=5000,
        games_per_update=30,
        save_interval=500,
        plot_interval=500  # Generate plots every 100 games
    )
    # Save final model
    trainer.save_model("final_model.pt")

if __name__ == "__main__":
    main()
