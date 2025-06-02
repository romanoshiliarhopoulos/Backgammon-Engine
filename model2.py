#!/usr/bin/env python3
import random
import sys, os
import time
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

import torch
import matplotlib.pyplot as plt # type: ignore
import backgammon_env as bg  # type: ignore
from BckgammonNet    import BackgammonNet
from encode_state    import build_legal_mask
import torch.nn.functional as F
from torch.distributions import Categorical

#!/usr/bin/env python3
import sys
import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # type: ignore
from torch.distributions import Categorical

# ensure C++ extension is on path
d = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(d, "build"), os.path.join(d, "pysrc")]
import backgammon_env as bg  # type: ignore

def plot_all_metrics(win_history, episode_rewards, episode_turns, episode_losses, num_episodes):
    """
    Create comprehensive plots for all training metrics and save with unique timestamps.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    # Convert to numpy arrays for easier manipulation
    wins = np.array(win_history, dtype=np.float32)
    rewards = np.array(episode_rewards, dtype=np.float32)
    turns = np.array(episode_turns, dtype=np.float32)
    losses = np.array(episode_losses, dtype=np.float32)
    episodes = np.arange(1, len(wins) + 1)
    
    # Calculate moving averages (window size = 10% of episodes, minimum 5)
    window_size = max(5, len(wins) // 10)
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 1. Win Rate Analysis
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Cumulative Win Rate
    plt.subplot(2, 2, 1)
    cum_wins = np.cumsum(wins)
    cum_win_rate = cum_wins / episodes
    plt.plot(episodes, cum_win_rate, 'b-', alpha=0.7, label='Cumulative Win Rate')
    
    # Add moving average if we have enough data
    if len(cum_win_rate) >= window_size:
        ma_episodes = episodes[window_size-1:]
        ma_win_rate = moving_average(cum_win_rate, window_size)
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
        windowed_episodes = episodes[window_size-1:]
        windowed_win_rate = moving_average(wins.astype(float), window_size)
        plt.plot(windowed_episodes, windowed_win_rate, 'g-', linewidth=2)
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random Performance')
    
    plt.xlabel('Episode')
    plt.ylabel(f'Win Rate (Last {window_size} Episodes)')
    plt.title('Recent Win Rate Trend')
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 2. Cumulative Reward Analysis
    plt.subplot(2, 2, 3)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(episodes, cumulative_rewards, 'purple', alpha=0.7, label='Cumulative Reward')
    
    if len(rewards) >= window_size:
        ma_episodes = episodes[window_size-1:]
        ma_rewards = moving_average(cumulative_rewards, window_size)
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
        ma_episodes = episodes[window_size-1:]
        ma_avg_rewards = moving_average(avg_reward_per_episode, window_size)
        plt.plot(ma_episodes, ma_avg_rewards, 'red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward per Episode')
    plt.title('Sample Efficiency')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/training_metrics_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 4. Game Length and Learning Stability
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Turns per Episode
    plt.subplot(1, 2, 1)
    plt.plot(episodes, turns, 'b-', alpha=0.6, label='Turns per Episode')
    
    if len(turns) >= window_size:
        ma_episodes = episodes[window_size-1:]
        ma_turns = moving_average(turns, window_size)
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
        ma_episodes = episodes[window_size-1:]
        ma_losses = moving_average(losses, window_size)
        plt.plot(ma_episodes, ma_losses, 'darkgreen', linewidth=2, label=f'Moving Avg ({window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/game_analysis_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 5. Performance Summary Statistics
    plt.figure(figsize=(10, 6))
    
    # Calculate performance milestones
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
    
    # Performance metrics text
    final_win_rate = cum_win_rate[-1]
    avg_turns = np.mean(turns)
    total_reward = np.sum(rewards)
    final_loss = losses[-1]
    
    # Create summary plot
    plt.subplot(1, 2, 1)
    performance_data = [final_win_rate, avg_turns/100, total_reward/num_episodes, (1-final_loss)]
    performance_labels = ['Final Win Rate', 'Avg Turns/100', 'Avg Reward', 'Learning Stability']
    colors = ['green', 'blue', 'purple', 'orange']
    
    bars = plt.bar(performance_labels, performance_data, color=colors, alpha=0.7)
    plt.title('Final Performance Summary')
    plt.ylabel('Normalized Values')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, performance_data):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Milestone achievements
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
        
        # Add value labels
        for i, episodes in enumerate(milestone_episodes):
            plt.text(episodes + max(milestone_episodes)*0.01, i, 
                    f'{episodes}', va='center', ha='left')
    else:
        plt.text(0.5, 0.5, 'No milestones\nachieved yet', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Learning Milestones')
    
    plt.tight_layout()
    plt.savefig(f"figures/performance_summary_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print summary statistics
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Total Episodes: {num_episodes}")
    print(f"Final Win Rate: {final_win_rate:.3f}")
    print(f"Average Game Length: {avg_turns:.1f} turns")
    print(f"Total Reward Earned: {total_reward:.2f}")
    print(f"Average Reward per Episode: {total_reward/num_episodes:.3f}")
    print(f"Final Training Loss: {final_loss:.4f}")
    
    if episodes_to_50_percent:
        print(f"Episodes to 50% win rate: {episodes_to_50_percent}")
    if episodes_to_60_percent:
        print(f"Episodes to 60% win rate: {episodes_to_60_percent}")
    if episodes_to_70_percent:
        print(f"Episodes to 70% win rate: {episodes_to_70_percent}")
    
    return timestamp
# -- feature encoding (unchanged) --
def encode_state(board, jailed, borne_off, turn):
    board_pts = torch.tensor(board, dtype=torch.float32)
    p1 = torch.clamp(board_pts, min=0).unsqueeze(0)
    p2 = torch.clamp(-board_pts, min=0).unsqueeze(0)
    jail_plane = torch.full((1,24), float(jailed))
    off_plane  = torch.full((1,24), float(borne_off))
    if turn == 1:
        cur = torch.cat([p1, jail_plane, off_plane], dim=0)
        opp = torch.cat([p2,
                         torch.full((1,24), float(board_pts.min().abs())),
                         torch.full((1,24), float(board_pts.max().abs()))], dim=0)
    else:
        cur = torch.cat([p2, jail_plane, off_plane], dim=0)
        opp = torch.cat([p1,
                         torch.full((1,24), float(board_pts.max().abs())),
                         torch.full((1,24), float(board_pts.min().abs()))], dim=0)
    return torch.cat([cur, opp], dim=0)


# -- network that outputs T-step logits at once --
class SeqBackgammonNet(nn.Module):
    def __init__(self, n_channels=6, hidden_dim=128, max_steps=4):
        super().__init__()
        self.conv1       = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2       = nn.Conv1d(32,       64, kernel_size=3, padding=1)
        self.fc_feat     = nn.Linear(64*24, hidden_dim)
        self.max_steps   = max_steps
        self.S           = 26
        self.N           = self.S * self.S
        # predict all T * (origin→dest) logits
        self.policy_logits = nn.Linear(hidden_dim, max_steps*self.N)
        self.value_head    = nn.Linear(hidden_dim, 1)

    def forward(self, x, masks=None):
        b = x.size(0)
        h = F.relu(self.conv1(x))       # [b,32,24]
        h = F.relu(self.conv2(h))       # [b,64,24]
        h = h.view(b, -1)               # [b,64*24]
        h = F.relu(self.fc_feat(h))     # [b,hidden_dim]

        logits = self.policy_logits(h)  # [b, T*676]
        logits = logits.view(b, self.max_steps, self.N)
        if masks is not None:
            logits = logits.masked_fill(~masks, float('-inf'))
        return logits, torch.tanh(self.value_head(h)).squeeze(-1)

def plot_win_rate(win_history):
    """
    Given win_history as a list of 0/1 (length = num_episodes),
    plot the cumulative win rate (episode on x−axis, win rate on y−axis).
    """
    wins = np.array(win_history, dtype=np.float32)      # e.g. [0,1,0,1,1,…]
    episodes = np.arange(1, len(wins) + 1)              # [1, 2, 3, …, num_episodes]
    cum_wins = np.cumsum(wins)                          # [wins[0], wins[0]+wins[1], …]
    cum_win_rate = cum_wins / episodes                  # [wins[0]/1, (wins[0]+wins[1])/2, …]

    plt.figure(figsize=(6,4))
    plt.plot(episodes, cum_win_rate, marker='o', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Win Rate vs. RandomBot")
    plt.title("RLAgent Win Rate Over Time")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/win_rate_1.png", dpi=300, bbox_inches="tight")
    plt.show()

# -- build a per-step mask from legalTurnSequences --
def build_sequence_mask(game, 
                        curr_player,
                        batch_size=1, 
                        device='cpu', 
                        max_steps=4):
    S = 26
    N = S*S
    mask = torch.zeros((batch_size, max_steps, N),
                       dtype=torch.bool, device=device)
    die1, die2 = game.get_last_dice()
    player     = game.getTurn()

    # get the combined list of sequences (exactly what C++ does)
    seqs = game.legalTurnSequences(player, die1, die2)

    dice_orders = []
    if die1 == die2:
        # doubles: every seq is four pips of die1
        dice_orders = [[die1]*4 for _ in seqs]
    else:
        # re‑build the two branches separately so we know which came first
        first_branch = []
        for m1 in game.legalMoves(player, die1):
            g1 = game.clone()
            # use the Python Player object!
            ok, err = g1.tryMove(curr_player, die1, m1[0], m1[1])
            for m2 in g1.legalMoves(player, die2):
                first_branch.append([m1, m2])

        second_branch = []
        for m1 in game.legalMoves(player, die2):
            g1 = game.clone()
            ok, err = g1.tryMove(curr_player, die2, m1[0], m1[1])
            for m2 in g1.legalMoves(player, die1):
                second_branch.append([m1, m2])

        if len(seqs) == 2 and len(first_branch) == 0 and len(second_branch) == 0:
            # C++ returned exactly two identical “dummy” sequences of the form {m1, {0,0}}.
            # We assign one ordering for “die1→die2” and one ordering for “die2→die1”.
            dice_orders = [[die1, die2], [die2, die1]]
        else:
            # Otherwise, do it exactly as before:
            dice_orders = [[die1, die2]] * len(first_branch) \
                        + [[die2, die1]] * len(second_branch)

            # As a safety, if C++ returned more seqs than we reconstructed,
            # fill in any missing entries with the default ordering [die1,die2].
            if len(dice_orders) < len(seqs):
                extra = len(seqs) - len(dice_orders)
                dice_orders += [[die1, die2]] * extra

    # fill the mask
    for b in range(batch_size):
        for i, seq in enumerate(seqs):
            for t, (o, d) in enumerate(seq):
                if t < max_steps:
                    mask[b, t, o*S + d] = True

    return mask, seqs, dice_orders


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    net = SeqBackgammonNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    num_episodes = 2000
    gamma, c1    = 0.99, 0.5

    # Track additional metrics
    episode_turns = []
    episode_rewards = []  # Track total reward per episode
    episode_losses = []   # Track loss per episode
    win_history = []  
    for ep in range(1, num_episodes+1):
        print(f"\n=== Episode {ep}/{num_episodes} ===")
        game = bg.Game(4)
        p1   = bg.Player("RLAgent",   bg.PlayerType.PLAYER1)
        p2   = bg.Player("RandomBot", bg.PlayerType.PLAYER2)
        game.setPlayers(p1, p2)

        logps, vals, rews = [], [], []
        agent_rews = []            # track rewards only for RLAgent’s turns

        turn_count = 0
        episode_total_reward = 0.0  # Track total reward for this episode

        while True:
            #game.printGameBoard()
            turn_count += 1
            die1, die2 = game.roll_dice()
            idx = game.getTurn()
            player = p1 if idx==0 else p2
            #print(f"[Turn {turn_count}] Rolled ({die1},{die2}) for P{idx}")

            # mask + sequences
            mask, seqs, dice_orders = build_sequence_mask(game, player, batch_size=1,
                                                device=device,
                                                max_steps=net.max_steps)
            
            if idx == 0:
                # encode
                board      = list(game.getGameBoard())
                ja1, bo1   = game.getJailedCount(0), game.getBornOffCount(0)
                ja2, bo2   = game.getJailedCount(1), game.getBornOffCount(1)
                jailed, borne = (ja1, bo1) if idx==0 else (ja2, bo2)
                state = encode_state(board, jailed, borne, idx)
                state = state.unsqueeze(0).to(device)

                # mask + sequences
                #mask, seqs, dice_orders = build_sequence_mask(game, player, batch_size=1,
                 #                               device=device,
                 #                               max_steps=net.max_steps)
                # if no legal sequences, first see if the game really is finished:
                is_over, winner = game.is_game_over()
                if is_over:
                    break
                # otherwise it was just a skipped turn (no moves), so hand turn to opponent:
                if len(seqs) == 0:
                    game.setTurn(1-idx)
                    continue

                # forward for raw logits + value
                logits, value = net(state, masks=mask)
                # compute sequence‐level logits by summing per‑step logits
                M = len(seqs)
                seq_logits = torch.empty((M,), device=device)
                for i, seq in enumerate(seqs):
                    s = 0.0
                    for t,(o,d) in enumerate(seq):
                        s = s + logits[0, t, o*net.S + d]
                    seq_logits[i] = s

                # sample one entire sequence
                probs_seq = F.softmax(seq_logits, dim=0)
                dist_seq  = Categorical(probs_seq)
                choice    = dist_seq.sample().item()
                #print(f"→ Chosen turn‐sequence [{choice}]: ", end="")
                #for (o,d) in seqs[choice]:
                    #print(f"{o}->{d} ", end="")
                #print()  # newline

                logp_seq = dist_seq.log_prob(torch.tensor(choice, device=device))
                 # detach the log‐prob so that only this scalar remains on device,
                # but it’s still a leaf with requires_grad=True
                logps.append(logp_seq.detach().requires_grad_(True))
                vals.append(value)
                

                #have a random play for randomBot
            else:
                if len(seqs) == 0:
                    game.setTurn(1 - idx)   # give turn back to RLAgent
                    continue

                # Otherwise, pick uniformly at random from the non‐empty seqs list:
                choice = random.randrange(len(seqs))
                #print(f"→ [RandomBot] Chosen sequence #{choice}:", end=" ")
                #for (o, d) in seqs[choice]:
                    #print(f"{o}->{d} ", end="")
                #print()
    

            #execute sequence
            if die1 == die2:
                # doubles → you get four moves of that pip
                remaining_dice = [die1] * 4
            else:
                # use exactly the same order C++ used
                remaining_dice = dice_orders[choice].copy()

            off_idx = net.S - 1        # 25, the “borne off” index

            for (o, d) in seqs[choice]:
                if o == 0 and d == 0:
                    continue
                pip = abs(d - o)
                chosen_die = None

                # look for an exact match or, on a bearing‑off, any die ≥ pip
                for i, die in enumerate(remaining_dice):
                    is_bearing_off = (idx == 0 and d == off_idx) or (idx == 1 and d == 0)
                    if pip == die or (is_bearing_off and die > pip):
                        chosen_die = die
                        remaining_dice.pop(i)
                        break

                assert chosen_die is not None, (
                    f"No valid die for move {o}->{d} (needed {pip}, had {remaining_dice})"
                )

                ok, err = game.tryMove(player, chosen_die, o, d)
                assert ok, f"Illegal {o}->{d} with die={chosen_die}: {err}"

            # record one logp+value+reward for the full-turn action
            
            is_over, winner = game.is_game_over()
            r = 1.0 if (is_over and winner == 0) else 0.0

            rews.append(r)
            if idx == 0:
                agent_rews.append(r)   #only track RLAgent’s reward here

            if not is_over:
                game.setTurn(1-idx)
                continue
            else:
                break

        episode_turns.append(turn_count)
        print(f"Episode done in {turn_count} turns.")
        print(f"Winner: {game.is_game_over()[1]}")
            
        # Determine winner correctly
        is_over, winner = game.is_game_over()
        rl_won = 1 if winner == 0 else 0
        win_history.append(rl_won)
            
        # Calculate total episode reward correctly
        episode_total_reward = sum(rews)
        episode_rewards.append(episode_total_reward)

        # returns & update
        returns = []
        G = 0.0
        for r in reversed(agent_rews):
            G = r + gamma*G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=device)

        logps = torch.stack(logps)
        vals   = torch.cat(vals)
        advs   = returns - vals.detach()

        policy_loss = -(logps * advs).mean()
        value_loss  = F.mse_loss(vals, returns)
        loss = policy_loss + c1*value_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
            
        # Record the loss for this episode
        episode_losses.append(loss.item())
        print(f"Loss: {loss.item():.4f}")


     # Call comprehensive plotting function
    print("\nGenerating training analysis plots...")
    timestamp = plot_all_metrics(win_history, episode_rewards, episode_turns, episode_losses, num_episodes)
    print(f"Plots saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()