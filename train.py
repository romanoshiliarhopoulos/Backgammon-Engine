#!/usr/bin/env python3
import random
import sys, os
import time

root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]
import encode_state #type: ignore
import torch
import matplotlib.pyplot as plt # type: ignore
import backgammon_env as bg  # type: ignore
from BckgammonNet    import SeqBackgammonNet
import encode_state
from encode_state import build_sequence_mask
import torch.nn.functional as F
from torch.distributions import Categorical
import cProfile, pstats


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
from tqdm import trange 

# ensure C++ extension is on path
d = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(d, "build"), os.path.join(d, "pysrc")]
import backgammon_env as bg  # type: ignore


# ----------------------------------------
# plot_all_metrics (handles empty data gracefully)
# ----------------------------------------

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


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    net = SeqBackgammonNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    global num_completed
    num_completed = 0
    gamma, c1 = 0.99, 0.5
    num_episodes = 1000

    episode_turns = []
    episode_rewards = []
    episode_losses = []
    win_history = []

    encode_state._seq_cache_hits = 0
    encode_state._seq_cache_misses = 0

    try:
        for ep in trange(1, num_episodes + 1, desc="Training", unit="ep"):
            game = bg.Game(4)
            p1 = bg.Player("RLAgent", bg.PlayerType.PLAYER1)
            p2 = bg.Player("RandomBot", bg.PlayerType.PLAYER2)
            game.setPlayers(p1, p2)

            logps, vals, rews = [], [], []
            agent_rews = []
            turn_count = 0

            while True:
                turn_count += 1
                die1, die2 = game.roll_dice()
                idx = game.getTurn()
                player = p1 if idx == 0 else p2

                # Fetch mask, seqs, dice_orders, and index tensors
                mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game,
                    player,
                    batch_size=1,
                    device=device,
                    max_steps=net.max_steps
                )

                if idx == 0:
                    # Encode state
                    board = list(game.getGameBoard())
                    ja1, bo1 = game.getJailedCount(0), game.getBornOffCount(0)
                    ja2, bo2 = game.getJailedCount(1), game.getBornOffCount(1)
                    jailed, borne = (ja1, bo1) if idx == 0 else (ja2, bo2)
                    state = encode_state.encode_state(board, jailed, borne, idx)
                    state = state.unsqueeze(0).to(device)

                    is_over, winner = game.is_game_over()
                    if is_over:
                        break

                    if len(seqs) == 0:
                        game.setTurn(1 - idx)
                        continue

                    logits, value = net(state, masks=mask)  # logits: [1, T, N]

                    # Vectorized sequence scoring
                    step_logits = logits.squeeze(0)         # [T, N]
                    T, N = step_logits.shape
                    flat_logits = step_logits.view(-1)      # [T*N]

                    index_tensor = all_t * N + all_flat     # [M, T]
                    gathered = flat_logits[index_tensor]     # [M, T]
                    gathered = gathered * valid_mask.float()
                    seq_logits = gathered.sum(dim=1)        # [M]

                    # Safe softmax: guard against nan/inf
                    probs_seq = F.softmax(seq_logits, dim=0)
                    if torch.isnan(probs_seq).any():
                        M = seq_logits.size(0)
                        probs_seq = torch.ones(M, device=device) / M
                    else:
                        probs_seq = probs_seq.clamp(min=1e-8)
                        probs_seq = probs_seq / probs_seq.sum()

                    choice_tensor = torch.multinomial(probs_seq, num_samples=1)
                    choice = choice_tensor.item()
                    logp_seq = torch.log(probs_seq[choice]).unsqueeze(0)

                    logps.append(logp_seq)
                    vals.append(value)

                else:
                    if len(seqs) == 0:
                        game.setTurn(1 - idx)
                        continue
                    choice = random.randrange(len(seqs))

                # Execute chosen sequence
                if die1 == die2:
                    remaining_dice = [die1] * 4
                else:
                    remaining_dice = dice_orders[choice].copy()

                off_idx = net.S - 1

                for (o, d) in seqs[choice]:
                    if o == 0 and d == 0:
                        continue
                    pip = abs(d - o)
                    chosen_die = None
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

                is_over, winner = game.is_game_over()
                r = 1.0 if (is_over and winner == 0) else 0.0
                rews.append(r)
                if idx == 0:
                    agent_rews.append(r)

                if not is_over:
                    game.setTurn(1 - idx)
                    continue
                else:
                    break

            episode_turns.append(turn_count)

            is_over, winner = game.is_game_over()
            rl_won = 1 if winner == 0 else 0
            win_history.append(rl_won)

            episode_total_reward = sum(rews)
            episode_rewards.append(episode_total_reward)

            # Compute returns
            returns = []
            G = 0.0
            for r in reversed(agent_rews):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, device=device)

            logps = torch.stack(logps)
            vals = torch.cat(vals)
            advs = returns - vals.detach()

            policy_loss = -(logps * advs).mean()
            value_loss = F.mse_loss(vals, returns)
            loss = policy_loss + c1 * value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            episode_losses.append(loss.item())
            num_completed += 1

    except Exception as e:
        print(f"\nTraining interrupted by error: {e}")
        print("Saving progress up to interruption...")
        timestamp = plot_all_metrics(
            win_history,
            episode_rewards,
            episode_turns,
            episode_losses,
            num_completed
        )
        print(f"Progress plots saved with timestamp: {timestamp}")

        model_path = f"model_checkpoint_{timestamp}.pth"
        torch.save(net.state_dict(), model_path)
        print(f"Model checkpoint saved to {model_path}")
        sys.exit(1)

    print("\nGenerating training analysis plots...")
    timestamp = plot_all_metrics(
        win_history,
        episode_rewards,
        episode_turns,
        episode_losses,
        num_completed
    )
    print(f"Plots saved with timestamp: {timestamp}")
    print(f"=== build_sequence_mask cache hits: {encode_state._seq_cache_hits}, misses: {encode_state._seq_cache_misses} ===")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats("cumtime")
    ps.print_stats(20)