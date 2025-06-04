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
# plot_all_metrics 
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

def pip_sum(board, player):
    """
    board: list of 24 integers, positive = P1 checkers, negative = P2 checkers
    player: 0 or 1
    Return the sum of “pip distances” for that players 15 checkers:
      - for P1, pip distance of a single checker on point i is (25 - i)
      - for P2, pip distance is (i)
    """
    s = 0
    if player == 0:  # P1
        for i, cnt in enumerate(board, start=1):
            if cnt > 0:
                s += (25 - i) * cnt
    else:  # P2
        for i, cnt in enumerate(board, start=1):
            if cnt < 0:
                s += i * abs(cnt)
    return s


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    net = SeqBackgammonNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4) 

    global num_completed
    num_completed = 0
    gamma, c1 = 0.99, 0.5
    num_episodes = 500

    episode_turns = []
    episode_rewards = []
    episode_losses = []
    win_history = []

    encode_state._seq_cache_hits = 0
    encode_state._seq_cache_misses = 0

    try:
        for ep in trange(1, num_episodes + 1, desc="Training", unit="ep"):
            game = bg.Game(4)
            p1 = bg.Player("Agent1", bg.PlayerType.PLAYER1)
            p2 = bg.Player("Agent2", bg.PlayerType.PLAYER2)
            game.setPlayers(p1, p2)

            # ─── TD(0) / Actor‐Critic bookkeeping ───
            prev_value = None                 # CHANGED: to store V(s_t)
            prev_logp = None                  # CHANGED: to store logπ(a_t|s_t)
            prev_r = 0.0                      # CHANGED: to store r_t from P1 action
            pending_update = False            # CHANGED: whether there's a pending TD update
            update_count = 0                  # CHANGED: count how many updates this episode
            loss_sum = 0.0                    # CHANGED: accumulate loss over updates
            rews = []                         # keep rewards per P1 move (for total reward)
            turn_count = 0

            while True:
                turn_count += 1
                die1, die2 = game.roll_dice()
                idx = game.getTurn()
                player = p1 if idx == 0 else p2

                # Fetch mask, seqs, dice_orders, and index tensors for the current player
                mask, seqs, dice_orders, all_t, all_flat, valid_mask = build_sequence_mask(
                    game,
                    player,
                    batch_size=1,
                    device=device,
                    max_steps=net.max_steps
                )

                # If there are no legal sequences, skip to the other player.
                if len(seqs) == 0:
                    game.setTurn(1 - idx)
                    continue

                # ────────────── Agent1 (idx == 0) ──────────────
                if idx == 0:
                    # Encode state for P1
                    board = list(game.getGameBoard())
                    ja1, bo1 = game.getJailedCount(0), game.getBornOffCount(0)
                    state = encode_state.encode_state(board, ja1, bo1, idx)
                    state = state.unsqueeze(0).to(device)

                    # Check for game over before picking an action
                    # Check if game ended on P2's previous move
                    is_over, winner = game.is_game_over()
                    if is_over:
                        if pending_update:
                            # CHANGED: terminal update for P1 when game ended on P2's move
                            td_target = 1.0 if (winner == 0) else -1.0
                            td_error = td_target - prev_value
                            value_loss = td_error.pow(2)
                            policy_loss = -prev_logp * td_error.detach()
                            loss = policy_loss + c1 * value_loss

                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                            opt.step()

                            loss_sum += loss.item()
                            update_count += 1
                            pending_update = False
                        break

                    # CHANGED: If there's a pending TD update from the last P1 action, do it now
                    if pending_update:
                        with torch.no_grad():
                            # Get value of current state V(s_{t+1})
                            _, V_t1 = net(state, masks=mask)
                        td_target = prev_r + gamma * V_t1
                        td_error = td_target - prev_value
                        value_loss = td_error.pow(2)
                        policy_loss = -prev_logp * td_error.detach()
                        loss = policy_loss + c1 * value_loss

                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        opt.step()

                        loss_sum += loss.item()
                        update_count += 1
                        pending_update = False

                    # Forward pass through network to get logits and value
                    logits, value = net(state, masks=mask)  # logits: [1, T, N]

                    # Vectorized sequence scoring (exact same as before)
                    step_logits = logits.squeeze(0)  # [T, N]
                    T, N = step_logits.shape
                    flat_logits = step_logits.view(-1)  # [T*N]

                    index_tensor = all_t * N + all_flat  # [M, T]
                    gathered = flat_logits[index_tensor]  # [M, T]
                    gathered = gathered * valid_mask.float()
                    seq_logits = gathered.sum(dim=1)  # [M]

                    # Safe softmax (avoid NaNs)
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

                    # CHANGED: store current policy/logp and value for later TD update
                    prev_value = value
                    prev_logp = logp_seq

                    # Save "previous" state variables for reward calculation (P1 perspective)
                    prev_board = list(game.getGameBoard())
                    prev_born = game.getBornOffCount(0)
                    prev_pip = pip_sum(prev_board, 0)
                    prev_jailed_opponent = game.getJailedCount(1)

                # ────────────── Agent2 (idx == 1) ──────────────
                else:
                    # Encode state for P2, but wrap in no_grad so we do not compute gradients
                    board = list(game.getGameBoard())
                    ja2, bo2 = game.getJailedCount(1), game.getBornOffCount(1)
                    state = encode_state.encode_state(board, ja2, bo2, idx)
                    state = state.unsqueeze(0).to(device)

                    # Forward pass under no_grad
                    with torch.no_grad():
                        logits, _ = net(state, masks=mask)  # we ignore the value for P2
                        step_logits = logits.squeeze(0)  # [T, N]
                        T, N = step_logits.shape
                        flat_logits = step_logits.view(-1)  # [T*N]

                        index_tensor = all_t * N + all_flat  # [M, T]
                        gathered = flat_logits[index_tensor]  # [M, T]
                        gathered = gathered * valid_mask.float()
                        seq_logits = gathered.sum(dim=1)  # [M]

                        # Safe softmax
                        probs_seq = F.softmax(seq_logits, dim=0)
                        if torch.isnan(probs_seq).any():
                            M = seq_logits.size(0)
                            probs_seq = torch.ones(M, device=device) / M
                        else:
                            probs_seq = probs_seq.clamp(min=1e-8)
                            probs_seq = probs_seq / probs_seq.sum()

                        choice_tensor = torch.multinomial(probs_seq, num_samples=1)
                        choice = choice_tensor.item()
                        # We do NOT append logp or value for P2

                # ────────────── Execute the chosen sequence ──────────────
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

                # ────────────── Compute reward for P1 perspective ──────────────
                is_over, winner = game.is_game_over()
                if is_over:
                    # Terminal reward: +1 if P1 wins; -1 if P2 wins
                    r_total = 1.0 if (winner == 0) else -1.0
                else:
                    # Born-off reward for P1
                    new_born = game.getBornOffCount(0)
                    r_born = 0.05 * (new_born - prev_born)

                    # Pip-sum reward for P1
                    new_board = list(game.getGameBoard())
                    new_pip = pip_sum(new_board, 0)
                    r_pip = 0.001 * (prev_pip - new_pip)

                    # Capture (hit) reward: if P1 just hit P2
                    r_hit = 0.0
                    if prev_jailed_opponent < game.getJailedCount(1):
                        r_hit = 0.15

                    # Time penalty (to encourage shorter games)
                    r_time = -0.001

                    r_total = r_born + r_pip + r_hit + r_time

                # Append reward for P1 actions (for episode total)
                if idx == 0:
                    rews.append(r_total)

                # ─── If P1 just acted, set up for the per-step TD update ───
                if idx == 0:
                    if is_over:
                        # CHANGED: Immediate TD update if game ended on P1's move
                        td_target = r_total
                        td_error = td_target - prev_value
                        value_loss = td_error.pow(2)
                        policy_loss = -prev_logp * td_error.detach()
                        loss = policy_loss + c1 * value_loss

                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        opt.step()

                        loss_sum += loss.item()
                        update_count += 1
                        pending_update = False
                        break
                    else:
                        # CHANGED: schedule a TD update once we see the next P1 state
                        prev_r = r_total
                        pending_update = True

                # Switch turn and continue
                game.setTurn(1 - idx)
                continue

            # ────────────── After episode ends ──────────────
            episode_turns.append(turn_count)

            is_over, winner = game.is_game_over()
            rl_won = 1 if winner == 0 else 0
            win_history.append(rl_won)

            episode_total_reward = sum(rews)
            episode_rewards.append(episode_total_reward)

            # CHANGED: record average per-step loss (or zero if no updates)
            if update_count > 0:
                episode_losses.append(loss_sum / update_count)
            else:
                episode_losses.append(0.0)

            num_completed += 1
            # ─── Free unused MPS memory ───
            torch.mps.empty_cache()


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
    profiler.print_stats(20)
    profiler = pstats.Stats(profiler).sort_stats("cumtime")