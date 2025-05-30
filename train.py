#!/usr/bin/env python3
import sys, os
import time
root = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(root, "build"), os.path.join(root, "pysrc")]

import torch
import matplotlib.pyplot as plt # type: ignore
import backgammon_env as bg  # type: ignore
from BckgammonNet    import BackgammonNet
from encode_state    import encode_state, build_legal_mask
import torch.nn.functional as F

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = BackgammonNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    num_episodes = 1000
    gamma        = 0.99
    c1           = 0.5

    #for some statistics
    num_turns_per_episode = []

    for episode in range(1, num_episodes+1):
        ep_start = time.time()
        print(f"\n=== Episode {episode}/{num_episodes} ===")
        game = bg.Game(4)
        p1   = bg.Player("RLAgent",  bg.PlayerType.PLAYER1)
        p2   = bg.Player("RandomBot",bg.PlayerType.PLAYER2)
        game.setPlayers(p1, p2)

        log_probs  = []
        values     = []
        rewards    = []

        turn_count = 0
        while True:
            game.printGameBoard()
            turn_count += 1

            # roll the dice for this turn
            die1, die2 = game.roll_dice()
            turn       = game.getTurn()
            print(f"[Ep {episode} Turn {turn_count}] Rolled dice: ({die1}, {die2}): player ({turn})")

            # 1) encode state
            board = list(game.getGameBoard())
            jailed_p1  = game.getJailedCount(0)
            jailed_p2  = game.getJailedCount(1)
            born_off_p1= game.getBornOffCount(0)
            born_off_p2= game.getBornOffCount(1)
            jailed, borne_off = (jailed_p1, born_off_p1) if turn==0 else (jailed_p2, born_off_p2)

            state = encode_state(board, jailed, borne_off, turn)
            state = state.unsqueeze(0).to(device)   # [1,6,24]

            # 2) get legal mask
            legal_mask = build_legal_mask(game, batch_size=1, device=device)
            legal = legal_mask[0].sum().item()
            print(f"[debug] found {legal} legal (o→d) pairs this turn")

            n_legal = legal_mask.sum().item()
            if n_legal == 0:
                print(f"[Ep {episode} Turn {turn_count}] ⚠️ No legal moves: skipping turn")
                game.printGameBoard()
                # swap turn manually, or let your C++ env handle it if you add a pass() call there
                # manually hand the turn to the other player
                turn = game.getTurn()
                game.setTurn(1 - turn)
                continue

            # 3) forward
            probs, value = net(state, legal_mask)

            # 4) sample action
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()            # int in [0,576)
            log_prob = dist.log_prob(action)
            o, d = divmod(action.item(), 26)
            print(f"[Ep {episode} Turn {turn_count}] Sampled action → move {o}→{d}, logp={log_prob.item():.3f}")

            # 5) step env
            current_player = p1 if turn==0 else p2
            success, err = game.tryMove(current_player, die1, o, d)
            if not success:
                print(f"   ⚠️  Invalid move: {err!r}, retrying turn {turn_count}")
                continue  # you may want to handle retries differently

            # 6) record
            log_probs.append(log_prob)
            values.append(value)

            is_over, winner = game.is_game_over()
            if is_over:
                reward = 1.0 if winner==turn else -1.0
                rewards.append(reward)
                print(f"[Ep {episode} Turn {turn_count}] Game over! Winner = Player {winner}, reward = {reward}")
                break
            else:
                rewards.append(0.0)  # no intermediate reward
            game.setTurn(1 - turn)

        # --- end of episode, do learning update ---
        num_turns_per_episode.append(turn_count)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=device)

        values    = torch.cat(values)
        log_probs = torch.stack(log_probs)
        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss  = F.mse_loss(values, returns)
        loss = policy_loss + c1 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_time = time.time() - ep_start
        print(f"=== Episode {episode} done in {turn_count} turns ({ep_time:.2f}s). Loss = {loss.item():.4f} ===")
    
    #plot the turn_counts per episode
    plt.plot(num_turns_per_episode, linestyle='--', marker='o', color='red')
    plt.xlabel('Index')
    plt.ylabel('Turns per game')
    plt.title('Trend of turns per game')
    plt.savefig('turns_trend.png',    # filename (extension sets the format)
            dpi=300,              # resolution in dots-per-inch
            bbox_inches='tight') 

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
