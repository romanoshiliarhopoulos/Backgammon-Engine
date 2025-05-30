#!/usr/bin/env python3
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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # type: ignore
from torch.distributions import Categorical

# ensure C++ extension is on path
d = os.path.dirname(__file__)
sys.path[:0] = [os.path.join(d, "build"), os.path.join(d, "pysrc")]
import backgammon_env as bg  # type: ignore

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

        assert len(seqs) == len(first_branch) + len(second_branch)
        dice_orders = [[die1, die2]] * len(first_branch) \
                    + [[die2, die1]] * len(second_branch)

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

    num_episodes = 30
    gamma, c1    = 0.99, 0.5
    episode_turns = []

    for ep in range(1, num_episodes+1):
        print(f"\n=== Episode {ep}/{num_episodes} ===")
        game = bg.Game(4)
        p1   = bg.Player("RLAgent",   bg.PlayerType.PLAYER1)
        p2   = bg.Player("RandomBot", bg.PlayerType.PLAYER2)
        game.setPlayers(p1, p2)

        logps, vals, rews = [], [], []
        turn_count = 0

        while True:
            game.printGameBoard()
            turn_count += 1
            die1, die2 = game.roll_dice()
            idx = game.getTurn()
            player = p1 if idx==0 else p2
            print(f"[Turn {turn_count}] Rolled ({die1},{die2}) for P{idx}")

            # encode
            board      = list(game.getGameBoard())
            ja1, bo1   = game.getJailedCount(0), game.getBornOffCount(0)
            ja2, bo2   = game.getJailedCount(1), game.getBornOffCount(1)
            jailed, borne = (ja1, bo1) if idx==0 else (ja2, bo2)
            state = encode_state(board, jailed, borne, idx)
            state = state.unsqueeze(0).to(device)

            # mask + sequences
            mask, seqs, dice_orders = build_sequence_mask(game, player, batch_size=1,
                                            device=device,
                                            max_steps=net.max_steps)
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
            logp_seq  = dist_seq.log_prob(torch.tensor(choice, device=device))
            print(f"→ Chosen turn‐sequence [{choice}]: ", end="")
            for (o,d) in seqs[choice]:
                print(f"{o}->{d} ", end="")
            print()  # newline

            if die1 == die2:
                # doubles → you get four moves of that pip
                remaining_dice = [die1] * 4
            else:
                # use exactly the same order C++ used
                remaining_dice = dice_orders[choice].copy()

            off_idx = net.S - 1        # 25, the “borne off” index

            for (o, d) in seqs[choice]:
                pip = abs(d - o)
                chosen_die = None

                # look for an exact match or, on a bearing‑off, any die ≥ pip
                for i, die in enumerate(remaining_dice):
                    is_bearing_off = (d == off_idx)
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
            logps.append(logp_seq)
            vals.append(value)
            is_over, winner = game.is_game_over()
            rews.append(1.0 if (is_over and winner==idx) else 0.0)
            if not is_over:
                game.setTurn(1-idx)
                continue
            else:
                break

        episode_turns.append(turn_count)
        print(f"Episode done in {turn_count} turns.")

        # returns & update
        returns = []
        G = 0.0
        for r in reversed(rews):
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

        print(f"Loss: {loss.item():.4f}")

    # plot
    plt.plot(episode_turns, '--o')
    plt.xlabel("Episode")
    plt.ylabel("Turns")
    plt.title("Turns per Game")
    plt.savefig("turns_trend.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()