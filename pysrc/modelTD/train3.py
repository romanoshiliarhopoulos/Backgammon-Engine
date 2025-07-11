import os
import sys

# ─── DLL fix: must be FIRST ─────────────────────────────────────────────────────
os.add_dll_directory("C:\\msys64\\ucrt64\\bin")

# ─── path bootstrap ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))

_search_paths = [
    os.path.join(_here, "..", "build"),
    os.path.join(_here, "..", "build", "Release"),
    os.path.join(_here, "..", "build", "Debug"),
]

for p in _search_paths:
    p = os.path.normpath(p)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

# ─── Now safe to import everything ─────────────────────────────────────────────
import torch
import torch.optim as optim
from model3 import TDGammonModel
from tqdm import trange
import tqdm
import matplotlib.pyplot as plt
import backgammon_env as bg


os.add_dll_directory("C:\\msys64\\ucrt64\\bin")
import backgammon_env as bg


def plot_all_metrics(game_length, td_loss, save_path=None):
    """Plots learning metrics """
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    
    # Plot game length
    axes[0].plot(game_length)
    axes[0].set_title("Game Length Over Episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Game Length")
    
    # Plot TD loss
    axes[1].plot(td_loss)
    axes[1].set_title("TD Loss Over Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("TD Loss")
    
    plt.tight_layout()
    
    # Save to PNG if requested
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()



def play_game(model, game_idx):
    """
    Plays a single game of backgammon using the provided model for move selection.

    Returns:
        winner: bg.PlayerType of the winning player
        states: list of torch.Tensor state encodings for PLAYER1's turns
        total_moves: Number of turns until end of game
    """
    device = next(model.parameters()).device
    model.eval()

    # Initialize game and players
    game = bg.Game(0)
    p1 = bg.Player("White", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Black", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    # Keep a mapping from turn code to the correct Player instance
    turn_player = {
        bg.PlayerType.PLAYER1: p1,
        bg.PlayerType.PLAYER2: p2
    }

    # Determine first turn by rolling dice
    p1_roll = game.roll_dice()
    p2_roll = game.roll_dice()
    while sum(p1_roll) == sum(p2_roll):
        p1_roll = game.roll_dice()
        p2_roll = game.roll_dice()
    if sum(p1_roll) > sum(p2_roll):
        game.setTurn(bg.PlayerType.PLAYER1)
    else:
        game.setTurn(bg.PlayerType.PLAYER2)

    states = [] # Store (state, player) tuples
    total_moves = 0

    # Main game loop
    while True:
        #game.printGameBoard()
        current_player = game.getTurn()
        state_encoding = model.encode_state(game).to(device)
        states.append((state_encoding, current_player))
        
        # Roll dice for this turn
        game.roll_dice()
        best_seq = model.make_move(game, game_idx)

        # Check for game end
        over, winner = game.is_game_over()
        if over:
            #print(f"Total moves: {total_moves}")
            return winner, states, total_moves

        # Switch turn
        next_turn = bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1 else bg.PlayerType.PLAYER1
        game.setTurn(next_turn)
        total_moves+=1



def train(num_games=1000, lr=1e-2):
    """
    Trains the TDGammonModel using TD(0) learning over a number of games.

    Args:
        num_games: Number of self-play games to simulate
        lr: Learning rate 

    Returns:
        model: The trained TDGammonModel instance
    """
    model = TDGammonModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    wins = 0
    #statistics for learning evaluation
    all_losses = []          # collects every single TD‐loss
    episode_losses = []      # collects one average loss per game
    number_of_turns = []     #holds the number of moves at each game

    for i in trange(1, num_games + 1, desc="Games"):
        winner, states, total_moves = play_game(model, i)

         # Debug: Check network outputs
        # In your training loop, add more detailed monitoring:
        
        number_of_turns.append(total_moves)
        if winner == bg.PlayerType.PLAYER1:
            wins += 1

        # Perform TD updates
        model.train()
        losses_this_episode = []

        # Update for intermediate states
        # Process all states in sequence
        for t in range(len(states) - 1):
            current_state, current_player = states[t]
            next_state, next_player = states[t + 1]
            
            model.eval()
            with torch.no_grad():
                v_next = model(next_state.unsqueeze(0))

            model.train()
            v_current = model(current_state.unsqueeze(0))
            
            # TD error: next prediction - current prediction
            td_error = v_next - v_current 
            # Adjust sign based on player perspective
            """if current_player == bg.PlayerType.PLAYER2:
                td_error = -td_error
            """
            loss = td_error.pow(2)
            losses_this_episode.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update for terminal state
        if states:
            final_state, final_player = states[-1]

            model.eval()
            v_final = model(final_state.unsqueeze(0))
            model.train()

            # Determine terminal reward
            terminal_target = torch.tensor([[1.0 if winner == bg.PlayerType.PLAYER1 else 0.0]])
            td_error = terminal_target - v_final
            
            loss = td_error.pow(2)
            losses_this_episode.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record losses
        all_losses.extend(losses_this_episode)
        if losses_this_episode:
            episode_losses.append(sum(losses_this_episode) / len(losses_this_episode))
    
    print(f"Training completed. Final win rate: {wins / num_games:.3f}")
    plot_all_metrics(number_of_turns, episode_losses, "plots.png")
    return model


def main():
    model = train(num_games=25000, lr=3e-4)
    # save the model
    torch.save(model.state_dict(), "tdgammon_model3000nosigmoid.pth")
    print("Model saved to tdgammon_model.pth")

if __name__ == "__main__":
    main()
