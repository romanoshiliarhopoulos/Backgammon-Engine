import torch
import torch.optim as optim
from model import TDLGammonModel
from tqdm import trange
import tqdm
import os
import backgammon_env as bg #type: ignore
import matplotlib.pyplot as plt

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

def load_previous_model(checkpoint_path="tdgammonNEW10k.pth"):
    """Load previous model weights if they exist, otherwise start fresh"""
    model = TDLGammonModel()
    
    if os.path.exists(checkpoint_path):
        print(f" Loading previous model from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            print(" Successfully loaded previous model weights")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Starting with fresh weights instead")
    else:
        print(f"No previous model found at {checkpoint_path}")
        print("Starting training from scratch")
    
    return model

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


def train(num_games = 1000, initial_lr = 0.1, checkpoint_path="tdgammonNEW10k.pth"):
    """
    Trains the TDGammonModel using TD(lambda) learning over a number of games.

    Args:
        num_games: Number of self-play games to simulate
        lr: Learning rate 

    Returns:
        model: The trained TDGammonModel instance
    """

    model = load_previous_model(checkpoint_path)


    #optimizer definition: SGD (Stochastic Grad Descent)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    # Definitions for learning statistics
    wins = 0
    all_losses = []          # collects every single TD‐loss
    episode_losses = []      # collects one average loss per game
    number_of_turns = []     #holds the number of moves at each game

    for i in trange(1, num_games + 1, desc="Games"):
        model.update_learning_params(i)

        #Reset traces at start of each game:
        for name in model.eligibility_traces:
            model.eligibility_traces[name].zero_()

        winner, states, total_moves = play_game(model, i)

        number_of_turns.append(total_moves)
        if winner == bg.PlayerType.PLAYER1:
            wins += 1

        
        model.train()
        losses_this_episode = []

        for t in range(len(states)-1):
            current_state, current_player = states[t]
            next_state, next_player = states[t + 1]

            model.eval()
            with torch.no_grad():
                #Feed the next game state through the neural network to get its value estimate.
                v_next = model(next_state.unsqueeze(0))
            
            model.train()

            #Feed the next next state through the neural network to get its value estimate.
            v_current = model(current_state.unsqueeze(0))

            td_error = (v_next - v_current).item()  # scalar
            
            # Zero gradients and compute gradients for current value
            optimizer.zero_grad()
            v_current.backward()
            
            # Update eligibility traces and params using TD(λ)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Update eligibility trace: e = λ * e + ∇V
                        model.eligibility_traces[name] = (
                            model.lambda_decay * model.eligibility_traces[name] + 
                            param.grad.data
                        )
                        
                        # Update param
                        param.add_(model.learning_rate * td_error * model.eligibility_traces[name])
            
            # Store loss
            losses_this_episode.append(td_error ** 2)
        
        #terminal state update at the end of the game
        if states:
            final_state, final_player = states[-1]
            model.train()
            v_final = model(final_state.unsqueeze(0))
            
            # Terminal reward
            terminal_reward = 1.0 if winner == bg.PlayerType.PLAYER1 else 0.0
            td_error = terminal_reward - v_final.item()
            
            optimizer.zero_grad()
            v_final.backward()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        model.eligibility_traces[name] = (
                            model.lambda_decay * model.eligibility_traces[name] + 
                            param.grad.data
                        )
                        param.add_(model.learning_rate * td_error * model.eligibility_traces[name])

        # record losses
        all_losses.extend(losses_this_episode)
        if losses_this_episode:
            episode_losses.append(sum(losses_this_episode) / len(losses_this_episode))

    print(f"Training completed. Final win rate: {wins / num_games:.3f}")
    plot_all_metrics(number_of_turns, episode_losses, "plots.png")
    return model


def main():
    model = train(num_games=20000)
    # save the model
    torch.save(model.state_dict(), "tdgammonNEW20k.pth")
    print("Model saved to tdgammon_model.pth")

if __name__ == "__main__":
    main()
