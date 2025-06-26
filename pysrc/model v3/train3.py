import torch
import torch.optim as optim
from model3 import TDGammonModel
from tqdm import trange
import tqdm
import backgammon_env as bg

def play_game(model):
    """
    Plays a single game of backgammon using the provided model for move selection.

    Returns:
        winner: bg.PlayerType of the winning player
        states: list of torch.Tensor state encodings for PLAYER1's turns
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

    states = []
    total_moves = 0

    # Main game loop
    while True:
        #game.printGameBoard()
        # Roll dice for this turn
        dice = game.roll_dice()
        #print(f"Dice: {dice} | Turn: {game.getTurn()}")
        # Get all legal move sequences for the current turn
        actions = game.legalTurnSequences(game.getTurn(), dice[0], dice[1])

        if actions:
            values = []
            # Evaluate each action by cloning and using the model
            for seq in actions:
                clone = game.clone()
                okay = True
                for origin, dest in seq:
                    die_used = abs(origin - dest)
                    player_obj = turn_player[clone.getTurn()]
                    success, _ = clone.tryMove(player_obj, int(die_used), origin, dest)
                    if not success:
                        okay = False
                        break
                if not okay:
                    # if clone move illegal, assign worst value
                    if game.getTurn() == bg.PlayerType.PLAYER1:
                        values.append(float('-inf'))
                    else:
                        values.append(float('inf'))
                    continue
                # encode and evaluate
                rep = model.encode_state(clone).to(device)
                with torch.no_grad():
                    val = model(rep.unsqueeze(0)).item()
                values.append(val)

            # Choose best move for each player
            if game.getTurn() == bg.PlayerType.PLAYER1:
                idx = max(range(len(values)), key=lambda i: values[i])
            else:
                idx = min(range(len(values)), key=lambda i: values[i])
            best_seq = actions[idx]

            # Apply best move to the actual game
            for origin, dest in best_seq:
                die_used = abs(origin - dest)
                player_obj = turn_player[game.getTurn()]
                game.tryMove(player_obj, int(die_used), origin, dest)

            # Record state for PLAYER1 after move
            if game.getTurn() == bg.PlayerType.PLAYER1:
                states.append(model.encode_state(game).to(device))

        # Check for game end
        over, winner = game.is_game_over()
        if over:
            #print(f"Total moves: {total_moves}")
            return winner, states

        # Switch turn
        next_turn = bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1 else bg.PlayerType.PLAYER1
        game.setTurn(next_turn)
        total_moves+=1


def train(num_games=1000, lr=1e-2):
    """
    Trains the TDGammonModel using TD(0) learning over a number of games.

    Args:
        num_games: Number of self-play games to simulate
        lr: Learning rate for the SGD optimizer

    Returns:
        model: The trained TDGammonModel instance
    """
    model = TDGammonModel()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    wins = 0
    for i in trange(1, num_games + 1, desc="Games"):
        winner, states = play_game(model)
        if winner == bg.PlayerType.PLAYER1:
            wins += 1

        # Perform TD updates
        model.train()
        # Update for intermediate states
        for t in range(len(states) - 1):
            v_t = model(states[t].unsqueeze(0))
            v_tp1 = model(states[t + 1].unsqueeze(0)).detach()
            loss = (v_tp1 - v_t).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update for terminal state
        if states:
            terminal_value = torch.tensor([[1.0]]) if winner == bg.PlayerType.PLAYER1 else torch.tensor([[0.0]])
            v_t = model(states[-1].unsqueeze(0))
            loss = (terminal_value - v_t).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            #tqdm.write(f"Game {i}, Win rate: {wins/i:.3f}")
            pass


    print(f"Training completed. Final win rate: {wins / num_games:.3f}")
    return model


def main():
    model = train(num_games=2000)
    # save the model
    torch.save(model.state_dict(), "tdgammon_model.pth")
    print("Model saved to tdgammon_model.pth")

if __name__ == "__main__":
    main()
