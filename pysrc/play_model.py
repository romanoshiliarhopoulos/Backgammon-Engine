import os
import sys
import random
import torch
import questionary

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'TD(λ) model')))

from model import TDLGammonModel
import backgammon_env as bg

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def printBanner():
    print(r"""
·············································································
: ____    _    ____ _  ______    _    __  __ __  __  ___  _   _             :
:| __ )  / \  / ___| |/ / ___|  / \  |  \/  |  \/  |/ _ \| \ | |            :
:|  _ \ / _ \| |   | ' | |  _  / _ \ | |\/| | |\/| | | | |  \| |            :
:| |_) / ___ | |___| . | |_| |/ ___ \| |  | | |  | | |_| | |\  |  _   _   _ :
:|____/_/   \_\____|_|\_\____/_/   \_|_|  |_|_|  |_|\___/|_| \_| (_) (_) (_):
·············································································
""")


def _model_compatible(path):
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        dummy = TDLGammonModel()
        dummy.load_state_dict(sd)
        return True
    except Exception:
        return False


def list_models():
    files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith('.pth'))
    choices = []
    for f in files:
        path = os.path.join(MODELS_DIR, f)
        size_kb = os.path.getsize(path) // 1024
        if _model_compatible(path):
            choices.append(questionary.Choice(title=f"{f[:-4]}  ({size_kb} KB)", value=f[:-4]))
    return choices


def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name + '.pth')
    model = TDLGammonModel()
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prompt_die_choice(d1, d2):
    while True:
        try:
            c = int(input(f"  Choose which die to play first ({d1} or {d2}): "))
            if c in (d1, d2):
                return c
            print(f"  Invalid — enter {d1} or {d2}.")
        except ValueError:
            print("  Invalid input.")


def prompt_pair(prompt):
    while True:
        try:
            parts = input(f"  {prompt} (origin dest): ").split()
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            print("  Enter two numbers, e.g.  3 8")
        except ValueError:
            print("  Both values must be integers.")


def play_game(player_name, model_name, model):
    print("\n  Press Enter to roll dice and determine who goes first...")
    input()

    agent_roll = random.randint(1, 6)
    player_roll = random.randint(1, 6)
    while agent_roll == player_roll:
        agent_roll = random.randint(1, 6)
        player_roll = random.randint(1, 6)

    print(f"  You rolled {player_roll}, CPU rolled {agent_roll}.")
    first = "You go first!" if player_roll > agent_roll else "CPU goes first!"
    print(f"  {first}\n")

    turn_code = 0 if player_roll > agent_roll else 1
    game = bg.Game(turn_code)
    p1 = bg.Player(player_name, bg.PlayerType.PLAYER1)
    p2 = bg.Player(f"CPU ({model_name})", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    game.printGameBoard()

    while True:
        turn = game.getTurn()
        dice = game.roll_dice()
        current_player = game.getPlayers(turn)

        if turn == bg.PlayerType.PLAYER2:
            print(f"\n  CPU rolled {dice[0]}, {dice[1]}")
            seq = model.make_move(game)
            if seq:
                moves_str = "  ".join(f"{o}→{d}" for o, d in seq)
                print(f"  CPU played: {moves_str}")
            else:
                print("  CPU has no legal moves.")
        else:
            print(f"\n  You rolled {dice[0]}, {dice[1]}")
            legal = game.legalTurnSequences(bg.PlayerType.PLAYER1, dice[0], dice[1])

            if not legal:
                print("  No legal moves — skipping your turn.")
            elif dice[0] != dice[1]:
                first_die = prompt_die_choice(dice[0], dice[1])
                second_die = dice[1] if first_die == dice[0] else dice[0]

                o1, d1 = prompt_pair(f"Move for die {first_die}")
                ok, err = game.tryMove(current_player, first_die, o1, d1)
                if not ok:
                    print(f"  Illegal move: {err}")
                else:
                    game.printGameBoard()
                    o2, d2 = prompt_pair(f"Move for die {second_die}")
                    ok2, err2 = game.tryMove(current_player, second_die, o2, d2)
                    if not ok2:
                        print(f"  Illegal move: {err2}")
            else:
                for i in range(4):
                    while True:
                        o, d = prompt_pair(f"Move {i+1}/4  (die={dice[0]})")
                        ok, err = game.tryMove(current_player, dice[0], o, d)
                        if not ok:
                            print(f"  Illegal move: {err}")
                        else:
                            game.printGameBoard()
                            break

        # switch turn
        next_turn = (bg.PlayerType.PLAYER2
                     if game.getTurn() == bg.PlayerType.PLAYER1
                     else bg.PlayerType.PLAYER1)
        game.setTurn(next_turn)

        game.printGameBoard()
        over, winner = game.is_game_over()
        if over:
            if winner == bg.PlayerType.PLAYER1:
                print("\n  YOU WIN! 🎉")
            else:
                print("\n  CPU wins. Better luck next time.")
            return


def main():
    printBanner()

    models = list_models()
    if not models:
        print("No .pth models found in models/. Train one first with  make train")
        sys.exit(1)

    player_name = questionary.text("Your name:").ask()
    if not player_name:
        sys.exit(0)

    model_name = questionary.select(
        "Select a model to play against:",
        choices=models,
    ).ask()
    if not model_name:
        sys.exit(0)

    print(f"\n  Loading {model_name}...")
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        sys.exit(1)
    print("  Ready.\n")

    while True:
        play_game(player_name, model_name, model)
        again = questionary.confirm("Play again?", default=True).ask()
        if not again:
            print("  Thanks for playing!")
            break


if __name__ == "__main__":
    main()
