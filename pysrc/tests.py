import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pysrc')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "TD(λ) model")))

from model import TDLGammonModel
import backgammon_env as bg


def test_basic_game():
    game = bg.Game(0)
    p1 = bg.Player("White", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Black", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    game.setGameBoard([-5, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5])
    game.printGameBoard()
    moves = game.legalTurnSequences(1, 6, 5)
    print(moves)
    assert isinstance(moves, list)


def test_model_loads():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model = TDLGammonModel()
    import torch
    state_dict = torch.load(os.path.join(models_dir, 'tdgammonNEW100k.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    assert model is not None


def test_model_makes_move():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    import torch
    model = TDLGammonModel()
    state_dict = torch.load(os.path.join(models_dir, 'tdgammonNEW100k.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    game = bg.Game(0)
    p1 = bg.Player("White", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Black", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    game.setTurn(bg.PlayerType.PLAYER1)
    game.roll_dice()
    seq = model.make_move(game)
    assert isinstance(seq, list)
