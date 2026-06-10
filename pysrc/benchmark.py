#!/usr/bin/env python3
"""Benchmark the latest (or specified) model against all compatible checkpoints + random."""

import argparse
import os
import random
import sys

import torch
from tqdm import trange

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'TD(λ) model')))

from model import TDLGammonModel
import backgammon_env as bg

_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


# ── helpers ───────────────────────────────────────────────────────────────────

def _model_compatible(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        TDLGammonModel().load_state_dict(sd)
        return True
    except Exception:
        return False


def _load_model(path):
    m = TDLGammonModel()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def _all_compatible_models():
    """Return list of (name, mtime) sorted newest-first, excluding incompatible."""
    files = [f for f in os.listdir(_models_dir) if f.endswith(".pth")]
    files = [f for f in files if _model_compatible(os.path.join(_models_dir, f))]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(_models_dir, f)), reverse=True)
    return files


def latest_compatible_model():
    files = _all_compatible_models()
    return files[0] if files else None


# ── game runners ──────────────────────────────────────────────────────────────

def _random_move(game, player):
    d1, d2 = game.get_last_dice()
    actions = game.legalTurnSequences(game.getTurn(), d1, d2)
    if not actions:
        return
    seq = actions[random.randint(0, len(actions) - 1)]
    for o, dst in seq:
        game.tryMove(player, abs(o - dst), o, dst)


def play_vs_random(model, num_games):
    """Model (P1) vs random (P2), alternating who starts. Returns model win rate.."""
    wins = 0
    total_len = 0
    p1 = bg.Player("RL agent", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Random",   bg.PlayerType.PLAYER2)

    for i in trange(num_games, desc="vs random", leave=False):
        game = bg.Game(0)
        game.setPlayers(p1, p2)
        game.setTurn(bg.PlayerType.PLAYER1 if i % 2 == 0 else bg.PlayerType.PLAYER2)
        length = 0

        while True:
            game.roll_dice()
            if game.getTurn() == bg.PlayerType.PLAYER1:
                model.make_move(game)
            else:
                _random_move(game, p2)
            over, winner = game.is_game_over()
            if over:
                if winner == bg.PlayerType.PLAYER1:
                    wins += 1
                total_len += length
                break
            game.setTurn(
                bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1
                else bg.PlayerType.PLAYER1
            )
            length += 1

    avg = total_len / num_games
    return wins / num_games, avg


def play_vs_model(model1, model2, num_games):
    """model1 (P1) vs model2 (P2), alternating start. Returns model1 win rate."""
    wins = 0
    total_len = 0
    p1 = bg.Player("Model1", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Model2", bg.PlayerType.PLAYER2)

    for i in trange(num_games, desc="vs model", leave=False):
        game = bg.Game(i % 2)
        game.setPlayers(p1, p2)
        length = 0

        while True:
            game.roll_dice()
            if game.getTurn() == bg.PlayerType.PLAYER1:
                model1.make_move(game)
            else:
                model2.make_move(game)
            over, winner = game.is_game_over()
            if over:
                if winner == bg.PlayerType.PLAYER1:
                    wins += 1
                total_len += length
                break
            game.setTurn(
                bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1
                else bg.PlayerType.PLAYER1
            )
            length += 1

    avg = total_len / num_games
    return wins / num_games, avg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark a model against all others")
    parser.add_argument("-m", "--model", default=None,
                        help="challenger checkpoint filename (default: latest compatible)")
    parser.add_argument("-n", "--games", type=int, default=200,
                        help="games per match-up (default: 200)")
    parser.add_argument("--no-random", action="store_true",
                        help="skip the vs-random baseline")
    args = parser.parse_args()

    # resolve challenger
    challenger_name = args.model or latest_compatible_model()
    if challenger_name is None:
        print("No compatible model found in models/.")
        sys.exit(1)
    challenger_path = os.path.join(_models_dir, challenger_name)
    if not os.path.isfile(challenger_path):
        print(f"Model not found: {challenger_path}")
        sys.exit(1)

    print(f"\nChallenger : {challenger_name}")
    print(f"Games/match: {args.games}")
    print(f"Models dir : {_models_dir}\n")

    challenger = _load_model(challenger_path)

    results = []

    # vs random baseline
    if not args.no_random:
        wr, avg_len = play_vs_random(challenger, args.games)
        results.append(("random baseline", wr, avg_len))

    # vs every other compatible model (sorted newest→oldest, skip self)
    opponents = _all_compatible_models()
    opponents = [f for f in opponents if f != challenger_name]

    if not opponents:
        print("No other compatible models found to compare against.")
    else:
        for opp_name in opponents:
            opp = _load_model(os.path.join(_models_dir, opp_name))
            wr, avg_len = play_vs_model(challenger, opp, args.games)
            results.append((opp_name, wr, avg_len))

    # ── summary table ─────────────────────────────────────────────────────────
    col = max(len(r[0]) for r in results) + 2
    print(f"\n{'─' * (col + 32)}")
    print(f"{'Opponent':<{col}} {'Win rate':>10}  {'Avg game len':>14}")
    print(f"{'─' * (col + 32)}")
    for name, wr, avg_len in results:
        bar = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        print(f"{name:<{col}} {wr*100:>9.1f}%  {avg_len:>14.1f}   {bar}")
    print(f"{'─' * (col + 32)}\n")


if __name__ == "__main__":
    main()
