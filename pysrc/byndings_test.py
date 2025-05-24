#!/usr/bin/env python3
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, os.path.abspath(build_dir))

try:
    import backgammon_env as bg # type: ignore
    print("✓ Module imported successfully")
    
    # Test basic game creation
    game = bg.Game(4)
    print("✓ Game created")
    
    # Test basic getters
    turn = game.getTurn()
    print(f"✓ Current turn: {turn}")
    
    board = game.getGameBoard()
    print(f"✓ Board retrieved, length: {len(board)}")
    
    # Test player creation
    p1 = bg.Player("TestPlayer1", bg.PlayerType.PLAYER1)
    p2 = bg.Player("TestPlayer2", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    print("✓ Players created and set")
    
    # Test legal moves
    moves = game.legalMoves(0, 6)
    print(f"✓ Legal moves for die 6: {len(moves)} moves found")
    
    # Test game over status
    is_over, winner = game.is_game_over()
    print(f"✓ Game over check: game_over={is_over}, winner={winner}")
    
    print("\n All basic tests passed!")
    game.printGameBoard()
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Test failed: {e}")
