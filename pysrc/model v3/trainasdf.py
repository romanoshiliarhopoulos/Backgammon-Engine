def play_game_collect_all_states(model):
    device = next(model.parameters()).device
    model.eval()
    
    game = bg.Game(0)
    p1 = bg.Player("White", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Black", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    
    # Initialize game...
    
    all_states = []  # Store (state, player) tuples
    total_moves = 0
    
    while True:
        # Record state before move
        current_player = game.getTurn()
        state_encoding = model.encode_state(game).to(device)
        all_states.append((state_encoding, current_player))
        
        best_seq = model.make_move(game)
        
        over, winner = game.is_game_over()
        if over:
            return winner, all_states, total_moves
        
        # Switch turn
        next_turn = bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1 else bg.PlayerType.PLAYER1
        game.setTurn(next_turn)
        total_moves += 1
