import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = BackgammonNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)

for episode in range(num_episodes):
    game = bg.Game(4)            # new game
    p1 = bg.Player("RLAgent", bg.PlayerType.PLAYER1)
    p2 = bg.Player("RandomBot", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    
    log_probs = []
    values    = []
    rewards   = []
    
    while True:
        # 1) encode state
        board = list(game.getGameBoard())  # length‑24 ints
        jailed     = game.getJailedCount()
        borne_off  = game.getBornOffCount()
        turn = game.getTurn()
        
        state = encode_state(board, jailed, borne_off, turn)
        state = state.unsqueeze(0).to(device)  # [1,6,24]
        
        # 2) get legal mask
        legal_mask = build_legal_mask(game, batch_size=1, device=device)
        
        # 3) forward
        probs, value = net(state, legal_mask)
        
        # 4) sample action
        m = torch.distributions.Categorical(probs)
        action = m.sample()           # int in [0,576)
        log_prob = m.log_prob(action)
        
        # 5) step env
        o, d = divmod(action.item(), 24)
        success, err = game.tryMove(p1 if turn==1 else p2, game.get_last_dice()[0], o, d)
        # you might need to handle invalid or multi‐step sequences properly
        
        # 6) record
        log_probs.append(log_prob)
        values.append(value)
        
        is_over, winner = game.is_game_over()
        if is_over:
            r = 1.0 if winner == turn else -1.0
            rewards.append(r)
            break
        else:
            rewards.append(0.0)  # intermediate reward 0
        
        # swap turn is handled inside C++ env
        
    # compute returns & advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, device=device)
    
    values = torch.cat(values)        # [T]
    log_probs = torch.stack(log_probs)  # [T]
    
    # advantage
    advantages = returns - values.detach()
    
    # losses
    policy_loss = -(log_probs * advantages).mean()
    value_loss  = F.mse_loss(values, returns)
    loss = policy_loss + c1 * value_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
