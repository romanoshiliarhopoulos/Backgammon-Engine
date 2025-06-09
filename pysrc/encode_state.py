import torch



# Pre‐allocate a reusable mask buffer of shape (1, max_steps, 26*26).
# We’ll zero it out in place each call, rather than recreating a new tensor.
_max_steps = 4
_S = 26
_N = _S * _S
_mask_buffer = torch.zeros((1, _max_steps, _N), dtype=torch.bool)


def encode_state(board, jailed, borne_off, turn):
    board_pts = torch.tensor(board, dtype=torch.float32)
    p1 = torch.clamp(board_pts, min=0).unsqueeze(0)
    p2 = torch.clamp(-board_pts, min=0).unsqueeze(0)
    
    # Get dice information from the last roll
    # You'll need to pass dice info to this function or get it from game state
    dice1_plane = torch.zeros((1, 24))  # Will need actual dice values
    dice2_plane = torch.zeros((1, 24))  # Will need actual dice values
    
    if turn == 1:
        # Player 1's perspective
        cur_pieces = p1
        cur_jail = torch.full((1, 24), float(jailed))
        cur_off = torch.full((1, 24), float(borne_off))
        
        opp_pieces = p2
        opp_jail = torch.full((1, 24), float(board_pts.min().abs()))
        opp_off = torch.full((1, 24), float(board_pts.max().abs()))
        
        turn_plane = torch.ones((1, 24))  # Player 1's turn
    else:
        # Player 2's perspective  
        cur_pieces = p2
        cur_jail = torch.full((1, 24), float(jailed))
        cur_off = torch.full((1, 24), float(borne_off))
        
        opp_pieces = p1
        opp_jail = torch.full((1, 24), float(board_pts.max().abs()))
        opp_off = torch.full((1, 24), float(board_pts.min().abs()))
        
        turn_plane = torch.zeros((1, 24))  # Player 2's turn

    # Stack all 9 channels
    return torch.cat([
        cur_pieces,    # Channel 0: Current player pieces
        cur_jail,      # Channel 1: Current player jailed count
        cur_off,       # Channel 2: Current player borne off count
        opp_pieces,    # Channel 3: Opponent pieces
        opp_jail,      # Channel 4: Opponent jailed count  
        opp_off,       # Channel 5: Opponent borne off count
        dice1_plane,   # Channel 6: First die value
        dice2_plane,   # Channel 7: Second die value
        turn_plane     # Channel 8: Whose turn (1=player1, 0=player2)
    ], dim=0)


_seq_cache = {}          # Cache: key → (mask_tensor_on_device, seqs, dice_orders)
_seq_cache_hits = 0
_seq_cache_misses = 0

def build_sequence_mask(game,
                        curr_player,
                        batch_size=1,
                        device='mps',
                        max_steps=_max_steps):
    """
    Returns:
      mask_on_device:   Tensor[1, max_steps, S*S] (bool)
      seqs:             list of legal sequences for this turn
      dice_orders:      list of dice_order lists for each sequence
      all_t:            LongTensor[M, max_steps] of timestep indices
      all_flat:         LongTensor[M, max_steps] of (o*S + d) indices
      valid_mask:       BoolTensor[M, max_steps] where True indicates a real step
    """
    global _seq_cache_hits, _seq_cache_misses, _mask_buffer

    assert batch_size == 1, "This implementation only supports batch_size=1."

    #Compute the cache key: (board_tuple, die1, die2, player)
    board = tuple(game.getGameBoard())
    die1, die2 = game.get_last_dice()
    player = game.getTurn()
    key = (board, die1, die2, player)

    # Cache hit? used cached info
    if key in _seq_cache:
        _seq_cache_hits += 1
        (
            cached_mask,
            cached_seqs,
            cached_orders,
            cached_all_t,
            cached_all_flat,
            cached_valid_mask
        ) = _seq_cache[key]
        return (
            cached_mask.clone(),
            cached_seqs,
            cached_orders,
            cached_all_t,
            cached_all_flat,
            cached_valid_mask
        )

    #Cache miss: build from scratch
    _seq_cache_misses += 1

    # Zero out buffer
    _mask_buffer.zero_()

    #  get all  Legal sequences from game engine
    seqs = game.legalTurnSequences(player, die1, die2)

    # Reconstruct dice_orders
    if die1 == die2:
        dice_orders = [[die1] * 4 for _ in seqs]
    else:
        first_moves = game.legalMoves(player, die1)
        cnt1 = 0
        for m1 in first_moves:
            g1 = game.clone()
            ok, err = g1.tryMove(curr_player, die1, m1[0], m1[1])
            next_moves = g1.legalMoves(player, die2)
            if not next_moves:
                cnt1 += 2
            else:
                cnt1 += len(next_moves)
        total_seqs = len(seqs)
        cnt2 = total_seqs - cnt1
        dice_orders = [[die1, die2]] * cnt1 + [[die2, die1]] * cnt2

    # Fill the mask buffer based on seqs
    for i, seq in enumerate(seqs):
        for t, (o, d) in enumerate(seq):
            if t < max_steps:
                _mask_buffer[0, t, o * _S + d] = True

    # 3d) Move mask to device and detach
    mask_on_device = _mask_buffer.to(device)
    mask_to_store = mask_on_device.detach()

    # 4) Build index tensors for vectorized scoring
    M = len(seqs)
    all_t = torch.zeros((M, max_steps), dtype=torch.long)
    all_flat = torch.zeros((M, max_steps), dtype=torch.long)
    valid_mask = torch.zeros((M, max_steps), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        for t, (o, d) in enumerate(seq):
            all_t[i, t] = t
            all_flat[i, t] = o * _S + d
            valid_mask[i, t] = True

    all_t = all_t.to(device)
    all_flat = all_flat.to(device)
    valid_mask = valid_mask.to(device)

    # 5) Cache everything
    _seq_cache[key] = (
        mask_to_store,
        seqs,
        dice_orders,
        all_t,
        all_flat,
        valid_mask
    )

    return (
        mask_on_device,
        seqs,
        dice_orders,
        all_t,
        all_flat,
        valid_mask
    )
