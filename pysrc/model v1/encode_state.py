import torch
import logging

logger = logging.getLogger(__name__)


# Pre‐allocate a reusable mask buffer of shape (1, max_steps, 26*26).
# We’ll zero it out in place each call, rather than recreating a new tensor.
_max_steps = 4
_S = 26
_N = _S * _S
_mask_buffer = torch.zeros((1, _max_steps, _N), dtype=torch.bool)


def encode_state(board, pieces, turn, dice1=0, dice2=0):
    """
    Encode game state into 9-channel tensor for neural network input
    
    Args:
        board: List of 24 integers representing piece counts on each point
        pieces: Pieces object from the game
        turn: Current player turn (0 or 1)
        dice1, dice2: Current dice values
    
    Returns:
        torch.Tensor of shape [9, 24] representing the encoded state
    """
    board_pts = torch.tensor(board, dtype=torch.float32)
    p1 = torch.clamp(board_pts, min=0).unsqueeze(0)
    p2 = torch.clamp(-board_pts, min=0).unsqueeze(0)
    
    # Create dice planes
    dice1_plane = torch.full((1, 24), float(dice1))
    dice2_plane = torch.full((1, 24), float(dice2))
    
    # Get piece counts for both players
    p1_jailed = pieces.numJailed(0)
    p1_borne_off = pieces.numFreed(0)
    p2_jailed = pieces.numJailed(1)
    p2_borne_off = pieces.numFreed(1)
    
    if turn == 0:  # Player 1's turn
        cur_pieces = p1
        cur_jail = torch.full((1, 24), float(p1_jailed))
        cur_off = torch.full((1, 24), float(p1_borne_off))
        
        opp_pieces = p2
        opp_jail = torch.full((1, 24), float(p2_jailed))
        opp_off = torch.full((1, 24), float(p2_borne_off))
        
        turn_plane = torch.ones((1, 24))
    else:  # Player 2's turn
        cur_pieces = p2
        cur_jail = torch.full((1, 24), float(p2_jailed))
        cur_off = torch.full((1, 24), float(p2_borne_off))
        
        opp_pieces = p1
        opp_jail = torch.full((1, 24), float(p1_jailed))
        opp_off = torch.full((1, 24), float(p1_borne_off))
        
        turn_plane = torch.zeros((1, 24))

    return torch.cat([
        cur_pieces, cur_jail, cur_off,
        opp_pieces, opp_jail, opp_off,
        dice1_plane, dice2_plane, turn_plane
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
    Simplified version that trusts the C++ engine's legal sequence generation
    """
    global _seq_cache_hits, _seq_cache_misses, _mask_buffer

    assert batch_size == 1, "This implementation only supports batch_size=1."

    # Compute the cache key
    board = tuple(game.getGameBoard())
    die1, die2 = game.get_last_dice()
    player_num = curr_player.getNum()
    key = (board, die1, die2, player_num)

    # Check cache
    if key in _seq_cache:
        _seq_cache_hits += 1
        cached_data = _seq_cache[key]
        return (
            cached_data[0].clone(),
            cached_data[1],
            cached_data[2],
            cached_data[3],
            cached_data[4],
            cached_data[5]
        )

    _seq_cache_misses += 1
    _mask_buffer.zero_()

    # Get legal sequences from C++ engine
    try:
        seqs = game.legalTurnSequences(player_num, die1, die2)
        logger.debug(f"C++ engine returned {len(seqs)} sequences for player {player_num}, dice [{die1}, {die2}]")
    except Exception as e:
        logger.error(f"Error getting legal sequences: {e}")
        seqs = []

    if not seqs:
        # Return empty structures
        empty_mask = torch.zeros((1, max_steps, _N), dtype=torch.bool).to(device)
        empty_tensors = torch.zeros((0, max_steps), dtype=torch.long).to(device)
        empty_valid = torch.zeros((0, max_steps), dtype=torch.bool).to(device)
        return (empty_mask, [], [], empty_tensors, empty_tensors, empty_valid)

    # Create simple dice orders - trust that sequences are already legal
    dice_orders = []
    if die1 == die2:
        # Doubles: use the same die for all moves
        for seq in seqs:
            dice_order = [die1] * len(seq)
            dice_orders.append(dice_order)
    else:
        # Non-doubles: create plausible dice orders without complex reconstruction
        for seq in seqs:
            dice_order = []
            dice_remaining = [die1, die2]
            
            for i, (origin, dest) in enumerate(seq):
                # Simple heuristic: use the die that matches the move distance if possible
                move_distance = abs(dest - origin)
                
                if move_distance in dice_remaining:
                    dice_order.append(move_distance)
                    dice_remaining.remove(move_distance)
                elif dice_remaining:
                    # Use any remaining die (for bearing off, etc.)
                    dice_order.append(dice_remaining[0])
                    dice_remaining.remove(dice_remaining[0])
                else:
                    # This shouldn't happen, but handle gracefully
                    dice_order.append(die1 if i % 2 == 0 else die2)
            
            dice_orders.append(dice_order)

    # Validate sequences by actually testing them
    validated_seqs = []
    validated_dice_orders = []
    
    for seq, dice_order in zip(seqs, dice_orders):
        if len(seq) <= max_steps:
            # Test the sequence on a cloned game
            temp_game = game.clone()
            valid = True
            
            for i, (origin, dest) in enumerate(seq):
                if i < len(dice_order):
                    success, error = temp_game.tryMove(curr_player, dice_order[i], origin, dest)
                    if not success:
                        # Try the other die if available
                        other_die = die2 if dice_order[i] == die1 else die1
                        success, error = temp_game.tryMove(curr_player, other_die, origin, dest)
                        if success:
                            dice_order[i] = other_die  # Update the dice order
                        else:
                            logger.debug(f"Sequence validation failed: {seq} - {error}")
                            valid = False
                            break
            
            if valid:
                validated_seqs.append(seq)
                validated_dice_orders.append(dice_order)

    seqs = validated_seqs
    dice_orders = validated_dice_orders

    if not seqs:
        empty_mask = torch.zeros((1, max_steps, _N), dtype=torch.bool).to(device)
        empty_tensors = torch.zeros((0, max_steps), dtype=torch.long).to(device)
        empty_valid = torch.zeros((0, max_steps), dtype=torch.bool).to(device)
        return (empty_mask, [], [], empty_tensors, empty_tensors, empty_valid)

    # Fill mask buffer
    for i, seq in enumerate(seqs):
        for t, (o, d) in enumerate(seq):
            if t < max_steps and 0 <= o < _S and 0 <= d < _S:
                _mask_buffer[0, t, o * _S + d] = True

    mask_on_device = _mask_buffer.to(device)
    mask_to_store = mask_on_device.detach().clone()

    # Build index tensors
    M = len(seqs)
    all_t = torch.zeros((M, max_steps), dtype=torch.long)
    all_flat = torch.zeros((M, max_steps), dtype=torch.long)
    valid_mask = torch.zeros((M, max_steps), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        for t, (o, d) in enumerate(seq):
            if t < max_steps:
                all_t[i, t] = t
                all_flat[i, t] = o * _S + d
                valid_mask[i, t] = True

    all_t = all_t.to(device)
    all_flat = all_flat.to(device)
    valid_mask = valid_mask.to(device)

    # Cache the results
    _seq_cache[key] = (
        mask_to_store,
        seqs,
        dice_orders,
        all_t,
        all_flat,
        valid_mask
    )

    return (mask_on_device, seqs, dice_orders, all_t, all_flat, valid_mask)

def clear_sequence_cache():
    """Clear the sequence cache - useful for debugging"""
    global _seq_cache, _seq_cache_hits, _seq_cache_misses
    _seq_cache.clear()
    _seq_cache_hits = 0
    _seq_cache_misses = 0

def get_cache_stats():
    """Get cache statistics"""
    return {
        'hits': _seq_cache_hits,
        'misses': _seq_cache_misses,
        'size': len(_seq_cache)
    }