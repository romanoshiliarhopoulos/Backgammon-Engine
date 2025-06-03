import torch

_seq_cache = {}          # Cache: key → (mask_tensor_on_device, seqs, dice_orders)
_seq_cache_hits = 0
_seq_cache_misses = 0

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
    jail_plane = torch.full((1, 24), float(jailed))
    off_plane = torch.full((1, 24), float(borne_off))

    if turn == 1:
        cur = torch.cat([p1, jail_plane, off_plane], dim=0)
        opp = torch.cat([
            p2,
            torch.full((1, 24), float(board_pts.min().abs())),
            torch.full((1, 24), float(board_pts.max().abs()))
        ], dim=0)
    else:
        cur = torch.cat([p2, jail_plane, off_plane], dim=0)
        opp = torch.cat([
            p1,
            torch.full((1, 24), float(board_pts.max().abs())),
            torch.full((1, 24), float(board_pts.min().abs()))
        ], dim=0)

    return torch.cat([cur, opp], dim=0)


def build_sequence_mask(game,
                        curr_player,
                        batch_size=1,
                        device='cpu',
                        max_steps=_max_steps):
    global _seq_cache_hits, _seq_cache_misses, _mask_buffer

    # We assume batch_size is always 1 in your code. If you ever want batch_size>1,
    # you would need to pre‐allocate a larger buffer. For now, we just handle batch_size=1.
    assert batch_size == 1, "This implementation only supports batch_size=1."

    # 1) Compute the cache key: (board_tuple, die1, die2, player)
    board = tuple(game.getGameBoard())  # 24‐int tuple
    die1, die2 = game.get_last_dice()
    player = game.getTurn()
    key = (board, die1, die2, player)

    # 2) If we have a cache hit, just return a clone of the stored mask + the stored seqs/orders.
    if key in _seq_cache:
        _seq_cache_hits += 1
        #print("Cache hit!")
        cached_mask, cached_seqs, cached_orders = _seq_cache[key]
        # Note: cached_mask is already on the correct device and has shape [1, max_steps, N].
        return cached_mask.clone(), cached_seqs, cached_orders

    # 3) Otherwise, we do the “slow” logic once, then store in cache.
    _seq_cache_misses += 1
    #print("Cache miss")

    # Zero out our pre‐allocated mask buffer (in place)
    _mask_buffer.zero_()

    # 3a) Ask C++ for the raw list of legal turn‐sequences
    seqs = game.legalTurnSequences(player, die1, die2)

    # 3b) Reconstruct dice_orders by doing exactly one clone() per m1,
    # so that we don’t clone inside the inner loop over m2. This matches what C++ did.
    if die1 == die2:
        # Doubles: each sequence has four identical pips, so dice_orders is [die1,die1,die1,die1]
        dice_orders = [[die1] * 4 for _ in seqs]
    else:
        # Count how many sequences C++ returned that came from (die1→die2) vs (die2→die1).
        first_moves = game.legalMoves(player, die1)
        cnt1 = 0
        for m1 in first_moves:
            g1 = game.clone()
            ok, err = g1.tryMove(curr_player, die1, m1[0], m1[1])
            next_moves = g1.legalMoves(player, die2)
            if not next_moves:
                # C++ would have emitted two “dummy” sequences [m1,{0,0}]
                cnt1 += 2
            else:
                cnt1 += len(next_moves)

        total_seqs = len(seqs)
        cnt2 = total_seqs - cnt1
        dice_orders = [[die1, die2]] * cnt1 + [[die2, die1]] * cnt2

    # 3c) Fill the mask buffer based on seqs:
    # Each seq is a list of up to max_steps pairs (o,d). We mark mask[b, t, o*S + d] = True.
    for i, seq in enumerate(seqs):
        for t, (o, d) in enumerate(seq):
            if t < max_steps:
                _mask_buffer[0, t, o * _S + d] = True

    # 3d) Move the mask buffer to the requested device (once), and store in cache.
    mask_on_device = _mask_buffer.to(device)

    # Detach the mask so that we store a Tensor that’s not requiring grad.
    # We could also call .cpu() here and re‐to(device) later, but storing on 'device'
    # avoids an extra copy on every cache hit.
    mask_to_store = mask_on_device.detach()

    _seq_cache[key] = (mask_to_store, seqs, dice_orders)
    return mask_on_device, seqs, dice_orders
