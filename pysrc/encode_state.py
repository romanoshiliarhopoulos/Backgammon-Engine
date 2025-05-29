import torch

def encode_state(board, jailed, borne_off, turn):
    """
    board: list[24] of signed ints: +n for P1 checkers, –n for P2 checkers
    jailed, borne_off: ints for the current player
    turn: 1 or 2
    returns: Tensor of shape [batch, 6, 24]
    """
    # board_pts: shape [24]
    board_pts = torch.tensor(board, dtype=torch.float32)
    
    # two occupancy planes
    p1 = torch.clamp(board_pts, min=0).unsqueeze(0)      # [1,24]
    p2 = torch.clamp(-board_pts, min=0).unsqueeze(0)     # [1,24]
    
    # broadcast jail/off counts across 24 points
    jail_plane = torch.full((1,24), float(jailed))
    off_plane  = torch.full((1,24), float(borne_off))
    
    # now stack: current‐player vs opponent
    if turn == 1:
        cur = torch.cat([p1, jail_plane, off_plane], dim=0)
        opp = torch.cat([p2,
                         torch.full((1,24), float(board_pts.min().abs())),  # you’d pull opponent’s counts if env gave them
                         torch.full((1,24), float(board_pts.max().abs()))], dim=0)
    else:
        cur = torch.cat([p2, jail_plane, off_plane], dim=0)
        opp = torch.cat([p1,
                         torch.full((1,24), float(board_pts.max().abs())),
                         torch.full((1,24), float(board_pts.min().abs()))], dim=0)
    
    return torch.cat([cur, opp], dim=0)  # [6,24]

def build_legal_mask(game, batch_size=1, device='cpu'):
    """
    Returns a [batch_size, 576] mask where True indicates that index
    corresponds to a legal (origin,dest) pair for the current dice roll.
    Here we assume batch_size=1 for simplicity.
    """
    legal = torch.zeros((batch_size, 24*24), dtype=torch.bool, device=device)
    for b in range(batch_size):
        # you’d pull dice from your env; for simplicity assume they’re stored
        dice = game.get_last_dice()  # e.g., [die1, die2]
        seqs = game.legalTurnSequences(dice[0], dice[1])
        # flatten all moves in all sequences
        legal_moves = set((o,d) for seq in seqs for (o,d) in seq)
        for (o,d) in legal_moves:
            legal[b, o*24 + d] = True
    return legal
