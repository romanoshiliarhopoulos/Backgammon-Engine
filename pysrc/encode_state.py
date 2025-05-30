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

import torch

import torch

def build_legal_mask(game, batch_size=1, device="cpu", max_steps=4):
    """
    Returns a BoolTensor of shape [batch_size, max_steps, 26*26].
    For each legalTurnSequence seq = [(o0,d0), (o1,d1), …],
    we set mask[b, step, o_step*26 + d_step] = True
    for every sequence and every step < len(seq).
    """
    S = 26
    N = S*S
    mask = torch.zeros((batch_size, max_steps, N),
                       dtype=torch.bool, device=device)

    die1, die2 = game.get_last_dice()
    player     = game.getTurn()
    seqs        = game.legalTurnSequences(player, die1, die2)

    for b in range(batch_size):
        for seq in seqs:
            for t, (o, d) in enumerate(seq):
                if t < max_steps:
                    mask[b, t, o*S + d] = True
    return mask