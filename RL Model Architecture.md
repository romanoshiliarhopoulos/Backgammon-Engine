RL Model Architecture Notes

Big Picture:

- Dual-Head Neural Network that outputs both move-probabilities and board evaluation.
  - Move-Probabilities: A vector that encodes how good each possible move is. A higher π(s) means this move is more likely to lead to a win under optimal play
  - Board Evaluation: Outputs a single scalar, the network’s estimate of the expected outcome (+1 win, -1 loss)
- Deep Convolutional neural network with 10 residual blocks and 64 filters each.

——————————————————————————————————————————————————————————————————————————————

Game State Representation: (Represent the entire game-state as a single tensor)

- Tensor representing:
  _ Game Board (a length‑24 array of signed ints, where + is player 1s checkers and - is player 2s checkers)
  _ Bar/ Jailed piece counts (for both players)
  _ Borne-off / Freed piece counts (for both players)]
  _ Current Player to move (turn of each player) \* Dice roll (passing the values of two dice rolls)

All counts are normalized so that our first-layer activations have to learn weights and biases that are within [0,1]

Plane ↓ \ Point → 1 2 3 4 5 6 7 8 … 24
───────────────────────────────── 0. P1 count 2/15 0 0 0 0 5/15 0 0 … 0

1.  P2 count 0 0 0 0 0 0 0 3/15 … 0
    ─────────────────────────────────
2.  P1 bar 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 … 1/15
3.  P2 bar 0 0 0 0 0 0 0 0 … 0
    ─────────────────────────────────
4.  P1 borne‑off 0 0 0 0 0 0 0 0 … 0
5.  P2 borne‑off 2/15 2/15 2/15 2/15 2/15 2/15 2/15 2/15 … 2/15
    ─────────────────────────────────
6.  Die₁ = 3/6 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 … 0.5
7.  Die₂ = 5/6 ≈0.83 0.83 0.83 0.83 0.83 0.83 0.83 0.83 … 0.83
    ─────────────────────────────────
8.  To‑move flag 1 1 1 1 1 1 1 1 … 1

——————————————————————————————————————————————————————————————————————————————

Model output:

- MOVE ENCODING (Policy Output): Encodes which move is the most likely to lead to a win
  _ Each entry πi(s)\pi_i(s)πi​(s) in the A=1 352‑dim policy vector is the network’s prior probability for one specific two‑move turn sequence. Concretely, we encode each composite action i as a triple:
  _ m1∈{0,…,25} : the “origin index” for the first sub‑move
  _ m2∈{0,…,25}: the “origin index” for the second sub‑move
  _ b∈{0,1}: which die is used first
  _ Here’s how to interpret those numbers:
  _ Origin indices 0–23 = board points 1–24
  _ Index 24 = bear‑off
  _ Index 25 = “no move” (skip the die)
  _ Die values (d1,d2)(d_1,d_2)(d1​,d2​) come from your environment each turn.
  _ Example:
  _ Dice roll (3,5) and Player 1’s turn
  _ Suppose highest prior is π107 = 0.08
  _ We decode i = 107 as (m1 = 4, m2 =7, d = 0) using the following flattenning formula: i =(m1 _ 26 +m2) _2 + b
  _ Since b = 0, we first play die 3 and then die 5 \* Move1: 5 -> 8 and Move2 from 8 ->13

- VALUE OUTPUT: a single scalar estimating the expected outcome from the given state.
  _ Activation function: V(s)=tanh(raw_value_score)∈[−1,1].
  _ where +1 means sure win for the player to move
  _ -1 means sure loss
  _ Intermediate values quantify the network’s estimated advantage \* Will be trained to minimize Mean-Squared-Error (MSE) to between prediction and actual outcome.

——————————————————————————————————————————————————————————————————————————————

TRAINING PIPELINE AND SELF PLAY
