import torch
import torch.optim as optim
from model import TDLGammonModel
from tqdm import trange
import tqdm
import os
import sys
import time
import argparse
import threading
import multiprocessing as mp
from collections import deque
import backgammon_env as bg #type: ignore

def plot_all_metrics(game_length, td_loss, save_path=None):
    """Plots learning metrics """
    import matplotlib.pyplot as plt  # lazy: keeps worker process spawn fast
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    
    # Plot game length
    axes[0].plot(game_length)
    axes[0].set_title("Game Length Over Episodes")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Game Length")
    
    # Plot TD loss
    axes[1].plot(td_loss)
    axes[1].set_title("TD Loss Over Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("TD Loss")
    
    plt.tight_layout()
    
    # Save to PNG if requested
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()

_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

def load_previous_model(checkpoint_path="tdgammonNEW10k.pth"):
    """Load previous model weights if they exist, otherwise start fresh"""
    model = TDLGammonModel()
    
    full_path = os.path.join(_models_dir, checkpoint_path)
    if os.path.exists(full_path):
        print(f" Loading previous model from {full_path}")
        try:
            state_dict = torch.load(full_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            print(" Successfully loaded previous model weights")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("Starting with fresh weights instead")
    else:
        print(f"No previous model found at {full_path}")
        print("Starting training from scratch")
    
    return model

def play_game(model, game_idx, epsilon=0.0):
    """
    Plays a single game of backgammon using the provided model for move selection.

    `epsilon` enables ε-greedy exploration during move selection (training only).

    Returns:
        winner: bg.PlayerType of the winning player
        states: list of np.float32[198] state encodings, one per turn
        total_moves: Number of turns until end of game
    """
    model.eval()

    # Initialize game and players
    game = bg.Game(0)
    p1 = bg.Player("White", bg.PlayerType.PLAYER1)
    p2 = bg.Player("Black", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    # Keep a mapping from turn code to the correct Player instance
    turn_player = {
        bg.PlayerType.PLAYER1: p1,
        bg.PlayerType.PLAYER2: p2
    }

    # Determine first turn by rolling dice
    p1_roll = game.roll_dice()
    p2_roll = game.roll_dice()
    while sum(p1_roll) == sum(p2_roll):
        p1_roll = game.roll_dice()
        p2_roll = game.roll_dice()
    if sum(p1_roll) > sum(p2_roll):
        game.setTurn(bg.PlayerType.PLAYER1)
    else:
        game.setTurn(bg.PlayerType.PLAYER2)

    states = [] # Store (state, player) tuples
    total_moves = 0

    # Main game loop
    while True:
        #game.printGameBoard()
        state_encoding = model.encode_state_np(game)
        states.append(state_encoding)
        
        # Roll dice for this turn
        game.roll_dice()
        best_seq = model.make_move(game, game_idx, epsilon=epsilon)

        # Check for game end
        over, winner = game.is_game_over()
        if over:
            #print(f"Total moves: {total_moves}")
            return winner, states, total_moves

        # Switch turn
        next_turn = bg.PlayerType.PLAYER2 if game.getTurn() == bg.PlayerType.PLAYER1 else bg.PlayerType.PLAYER1
        game.setTurn(next_turn)
        total_moves+=1


def apply_td_updates(model, optimizer, states, player1_won):
    """Applies one game's TD(lambda) update with eligibility traces.

    Single source of truth for the learning math, shared by the sequential and
    parallel trainers. `states` is a list of np.float32[198] encodings in the
    global frame, so V(s) ≈ P(PLAYER1 wins); the bootstrap target for state t is
    just V(s_{t+1}), and the terminal target is 1.0 if PLAYER1 won else 0.0.
    Traces are assumed reset by the caller. Returns per-step squared TD errors.
    """
    device = next(model.parameters()).device
    losses_this_episode = []

    def _trace_update(value_tensor, td_error):
        optimizer.zero_grad()
        value_tensor.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Eligibility trace: e = λ * e + ∇V ; then param += lr * δ * e
                    model.eligibility_traces[name] = (
                        model.lambda_decay * model.eligibility_traces[name] +
                        param.grad.data
                    )
                    param.add_(model.learning_rate * td_error * model.eligibility_traces[name])

    for t in range(len(states) - 1):
        current_state = torch.from_numpy(states[t]).to(device)
        next_state = torch.from_numpy(states[t + 1]).to(device)

        model.eval()
        with torch.no_grad():
            v_next = model(next_state.unsqueeze(0))

        model.train()
        v_current = model(current_state.unsqueeze(0))

        td_error = (v_next - v_current).item()
        _trace_update(v_current, td_error)
        losses_this_episode.append(td_error ** 2)

    # Terminal update toward the actual game outcome.
    if states:
        final_state = torch.from_numpy(states[-1]).to(device)
        model.train()
        v_final = model(final_state.unsqueeze(0))
        td_error = (1.0 if player1_won else 0.0) - v_final.item()
        _trace_update(v_final, td_error)

    return losses_this_episode


def train(num_games = 1000, initial_lr = 0.1, checkpoint_path="tdgammonNEW10k.pth"):
    """
    Trains the TDGammonModel using TD(lambda) learning over a number of games.

    Args:
        num_games: Number of self-play games to simulate
        lr: Learning rate 

    Returns:
        model: The trained TDGammonModel instance
    """

    model = load_previous_model(checkpoint_path)


    #optimizer definition: SGD (Stochastic Grad Descent)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    # Definitions for learning statistics
    wins = 0
    all_losses = []          # collects every single TD‐loss
    episode_losses = []      # collects one average loss per game
    number_of_turns = []     #holds the number of moves at each game

    for i in trange(1, num_games + 1, desc="Games"):
        model.update_learning_params(i)

        #Reset traces at start of each game:
        for name in model.eligibility_traces:
            model.eligibility_traces[name].zero_()

        winner, states, total_moves = play_game(model, i)

        number_of_turns.append(total_moves)
        if winner == bg.PlayerType.PLAYER1:
            wins += 1

        model.train()
        losses_this_episode = apply_td_updates(
            model, optimizer, states, winner == bg.PlayerType.PLAYER1
        )

        # record losses
        all_losses.extend(losses_this_episode)
        if losses_this_episode:
            episode_losses.append(sum(losses_this_episode) / len(losses_this_episode))

    print(f"Training completed. Final win rate: {wins / num_games:.3f}")
    plot_all_metrics(number_of_turns, episode_losses, "plots.png")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Parallel self-play
#
# Self-play games are independent, so we generate a "round" of N games in
# parallel from a single weight snapshot, then replay every trajectory through
# the SAME apply_td_updates() math in the main process before snapshotting the
# next round. This is mini-batched TD(λ): within a round games don't see each
# other's updates (mild, tunable staleness); across rounds learning is exact.
# ──────────────────────────────────────────────────────────────────────────────

_worker_models = {}  # reusable model slots per worker process ("a" train/cur, "b" opponent)


def _init_worker():
    """Pool initializer: pin each worker to a single torch thread (so N workers
    fill N cores without oversubscription) and build its reusable models. Workers
    ignore SIGINT so Ctrl-C is handled only by the main process (no traceback flood)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    global _worker_models
    _worker_models = {"a": TDLGammonModel().eval(), "b": TDLGammonModel().eval()}


def _play_one(args):
    """Worker task: load the round's weights, play one self-play game (with the
    round's exploration ε), return a picklable (player1_won, states, total_moves).
    states is np.float32[198] list."""
    state_dict, game_idx, epsilon = args
    m = _worker_models["a"]
    m.load_state_dict(state_dict)
    winner, states, total_moves = play_game(m, game_idx, epsilon=epsilon)
    return (winner == bg.PlayerType.PLAYER1), states, total_moves


def _play_head_to_head(model_p1, model_p2, game_idx):
    """Plays one full game, model_p1 as PLAYER1 vs model_p2 as PLAYER2.
    Returns the winning PlayerType (0 = PLAYER1, 1 = PLAYER2)."""
    game = bg.Game(game_idx % 2)  # alternate who the constructor seats first
    p1 = bg.Player("P1", bg.PlayerType.PLAYER1)
    p2 = bg.Player("P2", bg.PlayerType.PLAYER2)
    game.setPlayers(p1, p2)
    while True:
        game.roll_dice()
        turn = game.getTurn()
        (model_p1 if turn == bg.PlayerType.PLAYER1 else model_p2).make_move(game)
        over, winner = game.is_game_over()
        if over:
            return winner
        game.setTurn(bg.PlayerType.PLAYER2 if turn == bg.PlayerType.PLAYER1
                     else bg.PlayerType.PLAYER1)


def _play_eval(args):
    """Worker task for evaluation: play current model (A) vs opponent (B), with
    A seated as PLAYER1 or PLAYER2 per `a_is_p1`. Returns True iff A won."""
    sd_a, sd_b, game_idx, a_is_p1 = args
    ma, mb = _worker_models["a"], _worker_models["b"]
    ma.load_state_dict(sd_a)
    mb.load_state_dict(sd_b)
    if a_is_p1:
        return _play_head_to_head(ma, mb, game_idx) == bg.PlayerType.PLAYER1
    return _play_head_to_head(mb, ma, game_idx) == bg.PlayerType.PLAYER2


def _snapshot(model):
    """Detached CPU copy of the model weights (picklable for workers)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def evaluate_parallel(pool, cur_sd, opp_sd, num_games=100):
    """Win rate of the current model vs an opponent over `num_games`, sides
    alternated 50/50 to cancel any PLAYER1/PLAYER2 asymmetry. Uses the worker pool."""
    tasks = [(cur_sd, opp_sd, i, (i % 2 == 0)) for i in range(num_games)]
    results = pool.map(_play_eval, tasks)
    return sum(results) / len(results)


def train_parallel(num_games=20000, initial_lr=0.1,
                   checkpoint_path="tdgammonNEW10k.pth",
                   num_workers=24, round_size=None):
    """Parallel self-play trainer (non-interactive). Plays `round_size` games per
    round across `num_workers` processes, then applies TD(λ) updates from all
    trajectories. For the interactive experience use train_cli()."""
    if round_size is None:
        round_size = num_workers

    # The main process only ever does batch-1 forward/backward for the update;
    # extra torch threads just oversubscribe the cores the workers need. Pin it.
    torch.set_num_threads(1)

    model = load_previous_model(checkpoint_path)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    episode_losses = []
    number_of_turns = []

    ctx = mp.get_context("spawn")  # 'spawn' avoids fork/threads issues with torch
    pool = ctx.Pool(processes=num_workers, initializer=_init_worker)

    games_done = 0
    try:
        with tqdm.tqdm(total=num_games, desc="Games") as pbar:
            while games_done < num_games:
                n = min(round_size, num_games - games_done)
                state_dict = _snapshot(model)
                tasks = [(state_dict, games_done + j + 1, 0.0) for j in range(n)]

                model.train()
                k = 0
                for player1_won, states, total_moves in pool.imap_unordered(
                        _play_one, tasks, chunksize=1):
                    model.update_learning_params(games_done + k + 1)
                    for name in model.eligibility_traces:
                        model.eligibility_traces[name].zero_()
                    number_of_turns.append(total_moves)
                    losses = apply_td_updates(model, optimizer, states, player1_won)
                    if losses:
                        episode_losses.append(sum(losses) / len(losses))
                    k += 1
                    pbar.update(1)
                games_done += n
    finally:
        pool.close()
        pool.join()

    plot_all_metrics(number_of_turns, episode_losses, "plots.png")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Model discovery
# ──────────────────────────────────────────────────────────────────────────────

def _model_compatible(path):
    """True iff the checkpoint loads into the current TDLGammonModel architecture."""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        TDLGammonModel().load_state_dict(sd)
        return True
    except Exception:
        return False


def latest_compatible_model():
    """Most recently modified .pth in models/ that fits the current architecture,
    or None. The load test skips the old 3-layer checkpoints automatically."""
    if not os.path.isdir(_models_dir):
        return None
    files = [f for f in os.listdir(_models_dir) if f.endswith(".pth")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(_models_dir, f)), reverse=True)
    for f in files:
        if _model_compatible(os.path.join(_models_dir, f)):
            return f
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard control (non-blocking single keypresses during training)
# ──────────────────────────────────────────────────────────────────────────────

class _KeyListener:
    """Background reader for single keypresses on a TTY, without blocking the
    training loop. Collected keys are drained via pop()."""

    def __init__(self):
        self._keys = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None
        self.enabled = sys.stdin.isatty()

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # leaves ISIG on, so Ctrl-C still works
            while not self._stop:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._keys.append(ch.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def pop(self):
        with self._lock:
            keys = self._keys[:]
            self._keys.clear()
        return keys

    def stop(self):
        self._stop = True


# ──────────────────────────────────────────────────────────────────────────────
# Interactive training CLI
# ──────────────────────────────────────────────────────────────────────────────

def _banner():
    print(r"""
============================================================
   TD(λ) Backgammon — self-play training
============================================================
   While training:   [b] benchmark now    [s] save now
                     [q] quit and save
============================================================
""")


def train_cli(num_games, model_name=None, initial_lr=0.1, num_workers=None,
              round_size=None, num_checkpoints=10, lag_checkpoints=3,
              eval_games=100, out_name=None, eps_start=0.1, eps_end=0.0):
    """Interactive parallel self-play trainer.

    - Resumes from `model_name` (or the latest compatible checkpoint if None).
    - Periodically (≈ num_checkpoints times) reports win rate over `eval_games`
      against the frozen base model and a snapshot `lag_checkpoints` checkpoints
      ago, spaced out so evaluation stays a small fraction of total games.
    - Self-play uses ε-greedy exploration decaying linearly eps_start -> eps_end
      over the run (evaluation is always greedy).
    - Press [b] to benchmark on demand, [s] to save, [q] to quit and save.
    """
    if num_workers is None:
        num_workers = max(1, min(24, (os.cpu_count() or 2) - 2))
    if round_size is None:
        round_size = num_workers
    torch.set_num_threads(1)

    _banner()

    # Resolve the starting model.
    if model_name is None:
        model_name = latest_compatible_model()
    model = TDLGammonModel()
    if model_name:
        path = os.path.join(_models_dir, model_name if model_name.endswith(".pth")
                            else model_name + ".pth")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        print(f" Resuming from: {os.path.basename(path)}")
    else:
        print(" No compatible checkpoint found — starting from fresh weights.")
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    if out_name is None:
        out_name = f"tdgammon_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    out_path = os.path.join(_models_dir, out_name)

    # Frozen reference points for evaluation.
    base_sd = _snapshot(model)
    lag_snaps = deque(maxlen=lag_checkpoints + 1)
    lag_snaps.append(base_sd)

    # Space evaluations ~evenly, rounded to whole rounds, never more often than 1 round.
    eval_interval = max(round_size, num_games // max(1, num_checkpoints))
    eval_interval = max(round_size, (eval_interval // round_size) * round_size)

    number_of_turns = []
    episode_losses = []
    history = []  # (games_done, wr_vs_base, wr_vs_lag_or_None)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers, initializer=_init_worker)
    keys = _KeyListener()
    keys.start()

    def do_eval(tag):
        print(f"\n  [{tag}] running {eval_games}-game benchmark... ", end="", flush=True)
        cur = _snapshot(model)
        wr_base = evaluate_parallel(pool, cur, base_sd, eval_games)
        line = f"vs base: {wr_base*100:5.1f}%"
        wr_lag = None
        if len(lag_snaps) == lag_snaps.maxlen:
            print(f"{line} | vs-lag: ...", end="", flush=True)
            wr_lag = evaluate_parallel(pool, cur, lag_snaps[0], eval_games)
            line += f"  |  vs {lag_checkpoints} checkpoints ago: {wr_lag*100:5.1f}%"
        print(f"\n  [{tag}] after {games_done} games  |  {line}\n")
        history.append((games_done, wr_base, wr_lag))

    def save():
        torch.save(model.state_dict(), out_path)
        print(f"\n  Saved model -> {out_path}\n")

    print(f" Games: {num_games} | workers: {num_workers} | round: {round_size} | "
          f"eval every {eval_interval} games over {eval_games} games | "
          f"ε {eps_start:g}->{eps_end:g}"
          + ("" if keys.enabled else "  (keys disabled: stdin is not a TTY)"))

    games_done = 0
    next_eval = eval_interval
    quit_requested = False
    try:
        with tqdm.tqdm(total=num_games, desc="Training") as pbar:
            while games_done < num_games and not quit_requested:
                n = min(round_size, num_games - games_done)
                state_dict = _snapshot(model)
                # Linearly decay exploration across the run.
                epsilon = eps_start + (eps_end - eps_start) * (games_done / num_games)
                tasks = [(state_dict, games_done + j + 1, epsilon) for j in range(n)]

                model.train()
                k = 0
                for player1_won, states, total_moves in pool.imap_unordered(
                        _play_one, tasks, chunksize=1):
                    model.update_learning_params(games_done + k + 1)
                    for name in model.eligibility_traces:
                        model.eligibility_traces[name].zero_()
                    number_of_turns.append(total_moves)
                    losses = apply_td_updates(model, optimizer, states, player1_won)
                    if losses:
                        episode_losses.append(sum(losses) / len(losses))
                    k += 1
                    pbar.update(1)
                games_done += n

                # Handle keypresses collected during the round. Deduplicate so
                # that auto-repeat / mashing a key runs its action ONCE per round
                # (otherwise holding 'b' queues dozens of benchmarks).
                pressed = set(keys.pop())
                if "q" in pressed:
                    quit_requested = True
                    print("\n  [q] quitting — saving current model...")
                if "s" in pressed:
                    save()
                if "b" in pressed and not quit_requested:
                    try:
                        do_eval("manual benchmark")
                    except Exception as exc:
                        print(f"\n  [b] benchmark failed: {exc}\n")

                # Scheduled checkpoint evaluation.
                if games_done >= next_eval and not quit_requested:
                    do_eval("checkpoint")
                    lag_snaps.append(_snapshot(model))
                    next_eval += eval_interval
    except KeyboardInterrupt:
        print("\n  Interrupted — saving current model...")
    finally:
        keys.stop()
        pool.terminate()  # stop workers promptly; close()+join() can hang on Ctrl-C
        pool.join()

    save()
    if history:
        print("  Eval history (games | vs base | vs lagged):")
        for g, wb, wl in history:
            wl_s = f"{wl*100:5.1f}%" if wl is not None else "   -  "
            print(f"    {g:>7} | {wb*100:5.1f}% | {wl_s}")
    try:
        plot_all_metrics(number_of_turns, episode_losses, "plots.png")
    except Exception as e:
        print(f"  (skipped metrics plot: {e})")
    return model


def main():
    parser = argparse.ArgumentParser(description="TD(λ) Backgammon self-play training CLI")
    parser.add_argument("-n", "--num-games", type=int, default=None,
                        help="number of self-play games to train (prompts if omitted)")
    parser.add_argument("-m", "--model", default=None,
                        help="starting checkpoint name (default: latest compatible)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel self-play workers (default: cores-2, capped at 24)")
    parser.add_argument("--round-size", type=int, default=None,
                        help="games per weight snapshot (default: workers)")
    parser.add_argument("--checkpoints", type=int, default=10,
                        help="approximate number of scheduled evaluations (default: 10)")
    parser.add_argument("--lag", type=int, default=3,
                        help="evaluate against the snapshot this many checkpoints ago (default: 3)")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="games per win-rate evaluation (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--eps-start", type=float, default=0.1,
                        help="initial ε-greedy exploration during self-play (default: 0.1)")
    parser.add_argument("--eps-end", type=float, default=0.0,
                        help="final ε-greedy exploration, decayed to over the run (default: 0.0)")
    parser.add_argument("--out", default=None,
                        help="output checkpoint filename (default: timestamped tdgammonSYM_*)")
    args = parser.parse_args()

    num_games = args.num_games
    if num_games is None:
        if sys.stdin.isatty():
            resp = input("How many games to train? [20000]: ").strip()
            num_games = int(resp) if resp else 20000
        else:
            num_games = 20000

    train_cli(num_games=num_games, model_name=args.model, initial_lr=args.lr,
              num_workers=args.workers, round_size=args.round_size,
              num_checkpoints=args.checkpoints, lag_checkpoints=args.lag,
              eval_games=args.eval_games, out_name=args.out,
              eps_start=args.eps_start, eps_end=args.eps_end)


if __name__ == "__main__":
    main()
