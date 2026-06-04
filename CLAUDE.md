# Backgammon Engine — CLAUDE.md

NEVER COMMIT ANYTHING TO GITHUB< LEAVE IT ALL TO ME>
## How to use this file
This file is auto-loaded by Claude Code at the start of every session.
**If you discover something critical** — an env quirk, a breaking constraint, a non-obvious bug, an architectural invariant — append it to the [Critical Notes](#critical-notes) section before ending the session. Keep entries concise: one heading, one paragraph max. Do not add things already derivable from reading the code.

---

## Environment

- **OS:** Linux x86_64 (RHEL9). **Two RTX 4090 GPUs are present** (`/dev/nvidia0/1`, `torch.cuda.is_available()` is True). There is no `nvidia-smi` on PATH, which is why earlier notes wrongly said "no GPU." The TD(λ) net (198→128→1) is far too small to benefit from GPU — training is bound by per-move CPU/C++ overhead and parallelism, not matrix compute, so it runs on CPU.
- **Python:** 3.12.5 via Poetry venv at `.venv/`; Poetry 2.2.1
- **Torch:** 2.9.1+cu128 (CUDA build). It will fail to import unless `LD_LIBRARY_PATH` includes the nvidia libs bundled inside the venv. The `makefile` exports this automatically via a `find` over `.venv/lib/python3.12/site-packages/nvidia/*/lib`. When running Python outside of `make`, prefix with that path or it will crash with `libcusparseLt.so.0: cannot open shared object file`.
- **C++ standard:** C++17 (game.cpp uses structured bindings). CMakeLists is set to 17.
- **Filesystem:** Linux is case-sensitive. Includes must use `Pieces.hpp` not `pieces.hpp`.
- **pybind11:** fetched via CMake FetchContent — do not need to install separately. Also listed in `pyproject.toml` so `python -m pybind11 --includes` works in the makefile.

---

## Architecture

```
cppsrc/          C++17 game logic — Game, Player, Pieces
  backgammon_bindings.cpp  pybind11 module → build/backgammon_env.*.so
pysrc/
  TD(λ) model/
    model.py     TDLGammonModel: 2-layer MLP (198→128→1), TD(λ) RL
    train.py     self-play training loop; saves to models/
  play_model.py  interactive CLI (questionary arrow-key model picker)
  benchmark.py   model vs model / model vs random
  tests.py       pytest suite (3 tests)
models/          saved .pth weights
  tdgammonNEW*.pth        ← match current TDLGammonModel architecture
  bestModel.pth           ← OLD 3-layer arch, incompatible — do not load into TDLGammonModel
  tdgammon_model100.pth   ← OLD 3-layer arch, incompatible
```

The Python extension `backgammon_env` must be built before any Python code runs (`make build`). It lives in `build/`.

---

## Make Targets

| Target       | What it does |
|--------------|--------------|
| `make build` | CMake: builds `backgammon_env.so` + `backgammon_tests` binary |
| `make test`  | 36 C++ GoogleTests via ctest |
| `make pytest`| 3 Python tests via pytest |
| `make train` | Self-play TD(λ) training; saves new model to `models/` |
| `make play`  | Interactive CLI — arrow-key model picker, infers die from move |
| `make bench` | Benchmark two models head-to-head |
| `make run`   | Run the raw C++ CLI executable (`a.out`) |
| `make clean` | Remove `build/` and `a.out` |

---

## Known Issues

- **C++ test #31** (`LegalTurnSeq.trying_replicate_another`) has a pre-existing failure — `legalTurnSequences` returns 0 sequences when 2 are expected. Not introduced by any recent change.
- **`make test` exit code** is non-zero when test #31 fails, which makes `make` print an error. This is expected and not a build problem.

---

## Critical Notes

> Future agents: append critical discoveries here. Format — `### <short title>` followed by one concise paragraph. Only add things that are non-obvious and would take meaningful time to rediscover from the code alone.

### PLAYER1 could not bear off — fixed in `canFreePiece` (was the real cause of lopsided self-play)
`Game::canFreePiece` had a broken PLAYER1 branch: the overrun guard `dice > origin - 25` was always true, and the inner loop indexed `gameboard[i]` by point number (off-by-one) and read out of bounds at `gameboard[24]`. Effect: P1 essentially could never bear off — pure random-vs-random was P2 200/200 (P1 bore off ~0.3 checkers/game). This — not the value-function formulation — was why every self-play run looked P1-hopeless. Fixed to mirror the working P2 logic: exact pips for P1 = `25 - origin`, overrun only when `dice > 25 - origin`, checking P1 checkers on higher points (indices `origin..23`). After the fix random-vs-random is ~50/50. If self-play ever looks badly one-sided again, re-check bearing-off symmetry with a random-vs-random sanity test before blaming the model.

### Encoding is the original global PLAYER1/PLAYER2 frame (kept for checkpoint compatibility)
A symmetric side-to-move encoding was prototyped to remove the P1/P2 value asymmetry, but it was reverted because it broke all existing `tdgammonNEW*` checkpoints and — once the bear-off bug was fixed — was not actually needed (the real asymmetry was the engine bug, not the encoding). `_encode_states_np` therefore stays in the original global frame: P1 in point slots 0-3, P2 in 4-7, turn bits 192/193, `V ≈ P(PLAYER1 wins)`; `make_move` has P1 maximize and P2 minimize V; `apply_td_updates` uses target `V(s_{t+1})` and terminal reward 1/0 by `player1_won`. ε-greedy exploration WAS kept (`make_move(..., epsilon=)`, decayed in `train_cli`) — it does not change encoding semantics, so old checkpoints remain loadable.

### Training speed is dominated by per-candidate `clone()`, not the net
Each move evaluates every legal turn sequence by cloning the game (~31µs each) and running a batch-1 forward pass. Self-play does ~22k clones/game, so the cost is Python↔C++ crossings + torch call overhead, not matrix math. `Game::evaluateTurnSequences()` (C++) returns all candidate sequences AND their resulting board states in one call; `model.make_move` then encodes them with one vectorized NumPy pass (`_encode_states_np`) and scores them in a single batched forward. This is ~5.4× faster per move and byte-identical to the old path. Don't reintroduce per-candidate `clone()`/`encode_state` loops.

### Self-play is parallelized with `multiprocessing` spawn — guard `__main__` carefully
`train_cli`/`train_parallel` use a `spawn` Pool; workers re-import `train.py`, so any training call must stay behind `if __name__ == "__main__"`. A duplicated tail once left a second `main()` that fired `train_parallel(20000)` on import — watch for that. Workers set `torch.set_num_threads(1)` and the main process pins to 1 thread too (batch-1 updates don't need threads and would oversubscribe the workers' cores). Parallel scaling is currently only ~1.8× (update phase + IPC are the bottleneck, not yet fully solved); the single-process fast path is the guaranteed win.

### `make train` is now an interactive CLI
`train.py` is an argparse CLI (`train_cli`): resumes from the latest compatible checkpoint if `-m` omitted, takes `-n` games, periodically evaluates win rate (100 games, sides alternated) vs the frozen base and a lagged snapshot, and supports live keys `[b]` benchmark / `[s]` save / `[q]` quit+save (TTY only). Eval/`make play` compatibility helpers: `latest_compatible_model()` skips the old 3-layer checkpoints.
