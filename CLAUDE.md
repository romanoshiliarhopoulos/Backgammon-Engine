# Backgammon Engine — CLAUDE.md

## How to use this file
This file is auto-loaded by Claude Code at the start of every session.
**If you discover something critical** — an env quirk, a breaking constraint, a non-obvious bug, an architectural invariant — append it to the [Critical Notes](#critical-notes) section before ending the session. Keep entries concise: one heading, one paragraph max. Do not add things already derivable from reading the code.

---

## Environment

- **OS:** Linux x86_64 (RHEL9), no GPU, no CUDA drivers
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
