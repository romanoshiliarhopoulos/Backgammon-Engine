SRCDIR    := cppsrc
BUILD_DIR := build
TARGET    := a.out

CXX      := g++
CXXFLAGS := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function -I$(SRCDIR)

# All nvidia libs bundled with torch (needed for dlopen at import time).
# Use ABSOLUTE paths ($(CURDIR)) — recipes cd into the training dir, so relative
# entries would no longer resolve and torch would fail to import its CUDA libs.
NVIDIA_LIBS := $(shell find $(CURDIR)/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH := $(NVIDIA_LIBS)$(LD_LIBRARY_PATH)

PYTHON     := poetry run python
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Debug \
               -DPYTHON_EXECUTABLE=$(shell poetry run which python)

# ── Training configuration ──────────────────────────────────────────────────
# Override any of these on the command line, e.g.
#   make train-bg GAMES=50000 WORKERS=24 CHECKPOINTS=20 EPS_START=0.15
SESSION   ?= bgtrain
TRAINDIR  := $(CURDIR)/pysrc/TD(λ) model
LOGFILE   := $(CURDIR)/train.log

GAMES       ?= 20000
MODEL       ?=
WORKERS     ?=
ROUND       ?=
CHECKPOINTS ?=
LAG         ?=
EVALGAMES   ?=
LR          ?=
EPS_START   ?=
EPS_END     ?=
OUT         ?=

# Only emit a flag when its variable is set, so train.py's own defaults apply.
TRAIN_ARGS := -n $(GAMES) \
  $(if $(MODEL),-m $(MODEL)) \
  $(if $(WORKERS),--workers $(WORKERS)) \
  $(if $(ROUND),--round-size $(ROUND)) \
  $(if $(CHECKPOINTS),--checkpoints $(CHECKPOINTS)) \
  $(if $(LAG),--lag $(LAG)) \
  $(if $(EVALGAMES),--eval-games $(EVALGAMES)) \
  $(if $(LR),--lr $(LR)) \
  $(if $(EPS_START),--eps-start $(EPS_START)) \
  $(if $(EPS_END),--eps-end $(EPS_END)) \
  $(if $(OUT),--out $(OUT))

SRCS := $(filter-out \
           $(SRCDIR)/tests.cpp \
           $(SRCDIR)/backgammon_bindings.cpp, \
         $(wildcard $(SRCDIR)/*.cpp))

.PHONY: all run build test pytest clean cmake_configure cmake_build train bench play git \
        train-bg train-attach train-log train-status train-stop train-kill

all: $(TARGET) build

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	clear
	./$(TARGET)

cmake_configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

cmake_build: cmake_configure
	cmake --build $(BUILD_DIR) -- -j$(shell nproc)

build: cmake_build

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

pytest:
	$(PYTHON) -m pytest pysrc/tests.py -v

# Foreground interactive training (dies if the SSH session drops).
train: build
	cd "$(TRAINDIR)" && $(PYTHON) train.py $(TRAIN_ARGS)

# Start training inside a detached tmux session so it survives SSH disconnects.
# LD_LIBRARY_PATH is embedded because a pre-existing tmux server would otherwise
# not inherit it. Pane output is also streamed to $(LOGFILE).
train-bg: build
	@if tmux has-session -t $(SESSION) 2>/dev/null; then \
	  echo "tmux session '$(SESSION)' already exists. 'make train-attach' to view, or 'make train-kill' first."; \
	  exit 1; \
	fi
	@: > "$(LOGFILE)"
	@tmux new-session -d -s $(SESSION) -c "$(TRAINDIR)" \
	  "LD_LIBRARY_PATH='$(LD_LIBRARY_PATH)' $(PYTHON) train.py $(TRAIN_ARGS)"
	@tmux pipe-pane -o -t $(SESSION) "cat >> '$(LOGFILE)'"
	@echo "Training started in tmux session '$(SESSION)'  (args: $(TRAIN_ARGS))"
	@echo "  make train-attach   # live view + [b]/[s]/[q] keys  (detach: Ctrl-b then d)"
	@echo "  make train-status   # quick snapshot without attaching"
	@echo "  make train-log      # follow the log file"
	@echo "  make train-stop     # graceful: save current model and exit"
	@echo "  make train-kill     # hard stop, no save"

# Re-attach to the live training session (use after re-SSHing).
train-attach:
	@tmux attach -t $(SESSION)

# Follow the rolling log (Ctrl-C stops following, not training).
train-log:
	@tail -n 60 -f "$(LOGFILE)"

# Snapshot the current pane without attaching.
train-status:
	@if tmux has-session -t $(SESSION) 2>/dev/null; then \
	  echo "Session '$(SESSION)' is RUNNING:"; \
	  tmux capture-pane -p -t $(SESSION) | grep -v '^$$' | tail -n 15; \
	else \
	  echo "No training session '$(SESSION)' running. Last log lines:"; \
	  tail -n 15 "$(LOGFILE)" 2>/dev/null || echo "  (no log yet)"; \
	fi

# Graceful stop: send 'q' so train.py saves the current model, then exits.
train-stop:
	@if tmux has-session -t $(SESSION) 2>/dev/null; then \
	  tmux send-keys -t $(SESSION) q; \
	  echo "Sent quit+save to '$(SESSION)'. It saves after the current round, then the session ends."; \
	else echo "No training session '$(SESSION)' running."; fi

# Hard stop (no save) — kills the tmux session immediately.
train-kill:
	@tmux has-session -t $(SESSION) 2>/dev/null && { tmux kill-session -t $(SESSION); echo "Killed '$(SESSION)' (no save)."; } || echo "No session '$(SESSION)'."

bench:
	$(PYTHON) pysrc/benchmark.py

play:
	$(PYTHON) pysrc/play_model.py

git:
	git add .
	git commit -m "new"
	git push

clean:
	rm -rf $(TARGET) *.o $(BUILD_DIR)
