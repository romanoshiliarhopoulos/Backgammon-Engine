SRCDIR    := cppsrc
BUILD_DIR := build
TARGET    := a.out

CXX      := g++
CXXFLAGS := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function -I$(SRCDIR)

# All nvidia libs bundled with torch (needed for dlopen at import time)
NVIDIA_LIBS := $(shell find .venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH := $(NVIDIA_LIBS)$(LD_LIBRARY_PATH)

PYTHON     := poetry run python
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Debug \
               -DPYTHON_EXECUTABLE=$(shell poetry run which python)

SRCS := $(filter-out \
           $(SRCDIR)/tests.cpp \
           $(SRCDIR)/backgammon_bindings.cpp, \
         $(wildcard $(SRCDIR)/*.cpp))

.PHONY: all run build test pytest clean cmake_configure cmake_build train bench play git

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

train:
	cd "pysrc/TD(λ) model" && $(PYTHON) train.py

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
