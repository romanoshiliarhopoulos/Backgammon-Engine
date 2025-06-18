# --- Source directory (where all your .cpp/.hpp now live) ---
SRCDIR       := cppsrc

# --- Pull in pybind11’s include flags via: python3 -m pybind11 --includes ---
PYBIND11_INC := $(shell python3 -m pybind11 --includes)

# --- Compiler settings ---
CXX          := g++
CXXFLAGS     := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function \
                 -I$(SRCDIR) $(PYBIND11_INC)

# --- Collect all .cpp files under cppsrc/ except tests.cpp and the pybind bindings ---
SRCS         := $(filter-out \
                   $(SRCDIR)/tests.cpp \
                   $(SRCDIR)/backgammon_bindings.cpp, \
                 $(wildcard $(SRCDIR)/*.cpp))
TARGET       := a.out

# --- CMake build variables ---
BUILD_DIR    := build
CMAKE_FLAGS  := -DCMAKE_BUILD_TYPE=Debug

.PHONY: all run build test clean cmake_configure cmake_build

# Default: build both the CLI  the CMake 
all: $(TARGET) build

# 1) Direct build of your CLI executable (excludes tests.cpp)
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	clear
	./$(TARGET)

# 2) CMake configure step
cmake_configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

# 3) CMake build step (depends on configure)
cmake_build: cmake_configure
	cmake --build $(BUILD_DIR)

build: cmake_build


# 4) Run all GoogleTest cases via 
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure


# Cleans both the direct build artifacts and the CMake 
clean:
	rm -rf $(TARGET) *.o $(BUILD_DIR)

make train:
	clear
	/Users/romanos/miniconda3/bin/python /Users/romanos/Backgammon_Engine/train.py
make bench:
	clear
	/Users/romanos/miniconda3/bin/python "/Users/romanos/Backgammon_Engine/pysrc/model v2/benchmark.py"

make train2:
	clear
	/Users/romanos/miniconda3/bin/python "/Users/romanos/Backgammon_Engine/pysrc/model v2/train model2.py"