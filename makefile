# --- Source directory (where all your .cpp/.hpp now live) ---
SRCDIR       := cppsrc

# --- Pull in pybind11’s include flags via: python3 -m pybind11 --includes ---
PYBIND11_INC := $(shell python3 -m pybind11 --includes)

PYTHON := /Users/romanos/miniconda3/bin/python


# --- Compiler settings ---
CXX          := g++
# ↓ Added -fsanitize=address,undefined here to catch OOB / UB in your C++ extension
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
# ↓ Pass ASan into CMake as well
CMAKE_FLAGS  := -DCMAKE_BUILD_TYPE=Debug \
                -DCMAKE_CXX_FLAGS="-g -fsanitize=address,undefined"

.PHONY: all run build test clean cmake_configure cmake_build train

# Default: build both the CLI and the CMake extension (and then run training)
all: $(TARGET) build

# Direct build of  CLI executable (excludes tests.cpp)
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	clear
	./$(TARGET)

#CMake configure step
cmake_configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

# 3) CMake build step (depends on configure)
cmake_build: cmake_configure
	cmake --build $(BUILD_DIR)

# Build  CMake extension, then automatically kick off Python training under ASan
build: cmake_build
	@echo "→ Rebuild complete. Now running Python training under AddressSanitizer…"

#Run all GoogleTest cases via CTest
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure


clean:
	rm -rf $(TARGET) *.o $(BUILD_DIR)

