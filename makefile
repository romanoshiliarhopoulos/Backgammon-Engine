# Makefile for both direct CLI build and CMake+CTest

# --- Direct CLI build variables ---
CXX       := g++
CXXFLAGS  := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function

SRCS      := main.cpp game.cpp Pieces.cpp player.cpp
TARGET    := a.out

# --- CMake build variables ---
BUILD_DIR    := build
CMAKE_FLAGS  := -DCMAKE_BUILD_TYPE=Debug

.PHONY: all       run     build   test    clean cmake_configure cmake_build

# Default: build both the CLI app and the CMake project
all: $(TARGET) build

# 1) Direct build of your CLI executable
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

# make build is an alias for cmake_build
build: cmake_build

# 4) Run all GoogleTest cases via CTest
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

# Cleans both the direct build artifacts and the CMake tree
clean:
	rm -rf $(TARGET) *.o $(BUILD_DIR)
