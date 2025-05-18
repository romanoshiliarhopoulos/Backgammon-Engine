# --- Source directory (where all your .cpp/.hpp now live) ---
SRCDIR       := cppsrc

# --- Compiler settings ---
CXX          := g++
CXXFLAGS     := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function \
                 -I$(SRCDIR)

# --- Collect all .cpp files under cppsrc/ except tests.cpp ---
SRCS         := $(filter-out $(SRCDIR)/tests.cpp,$(wildcard $(SRCDIR)/*.cpp))
TARGET       := a.out

# --- CMake build variables ---
BUILD_DIR    := build
CMAKE_FLAGS  := -DCMAKE_BUILD_TYPE=Debug

.PHONY: all run build test clean cmake_configure cmake_build

# Default: build both the CLI app and the CMake project
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

# 4) Run all GoogleTest cases via CTest
test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

# Cleans both the direct build artifacts and the CMake tree
clean:
	rm -rf $(TARGET) *.o $(BUILD_DIR)
