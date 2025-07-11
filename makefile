# --- Source directory ---
SRCDIR       := cppsrc

# Use the system Python where pybind11 is installed
PYTHON_CMD := "C:/Users/roman/AppData/Local/Programs/Python/Python310/python.exe"
PYBIND11_INC := $(shell $(PYTHON_CMD) -m pybind11 --includes)

# --- Detect OS for cross-platform commands ---
ifeq ($(OS),Windows_NT)
    CLEAR_CMD := cls
    RM_CMD := del /Q
    MKDIR_CMD := mkdir
else
    CLEAR_CMD := clear
    RM_CMD := rm -rf
    MKDIR_CMD := mkdir -p
endif

# --- Compiler settings ---
CXX          := g++
CXXFLAGS     := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function \
                 -I$(SRCDIR) $(PYBIND11_INC)

# --- Collect all .cpp files ---
SRCS         := $(filter-out \
                   $(SRCDIR)/tests.cpp \
                   $(SRCDIR)/backgammon_bindings.cpp, \
                 $(wildcard $(SRCDIR)/*.cpp))
TARGET       := a.out

# --- CMake build variables with explicit Python ---
BUILD_DIR    := build
CMAKE_FLAGS  := -DCMAKE_BUILD_TYPE=Debug \
                -G "MinGW Makefiles" \
                -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/gcc.exe \
                -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe \
                -DPYTHON_EXECUTABLE=$(PYTHON_CMD) \
                -DPython_EXECUTABLE=$(PYTHON_CMD) \
                -Dpybind11_DIR="$(shell $(PYTHON_CMD) -m pybind11 --cmakedir)"

.PHONY: all run build test clean cmake_configure cmake_build

all: $(TARGET) build

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	cls
	./$(TARGET)

cmake_configure:
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

cmake_build: cmake_configure
	cmake --build $(BUILD_DIR)

build: cmake_build

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
ifeq ($(OS),Windows_NT)
	if exist $(TARGET) del $(TARGET)
	if exist *.o del *.o
	if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)
else
	rm -rf $(TARGET) *.o $(BUILD_DIR)
endif

train:
	$(CLEAR_CMD)
	python train.py

train2:
	$(CLEAR_CMD)
	python "pysrc/model v2/train model2.py"

# Add this new target
install_module: build
	if exist build\backgammon_env.cp310-win_amd64.pyd copy build\backgammon_env.cp310-win_amd64.pyd .\backgammon_env.pyd

train3: install_module
	cmd /c "set PATH=C:\msys64\ucrt64\bin;C:\msys64\ucrt64\lib;C:\msys64\usr\bin;%PATH% && set PYTHONPATH=C:\Users\roman\Documents\Backgammon-Engine;C:\Users\roman\Documents\Backgammon-Engine && C:/Users/roman/AppData/Local/Programs/Python/Python310/python.exe pysrc/modelTD/train3.py"


bench:
	python "pysrc/modelTD/benchmark.py"
