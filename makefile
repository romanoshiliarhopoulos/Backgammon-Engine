.PHONY: clean all run

CXX      := g++
CXXFLAGS := -std=c++17 -g -Wall -Wno-unused-variable -Wno-unused-function

SRCS     := main.cpp game.cpp Pieces.cpp player.cpp
TARGET   := a.out

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: all
	clear
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
