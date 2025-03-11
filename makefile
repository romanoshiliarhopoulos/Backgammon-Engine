run: 
	rm -f ./a.out
	g++ -std=c++17 -g -Wall main.cpp game.hpp game.cpp Pieces.cpp Pieces.hpp player.cpp player.hpp -lm -Wno-unused-variable -Wno-unused-function
	clear
	./a.out
tests:
	c++ -std=c++17 tests.cpp game.cpp Pieces.cpp player.cpp -o tests -I/opt/homebrew/include -L/opt/homebrew/lib -lgtest -lgtest_main -pthread
