run: 
	rm -f ./a.out
	g++ -std=c++17 -g -Wall main.cpp game.hpp game.cpp Pieces.cpp Pieces.hpp player.cpp player.hpp -lm -Wno-unused-variable -Wno-unused-function
	./a.out