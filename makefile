run: 
	rm -f ./a.out
	g++ -std=c++17 -g -Wall main.cpp game.hpp game.cpp Pieces.cpp Pieces.hpp player.cpp player.hpp -lm -Wno-unused-variable -Wno-unused-function
	clear
	./a.out
tests:
	g++ -std=c++17 -o my_test tests.cpp game.cpp Pieces.cpp player.cpp -lgtest -lgtest_main -pthread -lm -Wno-unused-variable -Wno-unused-function

	