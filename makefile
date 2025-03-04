run: 
	rm -f ./a.out
	g++ -std=c++17 -g -Wall main.cpp game.hpp game.cpp -lm -Wno-unused-variable -Wno-unused-function
	./a.out