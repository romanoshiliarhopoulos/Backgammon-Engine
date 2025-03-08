/*main.cpp*/
#include <iostream>
#include <chrono>
#include <thread>
#include "game.hpp"

using namespace std;

#include <iostream>

int main()
{
    cout << "Hello" << endl;
    Game game1 = Game(4);
    game1.printGameBoard();
    game1.clearGameboard(); // works!
    game1.printGameBoard();
    return 0;
}