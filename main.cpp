/*main.cpp*/
#include <iostream>
#include <chrono>
#include <thread>
#include "game.hpp"

using namespace std;

#include <iostream>

void clearGameboard(int gameboardLines)
{
    if (gameboardLines > 0)
    {
        // Move cursor up to the beginning of the gameboard
        std::cout << "\033[" << gameboardLines << "F";

        // Clear all lines
        for (int i = 0; i < gameboardLines; i++)
        {
            // Clear the entire line
            std::cout << "\033[2K";

            // Move to the next line (except for the last iteration)
            if (i < gameboardLines - 1)
            {
                std::cout << "\n";
            }
        }

        // Return cursor to beginning of the first line
        std::cout << "\r" << std::flush;
    }
}

int main()
{
    cout << "Hello" << endl;
    Game game1 = Game(4);
    game1.printGameBoard();
    cout << "◉ ◯" << endl;
    /*
    cout << "Leon" << endl;
    cout << "HAHA" << endl;
    clearGameboard(2);
    */
    return 0;
}