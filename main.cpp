/*main.cpp*/
#include <iostream>
#include <chrono>
#include <thread>
#include "game.hpp"

using namespace std;

#include <iostream>
void printBanner()
{
    std::cout << R"(
·············································································
: ____    _    ____ _  ______    _    __  __ __  __  ___  _   _             :
:| __ )  / \  / ___| |/ / ___|  / \  |  \/  |  \/  |/ _ \| \ | |            :
:|  _ \ / _ \| |   | ' | |  _  / _ \ | |\/| | |\/| | | | |  \| |            :
:| |_) / ___ | |___| . | |_| |/ ___ \| |  | | |  | | |_| | |\  |  _   _   _ :
:|____/_/   \_\____|_|\_\____/_/   \_|_|  |_|_|  |_|\___/|_| \_| (_) (_) (_):
·············································································
)" << std::endl;
}

int main()
{

    printBanner();
    Game game1 = Game(4);
    std::cout << "⚀ ⚁ ⚂ ⚃ ⚄ ⚅" << std::endl;

    // for the main game loop
    while (true)
    {
        string p1_name;
        string p2_name;
        cout << "Enter a name for Player 1: ";
        cin >> p1_name;
        cout << "Enter a name for Player 2: ";
        cin >> p2_name;
        cout << endl;

        // create the player objects
        Player p1(p1_name);
        Player p2(p2_name);

        // Rolling the dice to determin who goes first
        string response = "o";
        while (response != "r")
        {
            cout << "Press r for both to roll a dice: ";
            cin >> response;
        }
        

        break;
    }
    game1.printGameBoard();
    game1.clearGameboard(); // works!
    game1.printGameBoard();
    return 0;
}