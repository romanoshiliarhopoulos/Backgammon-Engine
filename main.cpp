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

    // for the main game loop
    string input;
    cout << "enter a num" << endl;
    cin >> input;
    cout << input << endl;
    game1.printGameBoard();
    game1.clearGameboard(); // works!
    game1.printGameBoard();
    return 0;
}