/*main.cpp*/
#include <iostream>
#include <chrono>
#include <thread>
#include <stdint.h>

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

int rollDice()
{
    // returns a number between 1 and 6

    return arc4random_uniform(5) + 1;
}

int main()
{
    printBanner();
    Game game = Game(4);

    // for the main game loop
    while (true)
    {
        string p1_name;
        string p2_name;
        cout << "Enter a name for Player 1: ";
        cin >> p1_name;
        cout << "Enter a name for Player 2: ";
        cin >> p2_name;
        while (p1_name == p2_name)
        {
            cout << "Names must be different: ";
            cin >> p2_name;
        }
        cout << endl;

        // create the player objects
        Player p1(p1_name, 0);
        Player p2(p2_name, 1);

        // starting the mechanics of the game
        Player *p1_addr = &p1;
        Player *p2_addr = &p2;

        Player *current_player;

        // Rolling the dice to determin who goes first
        string response = "o";
        while (response != "r")
        {
            cout << "Press r for both to roll a dice: ";
            cin >> response;
        }
        int p1_dice = rollDice();
        int p2_dice = rollDice();
        while (p2_dice == p1_dice)
        {
            // roll again,
            p1_dice = rollDice();
            p2_dice = rollDice();
        }
        cout << p1.getName() << " rolled a: " << p1_dice << ", " << p2.getName() << " rolled a: " << p2_dice << ". ";

        if (p1_dice > p2_dice)
        {
            // p1 goes first
            cout << p1.getName() << " goes first!" << endl;
            game.setTurn(Player::PLAYER1);
            current_player = p1_addr;
        }
        else
        {
            // p2 goes first
            cout << p1.getName() << " goes first!" << endl;
            game.setTurn(Player::PLAYER2);
            current_player = p2_addr;
        }
        string responseStrart;
        cout << "Press s to start or q to quit ... \n";
        cin >> responseStrart;
        while (responseStrart != "s" && responseStrart != "q")
        {
            cout << "Press s to start or q to quit ...";
            cin >> responseStrart;
        }
        if (responseStrart == "q")
        {
            return 0;
        }

        cout << "\n*-------------------------------------------------*" << endl;
        // for the game mechanics . . .

        while (!game.over())
        {
            cout << "Turn: " << current_player->getName() << endl;
            // player rolls both dice first

            if (current_player == p1_addr)
            {
                // next player must be p2
                current_player = p2_addr;
            }
            else
            {
                current_player = p1_addr;
            }
        }

        break;
    }
    game.printGameBoard();
    game.clearGameboard(); // works!
    game.printGameBoard();
    return 0;
}