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
        game.setPlayers(&p1, &p2); // pass the players onto the game class

        // starting the mechanics of the game
        Player *p1_addr = &p1;
        Player *p2_addr = &p2;

        Player *current_player;

        // Rolling the dice to determine who goes first
        string response = "o";
        while (response != "r")
        {
            cout << "Press r for both to roll a dice: ";
            cin >> response;
        }
        char c;
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
            current_player = p1_addr;
            cout << current_player->getName() << " goes first!" << endl;
            game.setTurn(Player::PLAYER1);
        }
        else
        {
            // p2 goes first
            current_player = p2_addr;
            cout << current_player->getName() << " goes first!" << endl;
            game.setTurn(Player::PLAYER2);
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
            std::cout << game.renderBoard();

            std::cout << "Press r to roll: ";
            while (std::cin >> c && c != 'r')
                std::cout << "Press r: ";
            int dice1 = rollDice(), dice2 = rollDice();
            std::cout << "Rolled " << dice1 << "," << dice2 << "\n";

            // Build moves and choice logic (replicates prior interactive flows)
            std::vector<std::pair<int, int>> moves;
            bool firstDiceFirst = true;
            if (dice1 != dice2)
            {
                int choice;
                std::cout << "Choose die first (" << dice1 << " or " << dice2 << "): ";
                std::cin >> choice;
                firstDiceFirst = (choice == dice1);
                // then read two origin/dest pairs
                for (int i = 0; i < 2; i++)
                {
                    int o, d;
                    std::cout << "Origin (" << (i == 0 ? (firstDiceFirst ? dice1 : dice2) : (firstDiceFirst ? dice2 : dice1)) << "): ";
                    std::cin >> o;
                    std::cout << "Destination: ";
                    std::cin >> d;
                    moves.emplace_back(o, d);
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    int o, d;
                    std::cout << "Origin (" << dice1 << "): ";
                    std::cin >> o;
                    std::cout << "Destination: ";
                    std::cin >> d;
                    moves.emplace_back(o, d);
                }
            }
            std::string err;
            if (!game.movePieces((game.getTurn() == Player::PLAYER1 ? &p1 : &p2),
                                 dice1, dice2, moves,
                                 firstDiceFirst, err))
            {
                std::cout << "Error: " << err << "\n";
                continue; // retry same player
            }
            // switch turn
            game.setTurn(game.getTurn() == Player::PLAYER1 ? Player::PLAYER2 : Player::PLAYER1);
        }

        break;
    }

    return 0;
}