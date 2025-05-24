// main.cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib> // for arc4random_uniform
#include <array>
#include <utility>
#include <string>

#include "game.hpp"

using namespace std;

void printBanner()
{
    cout << R"(
·············································································
: ____    _    ____ _  ______    _    __  __ __  __  ___  _   _             :
:| __ )  / \  / ___| |/ / ___|  / \  |  \/  |  \/  |/ _ \| \ | |            :
:|  _ \ / _ \| |   | ' | |  _  / _ \ | |\/| | |\/| | | | |  \| |            :
:| |_) / ___ | |___| . | |_| |/ ___ \| |  | | |  | | |_| | |\  |  _   _   _ :
:|____/_/   \_\____|_|\_\____/_/   \_|_|  |_|_|  |_|\___/|_| \_| (_) (_) (_):
·············································································
)" << endl;
}

int rollDice()
{
    // original logic: returns 1..5
    return arc4random_uniform(5) + 1;
}

/// Prompt for two ints (origin, destination) on one line.
pair<int, int> promptPair(const string &prompt)
{
    int o, d;
    cout << prompt << " (origin dest): ";
    cin >> o >> d;
    return {o, d};
}

/// Ask which die to play first; only accepts a==d1 or a==d2.
int promptDieChoice(int d1, int d2)
{
    int c;
    cout << "Choose which die to play first (" << d1 << " or " << d2 << "): ";
    cin >> c;
    while (c != d1 && c != d2)
    {
        cout << "Invalid.  Enter " << d1 << " or " << d2 << ": ";
        cin >> c;
    }
    return c;
}

int main()
{
    printBanner();
    Game game(4);
    int playerWon = -1;
    string p1_name, p2_name;

    while (true)
    {
        // —— set up players ——

        cout << "Enter a name for Player 1: ";
        cin >> p1_name;
        cout << "Enter a name for Player 2: ";
        cin >> p2_name;
        while (p1_name == p2_name)
        {
            cout << "Names must be different.  Enter Player 2 name: ";
            cin >> p2_name;
        }
        cout << endl;

        Player p1(p1_name, Player::PLAYER1), p2(p2_name, Player::PLAYER2);
        game.setPlayers(&p1, &p2);

        // —— decide who goes first ——
        cout << "Press r for both to roll a dice: ";
        string rollBoth;
        cin >> rollBoth;
        while (rollBoth != "r")
        {
            cout << "Press r to roll: ";
            cin >> rollBoth;
        }

        int p1_roll = rollDice(), p2_roll = rollDice();
        while (p1_roll == p2_roll)
        {
            p1_roll = rollDice();
            p2_roll = rollDice();
        }

        Player *current_player = nullptr;
        if (p1_roll > p2_roll)
        {
            cout << p1.getName() << " rolled " << p1_roll
                 << ", " << p2.getName() << " rolled " << p2_roll
                 << ". " << p1.getName() << " goes first!\n";
            game.setTurn(Player::PLAYER1);
            current_player = &p1;
        }
        else
        {
            cout << p1.getName() << " rolled " << p1_roll
                 << ", " << p2.getName() << " rolled " << p2_roll
                 << ". " << p2.getName() << " goes first!\n";
            game.setTurn(Player::PLAYER2);
            current_player = &p2;
        }

        cout << "Press s to start or q to quit ... ";
        string startOrQuit;
        cin >> startOrQuit;
        while (startOrQuit != "s" && startOrQuit != "q")
        {
            cout << "Press s to start or q to quit ... ";
            cin >> startOrQuit;
        }
        if (startOrQuit == "q")
            return 0;

        cout << "\n*-------------------------------------------------*" << endl;

        // —— main turn loop ——
        while (!game.over(&playerWon))
        {
            game.printGameBoard();

            // roll once per turn
            cout << "Turn: " << current_player->getName() << ". Press r to roll: ";
            string rollNow;
            cin >> rollNow;
            while (rollNow != "r")
            {
                cout << "Press r to roll: ";
                cin >> rollNow;
            }
            int d1 = rollDice(), d2 = rollDice();
            cout << "Dice: " << d1 << ", " << d2 << endl;

            // play dice until succesful move is completed
            bool turnDone = false;
            while (!turnDone)
            {
                string err;

                if (d1 != d2)
                {
                    // two‑move case
                    int first = promptDieChoice(d1, d2);
                    int second = (first == d1 ? d2 : d1);

                    // first move
                    auto [o1, dest1] = promptPair("Move for die " + to_string(first));
                    if (!game.tryMove(current_player, first, o1, dest1, err))
                    {
                        cout << "Error: " << err << "\n";
                        continue; // back to while(!turnDone) with same d1/d2
                    }
                    game.printGameBoard();

                    // second move
                    auto [o2, dest2] = promptPair("Move for die " + to_string(second));
                    if (!game.tryMove(current_player, second, o2, dest2, err))
                    {
                        cout << "Error: " << err << "\n";
                        continue; // back to while(!turnDone), same dice
                    }
                    game.printGameBoard();
                    turnDone = true; // both moves succeeded
                }
                else
                {
                    // doubles: four moves
                    for (int i = 0; i < 4; ++i)
                    {
                        auto [o, dst] = promptPair(
                            "Move " + to_string(i + 1) + " of 4 (die=" + to_string(d1) + ")");
                        if (!game.tryMove(current_player, d1, o, dst, err))
                        {
                            cout << "Error: " << err << "\n";
                            --i; // retry this same sub‑move
                            continue;
                        }
                        game.printGameBoard();
                    }
                    turnDone = true;
                }
            }

            // swap players once turnDone == true
            current_player = (current_player == &p1 ? &p2 : &p1);
        }

        break; // exit after someone wins
    }

    // final board redraw
    game.over(&playerWon);
    if (playerWon == Player::PLAYER1)
    {
        cout << p1_name << " WON!!!!" << endl;
    }
    else
    {
        cout << p2_name << " WON!!!!" << endl;
    }
    game.printGameBoard();
    return 0;
}
