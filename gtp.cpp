/* game.hpp */
#pragma once
#include <vector>
#include <string>
#include <utility>
#include "player.hpp"
#include "pieces.hpp"

class Game
{
public:
    Game(int player);
    void populateBoard();
    int getTurn() const;
    void setTurn(int player);
    void setPlayers(Player *p1, Player *p2);

    // Programmatic move API (no I/O)
    bool moveOne(Player *currentPlayer,
                 int dice,
                 int origin,
                 int destination,
                 std::string &error);

    bool movePieces(Player *currentPlayer,
                    int dice1,
                    int dice2,
                    const std::vector<std::pair<int, int>> &moves,
                    bool firstDiceFirst,
                    std::string &error);

    // Render board to string (no direct stdout)
    std::string renderBoard() const;

    bool over() const;

private:
    int turn;
    std::vector<int> gameboard;
    Pieces pieces;
    Player *p1;
    Player *p2;

    bool isValidOrigin(int multi, int idx) const;
    bool isValidDestination(int multi, int idx);
};

/* game.cpp */
#include "game.hpp"
#include <cmath>
#include <sstream>

Game::Game(int player) : turn(player), gameboard(), pieces(), p1(nullptr), p2(nullptr)
{
    populateBoard();
}

void Game::populateBoard()
{
    gameboard.assign({2, 0, 0, 0, 0, -5,
                      0, -3, 0, 0, 0, 5,
                      -5, 0, 0, 0, 3, 0,
                      5, 0, 0, 0, 0, -2});
}

int Game::getTurn() const
{
    return turn;
}

void Game::setTurn(int player)
{
    turn = player;
}

void Game::setPlayers(Player *a, Player *b)
{
    p1 = a;
    p2 = b;
}

bool Game::moveOne(Player *currentPlayer,
                   int dice,
                   int origin,
                   int destination,
                   std::string &error)
{
    int multi = (currentPlayer == p2) ? -1 : 1;

    // Origin validity
    if (!isValidOrigin(multi, origin))
    {
        error = "Invalid origin";
        return false;
    }
    // Direction
    int diff = origin - destination;
    if (diff * (-multi) < 0)
    {
        error = "Invalid move direction";
        return false;
    }
    // Dice distance
    if (dice != std::abs(diff))
    {
        error = "Dice value mismatch";
        return false;
    }
    // Destination validity (including hits)
    if (!isValidDestination(multi, destination))
    {
        error = "Invalid destination";
        return false;
    }
    // Perform move
    if (origin == 0)
    {
        // leaving jail
        if (multi == 1)
            pieces.removeJailedPiece(Player::PLAYER1);
        else
            pieces.removeJailedPiece(Player::PLAYER2);
        gameboard[destination - 1] += multi;
    }
    else
    {
        gameboard[origin - 1] -= multi;
        gameboard[destination - 1] += multi;
    }
    return true;
}

bool Game::movePieces(Player *currentPlayer,
                      int dice1,
                      int dice2,
                      const std::vector<std::pair<int, int>> &moves,
                      bool firstDiceFirst,
                      std::string &error)
{
    if (dice1 != dice2)
    {
        if (moves.size() != 2)
        {
            error = "Expected 2 moves for non-doubles";
            return false;
        }
        int first = firstDiceFirst ? dice1 : dice2;
        int second = firstDiceFirst ? dice2 : dice1;
        if (!moveOne(currentPlayer, first, moves[0].first, moves[0].second, error))
            return false;
        if (!moveOne(currentPlayer, second, moves[1].first, moves[1].second, error))
            return false;
    }
    else
    {
        if (moves.size() != 4)
        {
            error = "Expected 4 moves for doubles";
            return false;
        }
        for (auto &mv : moves)
        {
            if (!moveOne(currentPlayer, dice1, mv.first, mv.second, error))
                return false;
        }
    }
    return true;
}

std::string Game::renderBoard() const
{
    std::ostringstream out;
    int totalLines = 20;
    std::vector<int> board = gameboard;

    out << "\n"
        << p1->getName() << ": ◉    |      "
        << p2->getName() << ": ◯\n\n";
    out << "   12 11 10 9  8  7     6  5  4  3  2  1\n";
    out << "*-----------------------------------------*\n";
    // Top half
    for (int i = 0; i < 8; ++i)
    {
        out << "|  ";
        for (int j = 11; j >= 0; --j)
        {
            if (j == 5)
                out << "|  ";
            if (i == 0 && board[j] == 0)
                out << "'  ";
            else if (board[j] > 0)
            {
                out << "◉  ";
                --board[j];
            }
            else if (board[j] < 0)
            {
                out << "◯  ";
                ++board[j];
            }
            else
                out << "   ";
        }
        out << "|\n";
    }
    // Bottom half
    for (int i = 8; i >= 1; --i)
    {
        out << "|  ";
        for (int j = 12; j < 24; ++j)
        {
            if (j == 18)
                out << "|  ";
            if (std::abs(board[j]) == i)
            {
                if (board[j] > 0)
                {
                    out << "◉  ";
                    --board[j];
                }
                else
                {
                    out << "◯  ";
                    ++board[j];
                }
            }
            else
            {
                out << (i == 1 ? "'  " : "   ");
            }
        }
        out << "|\n";
    }
    out << "* ----------------------------------------*\n";
    out << "  13 14  15 16 17 18 | 19 20  21 22 23 24\n";
    out << "\nJail: ◉ x" << pieces.numJailed(0)
        << "  |  ◯ x" << pieces.numJailed(1) << "\n";
    return out.str();
}

bool Game::over() const
{
    return pieces.numFreed(0) == 15 || pieces.numFreed(1) == 15;
}

/* main.cpp */
#include <iostream>
#include <string>
#include <vector>
#include "game.hpp"

int rollDice()
{
    return arc4random_uniform(6) + 1; // 1..6
}

int main()
{
    // Banner (unchanged)
    std::cout << "...banner ascii..." << std::endl;

    // Player names
    std::string name1, name2;
    std::cout << "Enter name for Player 1: ";
    std::cin >> name1;
    std::cout << "Enter name for Player 2: ";
    std::cin >> name2;
    while (name1 == name2)
    {
        std::cout << "Names must differ. Enter Player 2: ";
        std::cin >> name2;
    }

    Player p1(name1, 0), p2(name2, 1);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // Decide who starts
    std::cout << "Press r to roll: ";
    char c;
    while (std::cin >> c && c != 'r')
        std::cout << "Press r: ";
    int d1 = rollDice(), d2 = rollDice();
    while (d1 == d2)
    {
        d1 = rollDice();
        d2 = rollDice();
    }
    if (d1 > d2)
    {
        game.setTurn(Player::PLAYER1);
        std::cout << p1.getName() << " starts!\n";
    }
    else
    {
        game.setTurn(Player::PLAYER2);
        std::cout << p2.getName() << " starts!\n";
    }

    // Game loop
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
                std::cout << "Origin (" << dice1 << // Build moves and choice logic (replicates prior interactive flows)
        std::vector<std::pair<int, int>> moves; "): ";
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

    // Final board & result
    std::cout << game.renderBoard();
    std::cout << "Game over! Winner: "
              << (game.over() && game.getTurn() == Player::PLAYER2 ? p1.getName() : p2.getName())
              << "\n";
    return 0;
}
