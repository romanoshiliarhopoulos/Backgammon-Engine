/*game.cpp*/
#include "game.hpp"

// constructor
Game::Game(int player)
{
    if (player % 2 == 0)
    {
        setTurn(Player::PLAYER1);
    }
    else
    {
        setTurn(Player::PLAYER2);
    }
    this->pieces = Pieces(); // initializes jailedpieces to be equal to zero
    populateBoard();
}

/// @brief
/// @param player is an enum of which players turn it is
void Game::setTurn(int player)
{
    // cout << "PLAYER: " << player << endl;
    this->turn = player;
}

vector<int> Game::getGameBoard()
{
    return this->gameboard;
}

/// @brief Mutates the gameBoard to correctly populate board. Player 1 positive, Player2 negative
void Game::populateBoard()
{
    /*
        5   0   0   0   -3   0 | -5   0   0   0   0   2


        -5   0   0   0   3   0 | 5   0   0   0   0   -2

    */

    // this->gameboard = {2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2};
    gameboard.assign({2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2});
}
/// @brief
/// @return the player whose turn it is
int Game::getTurn()
{
    return this->turn; // interesting.
}

void Game::printGameBoard()
{
    /*
       | 12 11 10 9  8  7     6  5  4  3  2  1
       *-----------------------------------------*
       |  ◉  '  '  '  ◯  '  |  ◯  '  '  '  '  ◉  |
       |  ◉           ◯     |  ◯              ◉  |
       |  ◉           ◯     |  ◯                 |
       |  ◉                 |  ◯                 |
       |  ◉                 |  ◯                 |
       |                    |                    |
       |                    |                    |
       |                    |                    |
       |                    |                    |
       |                    |                    |
       |                    |                    |
       |  ◯                 |  ◉                 |
       |  ◯                 |  ◉                 |
       |  ◯           ◉     |  ◉                 |
       |  ◯           ◉     |  ◉              ◯  |
       |  ◯  '  '  '  ◉  '  |  ◉  '  '  '  '  ◯  |
       * ----------------------------------------*
         13 14  15 16 17 18 | 19 20  21 22 23 24 |

        info: total Lines: 20 (16 game lines and 4 of decors/indeces)
    */
    int totalLines = 20;

    vector<int> board = this->gameboard;

    // before this point, I want to print the symbol for each player...
    cout << "\n"
         << this->p1->getName() << ": " << "◉    |      ";
    cout << this->p2->getName() << ": " << "◯" << endl;
    cout << "\n\n";

    cout << "   12 11 10 9  8  7     6  5  4  3  2  1" << endl;
    cout << "*-----------------------------------------*" << endl;
    // for the top half. first iteration
    for (int i = 0; i < 8; i++)
    {
        cout << "|  ";
        for (int j = 11; j > -1; j--)
        {
            if (j == 5)
            {
                cout << "|  ";
            }
            if (i == 0 && board[j] == 0)
            {
                // for the first iteration, print ' if equal to zero
                cout << "'";
            }
            else if (board[j] > 0)
            {
                // player 1 has something there
                cout << "◉";
                board[j] -= 1; // reduce the number by 1
            }
            else if (board[j] < 0)
            {
                // player 2 has something there
                cout << "◯";
                board[j] += 1;
            }
            else
            {
                // print empty space if its zero and not the first line
                cout << " ";
            }
            cout << "  "; // add two empty spaces after the piece
        }
        cout << "|" << endl;
    }

    // to print the bottom half of the board
    for (int i = 8; i > 0; i--)
    {
        cout << "|  ";
        for (int j = 12; j < 24; j++)
        {
            if (j == 18)
            {
                cout << "|  ";
            }

            if (abs(board[j]) == i)
            {
                // need to paint a piece
                if (board[j] > 0)
                {
                    // paint player 1 piece
                    cout << "◉";
                    board[j] -= 1;
                }
                else if (board[j] < 0)
                {
                    // paint a player 2 piece
                    cout << "◯";
                    board[j] += 1;
                }
            }
            else
            {
                if (i == 1)
                {
                    cout << "'";
                }
                else
                {
                    cout << " ";
                }
            }
            cout << "  ";
        }
        cout << "|" << endl;
    }
    cout << "* ----------------------------------------*" << endl;
    cout << "  13 14  15 16 17 18 | 19 20  21 22 23 24 " << endl;
    cout << endl;
    cout << "         Jail: ◉ x" << pieces.numJailed(0) << "  |  ◯ x" << pieces.numJailed(1) << endl;
    cout << "\n\n"
         << endl;
}

// assesses wether a player has won and the game is over...
bool Game::over(int *player)
{
    Pieces pieces = this->pieces;

    // evaluating if player 1 has won. Condition: all their pieces are freed
    if (pieces.numFreed(0) == 15)
    {
        *player = 0;
        return true; // player 1 wins
    }
    else if (pieces.numFreed(1) == 15)
    {
        *player = 1;
        return true; // player 2 wins
    }
    else
    {
        return false;
    }
}

void Game::setPlayers(Player *p1, Player *p2)
{
    this->p1 = p1;
    this->p2 = p2;
};

// helper function to determing if an index is a valid origin (move pieces from that index to somewhere else)
bool Game::isValidOrigin(int multi, int idx)
{
    // check for jailed pieces first
    if (multi == -1)
    {
        // it is player 2's move
        if (pieces.numJailed(Player::PLAYER2) > 0 && idx != 0)
        {
            cout << p2->getName() << " has a jailed piece, origin must be 0!" << endl;
            return false;
        }
        else if (pieces.numJailed(Player::PLAYER2) > 0 && idx == 0)
        {
            return true;
        }
    }
    else if (multi == 1)
    {
        // it is player 1's move
        if (pieces.numJailed(Player::PLAYER1) > 0 && idx != 0)
        {
            cout << p1->getName() << " has a jailed piece, origin must be 0!" << endl;
            return false;
        }
        else if (pieces.numJailed(Player::PLAYER1) > 0 && idx == 0)
        {
            return true;
        }
    }

    if (this->gameboard[idx - 1] * multi > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}
// helper function to see if a player can move a piece there
bool Game::isValidDestination(int multi, int idx)
{
    if (this->gameboard[idx - 1] * multi >= 0)
    {
        return true;
    }
    else if (this->gameboard[idx - 1] * multi == -1)
    {
        // oponent has a single open piece there....
        this->gameboard[idx - 1] += multi; // removes oponent piece
        // adds oponent piece to jail
        if (multi == -1)
        {
            pieces.addJailedPiece(Player::PLAYER1);
        }
        else
        {
            pieces.addJailedPiece(Player::PLAYER2);
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool Game::movePieces(Player *currentPlayer,
                      int diceValue,
                      const std::array<std::pair<int, int>, 4> &moves,
                      std::string &err)
{
    for (auto [o, d] : moves)
    {
        if (!tryMove(currentPlayer, diceValue, o, d, err))
            return false;
        printGameBoard();
    }
    return true;
}

bool Game::tryMove(Player *currentPlayer,
                   int dice,
                   int origin,
                   int destination,
                   std::string &err)
{
    int multi = (currentPlayer == p2) ? -1 : +1;

    // jailed‑piece checks:
    if (!isValidOrigin(multi, origin))
    {
        err = "You have a jailed piece; origin must be 0.";
        return false;
    }

    int diff = origin - destination;
    if (diff * (-multi) < 0)
    {
        err = "Cannot move in that direction.";
        return false;
    }
    if (dice != std::abs(diff))
    {
        err = "Move does not match dice.";
        return false;
    }
    if (!isValidDestination(multi, destination))
    {
        err = "Destination blocked by opponent.";
        return false;
    }

    // perform the move
    if (origin == 0)
    {
        // freeing from jail
        pieces.removeJailedPiece(multi > 0 ? Player::PLAYER1 : Player::PLAYER2);
        this->gameboard[destination - 1] += multi;
    }
    else
    {
        this->gameboard[origin - 1] -= multi;
        this->gameboard[destination - 1] += multi;
    }

    return true;
}
