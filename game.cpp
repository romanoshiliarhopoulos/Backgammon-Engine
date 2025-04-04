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
}

void Game::clearGameboard()
{

    int gameboardLines = 21;

    // Move cursor up by exact number of lines in the board
    std::cout << "\033[" << gameboardLines << "A";

    // Move cursor to beginning of that line
    std::cout << "\r" << std::flush;
}

// assesses wether a player has won and the game is over...
bool Game::over()
{
    Pieces pieces = this->pieces;

    // evaluating if player 1 has won. Condition: all their pieces are freed
    if (pieces.numFreed(0) == 15)
    {
        return true; // player 1 wins
    }
    else if (pieces.numFreed(1) == 15)
    {
        return true; // player 2 wins
    }
    else
    {
        return false;
    }
}

void Game::setPlayers(Player p1, Player p2)
{
    this->p1 = &p1;
    this->p2 = &p2;
};

// helper function to determing if an index is a valid origin (move pieces from that index to somewhere else)
bool Game::isValidOrigin(int multi, int idx)
{
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
    else
    {
        return false;
    }
}
void Game::movePieces(Player *currentPlayer, int dice1, int dice2)
{
    int multi = 1; // used to convert all pieces in array to the same sign for operations.
    // first we want to determine wether its player one or player two thats playing.
    if (currentPlayer == this->p2)
    {
        // if the pointers point to the same address
        multi = -1;
    }

    // if you have not rolled a "double"
    if (dice1 != dice2)
    {
        cout << "Move: ";
        int index_before_move;
        cin >> index_before_move;

        bool validO = isValidOrigin(multi, index_before_move);
        while (!validO)
        {
            // this player has no pieces there!!!
            cout << "Player: " << currentPlayer->getName() << " has no piece there, enter index: ";
            cin >> index_before_move;
        }
        cout << "To: ";
        int index_to_move_to;
        cin >> index_to_move_to;

        bool validD = isValidDestination(multi, index_to_move_to);
        while (!validD)
        {
            // this player has no pieces there!!!
            cout << "Player: " << currentPlayer->getName() << " has no piece there, enter index: ";
            cin >> index_to_move_to;
        }

        //by this point we have correct origin and destination indeces
    }
};
