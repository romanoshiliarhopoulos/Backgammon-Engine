/*game.cpp*/
#include "game.hpp"
#include <cassert>

// default constructor for copies
Game::Game()
{
}

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
Pieces &Game::getPieces()
{
    return this->pieces;
}
/// @brief
/// @param player is an enum of which players turn it is
void Game::setTurn(int player)
{
    // cout << "PLAYER: " << player << endl;
    this->turn = player;
}
Game Game::clone() const
{
    Game copy;
    copy.gameboard = this->gameboard;
    copy.pieces = this->pieces; // Ensure deep copy
    copy.turn = this->turn;
    copy.p1 = this->p1; // Pointer copy is OK here
    copy.p2 = this->p2; // Pointer copy is OK here
    return copy;
}

// computes all possible moves given a die value. Returns a vector of pairs(origin,destination)
vector<pair<int, int>> Game::legalMoves(int player, int die)
{
    vector<pair<int, int>> toReturn;
    int multi = player == 0 ? 1 : -1;
    for (int i = 0; i <= 25; i++)
    {
        bool validOrigin = isValidOrigin(multi, i);
        if (validOrigin)
        {
            int dest = i + multi * die;
            if (dest > 25)
            {
                dest = 25;
            }
            if (dest < 0)
            {
                dest = 0;
            }
            if (isValidDestination(multi, dest))
            {
                toReturn.emplace_back(i, dest);
            }
        }
    }
    return toReturn;
}

// faster way to collect all possible moves when rolling a double die:
// You only generate as many branches as remain legal after each simulated move.
void collectDoubles(Player *player,
                    int die,
                    int depth,
                    Game &state,
                    vector<pair<int, int>> &current,
                    vector<vector<pair<int, int>>> &out)
{
    if (depth == 4)
    {
        out.push_back(current);
        return;
    }
    for (auto m : state.legalMoves(player->getNum(), die))
    {
        Game next = state.clone();
        string err;
        next.tryMove(player, die, m.first, m.second, err);
        current.push_back(m);
        collectDoubles(player, die, depth + 1, next, current, out);
        current.pop_back();
    }
}
/// Returns all legal turn sequences for this player and the two dice.
/// Each element is a vector of (origin→dest) pairs in the order they must be played.
vector<vector<pair<int, int>>> Game::legalTurnSequences(int player, int die1, int die2)
{
    vector<vector<pair<int, int>>> sequences;
    Player *curr_player = player == 0 ? this->p1 : this->p2;

    if (die1 != die2)
    {
        // roll die1 then die2
        for (auto m1 : legalMoves(player, die1))
        {
            Game g1 = this->clone();
            string err;
            g1.tryMove(curr_player, die1, m1.first, m1.second, err);
            auto nextMoves = g1.legalMoves(player, die2);
            if (nextMoves.empty())
            {
                // if there are no more legal moves to make after the first one, just return there
                sequences.push_back({m1, {0, 0}});
                sequences.push_back({m1, {0, 0}});
                return sequences;
            }
            else
            {
                for (auto m2 : g1.legalMoves(player, die2))
                {
                    sequences.push_back({m1, m2});
                }
            }
        }
        // calculate all possible moves when moving die 2 first
        for (auto m1 : legalMoves(player, die2))
        {
            Game g1 = this->clone();
            string err;
            g1.tryMove(curr_player, die2, m1.first, m1.second, err); // make the move

            for (auto m2 : g1.legalMoves(player, die1))
            {
                sequences.push_back({m1, m2});
            }
        }
    }
    else
    {
        // case where we have a double.
        vector<pair<int, int>> curr;
        collectDoubles(curr_player, die1, 0, *this, curr, sequences);
    }
    return sequences;
}

vector<int> Game::getGameBoard()
{
    return this->gameboard;
}

int Game::getJailedCount(int player)
{
    return pieces.numJailed(player);
}

int Game::getBornOffCount(int player)
{
    return pieces.numFreed(player);
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
    cout << "\n         Free: ◉ x" << pieces.numFreed(0) << "  |  ◯ x" << pieces.numFreed(1) << endl;

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
        if (pieces.numJailed(Player::PLAYER2) > 0 && idx != 25)
        {
            return false;
        }
        else if (pieces.numJailed(Player::PLAYER2) > 0 && idx == 25)
        {
            return true;
        }
    }
    else if (multi == 1)
    {
        // it is player 1's move
        if (pieces.numJailed(Player::PLAYER1) > 0 && idx != 0)
        {
            return false;
        }
        else if (pieces.numJailed(Player::PLAYER1) > 0 && idx == 0)
        {
            return true;
        }
    }
    // Check bounds for regular board positions
    if (idx < 1 || idx > 24)
    {
        return false;
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
    if (idx == 0 || idx == 25)
    {
        // if player want to "free" a piece.
        return canFreePiece(multi);
    }
    // Check bounds for regular board positions
    if (idx < 1 || idx > 24)
    {
        return false;
    }
    if (this->gameboard[idx - 1] * multi >= 0)
    {
        return true;
    }
    else if (this->gameboard[idx - 1] * multi == -1)
    {
        // oponent has a single open piece there.... can capture
        return true;
    }
    else
    {
        return false;
    }
}

// helper function to see if a player can free a piece!
bool Game::canFreePiece(int multi)
{
    int player = (multi == +1) ? Player::PLAYER1  // enum value 0
                               : Player::PLAYER2; // enum value 1
    if (pieces.numJailed(player) != 0)
    {
        return false;
    }
    else
    {
        // need to check that all player's pieces on the game board are where they should be. (idx 19-24 for p1 and 1-6 for p2)
        for (int i = 1; i <= 24; i++)
        {
            if (player == Player::PLAYER1)
            {
                // ensure that p1 has all their pieces in the last 5 slots
                if (i < 19 && this->gameboard[i - 1] > 0)
                {
                    return false;
                }
            }
            else
            {
                // ensure that p2 has all their peices in their last 5 slots
                if (i > 6 && this->gameboard[i - 1] < 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
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
        err = "Invalid origin";
        return false;
    }
    if (origin < 0 || origin > 25)
    {
        err = "Origin out of range";
        return false;
    }
    if (destination < 0 || destination > 25)
    {
        err = "Destination out of range";
        return false;
    }

    int diff = origin - destination;
    // Special case: bearing off moves
    if (destination != 0 && destination != 25)
    // P1 bears off to 0, P2 bears off to 25
    // Don't apply normal direction rules for bearing off
    {
        // Normal direction check for regular moves
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
            err = "Invalid destination.";
            return false;
        }

        // Handle move execution with captures
        if (origin == 0 || origin == 25)
        {
            // Moving from jail
            pieces.removeJailedPiece(multi > 0 ? Player::PLAYER1 : Player::PLAYER2);
        }
        else
        {
            // Moving from regular point
            this->gameboard[origin - 1] -= multi;
        }
    }

    if (destination == 0 || destination == 25)
    {
        // Bearing off
        // origin must be on‐board to bear off
        if (origin == 0 || origin == 25 || origin < 0 || origin > 25)
        {
            err = "Cannot bear off from jail";
            return false;
        }
        pieces.freePiece(multi > 0 ? Player::PLAYER1 : Player::PLAYER2);
        this->gameboard[origin - 1] -= multi;
        return true;
    }
    else
    {
        // Check for capture before placing piece
        if (this->gameboard[destination - 1] * multi == -1)
        {
            // Capture opponent blot
            this->gameboard[destination - 1] = 0; // Remove opponent piece
            pieces.addJailedPiece(multi > 0 ? Player::PLAYER2 : Player::PLAYER1);
        }
        // Place piece
        this->gameboard[destination - 1] += multi;
    }

    return true;
}

array<int, 2> Game::rollDice()
{
    last_dice[0] = die(rng);
    last_dice[1] = die(rng);
    return last_dice;
}

array<int, 2> Game::getLastDice() const
{
    return last_dice;
}