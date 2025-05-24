/*game.hpp*/
#include <vector>
#include "player.hpp"
#include <iostream>
#include <string>
#include "Pieces.hpp"
#include <cmath>
#include <cstdlib>
#include <unordered_set>

using namespace std;
class Game
{

    // in game.hpp

public:
public:
    vector<int> gameboard = vector<int>(24, 0); // creates a vector of size 24 with values intialized to 0.
    Pieces pieces;                              // holds the number of jailed and freedPieces of each player

    Player *p1;
    Player *p2;
    int turn;

public:
    // constructor
    Game(int player);

    // populates the gameboard with appropriate number of pieces for each player
    void populateBoard();

    // accessor and mutator methods.
    void setTurn(int turn);
    int getTurn();

    vector<int> getGameBoard();

    /// @brief prints the gameboard of the current game instance
    void printGameBoard();

    // assesses wether a player has won and the game is over
    bool over(int *player);

    // sets the current players
    void setPlayers(Player *p1, Player *p2);

    bool movePieces(Player *currentPlayer,
                    int diceValue,
                    const std::array<std::pair<int, int>, 4> &moves,
                    std::string &err);

    bool tryMove(Player *currentPlayer,
                 int dice,
                 int origin,      // 0..24 as before
                 int destination, // 1..24
                 std::string &err);

    // helper functions for movePieces
    bool isValidOrigin(int multi, int idx);
    bool isValidDestination(int multi, int idx);

    bool canFreePiece(int player);
};