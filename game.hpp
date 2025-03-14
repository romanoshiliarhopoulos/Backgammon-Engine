/*game.hpp*/
#include <vector>
#include "player.hpp"
#include <iostream>
#include <string>
#include "Pieces.hpp"

using namespace std;
class Game
{
private:
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

    vector<int> getGame();

    /// @brief prints the gameboard of the current game instance
    void printGameBoard();

    /// @brief erases the current gameboard from the terminal
    void clearGameboard();

    // assesses wether a player has won and the game is over
    bool over();

    // sets the current players
    void setPlayers(Player p1, Player p2);

    // called within the game loop, to make a move on the pieces....
    void movePieces(Player *currentPlayer, int dice1, int dice2);
    // helper functions for movePieces
    bool isValidOrigin(int multi, int idx);
};