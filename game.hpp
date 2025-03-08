/*game.hpp*/
#include <vector>
#include "player.hpp"
#include <iostream>
#include "Pieces.hpp"

using namespace std;
class Game
{
private:
    vector<int> gameboard{24, 0}; // creates a vector of size 24 with values intialized to 0.
    Pieces pieces; //holds the number of jailed and freedPieces of each player

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
};