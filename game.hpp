/*game.hpp*/
#include <vector>
#include "player.hpp"
#include <iostream>
#include "jailedPiece.hpp"

using namespace std;
class Game
{
private:
    vector<int> gameboard{24, 0}; // creates a vector of size 24 with values intialized to 0.

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
};