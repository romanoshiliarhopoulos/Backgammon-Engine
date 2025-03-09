/*player.hppcle*/
#pragma once
#include <string>

using namespace std;
class Player
{
private:
    string name;

public:
    // Player 1 represented as zero and player2 represented as
    enum PLAYERS
    {
        PLAYER1 = 0,
        PLAYER2
    };

    // constructor
    Player(string name);

    //returns the player's name
    string getName();
};