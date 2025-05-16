/*player.hppcle*/
#pragma once
#include <string>

using namespace std;
class Player
{
private:
    string name;
    int player_num;

public:
    // Player 1 represented as zero and player2 represented as
    enum PLAYERS
    {
        PLAYER1 = 0,
        PLAYER2
    };

    // constructor
    Player(string name, int num);

    //returns the player's name
    string getName();

    //returns the player's num 0 for p1 or 1 for p2
    int getNum();
};