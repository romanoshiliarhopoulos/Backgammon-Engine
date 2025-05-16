/*player.cpp*/
#include "player.hpp"

Player::Player(string name, int num)
{
    this->name = name;
    this->player_num = num;
}

string Player::getName()
{
    return this->name;
}
int Player::getNum()
{
    return this->player_num;
}