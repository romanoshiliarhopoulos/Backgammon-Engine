/*player.cpp*/
#include "player.hpp"

Player::Player(string name)
{
    this->name = name;
}

string Player::getName()
{
    return this->name;
}