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
}

/// @brief
/// @param player is an enum of which players turn it is
void Game::setTurn(int player)
{
    cout << "PLAYER: " << player << endl;
}


int Game::getTurn()
{
    return this->turn;//interesting.
}