/*jailedPiece.hpp*/

#include "jailedPiece.hpp"

// constructor
JailedPieces::JailedPieces()
{
    this->numPieces_p1 = 0;
    this->numPieces_p2 = 0;
}

/// @brief returns true if the current player has a jailed piece
/// @param player enum (either player 1 or 0)
/// @return
bool JailedPieces::hasJailedPiece(int player)
{
    if (player == Player::PLAYER1)
    {
        return this->numPieces_p1 > 0;
    }
    else
    {
        return this->numPieces_p2 > 0;
    }
}

/// @brief increments jailedPiece counter by 1
/// @param player player 1 or player 0
void JailedPieces::addJailedPiece(int player)
{
}

/// @brief removes a jailedpiece
/// @param player player 1 or player 0
void JailedPieces::removeJailedPiece(int player)
{
}