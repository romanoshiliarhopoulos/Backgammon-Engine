/*jailedPiece.hpp*/

#include "Pieces.hpp"

// constructor
Pieces::Pieces()
{
    this->numPieces_p1 = 0; // number of jailed pieces
    this->numPieces_p2 = 0;
    this->freedPieces_p1 = 0; // number of 'freed' pieces
    this->freedPieces_p2 = 0;
}

/// @brief returns true if the current player has a jailed piece
/// @param player enum (either player 1 or 0)
/// @return
bool Pieces::hasJailedPiece(int player)
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
void Pieces::addJailedPiece(int player)
{
    if (player == Player::PLAYER1)
    {
        this->numPieces_p1 += 1;
    }
    else
    {
        this->numPieces_p2 += 1;
    }
}

/// @brief removes a jailedpiece
/// @param player player 1 or player 0
void Pieces::removeJailedPiece(int player)
{
    if (player == Player::PLAYER1 && this->numPieces_p1 > 0)
    {
        this->numPieces_p1 -= 1;
    }
    else
    {
        this->numPieces_p2 -= 1;
    }
}

/// @param player enum (either 1 or 0)
/// @return returns the number of freed pieces based on given player
int Pieces::numFreed(int player)
{
    if (player == Player::PLAYER1)
    {
        return this->freedPieces_p1;
    }
    else
    {
        return this->numPieces_p2;
    }
}

/// @brief Adds one more to the number of freed pieces to the given player
/// @param player enum, either 0 or 1
void Pieces::freePiece(int player)
{
    if (player == Player::PLAYER1)
    {
        this->freedPieces_p1 += 1;
    }
    else
    {
        this->freedPieces_p2 += 1;
    }
}
