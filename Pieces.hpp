/*jailedPiece.hpp*/
#pragma once
#include "player.hpp"

class Pieces
{
private:
    int numPieces_p1;
    int numPieces_p2;

    int freedPieces_p1;
    int freedPieces_p2;

public:
    /// @brief Constructor of JailedPieces object
    Pieces();

    // methods

    /// @brief returns true if the current player has a jailed piece
    /// @param player enum (either player 1 or 0)
    /// @return
    bool hasJailedPiece(int player);

    /// @brief increments jailedPiece counter by 1
    /// @param player player 1 or player 0
    void addJailedPiece(int player);

    /// @brief removes a jailedpiece
    /// @param player player 1 or player 0
    void removeJailedPiece(int player);

    /// @brief 
    /// @param player enum (either 1 or 0)
    /// @return returns the number of freed pieces based on given player
    int numFreed(int player);

    /// @brief Adds one more to the number of freed pieces to the given player
    /// @param player enum, either 0 or 1
    void freePiece(int player);
};