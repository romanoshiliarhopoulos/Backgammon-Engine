/*jailedPiece.hpp*/

#include "player.hpp"

class JailedPieces
{
private:
    int numPieces_p1;
    int numPieces_p2;

public:
    /// @brief Constructor of JailedPieces object
    JailedPieces();

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
};