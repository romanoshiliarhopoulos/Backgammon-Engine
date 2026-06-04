/*game.hpp*/
#include <vector>
#include "player.hpp"
#include <iostream>
#include <string>
#include "Pieces.hpp"
#include <cmath>
#include <cstdlib>
#include <unordered_set>
#include <random>
#include <array>
#include <utility>

using namespace std;

/// Result of evaluating every legal turn sequence at once.
/// `sequences[k]` is the k-th legal turn (a list of origin->dest moves).
/// `states[k]` is the resulting game state after playing sequence k:
///   indices 0..23  = resulting gameboard
///   index   24     = jailed count player 1
///   index   25     = jailed count player 2
///   index   26     = borne-off count player 1
///   index   27     = borne-off count player 2
struct TurnEval
{
    std::vector<std::vector<std::pair<int, int>>> sequences;
    std::vector<std::array<int, 28>> states;
};

class Game
{

    // in game.hpp

public:
public:
    vector<int> gameboard = vector<int>(24, 0); // creates a vector of size 24 with values intialized to 0.
    Pieces pieces;                              // holds the number of jailed and freedPieces of each player

    Player *p1;
    Player *p2;
    int turn;

    array<int, 2> last_dice{1, 1}; // default to {1,1}
    mt19937_64 rng{std::random_device{}()};
    uniform_int_distribution<int> die{1, 6};

public:
    // constructors
    Game(int player);
    Game();

    // populates the gameboard with appropriate number of pieces for each player
    void populateBoard();

    // accessor and mutator methods.
    void setTurn(int turn);
    int getTurn();

    vector<int> getGameBoard();

    Pieces &getPieces();

    /// @brief prints the gameboard of the current game instance
    void printGameBoard();

    // assesses wether a player has won and the game is over
    bool over(int *player);

    // sets the current players
    void setPlayers(Player *p1, Player *p2);

    bool movePieces(Player *currentPlayer,
                    int diceValue,
                    const std::array<std::pair<int, int>, 4> &moves,
                    std::string &err);

    bool tryMove(Player *currentPlayer,
                 int dice,
                 int origin,      // 0..24 as before
                 int destination, // 1..24
                 std::string &err);

    // helper functions for movePieces
    bool isValidOrigin(int multi, int idx);
    bool isValidDestination(int multi, int idx, int dice = 0, int origin = 0);

    bool canFreePiece(int player, int dice, int origin);

    // some more helper functions to expose game-state to RL model
    int getJailedCount(int player);
    int getBornOffCount(int player);
    /// Deep‑copy the entire game (board, pieces, turn, RNG)
    Game clone() const;

    /// Returns all legal moves for this player and one die.
    vector<pair<int, int>> legalMoves(int player, int die);

    /// Returns all legal turn sequences for this player and the two dice.
    /// Each element is a vector of (origin→dest) pairs in the order they must be played.
    vector<vector<pair<int, int>>> legalTurnSequences(int player, int die1, int die2);

    /// Enumerate all legal turn sequences AND the resulting state for each, in
    /// one call. Lets the RL model evaluate every candidate move with a single
    /// Python<->C++ crossing instead of one clone()+tryMove per candidate.
    TurnEval evaluateTurnSequences(int player, int die1, int die2);

    // creates a dice pair: rolling dice through the API
    array<int, 2> rollDice();
    array<int, 2> getLastDice() const;

    void setGameBoard(vector<int> gameboard);
    Player getPlayer(int num);

    void setFreed(int player, int num);

    void setDice(int d1, int d2);
};