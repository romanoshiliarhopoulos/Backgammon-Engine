// tests.cpp
#include <gtest/gtest.h>
#include "game.hpp"
#include "player.hpp"

// Note: these tests assume that Game::gameboard is accessible for setup.
// If it's private, you'll need to add a setter or make it protected for testing.

TEST(Player_functionality, newplayer)
{
    Player p1("romanos", Player::PLAYER1);
    ASSERT_EQ(p1.getName(), "romanos");
}

// ------ ValidOrigin tests ------

TEST(ValidOrigin, test_origin_boolean)
{
    Game game(0);
    ASSERT_TRUE(game.isValidOrigin(+1, 1));
    ASSERT_FALSE(game.isValidOrigin(-1, 1));
}

TEST(ValidOrigin, test_origin_for_player2)
{
    Game game(0);
    ASSERT_FALSE(game.isValidOrigin(+1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}

TEST(ValidOrigin, gameboard_with_no_pieces)
{
    Game game(0);
    ASSERT_FALSE(game.isValidOrigin(+1, 2));
    ASSERT_FALSE(game.isValidOrigin(-1, 2));
    ASSERT_FALSE(game.isValidOrigin(-1, 7));
}

TEST(ValidOrigin, test_p2_first_origin)
{
    Game game(1);
    ASSERT_FALSE(game.isValidOrigin(+1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}

// ------ ValidDestination tests ------

TEST(ValidDestination, with_friendly_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(+1, 1));
    ASSERT_TRUE(game.isValidDestination(+1, 12));
    ASSERT_TRUE(game.isValidDestination(-1, 6));
    ASSERT_TRUE(game.isValidDestination(-1, 13));
}

TEST(ValidDestination, with_no_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(+1, 2));
    ASSERT_TRUE(game.isValidDestination(+1, 11));
    ASSERT_TRUE(game.isValidDestination(-1, 2));
    ASSERT_TRUE(game.isValidDestination(-1, 22));
}

TEST(ValidDestination, with_enemy_blockade)
{
    Game game(0);
    ASSERT_FALSE(game.isValidDestination(+1, 6));
    ASSERT_FALSE(game.isValidDestination(+1, 8));
    ASSERT_FALSE(game.isValidDestination(-1, 19));
    ASSERT_FALSE(game.isValidDestination(-1, 17));
}

// ------ CapturePiece via tryMove tests ------

TEST(CapturePiece, singleBlotIsTakenAndJailed_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // place a single P2 blot on point 2
    game.gameboard[1] = -1;
    std::string err;

    // P1 rolls a 1 from 1→2
    bool ok = game.tryMove(&p1, 1, 1, 2, err);
    ASSERT_TRUE(ok) << "err=" << err;

    // blot captured, P1 checker sits there
    auto board = game.getGameBoard();
    EXPECT_EQ(board[1], +1);
    // P2 jailed one
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER2), 1);
}

TEST(CapturePiece, cannotCaptureTwoOrMore_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    std::string err;

    // Attempt P1 moving 5 spaces onto point 6 (blocked)
    bool ok = game.tryMove(&p1, 5, 1, 6, err);
    EXPECT_FALSE(ok);
    EXPECT_EQ(err, "Invalid destination.");
}

TEST(CapturePiece, invalidOrigin_with_jailed_piece_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    std::string err;
    // wrong origin
    EXPECT_FALSE(game.tryMove(&p1, 5, 12, 7, err));
    EXPECT_EQ(err, "Invalid origin");

    // correct jail exit (dice=5, origin=0→dest=5)
    // clear point 5 first
    game.gameboard[4] = 0;
    EXPECT_TRUE(game.tryMove(&p1, 5, 0, 5, err)) << err;
}

// ------ tryMove tests for moveOne behaviors ------

TEST(TryMove, FreesJailedPieceAndPlacesOnBoard)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(1);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    // clear point 2
    game.gameboard[1] = 0;

    std::string err;
    bool ok = game.tryMove(&p1, 2, 0, 2, err);
    EXPECT_TRUE(ok) << err;
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
    EXPECT_EQ(game.getGameBoard()[1], +1);
}

TEST(TryMove, RejectsNonZeroOriginWhenJailed)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    std::string err;
    // wrong origin
    EXPECT_FALSE(game.tryMove(&p1, 7, 5, 7, err));
    EXPECT_EQ(err, "Invalid origin");

    // correct exit to point 7
    game.gameboard[6] = 0;
    EXPECT_TRUE(game.tryMove(&p1, 7, 0, 7, err)) << err;
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
    EXPECT_EQ(game.getGameBoard()[6], +1);
}

TEST(TryMove, NormalMoveWhenNoJail)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // place 2 on point 3, clear point 5
    game.gameboard[2] = +2;
    game.gameboard[4] = 0;

    std::string err;
    bool ok = game.tryMove(&p1, 2, 3, 5, err);
    EXPECT_TRUE(ok) << err;

    auto b = game.getGameBoard();
    EXPECT_EQ(b[2], +1);
    EXPECT_EQ(b[4], +1);
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
}

// ------ over(&winner) test ------

TEST(GameOver, detects_winner)
{
    Game game(0);
    Player p1("Roma", Player::PLAYER1), p2("Romanos", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // free 15 pieces for P2
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER2);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER2);
}
// Test that freeing a piece requires no jailed pieces and all are in home
TEST(FreeingPiece, CannotFreeWhenNotInHomeOrJailed)
{
    Game game(4);
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // Initially, P1 has no jailed pieces but pieces are not all in home
    // Attempt to free from origin=0
    EXPECT_FALSE(game.isValidDestination(+1, 0));
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 0);

    // Now simulate a jailed piece and attempt free
    game.pieces.addJailedPiece(Player::PLAYER1);
    EXPECT_FALSE(game.isValidDestination(+1, 0));
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 0);
}

// Test that a single valid free increments numFreed
TEST(FreeingPiece, CanFreeWhenEligible)
{
    Game game(4);
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // Clear board and place one P1 checker in home region at point 19 (index 18)
    for (int i = 0; i < 24; ++i)
        game.gameboard[i] = 0;
    game.gameboard[18] = 1;

    // No jailed pieces, and only home region occupied
    EXPECT_TRUE(game.isValidDestination(+1, 0));
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 1);
}

// Test that Game::over detects winner by freed count
TEST(GameOver, DetectsPlayer1Winner)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    // Simulate P1 has freed 15 pieces
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER1);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER1);
}

TEST(GameOver, DetectsPlayer2Winner)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    // Simulate P2 has freed 15 pieces
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER2);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER2);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
