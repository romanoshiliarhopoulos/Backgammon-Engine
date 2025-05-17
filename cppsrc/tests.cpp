#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "player.hpp"
#include "game.hpp"
#include "gtest/gtest.h"

TEST(Player_functionality, newplayer)
{
    Player p1("romanos", 1);

    ASSERT_EQ(p1.getName(), "romanos");
};

// Test Suite for ValidOrigin with many cases
TEST(ValidOrigin, test_orginin_boolean)
{
    Game game(0); // populates initial gameboard
    ASSERT_TRUE(game.isValidOrigin(1, 1));
    ASSERT_FALSE(game.isValidOrigin(-1, 1));
}
TEST(ValidOrigin, test_orginin_boolean_p2)
{
    Game game(0); // populates initial gameboard
    ASSERT_FALSE(game.isValidOrigin(1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}
TEST(ValidOrigin, gameboard_with_no_pieces)
{
    Game game(0); // populates initial gameboard
    ASSERT_FALSE(game.isValidOrigin(1, 2));
    ASSERT_FALSE(game.isValidOrigin(-1, 2));

    ASSERT_FALSE(game.isValidOrigin(-1, 7));
}
TEST(ValidOrigin, test_p2_first_origin)
{
    Game game(1);
    ASSERT_FALSE(game.isValidOrigin(1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}

// to test about valid destination
TEST(ValidDestination, test_destination_with_friendly_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(1, 1));
    ASSERT_TRUE(game.isValidDestination(1, 12));

    ASSERT_TRUE(game.isValidDestination(-1, 6));
    ASSERT_TRUE(game.isValidDestination(-1, 13));
}
TEST(ValidDestination, test_destination_with_no_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(1, 2));
    ASSERT_TRUE(game.isValidDestination(1, 11));

    ASSERT_TRUE(game.isValidDestination(-1, 2));
    ASSERT_TRUE(game.isValidDestination(-1, 22));
}
TEST(ValidDestination, test_destination_with_enemy_piece)
{
    Game game(0);
    ASSERT_FALSE(game.isValidDestination(1, 6));
    ASSERT_FALSE(game.isValidDestination(1, 8));

    ASSERT_FALSE(game.isValidDestination(-1, 19));
    ASSERT_FALSE(game.isValidDestination(-1, 17));
}

TEST(CapturePiece, singleBlotIsTakenAndJailed)
{
    Game game(0);           // Player1 start
    game.gameboard[1] = -1; // p2 is at 1

    // sanity‐check initial jail counts are both zero
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER2), 0);
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);

    // Now P1 tries to move there
    bool ok = game.isValidDestination(/*multi=*/+1, /*idx=*/2);
    ASSERT_TRUE(ok);
    // make the move:
    ASSERT_EQ(game.gameboard[1], 0); // isValidDestination should remove the p2 piece

    game.gameboard[1] = 1; // simulates the rest of the moveOne function

    // the single P2 blot should be removed
    EXPECT_EQ(game.gameboard[1], /*P1’s checker now there*/ +1);

    // and one piece should have gone into P2’s jail
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER2), 1);
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
}

// You could also test that landing on a “too many” stack still fails:
TEST(CapturePiece, cannotCaptureTwoOrMore)
{
    Game game(0);
    // point 6 starts with five P2 checkers, so you can’t capture there
    ASSERT_FALSE(game.isValidDestination(+1, 6));
}

//
TEST(CapturePiece, invalidOrigin_with_jailed_piece)
{
    Game game(0);
    game.pieces.addJailedPiece(Player::PLAYER1); // add a jailed piece to player 1...
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER1), 1);
    ASSERT_FALSE(game.isValidOrigin(1, 12));
    ASSERT_TRUE(game.isValidOrigin(1, 0));
}

// Helper to restore cin after redirect
struct CinGuard
{
    std::streambuf *old;
    CinGuard(std::istream &in, std::istringstream &fake)
    {
        old = in.rdbuf(fake.rdbuf());
    }
    ~CinGuard()
    {
        std::cin.rdbuf(old);
    }
};

// Test that moveOne actually takes you out of jail (origin 0) and places a checker
TEST(MoveOne, FreesJailedPieceAndPlacesOnBoard)
{
    // set up two players
    Player p1("Alice", 1), p2("Bob", 2);
    Game game(1); // start as player 2, but we don't care
    game.setPlayers(&p1, &p2);

    // put one piece in jail for PLAYER1
    game.pieces.addJailedPiece(Player::PLAYER1);
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER1), 1);

    // pick a destination point that is empty: e.g. point index 2 (gameboard[1])
    game.gameboard[1] = 0;
    ASSERT_TRUE(game.isValidOrigin(+1, /*idx=*/0));
    ASSERT_TRUE(game.isValidDestination(+1, /*idx=*/2));

    // fake the two lines of console input: "0" (origin) then "2" (dest)
    std::istringstream fakeIn("0\n2\n");
    CinGuard guard(std::cin, fakeIn);

    // perform the move
    game.moveOne(&p1, /*dice=*/2);

    // after moveOne, jail count should have decreased
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0)
        << "Expected the jailed count to go down by one";

    // and exactly one checker should now sit on point 2
    EXPECT_EQ(game.gameboard[1], +1)
        << "Expected a single P1 checker at board index 1";
}

// Test that moveOne will reject a non‑zero origin when you still have a jailed piece
TEST(MoveOne, RejectsNonZeroOriginWhenJailed)
{
    Player p1("Alice", 1), p2("Bob", 2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // jail one piece for P1
    game.pieces.addJailedPiece(Player::PLAYER1);
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER1), 1);

    // try to move from origin=5 (illegal), then from origin=0 (legal), then dest=7
    // dest=7 must match dice roll 7
    std::istringstream fakeIn(
        "5\n" // first origin attempt → invalid
        "0\n" // second origin attempt → valid (jail)
        "7\n" // dest
    );
    CinGuard guard(std::cin, fakeIn);

    // point 7 is empty initially
    game.gameboard[6] = 0;
    game.moveOne(&p1, /*dice=*/7);

    // after a valid jail‐exit, jail count is zero
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);

    // and the checker wound up on point 7
    EXPECT_EQ(game.gameboard[6], +1);
}

// Test that if you have no jailed pieces, moveOne behaves like a normal move
TEST(MoveOne, NormalMoveWhenNoJail)
{
    Player p1("Alice", 1), p2("Bob", 2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // ensure P1 has no jail
    ASSERT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);

    // place two P1 checkers on point 3 (so valid origin) and leave point 5 empty
    game.gameboard[2] = +2;
    game.gameboard[4] = 0;

    // simulate origin=3, dest=5 with dice=2
    std::istringstream fakeIn("3\n5\n");
    CinGuard guard(std::cin, fakeIn);

    game.moveOne(&p1, /*dice=*/2);

    // moved one checker off point 3 → should now have 1 left
    EXPECT_EQ(game.gameboard[2], +1);
    // one arrives at point 5
    EXPECT_EQ(game.gameboard[4], +1);
    // jail count stays zero
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
}