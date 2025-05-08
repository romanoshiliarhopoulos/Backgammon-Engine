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
