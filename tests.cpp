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