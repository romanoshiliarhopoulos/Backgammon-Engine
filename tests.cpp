#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "player.hpp"
#include "gtest/gtest.h"

TEST(Player_functionality, newplayer)
{
    Player p1("romanos", 1);

    ASSERT_EQ(p1.getName(), "romanos");
};
