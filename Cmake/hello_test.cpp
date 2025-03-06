#include <gtest/gtest.h>

TEST(HelloTest, BasicAssertions)
{
    // Check that strings are not equal.
    EXPECT_STRNE("hello", "world");

    // Check an arithmetic expression.
    EXPECT_EQ(7 * 6, 42);
}
