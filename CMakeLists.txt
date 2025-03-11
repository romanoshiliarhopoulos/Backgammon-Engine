cmake_minimum_required(VERSION 3.14)
project(Backgammon-Engine VERSION 0.1.0 LANGUAGES C CXX)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the main library (excluding tests.cpp)
add_library(Backgammon-Engine 
    Pieces.cpp 
    game.cpp 
    player.cpp
)

# Setup GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test executable
enable_testing()
add_executable(backgammon_tests tests.cpp)
target_link_libraries(backgammon_tests
  PRIVATE
  GTest::gtest_main
  Backgammon-Engine
)

include(GoogleTest)
gtest_discover_tests(backgammon_tests)