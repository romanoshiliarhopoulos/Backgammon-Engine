cmake_minimum_required(VERSION 3.14)
project(Backgammon-Engine VERSION 0.1.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where all your sources and headers live
set(SRC_DIR "${CMAKE_SOURCE_DIR}/cppsrc")

# Grab every .cpp under cppsrc/
file(GLOB ALL_SOURCES "${SRC_DIR}/*.cpp")

# Separate out the test file from the library…
# grab every .cpp under cppsrc/
file(GLOB ALL_SOURCES "${SRC_DIR}/*.cpp")

# remove tests.cpp from the engine library
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/tests\\.cpp$")
# 1) Build your engine as a library
add_library(${PROJECT_NAME} ${ALL_SOURCES})

# Now it’s safe to specify include‐dirs for that library
target_include_directories(${PROJECT_NAME}
  PUBLIC
    "${SRC_DIR}"
)

# 2) Fetch & build GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 3) Build your test executable
add_executable(backgammon_tests
  "${SRC_DIR}/tests.cpp"
)
target_link_libraries(backgammon_tests
  PRIVATE
    GTest::gtest_main
    ${PROJECT_NAME}
)

# 4) Wire up CTest
enable_testing()
include(GoogleTest)
gtest_discover_tests(backgammon_tests)
