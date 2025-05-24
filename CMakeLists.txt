cmake_minimum_required(VERSION 3.14)
project(Backgammon-Engine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source directory
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/cppsrc)

# 1) Glob all .cpp in cppsrc
file(GLOB ALL_SOURCES
    "${SRCDIR}/*.cpp"
)

# 2) Exclude tests.cpp from the library list
#    (CMake ≥3.6 supports list(FILTER); otherwise use list(REMOVE_ITEM))
list(FILTER ALL_SOURCES
     EXCLUDE REGEX ".*/tests\\.cpp$"
)

# 3) Build your library from what's left
add_library(Backgammon-Engine
    ${ALL_SOURCES}
)
target_include_directories(Backgammon-Engine
    PUBLIC ${SRCDIR}
)

# 4) Now make your test executable from tests.cpp
add_executable(backgammon_tests
    ${SRCDIR}/tests.cpp
)
target_link_libraries(backgammon_tests
    PRIVATE GTest::gtest_main Backgammon-Engine
)

# 5) Fetch & hook up GoogleTest as before
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)
gtest_discover_tests(backgammon_tests)
