cmake_minimum_required(VERSION 3.14)
project(Backgammon-Engine VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force CMake to use specific Python (your system Python where pybind11 is installed)
set(Python_EXECUTABLE "C:/Users/roman/AppData/Local/Programs/Python/Python310/python.exe")
set(PYBIND11_PYTHON_VERSION 3.10)

# Source directory
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/cppsrc)

# Find all .cpp files in cppsrc
file(GLOB ALL_SOURCES "${SRCDIR}/*.cpp")

# Exclude specific files from the library
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/tests\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/backgammon_bindings\\.cpp$")

# Create the main library
add_library(Backgammon-Engine STATIC ${ALL_SOURCES})

# Set library properties
set_target_properties(Backgammon-Engine PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add include directories
target_include_directories(Backgammon-Engine
    PUBLIC ${SRCDIR}
)

# Find pybind11 - try to use the installed version first
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # If not found, fetch from GitHub
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Verify that backgammon_bindings.cpp exists
if(EXISTS "${SRCDIR}/backgammon_bindings.cpp")
    # Create the Python extension module
    pybind11_add_module(backgammon_env "${SRCDIR}/backgammon_bindings.cpp")
    
    # Link the library to the Python module
    target_link_libraries(backgammon_env PRIVATE Backgammon-Engine)
    
    # Set properties for the Python module
    set_target_properties(backgammon_env PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
else()
    message(WARNING "backgammon_bindings.cpp not found in ${SRCDIR}")
endif()

# Optional: Create a standalone executable from main.cpp
if(EXISTS "${SRCDIR}/main.cpp")
    add_executable(backgammon_main "${SRCDIR}/main.cpp")
    target_link_libraries(backgammon_main PRIVATE Backgammon-Engine)
    set_target_properties(backgammon_main PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endif()

# Google Test setup (optional, for testing)
if(EXISTS "${SRCDIR}/tests.cpp")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Create test executable
    add_executable(backgammon_tests "${SRCDIR}/tests.cpp")
    target_link_libraries(backgammon_tests PRIVATE 
        GTest::gtest_main 
        Backgammon-Engine
    )
    
    set_target_properties(backgammon_tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Enable testing
    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(backgammon_tests)
endif()

# Print configuration info
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
