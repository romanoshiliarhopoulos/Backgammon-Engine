add_test([=[Player_functionality.newplayer]=]  /Users/romanos/Backgammon_Engine/build/backgammon_tests [==[--gtest_filter=Player_functionality.newplayer]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Player_functionality.newplayer]=]  PROPERTIES WORKING_DIRECTORY /Users/romanos/Backgammon_Engine/build SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  backgammon_tests_TESTS Player_functionality.newplayer)
