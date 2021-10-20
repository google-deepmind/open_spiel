add_library (higc OBJECT
  channel.cc
  channel.h
  subprocess.h
  utils.cc
  utils.h
  referee.cc
  referee.h
)

add_executable(tournament tournament.cc ${OPEN_SPIEL_OBJECTS})
add_executable(random_bot bots/random_bot.cc ${OPEN_SPIEL_OBJECTS})
add_executable(referee_test referee_test.cc ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)

add_test(referee_test referee_test
         --bots_dir=${CMAKE_CURRENT_SOURCE_DIR}/bots
         --build_dir=${CMAKE_CURRENT_BINARY_DIR}
         --run_only_blocking)
# Make sure that we can import pyspiel and other python source files.
set_property(TEST referee_test
        PROPERTY ENVIRONMENT
        PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/../python:${CMAKE_CURRENT_SOURCE_DIR}/../..;
        TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR})
