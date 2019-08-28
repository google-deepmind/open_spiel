# Version >= 3.12 required for new FindPython module
# https://cmake.org/cmake/help/v3.12/release/3.12.html
cmake_minimum_required (VERSION 3.12)
project (open_spiel)

set (Python_TARGET_VERSION "" CACHE STRING "Specify a target Python version. \
Any Python library found with CMake modules FindPython2/FindPython3 that \
agrees  with the major version and is at least as high for minor version and \
patch number is accepted. If the variable is not set, we use the FindPython \
module which favours Python 3 over Python 2 if both are available.")
# set (CMAKE_CXX_COMPILER "/usr/bin/g++")
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-Werror -undefined dynamic_lookup")

# Position-independent code is needed for Python extension modules.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# Needed to disable Abseil tests.
set (BUILD_TESTING OFF)

# For now, let's enable all the tests.
enable_testing()

set (OPEN_SPIEL_CORE_FILES
  game_parameters.h
  game_parameters.cc
  spiel.h
  spiel.cc
  spiel_bots.h
  spiel_bots.cc
  matrix_game.h
  matrix_game.cc
  normal_form_game.h
  policy.h
  policy.cc
  simultaneous_move_game.h
  simultaneous_move_game.cc
  spiel_optional.h
  spiel_optional.cc
  spiel_utils.h
  spiel_utils.cc
)

# We add the subdirectory here so open_spiel_core can #include absl.
add_subdirectory (abseil-cpp)

# Just the core without any of the games
add_library(open_spiel_core OBJECT ${OPEN_SPIEL_CORE_FILES})
target_include_directories (open_spiel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
link_libraries(open_spiel_core absl::strings absl::str_format)

# Just the minimal base library: no games.
set (OPEN_SPIEL_CORE_OBJECTS $<TARGET_OBJECTS:open_spiel_core>)

set (OPEN_SPIEL_OBJECTS
  $<TARGET_OBJECTS:open_spiel_core>
  $<TARGET_OBJECTS:games>
  $<TARGET_OBJECTS:game_transforms>
  $<TARGET_OBJECTS:bridge_double_dummy_solver>
  $<TARGET_OBJECTS:algorithms>
)

# We have the parent of this directory in the include path, so that we can
# include for example "open_spiel/spiel.h" (assuming this directory is named
# open_spiel).
include_directories(..)

add_subdirectory (algorithms)
add_subdirectory (examples)
add_subdirectory (games)
add_subdirectory (game_transforms)
add_subdirectory (tests)
add_subdirectory (python)
