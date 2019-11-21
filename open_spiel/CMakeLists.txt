# Version >= 3.12 required for new FindPython module
# https://cmake.org/cmake/help/v3.12/release/3.12.html
cmake_minimum_required (VERSION 3.12)
project (open_spiel)

set (Python_TARGET_VERSION "" CACHE STRING "Specify a target Python version. \
Any Python library found with CMake modules FindPython2/FindPython3 that \
agrees  with the major version and is at least as high for minor version and \
patch number is accepted. If the variable is not set, we use the FindPython \
module which favours Python 3 over Python 2 if both are available.")
set (CMAKE_CXX_STANDARD 17)
if(APPLE)
  # On MacOS:
  #   -Werror causes problems with abseil on MacOS
  #   -undefined dynamic_lookup is necessary for pybind11 linking
  set (CMAKE_CXX_FLAGS "-w -undefined dynamic_lookup")
else()
  set (CMAKE_CXX_FLAGS "-Werror")
endif()

# Position-independent code is needed for Python extension modules.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

## Optional dependencies
# One can optionally build and link against specific external dependencies.
# We expect these arguments to be always defined, when building using any script
# in `open_spiel/scripts/`, thus, we emit a warning when it's not, with a
# conservative default.
# See the documentation in install.md.
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

set (BUILD_WITH_HANABI OFF CACHE BOOL "Build against the Hanabi game.")
if(NOT DEFINED ENV{BUILD_WITH_HANABI})
    message("${BoldRed}BUILD_WITH_HANABI not set. Defaults to OFF${ColourReset}")
    set (ENV{BUILD_WITH_HANABI} OFF)
endif()
set (BUILD_WITH_HANABI $ENV{BUILD_WITH_HANABI})
message("${BoldYellow}BUILD_WITH_HANABI: ${BUILD_WITH_HANABI} ${ColourReset}")

##



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
  spiel_utils.h
  spiel_utils.cc
  tensor_view.h
)

set (OPEN_SPIEL_QUERY_FILES query.cc query.h)

# We add the subdirectory here so open_spiel_core can #include absl.
add_subdirectory (abseil-cpp)

# Just the core without any of the games
add_library(open_spiel_core OBJECT ${OPEN_SPIEL_CORE_FILES})
target_include_directories (open_spiel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
link_libraries(open_spiel_core
  absl::container
  absl::flags
  absl::flags_parse
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

# Just the minimal base library: no games.
set (OPEN_SPIEL_CORE_OBJECTS $<TARGET_OBJECTS:open_spiel_core>)

# Query API objects
add_library(open_spiel_query OBJECT ${OPEN_SPIEL_QUERY_FILES})
target_include_directories (open_spiel_query PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set (OPEN_SPIEL_OBJECTS
  $<TARGET_OBJECTS:open_spiel_core>
  $<TARGET_OBJECTS:games>
  $<TARGET_OBJECTS:game_transforms>
  $<TARGET_OBJECTS:open_spiel_query>
  $<TARGET_OBJECTS:bridge_double_dummy_solver>
  $<TARGET_OBJECTS:algorithms>
)
if (BUILD_WITH_HANABI)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:hanabi_learning_environment>)
endif()

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
add_subdirectory (julia)