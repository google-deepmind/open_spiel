# Version >= 3.12 required for new FindPython module
# https://cmake.org/cmake/help/v3.12/release/3.12.html
cmake_minimum_required (VERSION 3.12)
project (open_spiel)

# Define some nice terminal colors.
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

set(CMAKE_CXX_STANDARD 17)

# Set default build type.
set (BUILD_TYPE $ENV{BUILD_TYPE})
if(NOT BUILD_TYPE)
   set(BUILD_TYPE Testing
       CACHE STRING "Choose the type of build: Debug Release Testing."
       FORCE)
endif()
message("${BoldYellow}Current build type is: ${BUILD_TYPE}${ColourReset}")

if(${BUILD_TYPE} STREQUAL "Debug")
  # Basic build for debugging (default).
  # -Og enables optimizations that do not interfere with debugging.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")
endif()

if(${BUILD_TYPE} STREQUAL "Testing")
  # A build used for running tests: keep all runtime checks (assert,
  # SPIEL_CHECK_*, SPIEL_DCHECK_*), but turn on some speed optimizations,
  # otherwise tests run for too long.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if(${BUILD_TYPE} STREQUAL "Release")
  # Optimized release build: turn off debug runtime checks (assert,
  # SPIEL_DCHECK_*) and turn on highest speed optimizations.
  # The difference in perfomance can be up to 10x higher.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
endif()

if(APPLE)
  # On MacOS:
  #   -undefined dynamic_lookup is necessary for pybind11 linking
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything -w -undefined dynamic_lookup")

  # On MacOS, we need this so that CMake will use the right Python if the user
  # has a virtual environment active
  set (CMAKE_FIND_FRAMEWORK LAST)
elseif(WIN32)
  # Setup for MSVC 2022.
  # No changes needed. In particular: do not use -Wno-everything.
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything")
endif()

# Position-independent code is needed for Python extension modules.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)


## Optional dependencies
# One can optionally build and link against specific external dependencies.
# We expect these arguments to be always defined, when building using any script
# in `open_spiel/scripts/`, thus, we emit a warning when it's not, with a
# conservative default.
# See the documentation in install.md.

# Use this macro to define optional dependencies.
# You can then use your chosen DEP_NAME as a variable to check if that
# dependency is enabled -- see code below.
macro(openspiel_optional_dependency DEP_NAME DEP_DEFAULT DEP_DESCRIPTION)
  set (${DEP_NAME} ${DEP_DEFAULT} CACHE BOOL ${DEP_DESCRIPTION})
  if(NOT DEFINED ENV{${DEP_NAME}})
    message("${BoldRed}${DEP_NAME} not set. Defaults to ${DEP_DEFAULT}${ColourReset}")
    set (ENV{${DEP_NAME}} ${DEP_DEFAULT})
  endif()
  set (${DEP_NAME} $ENV{${DEP_NAME}})
  message("${BoldYellow}${DEP_NAME}: ${${DEP_NAME}} ${ColourReset}")
  # If the dependency is on, pass in compiler flags to enable conditional code,
  # e.g. #if OPEN_SPIEL_BUILD_WITH_...
  if (${DEP_NAME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${DEP_NAME}")
  endif()
endmacro()

# List of all optional dependencies:
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ACPC           OFF
  "Build against the Universal Poker library.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_EIGEN          OFF
  "Build with support for Eigen in C++.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_GO             OFF
  "Build with support for Golang API.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_HANABI         OFF
  "Build against the Hanabi game.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_HIGC           OFF
  "Build with Hidden Information Game Competition tournament support.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_JULIA          OFF
  "Build binary for Julia.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBNOP         OFF
  "Build with support for libnop.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBTORCH       OFF
  "Build with support for libtorch.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC  OFF
  "Build with support for Tensorflow C++ API.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_PYTHON         ON
  "Build binary for Python.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_XINXIN         OFF
  "Build against xinxin Hearts program.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ROSHAMBO       OFF
  "Build against RoShamBo bots.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_GAMUT          OFF
  "Build with GAMUT generator integration.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_ORTOOLS        OFF
  "Build with C++ optimization library OR-Tools.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_RUST           OFF
  "Build with support for Rust API.")

if (WIN32)
  if (OPEN_SPIEL_BUILD_WITH_HIGC)
    message(FATAL_ERROR "HIGC not supported in Windows, please disable it.")
  endif()
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX              OFF
    "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH          OFF
    "Enable PyTorch.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_TENSORFLOW       OFF
    "Enable Tensorflow.")
else()
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX             AUTO
    "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH         AUTO
    "Enable PyTorch.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_TENSORFLOW      AUTO
    "Enable Tensorflow.")
endif()

openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTHON_MISC       OFF
  "Enable miscellaneous Python dependencies.")

openspiel_optional_dependency(OPEN_SPIEL_BUILDING_WHEEL           OFF
  "Building a Python wheel?")

# Needed to disable Abseil tests.
set (BUILD_TESTING OFF)

# For now, let's enable all the tests.
enable_testing()

set (OPEN_SPIEL_CORE_FILES
  action_view.h
  action_view.cc
  canonical_game_strings.cc
  canonical_game_strings.h
  game_parameters.cc
  game_parameters.h
  matrix_game.cc
  matrix_game.h
  normal_form_game.h
  observer.cc
  observer.h
  policy.cc
  policy.h
  simultaneous_move_game.cc
  simultaneous_move_game.h
  spiel.cc
  spiel.h
  spiel_bots.cc
  spiel_bots.h
  spiel_globals.h
  spiel_utils.cc
  spiel_utils.h
  tensor_game.cc
  tensor_game.h
)

# We add the subdirectory here so open_spiel_core can #include absl.
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory (abseil-cpp)

# Just the core without any of the games
add_library(open_spiel_core OBJECT ${OPEN_SPIEL_CORE_FILES})
target_include_directories (
  open_spiel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
link_libraries(open_spiel_core
  absl::algorithm
  absl::flags
  absl::flags_parse
  absl::flat_hash_map
  absl::optional
  absl::random_random
  absl::str_format
  absl::strings
  absl::time
)

# Just the minimal base library: no games.
set (OPEN_SPIEL_CORE_OBJECTS $<TARGET_OBJECTS:open_spiel_core>)

set (OPEN_SPIEL_OBJECTS
  $<TARGET_OBJECTS:open_spiel_core>
  $<TARGET_OBJECTS:bots>
  $<TARGET_OBJECTS:games>
  $<TARGET_OBJECTS:game_transforms>
  $<TARGET_OBJECTS:bridge_double_dummy_solver>
  $<TARGET_OBJECTS:algorithms>
  $<TARGET_OBJECTS:utils>
  $<TARGET_OBJECTS:tests>
)
if (OPEN_SPIEL_BUILD_WITH_HIGC)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS}
     $<TARGET_OBJECTS:higc>)
endif()
if (OPEN_SPIEL_BUILD_WITH_HANABI)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS}
      $<TARGET_OBJECTS:hanabi_learning_environment>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ACPC)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS}
      $<TARGET_OBJECTS:universal_poker_clib>
      $<TARGET_OBJECTS:universal_poker_lib>)
endif()
if (OPEN_SPIEL_BUILD_WITH_EIGEN)
  add_compile_definitions(OPEN_SPIEL_BUILD_WITH_EIGEN)
  # Add Eigen dependency.
  add_subdirectory(eigen/)
  # Now we can use #include "Eigen/Dense"
  # This is needed so that pybind11/eigen.h locates <Eigen/Core>
  include_directories(eigen/libeigen)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:eigen>)
endif()
if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:xinxin>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ROSHAMBO)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:roshambo>)
endif()
if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  include_directories(libnop/libnop/include)
  add_subdirectory(libnop)
endif()
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/libtorch")
  find_package(Torch REQUIRED)
  add_subdirectory(libtorch)
  include_directories(${TORCH_INCLUDE_DIRS})
  # Use following to link your_target_executable with torch libraries:
  # target_link_libraries(your_target_executable ${TORCH_LIBRARIES})
endif()
if (OPEN_SPIEL_BUILD_WITH_GAMUT)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:gamut>)
endif()
if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
  # Compile with OR-Tools headers and link against binary distribution,
  # downloaded from https://developers.google.com/optimization/install/cpp/linux
  # and assumed to be in $HOME/or-tools.
  # The flags were taken from the compilation of linear_programming.cc after
  # running make test_cc.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BOP -DUSE_GLOP -DUSE_CBC -DUSE_CLP -DUSE_SCIP -pthread")
  set(ORTOOLS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/ortools")
  set(ORTOOLS_INC_DIRS ${ORTOOLS_HOME} ${ORTOOLS_HOME}/include)
  set(ORTOOLS_LIB_DIRS ${ORTOOLS_HOME}/lib ${ORTOOLS_HOME}/lib64)
  set(ORTOOLS_LIBS z rt pthread ortools)
  set_target_properties(open_spiel_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  include_directories(${ORTOOLS_INC_DIRS})
  link_directories(${ORTOOLS_LIB_DIRS})
  # Use following to link your_target_executable with OrTools libraries:
  # target_link_libraries(your_target_executable ${ORTOOLS_LIBS})
endif()
if (OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  add_compile_definitions(OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  find_package(TensorflowCC REQUIRED)
endif()

# We have the parent of this directory in the include path, so that we can
# include for example "open_spiel/spiel.h" (assuming this directory is named
# open_spiel).
include_directories(..)

add_subdirectory (algorithms)
add_subdirectory (bots)
add_subdirectory (examples)
add_subdirectory (games)
add_subdirectory (game_transforms)
add_subdirectory (contrib)

if (OPEN_SPIEL_BUILD_WITH_GO)
  add_subdirectory(go)
endif()

if (OPEN_SPIEL_BUILD_WITH_RUST)
  add_subdirectory(rust)
endif()

if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_subdirectory (python)
  # HIGC needs pyspiel.so and corresponding PYTHONPATH to be set
  # in order to run its tests.
  if (OPEN_SPIEL_BUILD_WITH_HIGC)
    add_subdirectory (higc)
  endif()
endif()

add_subdirectory (utils)

if (OPEN_SPIEL_BUILD_WITH_JULIA)
  add_subdirectory (julia)
endif()

# Build a shared library, i.e. libopen_spiel.so. We generally only enable this
# for binary releases.
# Note that there are known problems when trying to use absl::flags within a
# shared library, hence is intentionally left out. To use ABSL flags, link with
# absl::flags and absl::flags_parse separately.
set (BUILD_SHARED_LIB OFF CACHE BOOL "Build a shared library?")
if(NOT DEFINED ENV{BUILD_SHARED_LIB})
  set (ENV{BUILD_SHARED_LIB} OFF)
endif()
set (BUILD_SHARED_LIB $ENV{BUILD_SHARED_LIB})
if (BUILD_SHARED_LIB)
  if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
    add_library(open_spiel SHARED ${OPEN_SPIEL_OBJECTS}
      # Optionally include files that use external dependencies, for example
      # linear program specification for finding Nash equilibria.
      $<TARGET_OBJECTS:open_spiel_ortools>
    )
  else()
    add_library(open_spiel SHARED ${OPEN_SPIEL_OBJECTS})
  endif()
  target_include_directories(open_spiel PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
  target_link_libraries(open_spiel PUBLIC
    absl::algorithm
    absl::flat_hash_map
    absl::optional
    absl::random_random
    absl::str_format
    absl::strings
    absl::time
    # Optionally link external dependencies, for example OrTools for solving
    # linear programs.
    ${ORTOOLS_LIBS}
  )
endif()

add_subdirectory (tests)
