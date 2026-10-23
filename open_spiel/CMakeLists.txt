# Root CMake file for OpenSpiel. This file is hand-written.
#
# Per-package CMakeLists.txt files (and targets.cmake for the targets of this
# directory) are generated from the Bazel BUILD files by
# scripts/generate_cmake.sh: edit the BUILD files, not the generated files.
# This file only holds what BUILD files do not describe: compiler flags per
# build type and compiler, the optional dependency switches, the external
# dependencies (cmake/dependencies.cmake), and the top-level directory list.

# Version >= 3.21 required for linking object files transitively through
# INTERFACE libraries ($<TARGET_OBJECTS:...> in target_link_libraries), which
# the generated files use to keep game registration (alwayslink) working.
# https://cmake.org/cmake/help/v3.21/command/target_link_libraries.html#linking-object-libraries-via-target-objects
cmake_minimum_required (VERSION 3.21)
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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

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
  if(MSVC)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20 /utf-8 /bigobj /DWIN32 /D_WINDOWS /GR /EHsc")
  endif()
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
# dependency is enabled -- see code below. The generated CMakeLists.txt files
# guard the targets that need an optional dependency with `if(DEP_NAME)`.
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
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_HANABI         OFF
  "Build against the Hanabi game.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_JULIA          OFF
  "Build binary for Julia.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBNOP         OFF
  "Build with support for libnop.")
openspiel_optional_dependency(OPEN_SPIEL_BUILD_WITH_LIBTORCH       OFF
  "Build with support for libtorch.")
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

if (WIN32)
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX              OFF
    "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH          OFF
    "Enable PyTorch.")
else()
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_JAX             AUTO
    "Enable JAX.")
  openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTORCH         AUTO
    "Enable PyTorch.")
endif()

openspiel_optional_dependency(OPEN_SPIEL_ENABLE_PYTHON_MISC       OFF
  "Enable miscellaneous Python dependencies.")

openspiel_optional_dependency(OPEN_SPIEL_BUILDING_WHEEL           OFF
  "Building a Python wheel?")

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH AND NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
  message(FATAL_ERROR
    "OPEN_SPIEL_BUILD_WITH_LIBTORCH requires OPEN_SPIEL_BUILD_WITH_LIBNOP")
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

# Needed to disable Abseil tests.
set (BUILD_TESTING OFF)

# For now, let's enable all the tests.
enable_testing()

# External dependencies: Abseil, nlohmann/json, Python and pybind11, and the
# optional ones enabled above. Each is exposed as a target (absl::*,
# nlohmann_json::nlohmann_json, open_spiel_ext::*) that the generated files
# link against.
include (cmake/dependencies.cmake)

# We have the parent of this directory in the include path, so that we can
# include for example "open_spiel/spiel.h" (assuming this directory is named
# open_spiel).
include_directories (..)

## OpenSpiel targets, generated from the BUILD files.
# Targets of this directory (the core library, test utilities, ...).
include (targets.cmake)

# Sub-packages. Each generated CMakeLists.txt adds its own sub-packages.
add_subdirectory (algorithms)
add_subdirectory (bots)
add_subdirectory (evaluation)
add_subdirectory (examples)
add_subdirectory (games)
add_subdirectory (game_transforms)
add_subdirectory (utils)

if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_subdirectory (python)
endif()

# Every OpenSpiel library (all games, algorithms, bots, ...), with the object
# files of the libraries that register games or bots. Used by the hand-written
# directories below and by the shared library.
get_property (OPEN_SPIEL_LIBRARIES GLOBAL PROPERTY OPEN_SPIEL_OBJECT_LIBRARIES)
add_library (open_spiel_full INTERFACE)
target_link_libraries (open_spiel_full INTERFACE ${OPEN_SPIEL_LIBRARIES})

## Hand-written directories that have no BUILD file.
if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  add_subdirectory (libnop)
endif()
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_subdirectory (libtorch)
  add_subdirectory (algorithms/alpha_zero_torch)
endif()
if (OPEN_SPIEL_BUILD_WITH_JULIA)
  add_subdirectory (julia)
endif()

if (BUILD_SHARED_LIB)
  # CMake needs at least one source file to create a library.
  set (OPEN_SPIEL_SHARED_LIB_SOURCE
       ${CMAKE_CURRENT_BINARY_DIR}/open_spiel_shared_lib.cc)
  file (WRITE ${OPEN_SPIEL_SHARED_LIB_SOURCE}
        "// Empty: the library consists of the OpenSpiel object files.\n")
  add_library (open_spiel SHARED ${OPEN_SPIEL_SHARED_LIB_SOURCE})
  target_include_directories (open_spiel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} abseil-cpp)
  # PRIVATE: users of the shared library must not link the object files again.
  target_link_libraries (open_spiel PRIVATE open_spiel_full)
  target_link_libraries (open_spiel PUBLIC
    absl::algorithm
    absl::flat_hash_map
    absl::optional
    absl::random_random
    absl::str_format
    absl::strings
    absl::time
  )
  if (OPEN_SPIEL_BUILD_WITH_ORTOOLS)
    # Optionally link external dependencies, for example OrTools for solving
    # linear programs.
    target_link_libraries (open_spiel PUBLIC ${ORTOOLS_LIBS})
  endif()
endif()

# Tests and binaries that only exist in the CMake build.
include (cmake/extra_targets.cmake)
