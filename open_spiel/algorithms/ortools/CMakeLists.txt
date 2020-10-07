# This is an optional dependency. Currently it does not work with OR-Tools
# and CMake. See https://github.com/deepmind/open_spiel/issues/398 for
# details.  If you are able to get this to work, please post a reply to
# the issue to let us know.
#
# You need to set BUILD_WITH_ORTOOLS to ON to include C++ Linear Programming,
# and have the OR-Tools github source cloned in algorithms/ortools/or-tools
# (i.e. git clone https://github.com/google/or-tools).
#
# Note that the OR-Tools CMake build is experimental. See CMake build
# instructions are here:
# https://github.com/google/or-tools/blob/stable/cmake/README.md

set(BUILD_DEPS ON)
add_subdirectory(or-tools)

add_library(open_spiel_ortools OBJECT
  lp_solver.cc
  lp_solver.h
)
target_link_libraries(open_spiel_ortools ortools::ortools)

add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(lp_solver_test ortools::ortools)
add_test(lp_solver_test lp_solver_test)
