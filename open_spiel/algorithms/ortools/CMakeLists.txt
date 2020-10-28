# This is an optional dependency. Currently it is only tested with the binary
# release of OR-Tools v8 (must be v8) on Linux x86_64 (Debian 10).
# See https://github.com/deepmind/open_spiel/issues/398 for details and
# discussion.
#
# You need to set BUILD_WITH_ORTOOLS to ON to include C++ Linear Programming.

add_library(open_spiel_ortools OBJECT
  lp_solver.cc
  lp_solver.h
  )
target_link_libraries(open_spiel_ortools ${ORTOOLS_LIBS})

add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(lp_solver_test ${ORTOOLS_LIBS})
add_test(lp_solver_test lp_solver_test)
