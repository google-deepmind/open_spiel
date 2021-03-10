# This is an optional dependency. Currently it is only tested with the binary
# release of OR-Tools v8 on Linux x86_64 (Debian 10).
#
# You need to set BUILD_WITH_ORTOOLS to ON to include C++ Linear Programming.
# See the main ../../CMakeLists for more instructions for setup, and
# https://github.com/deepmind/open_spiel/issues/398 for details and discussion
# of C++ optimization in OpenSpiel.

add_library(open_spiel_ortools OBJECT
  lp_solver.cc
  lp_solver.h
  sequence_form_lp.cc
  sequence_form_lp.h
)
target_link_libraries(open_spiel_ortools ${ORTOOLS_LIBS})

add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(lp_solver_test ${ORTOOLS_LIBS})
add_test(lp_solver_test lp_solver_test)

add_executable(sequence_form_lp_test sequence_form_lp_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(sequence_form_lp_test ${ORTOOLS_LIBS})
add_test(sequence_form_lp_test sequence_form_lp_test)
