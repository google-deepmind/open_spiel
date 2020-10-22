# This is an optional dependency. Currently it is only tested with the binary
# release of OR-Tools v8 (must be v8) on Linux x86_64 (Debian 10).
# See https://github.com/deepmind/open_spiel/issues/398 for details and
# discussion.
#
# You need to set BUILD_WITH_ORTOOLS to ON to include C++ Linear Programming.

# Binary option: compile with OR-Tools headers and link against binary
# distribution, downloaded from https://developers.google.com/optimization/install/cpp/linux
# and assumed to be in $HOME/or-tools. The flags were taken fro the compilation
# of linear_programming.cc after running make test_cc.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BOP -DUSE_GLOP -DUSE_CBC -DUSE_CLP -DUSE_SCIP")
set(ORTOOLS_HOME $ENV{HOME}/or-tools)
set(ORTOOLS_INC_DIRS ${ORTOOLS_HOME} ${ORTOOLS_HOME}/include)
set(ORTOOLS_LIB_DIRS ${ORTOOLS_HOME}/lib ${ORTOOLS_HOME}/lib64)
set(ORTOOLS_LIBS z rt pthread ortools)
add_library(open_spiel_ortools OBJECT
  lp_solver.cc
  lp_solver.h
  sequence_form_lp.cc
  sequence_form_lp.h
)
set_target_properties(open_spiel_ortools PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(open_spiel_ortools PUBLIC ${ORTOOLS_INC_DIRS})
link_directories(open_spiel_ortools ${ORTOOLS_LIB_DIRS})
target_link_libraries(open_spiel_ortools ${ORTOOLS_LIBS})

add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(lp_solver_test ${ORTOOLS_LIBS})
add_test(lp_solver_test lp_solver_test)

add_executable(sequence_form_lp_test sequence_form_lp_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
target_link_libraries(sequence_form_lp_test ${ORTOOLS_LIBS})
add_test(sequence_form_lp_test sequence_form_lp_test)

# Build OR-Tools from source option: compile with OR-Tools from source. Assumes
# the source in a subdirectory or-tools (or sym linked). Disable the above to
# use this. Note that the OR-Tools CMake build is experimental. See CMake build
# instructions are here:
# https://github.com/google/or-tools/blob/stable/cmake/README.md
# set(BUILD_DEPS ON)
#add_subdirectory(or-tools)
#add_library(open_spiel_ortools OBJECT
#  lp_solver.cc
#  lp_solver.h
#)
#target_link_libraries(open_spiel_ortools ortools::ortools)
#add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
#               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
#target_link_libraries(lp_solver_test ortools::ortools)
#add_test(lp_solver_test lp_solver_test)

