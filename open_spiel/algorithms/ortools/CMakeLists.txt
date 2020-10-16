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

# Option 1: compile with OR-Tools from source. Assumes the source in a
# subdirectory or-tools (or sym linked).
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

# Option 2: compile with OR-Tools headers and link against binary
# distribution, assumed to be in $HOME/or-tools. The flags were taken from
# the compilation of linear_programming.cc after running make test_cc, as seen
# on this page: https://developers.google.com/optimization/install/cpp/linux.
# The CMake command succeeds, but running make -j12 lp_solver_test results in
# compilation errors in absl::StrFormat, likely because of a version clash
# between our version and the one included in OR-Tools
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -DUSE_BOP -DUSE_GLOP -DUSE_CBC -DUSE_CLP -DUSE_SCIP")
#set(ORTOOLS_HOME $ENV{HOME}/or-tools)
#set(ORTOOLS_INC_DIRS ${ORTOOLS_HOME} ${ORTOOLS_HOME}/include)
#set(ORTOOLS_LIB_DIRS ${ORTOOLS_HOME}/lib ${ORTOOLS_HOME}/lib64)
#set(ORTOOLS_LIBS z rt pthread ortools)
#add_library(open_spiel_ortools OBJECT
#  lp_solver.cc
#  lp_solver.h
#)
#target_include_directories(open_spiel_ortools PUBLIC ${ORTOOLS_INC_DIRS})
#link_directories(open_spiel_ortools ${ORTOOLS_LIB_DIRS})
#target_link_libraries(open_spiel_ortools ${ORTOOLS_LIBS})
#add_executable(lp_solver_test lp_solver_test.cc ${OPEN_SPIEL_OBJECTS}
#               $<TARGET_OBJECTS:tests> $<TARGET_OBJECTS:open_spiel_ortools>)
#target_link_libraries(lp_solver_test ${ORTOOLS_LIBS})
#add_test(lp_solver_test lp_solver_test)

