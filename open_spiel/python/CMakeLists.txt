if(Python_TARGET_VERSION STREQUAL "")
  find_package(Python COMPONENTS Development)
  include_directories(SYSTEM ${Python_INCLUDE_DIRS})
elseif(Python_TARGET_VERSION VERSION_LESS "3.0.0")
  message(FATAL_ERROR
    "Only Python 3 is supported, not ${Python_TARGET_VERSION}")
elseif(Python_TARGET_VERSION VERSION_GREATER_EQUAL "3.0.0")
  # Look for the Interpreter component so that the linker uses the correct
  # libpython when a virtual environment is loaded
  find_package(Python3 COMPONENTS Interpreter Development)
  include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
  if(Python3_VERSION VERSION_LESS Python_TARGET_VERSION)
    message(FATAL_ERROR
      "Python found ${Python3_VERSION} < ${Python_TARGET_VERSION}")
  endif()
endif()

add_library(pyspiel MODULE
  pybind11/algorithms_trajectories.cc
  pybind11/algorithms_trajectories.h
  pybind11/bots.cc
  pybind11/bots.h
  pybind11/games_bridge.cc
  pybind11/games_bridge.h
  pybind11/games_negotiation.cc
  pybind11/games_negotiation.h
  pybind11/game_transforms.cc
  pybind11/game_transforms.h
  pybind11/policy.cc
  pybind11/policy.h
  pybind11/pyspiel.cc
  ${OPEN_SPIEL_OBJECTS})

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")

# Get all Python tests
file(GLOB_RECURSE py_tests ${CMAKE_CURRENT_SOURCE_DIR}/../*_test.py)
foreach(py_test_file IN LISTS py_tests)

get_filename_component(py_test ${py_test_file} NAME_WE)
add_test(NAME python_${py_test} COMMAND python3 ${py_test_file})

# We need two elements in the python path: CURRENT_BINARY_DIR to pick up
# pyspiel.so, and CURRENT_SOURCE_DIR for the Python source files. We use
# CURRENT_SOURCE_DIR/../.. so that the Python module imports are of the form:
#  from open_spiel.python import rl_environment.
set_property(TEST python_${py_test}
    PROPERTY ENVIRONMENT
    PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
    TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)

endforeach(py_test_file)

# Additional tests (running examples as tests)
# We don't generate these automatically because we may want custom parameters.

add_test(NAME python_examples_bridge_supervised_learning
         COMMAND python3
         ${CMAKE_CURRENT_SOURCE_DIR}/examples/bridge_supervised_learning.py
         --iterations 10
         --eval_every 5
         --data_path ${CMAKE_CURRENT_SOURCE_DIR}/examples/data/bridge)
set_property(TEST python_examples_bridge_supervised_learning
    PROPERTY ENVIRONMENT
    PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
    TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
