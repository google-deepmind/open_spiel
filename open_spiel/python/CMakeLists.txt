# Python is found in ../cmake/dependencies.cmake, before any target is defined.
if (OPEN_SPIEL_BUILDING_WHEEL)
  unset(OPEN_SPIEL_PYTHONPATH)
  if(DEFINED ENV{PYTHONPATH})
    set (OPEN_SPIEL_PYTHONPATH $ENV{PYTHONPATH})
  endif()
else()
  set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;)
endif()

# The Python bindings: the pyspiel module and the pybind_library targets it is
# made of. pybind11/CMakeLists.txt is generated from pybind11/BUILD.
add_subdirectory(pybind11)

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")
if (WIN32)
  # Python does not seem able to import the module without this change.
  set_target_properties(pyspiel PROPERTIES SUFFIX ".pyd")
endif()
# Put pyspiel in this directory of the build tree (which is on the PYTHONPATH
# of the tests below), unless the caller chose an output directory, as setup.py
# does when building a wheel.
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(pyspiel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Detect the Python ML frameworks.
if (OPEN_SPIEL_ENABLE_JAX STREQUAL AUTO)
  message(NOTICE "OPEN_SPIEL_ENABLE_JAX set to AUTO. Detecting Jax...")
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/find_jax.sh ${Python3_EXECUTABLE}
                  RESULT_VARIABLE JAX_RET_VAL
                  OUTPUT_VARIABLE JAX_OUT
                  ERROR_VARIABLE JAX_ERR)
  if (JAX_RET_VAL EQUAL 0)
    message(NOTICE "Found, version: ${JAX_OUT}")
    set(OPEN_SPIEL_ENABLE_JAX ON)
  else()
    message(NOTICE "Not found. Enable printing errors in python/CMakeLists.txt to see output.")
    set(OPEN_SPIEL_ENABLE_JAX OFF)
    # message(NOTICE "stdout: ${JAX_OUT}, stderr: ${JAX_ERR}")
  endif()
endif()

if (OPEN_SPIEL_ENABLE_PYTORCH STREQUAL AUTO)
  message(NOTICE "OPEN_SPIEL_ENABLE_PYTORCH set to AUTO. Detecting PyTorch...")
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/find_pytorch.sh ${Python3_EXECUTABLE}
                  RESULT_VARIABLE PYTORCH_RET_VAL
                  OUTPUT_VARIABLE PYTORCH_OUT
                  ERROR_VARIABLE PYTORCH_ERR)
  if (PYTORCH_RET_VAL EQUAL 0)
    message(NOTICE "Found, version: ${PYTORCH_OUT}")
    set(OPEN_SPIEL_ENABLE_PYTORCH ON)
  else()
    message(NOTICE "Not found. Enable printing errors in python/CMakeLists.txt to see output.")
    set(OPEN_SPIEL_ENABLE_PYTORCH OFF)
    # message(NOTICE "stdout: ${PYTORCH_OUT}, stderr: ${PYTORCH_ERR}")
  endif()
endif()

# Tests that are excluded when running via the installed wheel.
# Some bundle data not shipped with the wheel (e.g. playthroughs)
set(WHEEL_EXCLUDED_PYTHON_TESTS
  algorithms/response_graph_ucb_test.py
  examples/bridge_supervised_learning.py
  examples/hearts_supervised_learning.py
  games/tic_tac_toe_test.py
  ../integration_tests/playthrough_test.py)

# Tests of the BUILD files that CMake does not run. The meta_cfr directory has
# a typing.py, which shadows the standard library module when a file there is
# run as a script.
set(CMAKE_EXCLUDED_PYTHON_TESTS
  examples/meta_cfr/sequential_games/cfr_test.py
  examples/meta_cfr/sequential_games/meta_learning_test.py)

# Prefer to use PYBIN as the executable if it's defined (e.g. on GitHub CI)
if(NOT DEFINED ENV{PYBIN})
  set(PYBIN ${Python3_EXECUTABLE})
else()
  set(PYBIN $ENV{PYBIN})
endif()

# Adds the CTest test python/<py_test_file>, which runs the Python file
# <py_test_file> (relative to this directory) with the remaining arguments.
function(open_spiel_py_test py_test_file)
  if (py_test_file IN_LIST CMAKE_EXCLUDED_PYTHON_TESTS)
    return()
  endif()
  if (OPEN_SPIEL_BUILDING_WHEEL AND py_test_file IN_LIST WHEEL_EXCLUDED_PYTHON_TESTS)
    return()
  endif()
  add_test(NAME python/${py_test_file}
           COMMAND ${PYBIN} ${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file} ${ARGN})
  # We need two elements in the python path: CURRENT_BINARY_DIR to pick up
  # pyspiel.so, and CURRENT_SOURCE_DIR for the Python source files. We use
  # CURRENT_SOURCE_DIR/../.. so that the Python module imports are of the form:
  #  from open_spiel.python import rl_environment.
  set_property(TEST python/${py_test_file}
      PROPERTY ENVIRONMENT
      PYTHONPATH=${OPEN_SPIEL_PYTHONPATH};
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endfunction()

# Adds the CTest test python/<name> for a pytest_test, which runs
#   ${PYBIN} -m pytest <ARGS...> --deselect=<id> ... <FILES...>
# from the OpenSpiel source root's parent, like the Bazel macro, so that test
# ids look like open_spiel/python/foo_test.py::test_x. FILES and DESELECT ids
# are relative to the OpenSpiel root. pytest exits with 5 when it collects no
# test, which fails the CTest test.
function(open_spiel_pytest name)
  cmake_parse_arguments(PARSE_ARGV 1 PYTEST "" "" "FILES;DESELECT;ARGS")
  if (OPEN_SPIEL_BUILDING_WHEEL AND name IN_LIST WHEEL_EXCLUDED_PYTHON_TESTS)
    return()
  endif()
  set(files)
  foreach(file IN LISTS PYTEST_FILES)
    list(APPEND files open_spiel/${file})
  endforeach()
  set(deselect)
  foreach(id IN LISTS PYTEST_DESELECT)
    list(APPEND deselect --deselect=open_spiel/${id})
  endforeach()
  add_test(NAME python/${name}
           COMMAND ${PYBIN} -m pytest --rootdir=. --import-mode=importlib
                   -p no:cacheprovider ${PYTEST_ARGS} ${deselect} ${files}
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set_property(TEST python/${name}
      PROPERTY ENVIRONMENT
      PYTHONPATH=${OPEN_SPIEL_PYTHONPATH};
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..;
      PYTHONDONTWRITEBYTECODE=1)
endfunction()

# The Python tests of the BUILD files (py_test and friends), with the same
# conditions: generated by scripts/generate_cmake.sh.
include(python_tests.cmake)
