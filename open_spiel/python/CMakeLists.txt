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

add_library(pyspiel MODULE pybind11/pyspiel.cc ${OPEN_SPIEL_OBJECTS})

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
