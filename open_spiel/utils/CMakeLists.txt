add_library (utils OBJECT
  circular_buffer.h
  combinatorics.h
  combinatorics.cc
  data_logger.h
  data_logger.cc
  file.h
  file.cc
  functional.h
  init.h
  init.cc
  json.h
  json.cc
  logger.h
  lru_cache.h
  random.h
  random.cc
  run_python.h
  run_python.cc
  serializable_circular_buffer.h
  serialization.h
  stats.h
  tensor_view.h
  thread.h
  thread.cc
  threaded_queue.h
)
target_include_directories (utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_open_spiel_test(circular_buffer_test circular_buffer_test.cc)

add_open_spiel_test(combinatorics_test combinatorics_test.cc)

add_open_spiel_test(data_logger_test data_logger_test.cc)

add_open_spiel_test(file_test file_test.cc)

add_open_spiel_test(functional_test functional_test.cc)

add_open_spiel_test(json_test json_test.cc)

add_open_spiel_test(logger_test logger_test.cc)

add_open_spiel_test(lru_cache_test lru_cache_test.cc)

add_open_spiel_test(random_test random_test.cc)

if (OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_executable(run_python_test run_python_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:tests>)
  add_test(run_python_test run_python_test
           --python_command=${Python3_EXECUTABLE})
  set_property(TEST run_python_test
      PROPERTY ENVIRONMENT
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../..;
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if (OPEN_SPIEL_BUILD_WITH_LIBNOP)
  add_executable(serializable_circular_buffer_test
                 serializable_circular_buffer_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:tests>)
  add_test(serializable_circular_buffer_test serializable_circular_buffer_test)
endif()

add_open_spiel_test(stats_test stats_test.cc)

add_open_spiel_test(tensor_view_test tensor_view_test.cc)

# Failing on Ubuntu 18.04 since upgrade of abseil version (2021-05-17).
# Disabling while we look into it.
# add_open_spiel_test(thread_test thread_test.cc)

add_open_spiel_test(threaded_queue_test threaded_queue_test.cc)
