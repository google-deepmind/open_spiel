# To enable C++ AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC. See:
# https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md
if (OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  add_library (alpha_zero_random_environment OBJECT
          alpha_zero_random_environment.h
          alpha_zero_random_environment.cc
    device_manager.h
          pvpevaluator.h
          pvpevaluator.cc
          pvpnet.h
          pvpnet.cc
  )
  target_include_directories (alpha_zero_random_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(pvpnet_test pvpnet_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:alpha_zero> $<TARGET_OBJECTS:tests>)
  add_test(pvpnet_test vpnet_test)

  target_link_libraries(alpha_zero_random_environment TensorflowCC::TensorflowCC)
  target_link_libraries(pvpnet_test TensorflowCC::TensorflowCC)
endif()
