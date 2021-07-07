# To enable C++ Torch AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH.
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  if(NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
    message(FATAL_ERROR
      "alpha_zero_torch requires libnop (OPEN_SPIEL_BUILD_WITH_LIBNOP)")
  endif()

  add_library (alpha_zero_torch OBJECT
    alpha_zero.h
    alpha_zero.cc
    device_manager.h
    model.h
    model.cc
    vpevaluator.h
    vpevaluator.cc
    vpnet.h
    vpnet.cc
  )
  target_include_directories (alpha_zero_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(torch_model_test model_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_model_test torch_model_test)

  add_executable(torch_vpnet_test vpnet_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_vpnet_test torch_vpnet_test)

  target_link_libraries (alpha_zero_torch ${TORCH_LIBRARIES})
  target_link_libraries (torch_model_test ${TORCH_LIBRARIES})
  target_link_libraries (torch_vpnet_test ${TORCH_LIBRARIES})
endif ()
