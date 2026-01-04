# To enable C++ Torch DQN, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH. 
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_library (dqn_torch OBJECT
    dqn.h
    dqn.cc
    simple_nets.h
    simple_nets.cc
  )
  target_include_directories (dqn_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(dqn_torch_test dqn_torch_test.cc ${OPEN_SPIEL_OBJECTS}
                 $<TARGET_OBJECTS:dqn_torch> $<TARGET_OBJECTS:tests>)
  add_test(dqn_torch_test dqn_torch_test)

  target_link_libraries(dqn_torch ${TORCH_LIBRARIES})
  target_link_libraries(dqn_torch_test ${TORCH_LIBRARIES})
endif()
