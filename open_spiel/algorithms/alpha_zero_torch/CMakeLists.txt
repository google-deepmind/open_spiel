# C++ Torch AlphaZero. This directory has no BUILD file, so this file is
# hand-written. It is only added when OPEN_SPIEL_BUILD_WITH_LIBTORCH (which
# requires OPEN_SPIEL_BUILD_WITH_LIBNOP) is set; see the root CMakeLists.txt.

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
target_link_libraries (alpha_zero_torch PUBLIC
  open_spiel_full open_spiel_ext::libnop open_spiel_ext::torch)

add_executable(torch_model_test model_test.cc)
target_link_libraries(torch_model_test PRIVATE
  alpha_zero_torch open_spiel::basic_tests)
add_test(torch_model_test torch_model_test)

add_executable(torch_vpnet_test vpnet_test.cc)
target_link_libraries(torch_vpnet_test PRIVATE
  alpha_zero_torch open_spiel::basic_tests)
add_test(torch_vpnet_test torch_vpnet_test)

# Torch examples in ../../examples, which have no BUILD target either.
add_executable(alpha_zero_torch_example
  ../../examples/alpha_zero_torch_example.cc)
target_link_libraries(alpha_zero_torch_example PRIVATE alpha_zero_torch)

add_executable(alpha_zero_torch_game_example
  ../../examples/alpha_zero_torch_game_example.cc)
target_link_libraries(alpha_zero_torch_game_example PRIVATE alpha_zero_torch)

add_executable(dqn_torch_example ../../examples/dqn_torch_example.cc)
target_link_libraries(dqn_torch_example PRIVATE
  open_spiel_full open_spiel::algorithms::dqn_torch::dqn)
