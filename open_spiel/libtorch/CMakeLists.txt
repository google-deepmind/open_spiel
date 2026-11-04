# An integration test to make sure that we can link with torchlib.
add_executable(torch_integration_test torch_integration_test.cc)
target_link_libraries(torch_integration_test PRIVATE
  open_spiel_full open_spiel_ext::torch)
add_test(torch_integration_test torch_integration_test)
