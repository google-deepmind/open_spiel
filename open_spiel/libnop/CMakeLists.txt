# An integration test to make sure that we can link with libnop.
add_executable(libnop_integration_test libnop_integration_test.cc)
target_link_libraries(libnop_integration_test PRIVATE
  open_spiel_full open_spiel_ext::libnop)
add_test(libnop_integration_test libnop_integration_test)
