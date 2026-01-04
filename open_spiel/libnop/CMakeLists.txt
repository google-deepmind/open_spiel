# An integration test to make sure that we can link with libnop.
add_executable(libnop_integration_test
  libnop_integration_test.cc ${OPEN_SPIEL_OBJECTS})
add_test(libnop_integration_test libnop_integration_test)
