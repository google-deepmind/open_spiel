# Now we can use #include "open_spiel/spiel.h"
include_directories(../..)

# An integration test to make sure that we can link with torchlib.
add_executable(torch_integration_test
  torch_integration_test.cc ${OPEN_SPIEL_OBJECTS})
add_test(torch_integration_test torch_integration_test)

target_link_libraries(torch_integration_test ${TORCH_LIBRARIES})
