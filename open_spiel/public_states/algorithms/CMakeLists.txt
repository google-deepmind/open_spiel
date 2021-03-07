set(PUBLIC_STATES_ALGORITHMS_FILES
  cfr.h
  cfr.cc
  )

add_library(public_states_algorithms OBJECT ${PUBLIC_STATES_ALGORITHMS_FILES})

add_executable(public_states_cfr_test cfr_test.cc ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(public_states_cfr_test public_states_cfr_test)

