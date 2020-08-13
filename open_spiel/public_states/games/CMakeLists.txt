set(PUBLIC_STATES_GAME_FILES
  kuhn_poker.h
  kuhn_poker.cc
  )

add_library(public_states_games OBJECT ${PUBLIC_STATES_GAME_FILES})

add_executable(public_states_kuhn_poker_test kuhn_poker_test.cc ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(public_states_kuhn_poker_test public_states_kuhn_poker_test)
