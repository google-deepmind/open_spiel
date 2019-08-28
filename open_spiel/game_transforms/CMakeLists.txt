add_library (game_transforms OBJECT
  misere.cc
  misere.h
  turn_based_simultaneous_game.cc
  turn_based_simultaneous_game.h
)
target_include_directories (game_transforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(turn_based_simultaneous_game_test
               turn_based_simultaneous_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(turn_based_simultaneous_game_test turn_based_simultaneous_game_test)

add_executable(misere_test
               misere_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(misere_test misere_test)
