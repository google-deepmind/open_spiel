add_library (game_transforms OBJECT
  add_noise.cc
  add_noise.h
  cached_tree.cc
  cached_tree.h
  coop_to_1p.cc
  coop_to_1p.h
  efg_writer.cc
  efg_writer.h
  misere.cc
  misere.h
  normal_form_extensive_game.cc
  normal_form_extensive_game.h
  repeated_game.cc
  repeated_game.h
  restricted_nash_response.cc
  restricted_nash_response.h
  start_at.cc
  start_at.h
  turn_based_simultaneous_game.cc
  turn_based_simultaneous_game.h
  zerosum.cc
  zerosum.h
)
target_include_directories (game_transforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(restricted_nash_response_test
        restricted_nash_response_test.cc
        ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(restricted_nash_response_test restricted_nash_response_test)

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

add_executable(add_noise_test
               add_noise_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(add_noise_test add_noise_test)

add_executable(cached_tree_test
               cached_tree_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(cached_tree_test cached_tree_test)

add_executable(coop_to_1p_test
               coop_to_1p_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(coop_to_1p_test coop_to_1p_test)

add_executable(efg_writer_test
               efg_writer_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(efg_writer_test efg_writer_test)

add_executable(normal_form_extensive_game_test
               normal_form_extensive_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(normal_form_extensive_game_test normal_form_extensive_game_test)

add_executable(repeated_game_test
               repeated_game_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(repeated_game_test repeated_game_test)

add_executable(start_at_test
  start_at_test.cc
  ${OPEN_SPIEL_OBJECTS}
  $<TARGET_OBJECTS:tests>)
add_test(start_at_test start_at_test)

add_executable(zerosum_test
               zerosum_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(zerosum_test zerosum_test)
