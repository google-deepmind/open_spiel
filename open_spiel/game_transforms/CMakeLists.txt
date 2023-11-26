add_library (game_transforms OBJECT
  add_noise.cc
  add_noise.h
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

add_open_spiel_test(restricted_nash_response_test restricted_nash_response_test.cc)

add_open_spiel_test(turn_based_simultaneous_game_test turn_based_simultaneous_game_test.cc)

add_open_spiel_test(misere_test misere_test.cc)

add_open_spiel_test(add_noise_test add_noise_test.cc)

add_open_spiel_test(coop_to_1p_test coop_to_1p_test.cc)

add_open_spiel_test(efg_writer_test efg_writer_test.cc)

add_open_spiel_test(normal_form_extensive_game_test normal_form_extensive_game_test.cc)

add_open_spiel_test(repeated_game_test repeated_game_test.cc)

add_open_spiel_test(start_at_test start_at_test.cc)

add_open_spiel_test(zerosum_test zerosum_test.cc)
