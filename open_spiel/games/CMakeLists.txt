add_library (games OBJECT
  backgammon.cc
  backgammon.h
  blotto.cc
  blotto.h
  breakthrough.cc
  breakthrough.h
  bridge/bridge_scoring.cc
  bridge/bridge_scoring.h
  bridge_uncontested_bidding.cc
  bridge_uncontested_bidding.h
  catch.cc
  catch.h
  chess.cc
  chess.h
  chess/chess_board.cc
  chess/chess_board.h
  chess/chess_common.cc
  chess/chess_common.h
  coin_game.cc
  coin_game.h
  connect_four.cc
  connect_four.h
  coop_box_pushing.cc
  coop_box_pushing.h
  first_sealed_auction.cc
  first_sealed_auction.h
  go.cc
  go.h
  go/go_board.cc
  go/go_board.h
  goofspiel.cc
  goofspiel.h
  # Uncomment to enable Hanabi.
  # hanabi.cc
  # hanabi.h
  havannah.cc
  havannah.h
  hex.cc
  hex.h
  kuhn_poker.cc
  kuhn_poker.h
  laser_tag.cc
  laser_tag.h
  leduc_poker.cc
  leduc_poker.h
  liars_dice.cc
  liars_dice.h
  markov_soccer.cc
  markov_soccer.h
  matching_pennies_3p.cc
  matching_pennies_3p.h
  matrix_games.cc
  oshi_zumo.cc
  oshi_zumo.h
  oware.cc
  oware.h
  oware/oware_board.cc
  oware/oware_board.h
  pentago.cc
  pentago.h
  phantom_ttt.cc
  phantom_ttt.h
  pig.cc
  pig.h
  quoridor.cc
  quoridor.h
  tic_tac_toe.cc
  tic_tac_toe.h
  tiny_bridge.cc
  tiny_bridge.h
  tiny_hanabi.cc
  tiny_hanabi.h
  y.cc
  y.h
)
target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Uncomment to enable Hanabi.
# add_subdirectory (hanabi)

# Uncomment to build the Ludii demo
# add_subdirectory (ludii)

add_library(bridge_double_dummy_solver OBJECT
  bridge/double_dummy_solver/include/dll.h
  bridge/double_dummy_solver/include/portab.h
  bridge/double_dummy_solver/src/ABsearch.cpp
  bridge/double_dummy_solver/src/ABsearch.h
  bridge/double_dummy_solver/src/ABstats.cpp
  bridge/double_dummy_solver/src/ABstats.h
  bridge/double_dummy_solver/src/CalcTables.cpp
  bridge/double_dummy_solver/src/CalcTables.h
  bridge/double_dummy_solver/src/dds.cpp
  bridge/double_dummy_solver/src/dds.h
  bridge/double_dummy_solver/src/DealerPar.cpp
  bridge/double_dummy_solver/src/debug.h
  bridge/double_dummy_solver/src/dump.cpp
  bridge/double_dummy_solver/src/dump.h
  bridge/double_dummy_solver/src/File.cpp
  bridge/double_dummy_solver/src/File.h
  bridge/double_dummy_solver/src/Init.cpp
  bridge/double_dummy_solver/src/Init.h
  bridge/double_dummy_solver/src/LaterTricks.cpp
  bridge/double_dummy_solver/src/LaterTricks.h
  bridge/double_dummy_solver/src/Memory.cpp
  bridge/double_dummy_solver/src/Memory.h
  bridge/double_dummy_solver/src/Moves.cpp
  bridge/double_dummy_solver/src/Moves.h
  bridge/double_dummy_solver/src/Par.cpp
  bridge/double_dummy_solver/src/parallel.h
  bridge/double_dummy_solver/src/PBN.cpp
  bridge/double_dummy_solver/src/PBN.h
  bridge/double_dummy_solver/src/PlayAnalyser.cpp
  bridge/double_dummy_solver/src/PlayAnalyser.h
  bridge/double_dummy_solver/src/QuickTricks.cpp
  bridge/double_dummy_solver/src/QuickTricks.h
  bridge/double_dummy_solver/src/Scheduler.cpp
  bridge/double_dummy_solver/src/Scheduler.h
  bridge/double_dummy_solver/src/SolveBoard.cpp
  bridge/double_dummy_solver/src/SolveBoard.h
  bridge/double_dummy_solver/src/SolverIF.cpp
  bridge/double_dummy_solver/src/SolverIF.h
  bridge/double_dummy_solver/src/System.cpp
  bridge/double_dummy_solver/src/System.h
  bridge/double_dummy_solver/src/ThreadMgr.cpp
  bridge/double_dummy_solver/src/ThreadMgr.h
  bridge/double_dummy_solver/src/Timer.cpp
  bridge/double_dummy_solver/src/Timer.h
  bridge/double_dummy_solver/src/TimerGroup.cpp
  bridge/double_dummy_solver/src/TimerGroup.h
  bridge/double_dummy_solver/src/TimerList.cpp
  bridge/double_dummy_solver/src/TimerList.h
  bridge/double_dummy_solver/src/TimeStat.cpp
  bridge/double_dummy_solver/src/TimeStat.h
  bridge/double_dummy_solver/src/TimeStatList.cpp
  bridge/double_dummy_solver/src/TimeStatList.h
  bridge/double_dummy_solver/src/TransTable.h
  bridge/double_dummy_solver/src/TransTableL.cpp
  bridge/double_dummy_solver/src/TransTableL.h
  bridge/double_dummy_solver/src/TransTableS.cpp
  bridge/double_dummy_solver/src/TransTableS.h
)
target_include_directories (bridge_double_dummy_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bridge_double_dummy_solver PUBLIC DDS_NO_STATIC_INIT)

add_executable(backgammon_test backgammon_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(backgammon_test backgammon_test)

add_executable(blotto_test blotto_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(blotto_test blotto_test)

add_executable(breakthrough_test breakthrough_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(breakthrough_test breakthrough_test)

add_executable(bridge_test bridge_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(bridge_test bridge_test)

add_executable(catch_test catch_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(catch_test catch_test)

add_executable(chess_test chess_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(chess_test chess_test)

add_executable(coin_game_test coin_game_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(coin_game_test coin_game_test)

add_executable(connect_four_test connect_four_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(connect_four_test connect_four_test)

add_executable(coop_box_pushing_test coop_box_pushing_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(coop_box_pushing_test coop_box_pushing_test)

add_executable(first_sealed_auction_test first_sealed_auction_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(first_sealed_auction_test first_sealed_auction_test)

add_executable(go_test go_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(go_test go_test)

add_executable(goofspiel_test goofspiel_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(goofspiel_test goofspiel_test)

add_executable(havannah_test havannah_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(havannah_test havannah_test)

add_executable(hex_test hex_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(hex_test hex_test)

add_executable(kuhn_poker_test kuhn_poker_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(kuhn_poker_test kuhn_poker_test)

add_executable(leduc_poker_test leduc_poker_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(leduc_poker_test leduc_poker_test)

add_executable(liars_dice_test liars_dice_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(liars_dice_test liars_dice_test)

add_executable(markov_soccer_test markov_soccer_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(markov_soccer_test markov_soccer_test)

add_executable(matching_pennies_3p_test matching_pennies_3p_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(matching_pennies_3p_test matching_pennies_3p_test)

add_executable(matrix_games_test matrix_games_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(matrix_games_test matrix_games_test)

add_executable(oshi_zumo_test oshi_zumo_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(oshi_zumo_test oshi_zumo_test)

add_executable(oware_test oware_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(oware_test oware_test)

add_executable(pentago_test pentago_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(pentago_test pentago_test)

add_executable(phantom_ttt_test phantom_ttt_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(phantom_ttt_test phantom_ttt_test)

add_executable(pig_test pig_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(pig_test pig_test)

add_executable(quoridor_test quoridor_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(quoridor_test quoridor_test)

add_executable(tic_tac_toe_test tic_tac_toe_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tic_tac_toe_test tic_tac_toe_test)

add_executable(laser_tag_test laser_tag_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(laser_tag_test laser_tag_test)

add_executable(tiny_bridge_test tiny_bridge_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(tiny_bridge_test tiny_bridge_test)

add_executable(tiny_hanabi_test tiny_hanabi_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(tiny_hanabi_test tiny_hanabi_test)

add_executable(y_test y_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(y_test y_test)
