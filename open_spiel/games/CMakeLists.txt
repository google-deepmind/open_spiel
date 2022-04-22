set(GAME_SOURCES
  amazons.cc
  amazons.h
  backgammon.cc
  backgammon.h
  bargaining.cc
  bargaining.h
  battleship.cc
  battleship.h
  battleship_types.h
  battleship_types.cc
  blackjack.cc
  blackjack.h
  blotto.cc
  blotto.h
  breakthrough.cc
  breakthrough.h
  bridge.cc
  bridge.h
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
  cliff_walking.cc
  cliff_walking.h
  clobber.cc
  clobber.h
  coin_game.cc
  coin_game.h
  colored_trails.cc
  colored_trails.h
  colored_trails/colored_trails_utils.cc
  connect_four.cc
  connect_four.h
  coop_box_pushing.cc
  coop_box_pushing.h
  coordinated_mp.cc
  coordinated_mp.h
  cursor_go.cc
  cursor_go.h
  dark_chess.cc
  dark_chess.h
  dark_hex.cc
  dark_hex.h
  deep_sea.cc
  deep_sea.h
  efg_game.cc
  efg_game.h
  efg_game_data.cc
  efg_game_data.h
  first_sealed_auction.cc
  first_sealed_auction.h
  gin_rummy.cc
  gin_rummy.h
  gin_rummy/gin_rummy_utils.cc
  gin_rummy/gin_rummy_utils.h
  go.cc
  go.h
  go/go_board.cc
  go/go_board.h
  goofspiel.cc
  goofspiel.h
  havannah.cc
  havannah.h
  hearts.cc
  hearts.h
  hex.cc
  hex.h
  kriegspiel.cc
  kriegspiel.h
  kuhn_poker.cc
  kuhn_poker.h
  laser_tag.cc
  laser_tag.h
  leduc_poker.cc
  leduc_poker.h
  lewis_signaling.cc
  lewis_signaling.h
  liars_dice.cc
  liars_dice.h
  markov_soccer.cc
  markov_soccer.h
  matching_pennies_3p.cc
  matching_pennies_3p.h
  matrix_games.cc
  mfg/crowd_modelling.cc
  mfg/crowd_modelling.h
  mfg/crowd_modelling_2d.cc
  mfg/crowd_modelling_2d.h
  mfg/garnet.cc
  mfg/garnet.h
  morpion_solitaire.cc
  morpion_solitaire.h
  negotiation.cc
  negotiation.h
  nfg_game.cc
  nfg_game.h
  oh_hell.cc
  oh_hell.h
  oshi_zumo.cc
  oshi_zumo.h
  othello.cc
  othello.h
  oware.cc
  oware.h
  oware/oware_board.cc
  oware/oware_board.h
  pathfinding.cc
  pathfinding.h
  pentago.cc
  pentago.h
  phantom_ttt.cc
  phantom_ttt.h
  pig.cc
  pig.h
  quoridor.cc
  quoridor.h
  rbc.cc
  rbc.h
  sheriff.cc
  sheriff.h
  skat.cc
  skat.h
  solitaire.cc
  solitaire.h
  stones_and_gems.cc
  stones_and_gems.h
  tarok.cc
  tarok.h
  tarok/cards.cc
  tarok/cards.h
  tarok/contracts.cc
  tarok/contracts.h
  tic_tac_toe.cc
  tic_tac_toe.h
  tiny_bridge.cc
  tiny_bridge.h
  tiny_hanabi.cc
  tiny_hanabi.h
  trade_comm.cc
  trade_comm.h
  ultimate_tic_tac_toe.h
  ultimate_tic_tac_toe.cc
  y.cc
  y.h
)

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  set(GAME_SOURCES ${GAME_SOURCES} hanabi.cc hanabi.h)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  set(GAME_SOURCES ${GAME_SOURCES} universal_poker.cc universal_poker.h)
endif()

add_library (games OBJECT ${GAME_SOURCES})

target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  add_subdirectory(hanabi)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_subdirectory(universal_poker)
endif()
if (${OPEN_SPIEL_BUILD_WITH_GAMUT})
  add_subdirectory(gamut)
endif()


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

add_executable(amazons_test amazons_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(amazons_test amazons_test)

add_executable(backgammon_test backgammon_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(backgammon_test backgammon_test)

add_executable(bargaining_instance_generator bargaining_instance_generator.cc
               ${OPEN_SPIEL_OBJECTS})
add_executable(bargaining_test bargaining_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(bargaining_test bargaining_test)

add_executable(battleship_test battleship_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(battleship_test battleship_test)

add_executable(blackjack_test blackjack_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(blackjack_test blackjack_test)

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

add_executable(cliff_walking_test cliff_walking_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(cliff_walking_test cliff_walking_test)

add_executable(clobber_test clobber_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(clobber_test clobber_test)

add_executable(coin_game_test coin_game_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(coin_game_test coin_game_test)

add_executable(colored_trails_board_generator
               colored_trails/colored_trails_board_generator.cc
               ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)

add_executable(colored_trails_test colored_trails_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(colored_trails_test colored_trails_test)

add_executable(connect_four_test connect_four_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(connect_four_test connect_four_test)

add_executable(coop_box_pushing_test coop_box_pushing_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(coop_box_pushing_test coop_box_pushing_test)

add_executable(coordinated_mp_test coordinated_mp_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>
        $<TARGET_OBJECTS:algorithms>)
add_test(coordinated_mp_test coordinated_mp_test)

add_executable(crowd_modelling_test mfg/crowd_modelling_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(crowd_modelling_test crowd_modelling_test)

add_executable(crowd_modelling_2d_test mfg/crowd_modelling_2d_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(crowd_modelling_2d_test crowd_modelling_2d_test)

add_executable(cursor_go_test cursor_go_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(cursor_go_test cursor_go_test)

add_executable(dark_chess_test dark_chess_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(dark_chess_test dark_chess_test)

add_executable(dark_hex_test dark_hex_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(dark_hex_test dark_hex_test)

add_executable(deep_sea_test deep_sea_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(deep_sea_test deep_sea_test)

add_executable(efg_game_test efg_game_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(efg_game_test efg_game_test)

add_executable(first_sealed_auction_test first_sealed_auction_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(first_sealed_auction_test first_sealed_auction_test)

add_executable(garnet_test mfg/garnet_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(garnet_test garnet_test)

add_executable(gin_rummy_test gin_rummy_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(gin_rummy_test gin_rummy_test)

add_executable(go_test go_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(go_test go_test)

add_executable(goofspiel_test goofspiel_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(goofspiel_test goofspiel_test)

add_executable(havannah_test havannah_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(havannah_test havannah_test)

add_executable(hearts_test hearts_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(hearts_test hearts_test)

add_executable(hex_test hex_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(hex_test hex_test)

add_executable(kriegspiel_test kriegspiel_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(kriegspiel_test kriegspiel_test)

add_executable(kuhn_poker_test kuhn_poker_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(kuhn_poker_test kuhn_poker_test)

add_executable(leduc_poker_test leduc_poker_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(leduc_poker_test leduc_poker_test)

add_executable(lewis_signaling_test lewis_signaling_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(lewis_signaling_test lewis_signaling_test)

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

add_executable(morpion_solitaire_test morpion_solitaire_test.cc ${OPEN_SPIEL_OBJECTS}
        $<TARGET_OBJECTS:tests>)
add_test(morpion_solitaire_test morpion_solitaire_test)

add_executable(negotiation_test negotiation_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(negotiation_test negotiation_test)

add_executable(nfg_game_test nfg_game_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(nfg_game_test nfg_game_test)

add_executable(oh_hell_test oh_hell_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(oh_hell_test oh_hell_test)

add_executable(oshi_zumo_test oshi_zumo_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(oshi_zumo_test oshi_zumo_test)

add_executable(othello_test othello_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(othello_test othello_test)

add_executable(oware_test oware_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>
               $<TARGET_OBJECTS:algorithms>)
add_test(oware_test oware_test)

add_executable(pathfinding_test pathfinding_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(pathfinding_test pathfinding_test)

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

add_executable(rbc_test rbc_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(rbc_test rbc_test)

add_executable(sheriff_test sheriff_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(sheriff_test sheriff_test)

add_executable(skat_test skat_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(skat_test skat_test)

add_executable(solitaire_test solitaire_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(solitaire_test solitaire_test)

add_executable(stones_and_gems_test stones_and_gems_test.cc
               ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(stones_and_gems_test stones_and_gems_test)

add_executable(tarok_test tarok_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(tarok_test tarok_test)

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

add_executable(trade_comm_test trade_comm_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(trade_comm_test trade_comm_test)

add_executable(ultimate_tic_tac_toe_test ultimate_tic_tac_toe_test.cc
               ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
add_test(ultimate_tic_tac_toe_test ultimate_tic_tac_toe_test)

if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_executable(universal_poker_test universal_poker_test.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:tests>
          $<TARGET_OBJECTS:algorithms>)
  add_test(universal_poker_test universal_poker_test
           --subgames_data_dir=${CMAKE_CURRENT_SOURCE_DIR}/universal_poker/endgames)
endif()

add_executable(y_test y_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(y_test y_test)

