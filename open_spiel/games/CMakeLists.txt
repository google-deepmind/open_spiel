set(GAME_SOURCES)

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  set(GAME_SOURCES ${GAME_SOURCES} hanabi/hanabi.cc hanabi/hanabi.h)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  set(GAME_SOURCES ${GAME_SOURCES} universal_poker/universal_poker.cc universal_poker/universal_poker.h)
endif()

add_library (games OBJECT)

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

# Convenience macro to register a game and one of its tests.
macro(add_game)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCES LIBRARIES TEST_SOURCES TEST_LIBRARIES TEST_ARGS)
  cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Create a library that others can link against to play the game
  add_library("${_NAME}" ${_SOURCES})
  target_link_libraries("${_NAME}" open_spiel_core ${_LIBRARIES})
  
  # Register the game's source files.
  # This line is needed so that the variable ${OPEN_SPIEL_OBJECTS} includes
  # the source files of every game. We would like to replace usages of
  # ${OPEN_SPIEL_OBJECTS} with proper library linkage, and eventually remove
  # this line once we no longer rely on ${OPEN_SPIEL_OBJECTS}.
  list(APPEND GAME_SOURCES ${_SOURCES})
  
  add_game_test(NAME "${_NAME}_test"
    SOURCES ${_TEST_SOURCES}
    LIBRARIES "${_NAME}" ${_TEST_LIBRARIES}
    ARGS ${_TEST_ARGS})
endmacro()
  
# Build and register an OpenSpiel test
macro(add_game_test)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCES LIBRARIES ARGS)
  cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_game_executable(NAME "${_NAME}" SOURCES ${_SOURCES} LIBRARIES tests ${_LIBRARIES})
  add_test(NAME "${_NAME}" COMMAND "${_NAME}" ${_ARGS})
endmacro()

macro(add_game_executable)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCES LIBRARIES)
  cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable("${_NAME}" ${_SOURCES})
  target_link_libraries("${_NAME}" open_spiel_core ${_LIBRARIES})
endmacro()

add_game(NAME 2048
  SOURCES twenty_forty_eight/2048.cc twenty_forty_eight/2048.h
  TEST_SOURCES twenty_forty_eight/2048_test.cc)

add_game(NAME amazons
  SOURCES amazons/amazons.cc amazons/amazons.h
  TEST_SOURCES amazons/amazons_test.cc)

add_game(NAME backgammon
  SOURCES backgammon/backgammon.cc backgammon/backgammon.h
  TEST_SOURCES backgammon/backgammon_test.cc)

add_game(NAME bargaining
  SOURCES bargaining/bargaining.cc bargaining/bargaining.h
  LIBRARIES utils
  TEST_SOURCES bargaining/bargaining_test.cc
  TEST_LIBRARIES utils)
add_game_executable(NAME bargaining_instance_generator
  SOURCES bargaining/bargaining_instance_generator.cc
  LIBRARIES bargaining utils)

add_game(NAME battleship
  SOURCES
    battleship/battleship.cc
    battleship/battleship.h
    battleship/battleship_types.h
    battleship/battleship_types.cc
  TEST_SOURCES
    battleship/battleship_test.cc
  TEST_LIBRARIES
    algorithms game_transforms utils)

add_game(NAME blackjack
  SOURCES blackjack/blackjack.cc blackjack/blackjack.h
  TEST_SOURCES blackjack/blackjack_test.cc)

add_game(NAME blotto
  SOURCES blotto/blotto.cc blotto/blotto.h
  TEST_SOURCES blotto/blotto_test.cc)

add_game(NAME breakthrough
  SOURCES breakthrough/breakthrough.cc breakthrough/breakthrough.h
  TEST_SOURCES breakthrough/breakthrough_test.cc)

add_game(NAME bridge
  SOURCES
    bridge/bridge.cc
    bridge/bridge.h
    bridge/bridge_scoring.cc
    bridge/bridge_scoring.h
    bridge/bridge_uncontested_bidding.cc
    bridge/bridge_uncontested_bidding.h
  LIBRARIES
    bridge_double_dummy_solver
  TEST_SOURCES
    bridge/bridge_test.cc)

add_game(NAME catch
  SOURCES catch/catch.cc catch/catch.h
  TEST_SOURCES catch/catch_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME checkers
  SOURCES checkers/checkers.cc checkers/checkers.h
  TEST_SOURCES checkers/checkers_test.cc)

add_game(NAME chess
  SOURCES
    chess/chess.cc
    chess/chess.h
    chess/chess_board.cc
    chess/chess_board.h
    chess/chess_common.cc
    chess/chess_common.h
  TEST_SOURCES
    chess/chess_test.cc)

add_game(NAME cliff_walking
  SOURCES cliff_walking/cliff_walking.cc cliff_walking/cliff_walking.h
  TEST_SOURCES cliff_walking/cliff_walking_test.cc)

add_game(NAME clobber
  SOURCES clobber/clobber.cc clobber/clobber.h
  TEST_SOURCES clobber/clobber_test.cc)

add_game(NAME coin_game
  SOURCES coin_game/coin_game.cc coin_game/coin_game.h
  TEST_SOURCES coin_game/coin_game_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME colored_trails
  SOURCES
    colored_trails/colored_trails.cc
    colored_trails/colored_trails.h
    colored_trails/colored_trails_utils.cc
  LIBRARIES
    utils
  TEST_SOURCES
    colored_trails/colored_trails_test.cc
  TEST_LIBRARIES
    utils)
add_game_executable(NAME colored_trails_board_generator
  SOURCES colored_trails/colored_trails_board_generator.cc
  LIBRARIES colored_trails utils)

add_game(NAME connect_four
  SOURCES connect_four/connect_four.cc connect_four/connect_four.h
  TEST_SOURCES connect_four/connect_four_test.cc)

add_game(NAME coop_box_pushing
  SOURCES coop_box_pushing/coop_box_pushing.cc coop_box_pushing/coop_box_pushing.h
  TEST_SOURCES coop_box_pushing/coop_box_pushing_test.cc)

add_game(NAME coordinated_mp
  SOURCES coordinated_mp/coordinated_mp.cc coordinated_mp/coordinated_mp.h
  TEST_SOURCES coordinated_mp/coordinated_mp_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME crazy_eights
  SOURCES crazy_eights/crazy_eights.cc crazy_eights/crazy_eights.h
  TEST_SOURCES crazy_eights/crazy_eights_test.cc)

add_game(NAME crowd_modelling
  SOURCES
    dynamic_routing/dynamic_routing_data.cc
    dynamic_routing/dynamic_routing_data.h
    dynamic_routing/dynamic_routing_utils.cc
    dynamic_routing/dynamic_routing_utils.h
    mfg/crowd_modelling.cc
    mfg/crowd_modelling.h
    mfg/crowd_modelling_2d.cc
    mfg/crowd_modelling_2d.h
    mfg/dynamic_routing.cc
    mfg/dynamic_routing.h
    mfg/garnet.cc
    mfg/garnet.h
  TEST_SOURCES
    mfg/crowd_modelling_test.cc)

add_game_test(NAME crowd_modelling_2d_test
  SOURCES mfg/crowd_modelling_2d_test.cc
  LIBRARIES crowd_modelling)

add_game_test(NAME dynamic_routing_test
  SOURCES mfg/dynamic_routing_test.cc
  LIBRARIES crowd_modelling)

add_game_test(NAME dynamic_routing_data_test
  SOURCES dynamic_routing/dynamic_routing_data_test.cc
  LIBRARIES crowd_modelling)

add_game_test(NAME dynamic_routing_utils_test
  SOURCES dynamic_routing/dynamic_routing_utils_test.cc
  LIBRARIES crowd_modelling)

add_game_test(NAME garnet_test
  SOURCES mfg/garnet_test.cc
  LIBRARIES crowd_modelling)

add_game(NAME cursor_go
  SOURCES cursor_go/cursor_go.cc cursor_go/cursor_go.h
  LIBRARIES go
  TEST_SOURCES cursor_go/cursor_go_test.cc
  TEST_LIBRARIES go)

add_game(NAME dark_chess
  SOURCES dark_chess/dark_chess.cc dark_chess/dark_chess.h
  LIBRARIES chess
  TEST_SOURCES dark_chess/dark_chess_test.cc
  TEST_LIBRARIES chess)

add_game(NAME dark_hex
  SOURCES dark_hex/dark_hex.cc dark_hex/dark_hex.h
  LIBRARIES hex
  TEST_SOURCES dark_hex/dark_hex_test.cc)

add_game(NAME deep_sea
  SOURCES deep_sea/deep_sea.cc deep_sea/deep_sea.h
  TEST_SOURCES deep_sea/deep_sea_test.cc)

add_game(NAME dots_and_boxes
  SOURCES dots_and_boxes/dots_and_boxes.cc dots_and_boxes/dots_and_boxes.h
  TEST_SOURCES dots_and_boxes/dots_and_boxes_test.cc)

add_game(NAME dou_dizhu
  SOURCES
    dou_dizhu/dou_dizhu.cc
    dou_dizhu/dou_dizhu.h
    dou_dizhu/dou_dizhu_utils.cc
    dou_dizhu/dou_dizhu_utils.h
  TEST_SOURCES
    dou_dizhu/dou_dizhu_test.cc)
add_game_test(NAME dou_dizhu_utils_test
  SOURCES dou_dizhu/dou_dizhu_utils_test.cc
  LIBRARIES dou_dizhu)

add_game(NAME efg_game
  SOURCES
    efg_game/efg_game.cc
    efg_game/efg_game.h
    efg_game/efg_game_data.cc
    efg_game/efg_game_data.h
  LIBRARIES
    utils
  TEST_SOURCES
    efg_game/efg_game_test.cc
  TEST_LIBRARIES
    utils)

add_game(NAME euchre
  SOURCES euchre/euchre.cc euchre/euchre.h
  TEST_SOURCES euchre/euchre_test.cc)

add_game(NAME first_sealed_auction
  SOURCES first_sealed_auction/first_sealed_auction.cc first_sealed_auction/first_sealed_auction.h
  TEST_SOURCES first_sealed_auction/first_sealed_auction_test.cc)

add_game(NAME gin_rummy
  SOURCES
    gin_rummy/gin_rummy.cc
    gin_rummy/gin_rummy.h
    gin_rummy/gin_rummy_utils.cc
    gin_rummy/gin_rummy_utils.h
  LIBRARIES
    algorithms
    game_transforms
    utils
  TEST_SOURCES
    gin_rummy/gin_rummy_test.cc)

add_game(NAME go
  SOURCES
    go/go.cc
    go/go.h
    go/go_board.cc
    go/go_board.h
  TEST_SOURCES
    go/go_test.cc)

add_game(NAME goofspiel
  SOURCES goofspiel/goofspiel.cc goofspiel/goofspiel.h
  TEST_SOURCES goofspiel/goofspiel_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME havannah
  SOURCES havannah/havannah.cc havannah/havannah.h
  TEST_SOURCES havannah/havannah_test.cc)

add_game(NAME hearts
  SOURCES hearts/hearts.cc hearts/hearts.h
  TEST_SOURCES hearts/hearts_test.cc)

add_game(NAME hex
  SOURCES hex/hex.cc hex/hex.h
  TEST_SOURCES hex/hex_test.cc)

add_game(NAME kriegspiel
  SOURCES kriegspiel/kriegspiel.cc kriegspiel/kriegspiel.h
  LIBRARIES chess
  TEST_SOURCES kriegspiel/kriegspiel_test.cc)

add_game(NAME kuhn_poker
  SOURCES kuhn_poker/kuhn_poker.cc kuhn_poker/kuhn_poker.h
  TEST_SOURCES kuhn_poker/kuhn_poker_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME laser_tag
  SOURCES laser_tag/laser_tag.cc laser_tag/laser_tag.h
  TEST_SOURCES laser_tag/laser_tag_test.cc)

add_game(NAME leduc_poker
  SOURCES leduc_poker/leduc_poker.cc leduc_poker/leduc_poker.h
  TEST_SOURCES leduc_poker/leduc_poker_test.cc)

add_game(NAME lewis_signaling
  SOURCES lewis_signaling/lewis_signaling.cc lewis_signaling/lewis_signaling.h
  TEST_SOURCES lewis_signaling/lewis_signaling_test.cc)

add_game(NAME liars_dice
  SOURCES liars_dice/liars_dice.cc liars_dice/liars_dice.h
  TEST_SOURCES liars_dice/liars_dice_test.cc)

add_game(NAME maedn
  SOURCES maedn/maedn.cc maedn/maedn.h
  TEST_SOURCES maedn/maedn_test.cc)

add_game(NAME mancala
  SOURCES mancala/mancala.cc mancala/mancala.h
  TEST_SOURCES mancala/mancala_test.cc)

add_game(NAME markov_soccer
  SOURCES markov_soccer/markov_soccer.cc markov_soccer/markov_soccer.h
  TEST_SOURCES markov_soccer/markov_soccer_test.cc)

add_game(NAME matching_pennies_3p
  SOURCES matching_pennies_3p/matching_pennies_3p.cc matching_pennies_3p/matching_pennies_3p.h
  TEST_SOURCES matching_pennies_3p/matching_pennies_3p_test.cc)

add_game(NAME matrix_games
  SOURCES matrix_games/matrix_games.cc
  TEST_SOURCES matrix_games/matrix_games_test.cc)

add_game(NAME morpion_solitaire
  SOURCES morpion_solitaire/morpion_solitaire.cc morpion_solitaire/morpion_solitaire.h
  TEST_SOURCES morpion_solitaire/morpion_solitaire_test.cc)

add_game(NAME negotiation
  SOURCES negotiation/negotiation.cc negotiation/negotiation.h
  TEST_SOURCES negotiation/negotiation_test.cc)

add_game(NAME nfg_game
  SOURCES nfg_game/nfg_game.cc nfg_game/nfg_game.h
  TEST_SOURCES nfg_game/nfg_game_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME nine_mens_morris
  SOURCES nine_mens_morris/nine_mens_morris.cc nine_mens_morris/nine_mens_morris.h
  TEST_SOURCES nine_mens_morris/nine_mens_morris_test.cc)

add_game(NAME nim
  SOURCES nim/nim.cc nim/nim.h
  TEST_SOURCES nim/nim_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME oh_hell
  SOURCES oh_hell/oh_hell.cc oh_hell/oh_hell.h
  TEST_SOURCES oh_hell/oh_hell_test.cc)

add_game(NAME oshi_zumo
  SOURCES oshi_zumo/oshi_zumo.cc oshi_zumo/oshi_zumo.h
  TEST_SOURCES oshi_zumo/oshi_zumo_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME othello
  SOURCES othello/othello.cc othello/othello.h
  TEST_SOURCES othello/othello_test.cc)

add_game(NAME oware
  SOURCES oware/oware.cc oware/oware.h oware/oware_board.cc oware/oware_board.h
  TEST_SOURCES oware/oware_test.cc)

add_game(NAME pathfinding
  SOURCES pathfinding/pathfinding.cc pathfinding/pathfinding.h
  LIBRARIES utils
  TEST_SOURCES pathfinding/pathfinding_test.cc)

add_game(NAME pentago
  SOURCES pentago/pentago.cc pentago/pentago.h
  TEST_SOURCES pentago/pentago_test.cc)

add_game(NAME phantom_go
  SOURCES phantom_go/phantom_go.h phantom_go/phantom_go.cc phantom_go/phantom_go_board.h phantom_go/phantom_go_board.cc
  TEST_SOURCES phantom_go/phantom_go_test.cc)

add_game(NAME phantom_ttt
  SOURCES phantom_ttt/phantom_ttt.cc phantom_ttt/phantom_ttt.h
  LIBRARIES tic_tac_toe
  TEST_SOURCES phantom_ttt/phantom_ttt_test.cc)

add_game(NAME pig
  SOURCES pig/pig.cc pig/pig.h
  TEST_SOURCES pig/pig_test.cc)

add_game(NAME quoridor
  SOURCES quoridor/quoridor.cc quoridor/quoridor.h
  TEST_SOURCES quoridor/quoridor_test.cc)

add_game(NAME rbc
  SOURCES rbc/rbc.cc rbc/rbc.h
  LIBRARIES chess
  TEST_SOURCES rbc/rbc_test.cc)

add_game(NAME sheriff
  SOURCES sheriff/sheriff.cc sheriff/sheriff.h
  TEST_SOURCES sheriff/sheriff_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME skat
  SOURCES skat/skat.cc skat/skat.h
  TEST_SOURCES skat/skat_test.cc)

add_game(NAME solitaire
  SOURCES solitaire/solitaire.cc solitaire/solitaire.h
  TEST_SOURCES solitaire/solitaire_test.cc)

add_game(NAME stones_and_gems
  SOURCES stones_and_gems/stones_and_gems.cc stones_and_gems/stones_and_gems.h
  TEST_SOURCES stones_and_gems/stones_and_gems_test.cc)

add_game(NAME tarok
  SOURCES tarok/tarok.cc tarok/tarok.h tarok/cards.cc tarok/cards.h tarok/contracts.cc tarok/contracts.h
  TEST_SOURCES tarok/tarok_test.cc)

add_game(NAME tic_tac_toe
  SOURCES tic_tac_toe/tic_tac_toe.cc tic_tac_toe/tic_tac_toe.h
  TEST_SOURCES tic_tac_toe/tic_tac_toe_test.cc)

add_game(NAME tiny_bridge
  SOURCES tiny_bridge/tiny_bridge.cc tiny_bridge/tiny_bridge.h
  TEST_SOURCES tiny_bridge/tiny_bridge_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME tiny_hanabi
  SOURCES tiny_hanabi/tiny_hanabi.cc tiny_hanabi/tiny_hanabi.h
  TEST_SOURCES tiny_hanabi/tiny_hanabi_test.cc
  TEST_LIBRARIES algorithms game_transforms utils)

add_game(NAME trade_comm
  SOURCES trade_comm/trade_comm.cc trade_comm/trade_comm.h
  TEST_SOURCES trade_comm/trade_comm_test.cc)

add_game(NAME ultimate_tic_tac_toe
  SOURCES ultimate_tic_tac_toe/ultimate_tic_tac_toe.h ultimate_tic_tac_toe/ultimate_tic_tac_toe.cc
  LIBRARIES tic_tac_toe
  TEST_SOURCES ultimate_tic_tac_toe/ultimate_tic_tac_toe_test.cc)

add_game(NAME y
  SOURCES y/y.cc y/y.h
  TEST_SOURCES y/y_test.cc)

if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_executable(universal_poker_test universal_poker/universal_poker_test.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:tests>
          $<TARGET_OBJECTS:algorithms>)
  add_test(universal_poker_test universal_poker_test
           --subgames_data_dir=${CMAKE_CURRENT_SOURCE_DIR}/universal_poker/endgames)
endif()

target_sources(games PRIVATE ${GAME_SOURCES})
