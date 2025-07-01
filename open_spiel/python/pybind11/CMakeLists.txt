if (OPEN_SPIEL_BUILDING_WHEEL)
  # When building a wheel, need to use this. See:
  # https://github.com/joerick/cibuildwheel/issues/639#issuecomment-817872369
  message(NOTICE "Building Wheel Detected. Finding Python Interpeter Development.Module")
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  # find_package(Python3 COMPONENTS Interpreter Development)
  unset(OPEN_SPIEL_PYTHONPATH)
  if(DEFINED ENV{PYTHONPATH})
    set (OPEN_SPIEL_PYTHONPATH $ENV{PYTHONPATH})
  endif()
else()
  message(NOTICE "Not building wheel. Finding Python normally...")
  find_package(Python3 COMPONENTS Interpreter Development)
  set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}/..:${CMAKE_CURRENT_SOURCE_DIR}/../../..;)
endif()

message(NOTICE "Python executable: ${Python3_EXECUTABLE}")
message(NOTICE "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(NOTICE "Python library dirs: ${Python3_LIBRARY_DIRS}")
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})

if(Python3_VERSION VERSION_LESS "3.8.0")
  message(FATAL_ERROR
    "Python found ${Python3_VERSION} < 3.8.0")
endif()

# List of all Python bindings to add to pyspiel.
include_directories (../../.. ../../pybind11_abseil ../../../pybind11/include)
set(PYTHON_BINDINGS ${PYTHON_BINDINGS}
  algorithms_corr_dist.cc
  algorithms_corr_dist.h
  algorithms_trajectories.cc
  algorithms_trajectories.h
  bots.cc
  bots.h
  evaluation_sco.cc
  evaluation_sco.h
  games_backgammon.cc
  games_backgammon.h
  games_bargaining.cc
  games_bargaining.h
  games_blackjack.cc
  games_blackjack.h
  games_bridge.cc
  games_bridge.h
  games_chess.cc
  games_chess.h
  games_crazy_eights.cc
  games_crazy_eights.h
  games_colored_trails.cc
  games_colored_trails.h
  games_dots_and_boxes.cc
  games_dots_and_boxes.h
  games_euchre.cc
  games_euchre.h
  games_gin_rummy.cc
  games_gin_rummy.h
  games_kuhn_poker.cc
  games_kuhn_poker.h
  games_leduc_poker.cc
  games_leduc_poker.h
  games_negotiation.cc
  games_negotiation.h
  games_spades.cc
  games_spades.h
  games_tarok.cc
  games_tarok.h
  games_tic_tac_toe.cc
  games_tic_tac_toe.h
  games_tiny_bridge.cc
  games_tiny_bridge.h
  games_trade_comm.cc
  games_trade_comm.h
  game_transforms.cc
  game_transforms.h
  observer.cc
  observer.h
  policy.cc
  policy.h
  pybind11.h
  pyspiel.cc
  python_games.cc
  python_games.h
  python_policy.cc
  python_policy.h
  utils.cc
  utils.h
  )

if (OPEN_SPIEL_BUILD_WITH_ACPC)
  set(PYTHON_BINDINGS ${PYTHON_BINDINGS}
      games_universal_poker.cc
      games_universal_poker.h
  )
endif()

# Optional pyspiel sub-modules, which can specify their python bindings.
if (OPEN_SPIEL_BUILD_WITH_GAMUT)
  set (PYTHON_BINDINGS ${PYTHON_BINDINGS}
      ../games/gamut/gamut_pybind11.h
      ../games/gamut/gamut_pybind11.cc
      )
endif()
if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  set (PYTHON_BINDINGS ${PYTHON_BINDINGS}
      ../bots/xinxin/xinxin_pybind11.h
      ../bots/xinxin/xinxin_pybind11.cc
      )
endif()

