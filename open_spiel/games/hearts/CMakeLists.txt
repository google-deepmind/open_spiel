add_library(xinxin OBJECT
  hearts/Algorithm.cpp
  hearts/Algorithm.h
  hearts/CardGameState.cpp
  hearts/CardGameState.h
  hearts/CardProbabilityData.cpp
  hearts/CardProbabilityData.h
  hearts/Game.cpp
  hearts/Game.h
  hearts/GameState.cpp
  hearts/GameState.h
  hearts/Hearts.cpp
  hearts/Hearts.h
  hearts/HeartsGameData.cpp
  hearts/HeartsGameData.h
  hearts/HeartsGameHistories.cpp
  hearts/HeartsGameHistories.h
  hearts/Player.cpp
  hearts/Player.h
  hearts/ProblemState.cpp
  hearts/ProblemState.h
  hearts/States.cpp
  hearts/States.h
  hearts/Timer.cpp
  hearts/Timer.h
  hearts/UCT.cpp
  hearts/UCT.h
  hearts/algorithmStates.cpp
  hearts/algorithmStates.h
  hearts/fpUtil.cpp
  hearts/fpUtil.h
  hearts/hash.cpp
  hearts/hash.h
  hearts/iiGameState.cpp
  hearts/iiGameState.h
  hearts/iiMonteCarlo.cpp
  hearts/iiMonteCarlo.h
  hearts/mt_random.cpp
  hearts/mt_random.h
  xinxin_bot.cc
  xinxin_bot.h
)

target_include_directories (xinxin PUBLIC hearts)

add_executable (xinxin_bot_test xinxin_bot_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(xinxin_bot_test xinxin_bot_test)

add_executable (xinxin_game_generator xinxin_game_generator.cc ${OPEN_SPIEL_OBJECTS})
