set (HANABI_HOME $ENV{HOME}/hanabi-learning-environment)

add_library(hanabi_learning_environment OBJECT
  ${HANABI_HOME}/hanabi_lib/canonical_encoders.cc
  ${HANABI_HOME}/hanabi_lib/canonical_encoders.h
  ${HANABI_HOME}/hanabi_lib/hanabi_card.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_card.h
  ${HANABI_HOME}/hanabi_lib/hanabi_game.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_game.h
  ${HANABI_HOME}/hanabi_lib/hanabi_hand.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_hand.h
  ${HANABI_HOME}/hanabi_lib/hanabi_history_item.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_history_item.h
  ${HANABI_HOME}/hanabi_lib/hanabi_move.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_move.h
  ${HANABI_HOME}/hanabi_lib/hanabi_observation.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_observation.h
  ${HANABI_HOME}/hanabi_lib/hanabi_state.cc
  ${HANABI_HOME}/hanabi_lib/hanabi_state.h
  ${HANABI_HOME}/hanabi_lib/observation_encoder.h
  ${HANABI_HOME}/hanabi_lib/util.cc
  ${HANABI_HOME}/hanabi_lib/util.h
)

target_include_directories (hanabi_learning_environment PUBLIC ${HANABI_HOME})
target_include_directories (games PUBLIC ${HANABI_HOME})

add_executable(hanabi_test ../hanabi_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(hanabi_test hanabi_test)
target_include_directories (hanabi_test PUBLIC ${HANABI_HOME})
