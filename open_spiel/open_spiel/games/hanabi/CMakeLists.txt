add_library(hanabi_learning_environment OBJECT
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/canonical_encoders.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/canonical_encoders.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_card.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_card.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_game.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_game.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_hand.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_hand.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_history_item.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_history_item.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_move.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_move.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_observation.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_observation.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_state.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/hanabi_state.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/observation_encoder.h
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/util.cc
  hanabi-learning-environment/hanabi_learning_environment/hanabi_lib/util.h
)

target_include_directories (hanabi_learning_environment PUBLIC hanabi-learning-environment/hanabi_learning_environment)
target_include_directories (games PUBLIC hanabi-learning-environment/hanabi_learning_environment)

add_executable(hanabi_test hanabi_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(hanabi_test hanabi_test)
target_include_directories (hanabi_test PUBLIC hanabi-learning-environment/hanabi_learning_environment)
