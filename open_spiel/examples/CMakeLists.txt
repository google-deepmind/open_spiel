add_executable(benchmark_game benchmark_game.cc ${OPEN_SPIEL_OBJECTS})
add_test(benchmark_game_test benchmark_game --game=tic_tac_toe --sims=100 --attempts=2)

add_executable(cfr_example cfr_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(cfr_example_test cfr_example)

add_executable(cfr_multi_equilibria_example cfr_multi_equilibria_example.cc
               ${OPEN_SPIEL_OBJECTS})

add_executable(imperfect_recall_mccfr imperfect_recall_mccfr.cc
               ${OPEN_SPIEL_OBJECTS})

add_executable(example example.cc ${OPEN_SPIEL_OBJECTS})
add_test(example_test example --game=tic_tac_toe --seed=0)

add_executable(fsicfr_liars_dice fsicfr_liars_dice.cc ${OPEN_SPIEL_OBJECTS})

add_executable(gtp gtp.cc ${OPEN_SPIEL_OBJECTS})

add_executable(matrix_example matrix_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(matrix_example_test matrix_example)

add_executable(mcts_example mcts_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(mcts_example_test mcts_example)

add_executable(minimax_example minimax_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(minimax_example_test minimax_example)

add_executable(policy_iteration_example policy_iteration_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(policy_iteration_example_test policy_iteration_example)

add_executable(value_iteration_example value_iteration_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(value_iteration_example_test value_iteration_example)

add_executable(tabular_sarsa_example tabular_sarsa_example.cc ${OPEN_SPIEL_OBJECTS})

add_executable(tabular_q_learning_example tabular_q_learning_example.cc ${OPEN_SPIEL_OBJECTS})

add_executable(count_all_states count_all_states.cc ${OPEN_SPIEL_OBJECTS})

if (OPEN_SPIEL_BUILD_WITH_TENSORFLOW_CC)
  add_executable(alpha_zero_example alpha_zero_example.cc ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:alpha_zero>)
  add_test(alpha_zero_example_test alpha_zero_example)
  target_link_libraries(alpha_zero_example TensorflowCC::TensorflowCC)
endif ()

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_executable(alpha_zero_torch_example alpha_zero_torch_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch>)
  target_link_libraries (alpha_zero_torch_example ${TORCH_LIBRARIES})

  add_executable(alpha_zero_torch_game_example alpha_zero_torch_game_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch>
                $<TARGET_OBJECTS:bots>)
  target_link_libraries (alpha_zero_torch_game_example ${TORCH_LIBRARIES})

  add_executable(dqn_torch_example dqn_torch_example.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:dqn_torch>)
  target_link_libraries (dqn_torch_example ${TORCH_LIBRARIES})
endif ()
