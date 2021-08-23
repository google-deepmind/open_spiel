add_library (algorithms OBJECT
  best_response.cc
  best_response.h
  cfr.cc
  cfr.h
  cfr_br.cc
  cfr_br.h
  corr_dist.cc
  corr_dist.h
  corr_dist/afcce.cc
  corr_dist/afcce.h
  corr_dist/afce.cc
  corr_dist/afce.h
  corr_dist/efcce.cc
  corr_dist/efcce.h
  corr_dist/efce.cc
  corr_dist/efce.h
  corr_dist/cce.cc
  corr_dist/cce.h
  corr_dist/ce.cc
  corr_dist/ce.h
  corr_dev_builder.cc
  corr_dev_builder.h
  deterministic_policy.cc
  deterministic_policy.h
  evaluate_bots.cc
  evaluate_bots.h
  expected_returns.cc
  expected_returns.h
  external_sampling_mccfr.cc
  external_sampling_mccfr.h
  fsicfr.cc
  fsicfr.h
  get_all_histories.cc
  get_all_histories.h
  get_all_infostates.cc
  get_all_infostates.h
  get_all_states.cc
  get_all_states.h
  get_legal_actions_map.cc
  get_legal_actions_map.h
  history_tree.cc
  history_tree.h
  infostate_tree.h
  infostate_tree.cc
  is_mcts.cc
  is_mcts.h
  matrix_game_utils.cc
  matrix_game_utils.h
  nfg_writer.cc
  nfg_writer.h
  mcts.cc
  mcts.h
  minimax.cc
  minimax.h
  observation_history.h
  observation_history.cc
  oos.h
  oos.cc
  outcome_sampling_mccfr.cc
  outcome_sampling_mccfr.h
  policy_iteration.cc
  policy_iteration.h
  state_distribution.cc
  state_distribution.h
  tabular_best_response_mdp.cc
  tabular_best_response_mdp.h
  tabular_exploitability.cc
  tabular_exploitability.h
  tabular_q_learning.cc
  tabular_q_learning.h
  tabular_sarsa.cc
  tabular_sarsa.h
  tensor_game_utils.cc
  tensor_game_utils.h
  trajectories.cc
  trajectories.h
  value_iteration.cc
  value_iteration.h
)
target_include_directories (algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${OPEN_SPIEL_BUILD_WITH_ORTOOLS})
  add_subdirectory (ortools)
endif()

add_executable(best_response_test best_response_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(best_response_test best_response_test)

add_executable(cfr_test cfr_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(cfr_test cfr_test)

add_executable(cfr_br_test cfr_br_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(cfr_br_test cfr_br_test)

add_executable(corr_dist_test corr_dist_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(corr_dist_test corr_dist_test)

add_executable(corr_dev_builder_test corr_dev_builder_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(corr_dev_builder_test corr_dev_builder_test)

add_executable(deterministic_policy_test deterministic_policy_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(deterministic_policy_test deterministic_policy_test)

add_executable(evaluate_bots_test evaluate_bots_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(evaluate_bots_test evaluate_bots_test)

add_executable(external_sampling_mccfr_test external_sampling_mccfr_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(external_sampling_mccfr_test external_sampling_mccfr_test)

add_executable(get_all_histories_test get_all_histories_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(get_all_histories_test get_all_histories_test)

add_executable(get_all_states_test get_all_states_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(get_all_states_test get_all_states_test)

add_executable(get_legal_actions_map_test get_legal_actions_map_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(get_legal_actions_map_test get_legal_actions_map_test)

add_executable(history_tree_test history_tree_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(history_tree_test history_tree_test)

add_executable(infostate_tree_test   infostate_tree_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(infostate_tree_test infostate_tree_test)

add_executable(is_mcts_test is_mcts_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(is_mcts_test is_mcts_test)

add_executable(matrix_game_utils_test matrix_game_utils_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(matrix_game_utils_test matrix_game_utils_test)

add_executable(minimax_test minimax_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(minimax_test minimax_test)

add_executable(observation_history_test observation_history_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(observation_history_test observation_history_test)

add_executable(oos_test oos_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(oos_test oos_test)

add_executable(outcome_sampling_mccfr_test outcome_sampling_mccfr_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(outcome_sampling_mccfr_test outcome_sampling_mccfr_test)

add_executable(state_distribution_test state_distribution_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(state_distribution_test state_distribution_test)

add_executable(tabular_best_response_mdp_test tabular_best_response_mdp_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(tabular_best_response_mdp_test tabular_best_response_mdp_test)

add_executable(tabular_exploitability_test tabular_exploitability_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(tabular_exploitability_test tabular_exploitability_test)

add_executable(tensor_game_utils_test tensor_game_utils_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(tensor_game_utils_test tensor_game_utils_test)

add_executable(trajectories_test trajectories_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(trajectories_test trajectories_test)

add_subdirectory (alpha_zero)
add_subdirectory (alpha_zero_torch)
add_subdirectory (dqn_torch)
