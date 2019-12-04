add_library (algorithms OBJECT
  best_response.h
  best_response.cc
  cfr.h
  cfr.cc
  cfr_br.h
  cfr_br.cc
  deterministic_policy.h
  deterministic_policy.cc
  evaluate_bots.h
  evaluate_bots.cc
  expected_returns.h
  expected_returns.cc
  external_sampling_mccfr.h
  external_sampling_mccfr.cc
  get_all_states.h
  get_all_states.cc
  get_legal_actions_map.h
  get_legal_actions_map.cc
  history_tree.h
  history_tree.cc
  matrix_game_utils.h
  matrix_game_utils.cc
  mcts.h
  mcts.cc
  minimax.h
  minimax.cc
  outcome_sampling_mccfr.h
  outcome_sampling_mccfr.cc
  state_distribution.h
  state_distribution.cc
  tabular_exploitability.h
  tabular_exploitability.cc
  trajectories.h
  trajectories.cc
  value_iteration.h
  value_iteration.cc
)
target_include_directories (algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(best_response_test best_response_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(best_response_test best_response_test)

add_executable(cfr_test cfr_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(cfr_test cfr_test)

add_executable(cfr_br_test cfr_br_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(cfr_br_test cfr_br_test)

add_executable(deterministic_policy_test deterministic_policy_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(deterministic_policy_test deterministic_policy_test)

add_executable(evaluate_bots_test evaluate_bots_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(evaluate_bots_test evaluate_bots_test)

add_executable(external_sampling_mccfr_test external_sampling_mccfr_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(external_sampling_mccfr_test external_sampling_mccfr_test)

add_executable(get_all_states_test get_all_states_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(get_all_states_test get_all_states_test)

add_executable(get_legal_actions_map_test get_legal_actions_map_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(get_legal_actions_map_test get_legal_actions_map_test)

add_executable(history_tree_test history_tree_test.cc
        $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(history_tree_test history_tree_test)

add_executable(matrix_game_utils_test matrix_game_utils_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(matrix_game_utils_test matrix_game_utils_test)

add_executable(minimax_test minimax_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(minimax_test minimax_test)

add_executable(outcome_sampling_mccfr_test outcome_sampling_mccfr_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(outcome_sampling_mccfr_test outcome_sampling_mccfr_test)

add_executable(state_distribution_test state_distribution_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(state_distribution_test state_distribution_test)

add_executable(tabular_exploitability_test tabular_exploitability_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(tabular_exploitability_test tabular_exploitability_test)

add_executable(trajectories_test trajectories_test.cc
    $<TARGET_OBJECTS:algorithms> ${OPEN_SPIEL_OBJECTS})
add_test(trajectories_test trajectories_test)
