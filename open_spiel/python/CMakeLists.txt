# Python is found in ../cmake/dependencies.cmake, before any target is defined.
if (OPEN_SPIEL_BUILDING_WHEEL)
  unset(OPEN_SPIEL_PYTHONPATH)
  if(DEFINED ENV{PYTHONPATH})
    set (OPEN_SPIEL_PYTHONPATH $ENV{PYTHONPATH})
  endif()
else()
  set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;)
endif()

# The Python bindings: the pyspiel module and the pybind_library targets it is
# made of. pybind11/CMakeLists.txt is generated from pybind11/BUILD.
add_subdirectory(pybind11)

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")
if (WIN32)
  # Python does not seem able to import the module without this change.
  set_target_properties(pyspiel PROPERTIES SUFFIX ".pyd")
endif()
# Put pyspiel in this directory of the build tree (which is on the PYTHONPATH
# of the tests below), unless the caller chose an output directory, as setup.py
# does when building a wheel.
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(pyspiel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Detect the Python ML frameworks.
if (OPEN_SPIEL_ENABLE_JAX STREQUAL AUTO)
  message(NOTICE "OPEN_SPIEL_ENABLE_JAX set to AUTO. Detecting Jax...")
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/find_jax.sh ${Python3_EXECUTABLE}
                  RESULT_VARIABLE JAX_RET_VAL
                  OUTPUT_VARIABLE JAX_OUT
                  ERROR_VARIABLE JAX_ERR)
  if (JAX_RET_VAL EQUAL 0)
    message(NOTICE "Found, version: ${JAX_OUT}")
    set(OPEN_SPIEL_ENABLE_JAX ON)
  else()
    message(NOTICE "Not found. Enable printing errors in python/CMakeLists.txt to see output.")
    set(OPEN_SPIEL_ENABLE_JAX OFF)
    # message(NOTICE "stdout: ${JAX_OUT}, stderr: ${JAX_ERR}")
  endif()
endif()

if (OPEN_SPIEL_ENABLE_PYTORCH STREQUAL AUTO)
  message(NOTICE "OPEN_SPIEL_ENABLE_PYTORCH set to AUTO. Detecting PyTorch...")
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/find_pytorch.sh ${Python3_EXECUTABLE}
                  RESULT_VARIABLE PYTORCH_RET_VAL
                  OUTPUT_VARIABLE PYTORCH_OUT
                  ERROR_VARIABLE PYTORCH_ERR)
  if (PYTORCH_RET_VAL EQUAL 0)
    message(NOTICE "Found, version: ${PYTORCH_OUT}")
    set(OPEN_SPIEL_ENABLE_PYTORCH ON)
  else()
    message(NOTICE "Not found. Enable printing errors in python/CMakeLists.txt to see output.")
    set(OPEN_SPIEL_ENABLE_PYTORCH OFF)
    # message(NOTICE "stdout: ${PYTORCH_OUT}, stderr: ${PYTORCH_ERR}")
  endif()
endif()

if (OPEN_SPIEL_BUILD_WITH_XINXIN)
  set(PYTHON_TESTS ${PYTHON_TESTS} ../bots/xinxin/xinxin_bot_test.py)
endif()
if (OPEN_SPIEL_BUILD_WITH_GAMUT)
  set(PYTHON_TESTS ${PYTHON_TESTS} ../games/gamut/gamut_test.py)
endif()

# Python tests to run. Start with all the core tests here first, then
# conditionally add other tests based on what has been enabled/detected.
set(PYTHON_TESTS ${PYTHON_TESTS}
  ../integration_tests/api_test.py
  ../integration_tests/playthrough_test.py
  algorithms/action_value_test.py
  algorithms/action_value_vs_best_response_test.py
  algorithms/async_mcts_test.py
  algorithms/best_response_test.py
  algorithms/boltzmann_tabular_qlearner_test.py
  algorithms/cfr_br_test.py
  algorithms/cfr_test.py
  algorithms/discounted_cfr_test.py
  algorithms/elo_test.py
  algorithms/efr_test.py
  algorithms/evaluate_bots_test.py
  algorithms/expected_game_score_test.py
  algorithms/external_sampling_mccfr_test.py
  algorithms/fictitious_play_test.py
  algorithms/gambit_test.py
  algorithms/generate_playthrough_test.py
  algorithms/get_all_states_test.py
  algorithms/ismcts_agent_test.py
  algorithms/infostate_tree_test.py
  algorithms/mcts_agent_test.py
  algorithms/mcts_test.py
  algorithms/minimax_solver_test.py
  algorithms/minimax_test.py
  algorithms/nfg_utils_test.py
  algorithms/noisy_policy_test.py
  algorithms/outcome_sampling_mccfr_test.py
  algorithms/policy_aggregator_joint_test.py
  algorithms/policy_aggregator_test.py
  algorithms/psro_v2/strategy_selectors_test.py
  algorithms/projected_replicator_dynamics_test.py
  algorithms/random_agent_test.py
  algorithms/regret_matching_test.py
  algorithms/tabular_qlearner_test.py
  algorithms/sequence_form_utils_test.py
  algorithms/wolf_phc_test.py
  algorithms/mmd_dilated_test.py
  coalitional_games/shapley_values_test.py
  bots/bluechip_bridge_test.py
  bots/bluechip_bridge_uncontested_bidding_test.py
  bots/is_mcts_test.py
  bots/uniform_random_test.py
  egt/dynamics_test.py
  egt/heuristic_payoff_table_test.py
  egt/utils_test.py
  environments/catch_test.py
  environments/cliff_walking_test.py
  games/ant_foraging_test.py
  games/block_dominoes_test.py
  games/chat_game_test.py
  games/chat_games/envs/base_envs/base_envs_test.py
  games/data_test.py
  games/dynamic_routing_test.py
  games/dynamic_routing_utils_test.py
  games/hangman_test.py
  games/heuristics_test.py
  games/liars_poker_test.py
  games/pokerkit_wrapper_test.py
  games/pursuit_evasion_test.py
  games/team_dominoes_test.py
  games/tic_tac_toe_test.py
  gui/api_test.py
  gui/catalogue_test.py
  gui/cli_test.py
  gui/contract_test.py
  gui/controller_settings_test.py
  gui/controllers_test.py
  gui/errors_test.py
  gui/fake_session_test.py
  gui/frame_checks_test.py
  gui/frame_test.py
  gui/games_smoke_test.py
  gui/gil_safety_test.py
  gui/health_report_test.py
  gui/health_test.py
  gui/information_test.py
  gui/loader_test.py
  gui/plies_test.py
  gui/request_test.py
  gui/schema_test.py
  gui/seats_test.py
  gui/server_test.py
  gui/session_history_test.py
  gui/session_test.py
  gui/static_test.py
  gui/store_test.py
  gui/sweep_all_test.py
  gui/sweep_census_test.py
  gui/sweep_exceptions_test.py
  gui/sweep_test.py
  gui/views_test.py
  gui/wire_test.py
  mfg/algorithms/best_response_value_test.py
  mfg/algorithms/mirror_descent_test.py
  mfg/algorithms/greedy_policy_test.py
  mfg/algorithms/nash_conv_test.py
  mfg/algorithms/policy_value_test.py
  mfg/games/crowd_modelling_test.py
  mfg/games/predator_prey_test.py
  mfg/games/dynamic_routing_test.py
  mfg/games/normal_form_game_test.py
  tests/mfg_implementation_test/mfg_test.py
  tests/bot_test.py
  tests/game_transforms_test.py
  tests/games_bargaining_test.py
  tests/games_blackjack_test.py
  tests/games_bridge_test.py
  tests/games_catch_test.py
  tests/games_chess_test.py
  tests/games_connect_four_test.py
  tests/games_crazy_eights_test.py
  tests/games_crazyhouse_test.py
  tests/games_crossword_test.py
  tests/games_euchre_test.py
  tests/games_gin_rummy_test.py
  tests/games_go_test.py
  tests/games_gomoku_test.py
  tests/games_hearts_test.py
  tests/games_repeated_poker_test.py
  tests/games_sim_test.py
  tests/games_tic_tac_toe_test.py
  tests/games_universal_poker_test.py
  tests/policy_test.py
  tests/pyspiel_test.py
  tests/rl_environment_test.py
  tests/sampled_stochastic_games_test.py
  tests/sgf_reader_test.py
  tests/tensor_game_utils_test.py
  tests/utils_trajectories_test.py
  utils/file_logger_test.py
  utils/lru_cache_test.py
  utils/spawn_test.py
  voting/approval_test.py
  voting/base_test.py
  voting/borda_test.py
  voting/copeland_test.py
  voting/elo_test.py
  voting/kemeny_young_test.py
  voting/plurality_test.py
  voting/ranked_pairs_test.py
  voting/schulze_test.py
  voting/soft_condorcet_optimization_test.py
  voting/stv_test.py
)

# Add Jax tests if it is enabled.
if (OPEN_SPIEL_ENABLE_JAX)
  set (PYTHON_TESTS ${PYTHON_TESTS}
    algorithms/alpha_zero/evaluator_test.py
    algorithms/alpha_zero/model_test.py
    jax/deep_cfr_jax_test.py
    jax/dqn_jax_test.py
    jax/nfsp_jax_test.py
    jax/boltzmann_dqn_jax_test.py
    jax/opponent_shaping_jax_test.py
    jax/policy_gradient_jax_test.py
    coalitional_games/least_core_lagrangian_test.py
    mfg/algorithms/fictitious_play_test.py
  )
endif()

# Add PyTorch tests if is enabled.
if (OPEN_SPIEL_ENABLE_PYTORCH)
  set(PYTHON_TESTS ${PYTHON_TESTS}
      # RCFR PyTorch test causing problems. Disabled for now.
      # pytorch/rcfr_pytorch_test.py
      pytorch/dqn_pytorch_test.py
      pytorch/boltzmann_dqn_pytorch_test.py
      pytorch/nfsp_pytorch_test.py
      pytorch/deep_cfr_pytorch_test.py
      pytorch/escher_test.py
      pytorch/losses/rl_losses_pytorch_test.py
      pytorch/policy_gradient_pytorch_test.py
      pytorch/ppo_pytorch_test.py
      pytorch/neurd_pytorch_test.py
  )
endif()

# Add miscellaneous Python tests if enabled.
# These require extra dependencies like cvxopt, nashpy, matplotlib, etc.
if (OPEN_SPIEL_ENABLE_PYTHON_MISC)
  set(PYTHON_TESTS ${PYTHON_TESTS}
      algorithms/adidas_test.py
      algorithms/double_oracle_test.py
      algorithms/jpsro_test.py
      algorithms/lp_solver_test.py
      algorithms/nash_averaging_test.py
      algorithms/mip_nash_test.py
      algorithms/psro_v2/best_response_oracle_test.py
      algorithms/response_graph_ucb_test.py
      algorithms/sequence_form_lp_test.py
      algorithms/stackelberg_lp_test.py
      algorithms/tabular_multiagent_qlearner.py
      algorithms/value_iteration_test.py
      examples/mcp_tool_server/tools_test.py
      coalitional_games/least_core_lp_test.py
      coalitional_games/wvg_test.py
      egt/alpharank_test.py
      egt/alpharank_visualizer_test.py
      egt/visualization_test.py
      games/kuhn_poker_test.py
      voting/maximal_lotteries_test.py
      tests/matrix_game_utils_test.py
  )
endif()

# Tests that are excluded when running via the installed wheel.
# Some bundle data not shipped with the wheel (e.g. playthroughs)
set(WHEEL_EXCLUDED_PYTHON_TESTS
  algorithms/response_graph_ucb_test.py
  games/tic_tac_toe_test.py
  ../integration_tests/playthrough_test.py)

# Prefer to use PYBIN as the executable if it's defined (e.g. on GitHub CI)
if(NOT DEFINED ENV{PYBIN})
  set(PYBIN ${Python3_EXECUTABLE})
else()
  set(PYBIN $ENV{PYBIN})
endif()

# Create a python test.
foreach(py_test_file IN LISTS PYTHON_TESTS)
  if (NOT (OPEN_SPIEL_BUILDING_WHEEL AND ${py_test_file} IN_LIST WHEEL_EXCLUDED_PYTHON_TESTS))
    add_test(NAME python/${py_test_file} COMMAND ${PYBIN} ${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file})

    # We need two elements in the python path: CURRENT_BINARY_DIR to pick up
    # pyspiel.so, and CURRENT_SOURCE_DIR for the Python source files. We use
    # CURRENT_SOURCE_DIR/../.. so that the Python module imports are of the form:
    #  from open_spiel.python import rl_environment.
    set_property(TEST python/${py_test_file}
        PROPERTY ENVIRONMENT
        PYTHONPATH=${OPEN_SPIEL_PYTHONPATH};
        TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
  endif()
endforeach(py_test_file)

# Additional tests (running examples as tests)
# We don't generate these automatically because we may want custom parameters.
if (OPEN_SPIEL_ENABLE_JAX AND NOT OPEN_SPIEL_BUILDING_WHEEL)
  add_test(NAME python_examples_bridge_supervised_learning
           COMMAND ${PYBIN}
           ${CMAKE_CURRENT_SOURCE_DIR}/examples/bridge_supervised_learning.py
           --iterations 10
           --eval_every 5
           --data_path ${CMAKE_CURRENT_SOURCE_DIR}/examples/data/bridge)
  set_property(TEST python_examples_bridge_supervised_learning
      PROPERTY ENVIRONMENT
      PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
