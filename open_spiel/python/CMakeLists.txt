find_package(Python3 COMPONENTS Interpreter Development)
message(NOTICE "Python include dirs: ${Python3_INCLUDE_DIRS}")
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})

if(Python3_VERSION VERSION_LESS "3.6.0")
  message(FATAL_ERROR
    "Python found ${Python3_VERSION} < 3.6.0")
endif()

# List of all Python bindings to add to pyspiel.
set(PYTHON_BINDINGS ${PYTHON_BINDINGS}
  pybind11/algorithms_corr_dist.cc
  pybind11/algorithms_corr_dist.h
  pybind11/algorithms_trajectories.cc
  pybind11/algorithms_trajectories.h
  pybind11/bots.cc
  pybind11/bots.h
  pybind11/games_backgammon.cc
  pybind11/games_backgammon.h
  pybind11/games_bridge.cc
  pybind11/games_bridge.h
  pybind11/games_negotiation.cc
  pybind11/games_negotiation.h
  pybind11/games_tarok.cc
  pybind11/games_tarok.h
  pybind11/game_transforms.cc
  pybind11/game_transforms.h
  pybind11/observation_history.cc
  pybind11/observation_history.h
  pybind11/observer.cc
  pybind11/observer.h
  pybind11/policy.cc
  pybind11/policy.h
  pybind11/pyspiel.cc
  pybind11/python_games.cc
  pybind11/python_games.h
  )

# Optional pyspiel sub-modules, which can specify their python bindings.
if (BUILD_WITH_GAMUT)
  set (PYTHON_BINDINGS ${PYTHON_BINDINGS}
      ../games/gamut/gamut_pybind11.h
      ../games/gamut/gamut_pybind11.cc
      )
endif()
if (BUILD_WITH_PUBLIC_STATES)
  set (PYTHON_BINDINGS ${PYTHON_BINDINGS}
      ../public_states/pybind11/public_states.h
      ../public_states/pybind11/public_states.cc
      )
endif()
if (BUILD_WITH_XINXIN)
  set (PYTHON_BINDINGS ${PYTHON_BINDINGS}
      ../bots/xinxin/xinxin_pybind11.h
      ../bots/xinxin/xinxin_pybind11.cc
      )
endif()

add_library(pyspiel MODULE ${PYTHON_BINDINGS} ${OPEN_SPIEL_OBJECTS})

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")

# Optional pyspiel-related modules, which can specify their own python tests.
if (BUILD_WITH_EIGEN)
  add_library(pyspiel_eigen_test MODULE
    ../eigen/eigen_test_support.h
    ../eigen/pyeig.h
    ../eigen/pyspiel_eigen_test.cc
    ${OPEN_SPIEL_OBJECTS})

  # Without this, the binary is called `libpyspiel_eigen_test.so`
  set_target_properties(pyspiel_eigen_test PROPERTIES PREFIX "")

  set(PYTHON_TESTS ${PYTHON_TESTS}
    ../eigen/eigen_binding_test.py)
endif()
if (BUILD_WITH_PUBLIC_STATES)
  set(PYTHON_TESTS ${PYTHON_TESTS}
    ../integration_tests/public_states_api_test.py)
endif()
if (BUILD_WITH_XINXIN)
  set(PYTHON_TESTS ${PYTHON_TESTS} ../bots/xinxin/xinxin_bot_test.py)
endif()
if (BUILD_WITH_GAMUT)
  set(PYTHON_TESTS ${PYTHON_TESTS} ../games/gamut/gamut_test.py)
endif()


# List of all Python tests to run.
set(PYTHON_TESTS ${PYTHON_TESTS}
  ../integration_tests/api_test.py
  ../integration_tests/playthrough_test.py
  algorithms/action_value_test.py
  algorithms/action_value_vs_best_response_test.py
  algorithms/alpha_zero/evaluator_test.py
  algorithms/alpha_zero/model_test.py
  algorithms/best_response_test.py
  algorithms/cfr_br_test.py
  algorithms/cfr_test.py
  algorithms/deep_cfr_test.py
  algorithms/deep_cfr_tf2_test.py
  algorithms/discounted_cfr_test.py
  algorithms/double_oracle_test.py
  algorithms/dqn_test.py
  algorithms/eva_test.py
  algorithms/evaluate_bots_test.py
  algorithms/expected_game_score_test.py
  algorithms/exploitability_descent_test.py
  algorithms/exploitability_test.py
  algorithms/external_sampling_mccfr_test.py
  algorithms/fictitious_play_test.py
  algorithms/gambit_test.py
  algorithms/generate_playthrough_test.py
  algorithms/get_all_states_test.py
  algorithms/losses/rl_losses_test.py
  algorithms/lp_solver_test.py
  algorithms/masked_softmax_test.py
  algorithms/mcts_test.py
  algorithms/minimax_test.py
  algorithms/neurd_test.py
  algorithms/nfsp_test.py
  algorithms/noisy_policy_test.py
  algorithms/outcome_sampling_mccfr_test.py
  algorithms/policy_aggregator_joint_test.py
  algorithms/policy_aggregator_test.py
  algorithms/policy_gradient_test.py
  algorithms/projected_replicator_dynamics_test.py
  algorithms/psro_v2/best_response_oracle_test.py
  algorithms/psro_v2/strategy_selectors_test.py
  algorithms/random_agent_test.py
  algorithms/rcfr_test.py
  algorithms/sequence_form_lp_test.py
  algorithms/value_iteration_test.py
  bots/bluechip_bridge_test.py
  bots/bluechip_bridge_uncontested_bidding_test.py
  bots/is_mcts_test.py
  bots/uniform_random_test.py
  egt/alpharank_test.py
  egt/alpharank_visualizer_test.py
  egt/dynamics_test.py
  egt/heuristic_payoff_table_test.py
  egt/utils_test.py
  egt/visualization_test.py
  environments/catch_test.py
  environments/cliff_walking_test.py
  games/data_test.py
  games/kuhn_poker_test.py
  games/tic_tac_toe_test.py
  tests/bot_test.py
  tests/games_bridge_test.py
  tests/games_sim_test.py
  tests/matrix_game_utils_test.py
  tests/observation_history_test.py
  tests/policy_test.py
  tests/pyspiel_test.py
  tests/rl_environment_test.py
  tests/sampled_stochastic_games_test.py
  tests/tensor_game_utils_test.py
  utils/file_logger_test.py
  utils/lru_cache_test.py
  utils/spawn_test.py
  pytorch/rcfr_pytorch_test.py
  pytorch/dqn_pytorch_test.py
  pytorch/nfsp_pytorch_test.py
  pytorch/deep_cfr_pytorch_test.py
  pytorch/eva_pytorch_test.py
  pytorch/losses/rl_losses_pytorch_test.py
  pytorch/policy_gradient_pytorch_test.py
)


# Create a python test.
foreach(py_test_file IN LISTS PYTHON_TESTS)
  get_filename_component(py_test ${py_test_file} NAME_WE)
  add_test(NAME python_${py_test} COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file})

  # We need two elements in the python path: CURRENT_BINARY_DIR to pick up
  # pyspiel.so, and CURRENT_SOURCE_DIR for the Python source files. We use
  # CURRENT_SOURCE_DIR/../.. so that the Python module imports are of the form:
  #  from open_spiel.python import rl_environment.
  set_property(TEST python_${py_test}
      PROPERTY ENVIRONMENT
      PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endforeach(py_test_file)

# Additional tests (running examples as tests)
# We don't generate these automatically because we may want custom parameters.

add_test(NAME python_examples_bridge_supervised_learning
         COMMAND python3
         ${CMAKE_CURRENT_SOURCE_DIR}/examples/bridge_supervised_learning.py
         --iterations 10
         --eval_every 5
         --data_path ${CMAKE_CURRENT_SOURCE_DIR}/examples/data/bridge)
set_property(TEST python_examples_bridge_supervised_learning
    PROPERTY ENVIRONMENT
    PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
    TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
