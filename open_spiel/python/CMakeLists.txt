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
  set(OPEN_SPIEL_PYTHONPATH ${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;)
endif()

message(NOTICE "Python executable: ${Python3_EXECUTABLE}")
message(NOTICE "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(NOTICE "Python library dirs: ${Python3_LIBRARY_DIRS}")
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})

if(Python3_VERSION VERSION_LESS "3.6.0")
  message(FATAL_ERROR
    "Python found ${Python3_VERSION} < 3.6.0")
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

if (OPEN_SPIEL_ENABLE_TENSORFLOW STREQUAL AUTO)
  message(NOTICE "OPEN_SPIEL_ENABLE_TENSORFLOW set to AUTO. Detecting Tensorflow...")
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/find_tensorflow.sh ${Python3_EXECUTABLE}
                  RESULT_VARIABLE TENSORFLOW_RET_VAL
                  OUTPUT_VARIABLE TENSORFLOW_OUT
                  ERROR_VARIABLE TENSORFLOW_ERR)
  if (TENSORFLOW_RET_VAL EQUAL 0)
    message(NOTICE "Found, version: ${TENSORFLOW_OUT}")
    set(OPEN_SPIEL_ENABLE_TENSORFLOW ON)
  else()
    message(NOTICE "Not found. Enable printing errors in python/CMakeLists.txt to see output.")
    set(OPEN_SPIEL_ENABLE_TENSORFLOW OFF)
    # message(NOTICE "stdout: ${TENSORFLOW_OUT}, stderr: ${TENSORFLOW_ERR}")
  endif()
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
  pybind11/games_chess.cc
  pybind11/games_chess.h
  pybind11/games_kuhn_poker.cc
  pybind11/games_kuhn_poker.h
  pybind11/games_leduc_poker.cc
  pybind11/games_leduc_poker.h
  pybind11/games_negotiation.cc
  pybind11/games_negotiation.h
  pybind11/games_tarok.cc
  pybind11/games_tarok.h
  pybind11/game_transforms.cc
  pybind11/game_transforms.h
  pybind11/observer.cc
  pybind11/observer.h
  pybind11/policy.cc
  pybind11/policy.h
  pybind11/pybind11.h
  pybind11/pyspiel.cc
  pybind11/python_games.cc
  pybind11/python_games.h
  )

if (OPEN_SPIEL_BUILD_WITH_HIGC)
  set(PYTHON_BINDINGS ${PYTHON_BINDINGS}
      pybind11/referee.cc
      pybind11/referee.h
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

add_library(pyspiel MODULE ${PYTHON_BINDINGS} ${OPEN_SPIEL_OBJECTS})
target_link_directories(pyspiel PUBLIC ${Python3_LIBRARY_DIRS})

# Without this, the binary is called `libpyspiel.so`
set_target_properties(pyspiel PROPERTIES PREFIX "")
if (WIN32)
  # Python does not seem able to import the module without this change.
  set_target_properties(pyspiel PROPERTIES SUFFIX ".pyd")
endif()

# Optional pyspiel-related modules, which can specify their own python tests.
if (OPEN_SPIEL_BUILD_WITH_EIGEN)
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
  algorithms/best_response_test.py
  algorithms/cfr_br_test.py
  algorithms/cfr_test.py
  algorithms/evaluate_bots_test.py
  algorithms/expected_game_score_test.py
  algorithms/external_sampling_mccfr_test.py
  algorithms/fictitious_play_test.py
  algorithms/gambit_test.py
  algorithms/generate_playthrough_test.py
  algorithms/get_all_states_test.py
  algorithms/mcts_agent_test.py
  algorithms/mcts_test.py
  algorithms/minimax_test.py
  algorithms/noisy_policy_test.py
  algorithms/outcome_sampling_mccfr_test.py
  algorithms/policy_aggregator_joint_test.py
  algorithms/policy_aggregator_test.py
  algorithms/projected_replicator_dynamics_test.py
  algorithms/random_agent_test.py
  bots/bluechip_bridge_test.py
  bots/bluechip_bridge_uncontested_bidding_test.py
  bots/is_mcts_test.py
  bots/uniform_random_test.py
  egt/dynamics_test.py
  egt/heuristic_payoff_table_test.py
  egt/utils_test.py
  environments/catch_test.py
  environments/cliff_walking_test.py
  games/data_test.py
  games/dynamic_routing_test.py
  games/dynamic_routing_utils_test.py
  games/tic_tac_toe_test.py
  mfg/algorithms/best_response_value_test.py
  mfg/algorithms/mirror_descent_test.py
  mfg/algorithms/greedy_policy_test.py
  mfg/algorithms/nash_conv_test.py
  mfg/algorithms/policy_value_test.py
  mfg/games/crowd_modelling_test.py
  mfg/games/predator_prey_test.py
  mfg/games/dynamic_routing_test.py
  tests/mfg_implementation_test/mfg_test.py
  tests/bot_test.py
  tests/games_bridge_test.py
  tests/games_sim_test.py
  tests/policy_test.py
  tests/pyspiel_test.py
  tests/rl_environment_test.py
  tests/sampled_stochastic_games_test.py
  tests/tensor_game_utils_test.py
  utils/file_logger_test.py
  utils/lru_cache_test.py
  utils/spawn_test.py
)

# Add Jax tests if it is enabled.
if (OPEN_SPIEL_ENABLE_JAX)
  # Only current JAX test is the bridge supervised learning example below.
  set (PYTHON_TESTS ${PYTHON_TESTS}
    jax/deep_cfr_jax_test.py
    jax/dqn_jax_test.py
    jax/nfsp_jax_test.py
    mfg/algorithms/fictitious_play_test.py
  )
endif()

# Add PyTorch tests if is enabled.
if (OPEN_SPIEL_ENABLE_PYTORCH)
  set(PYTHON_TESTS ${PYTHON_TESTS}
      pytorch/rcfr_pytorch_test.py
      pytorch/dqn_pytorch_test.py
      pytorch/nfsp_pytorch_test.py
      pytorch/deep_cfr_pytorch_test.py
      pytorch/eva_pytorch_test.py
      pytorch/losses/rl_losses_pytorch_test.py
      pytorch/policy_gradient_pytorch_test.py
  )
endif()

# Add Tensorflow tests if is enabled.
if (OPEN_SPIEL_ENABLE_TENSORFLOW)
  set(PYTHON_TESTS ${PYTHON_TESTS}
      algorithms/alpha_zero/evaluator_test.py
      algorithms/alpha_zero/model_test.py
      algorithms/deep_cfr_test.py
      algorithms/deep_cfr_tf2_test.py
      algorithms/discounted_cfr_test.py
      algorithms/dqn_test.py
      algorithms/eva_test.py
      algorithms/exploitability_descent_test.py
      algorithms/exploitability_test.py
      algorithms/losses/rl_losses_test.py
      algorithms/masked_softmax_test.py
      algorithms/neurd_test.py
      algorithms/nfsp_test.py
      algorithms/policy_gradient_test.py
      algorithms/psro_v2/strategy_selectors_test.py
      algorithms/rcfr_test.py
  )
  if (OPEN_SPIEL_ENABLE_PYTHON_MISC)
    set(PYTHON_TESTS ${PYTHON_TESTS}
        algorithms/psro_v2/best_response_oracle_test.py
    )
  endif()
endif()

# Add miscellaneous Python tests if enabled.
# These require extra dependencies like cvxopt, nashpy, or matplotlib
if (OPEN_SPIEL_ENABLE_PYTHON_MISC)
  set(PYTHON_TESTS ${PYTHON_TESTS}
      algorithms/adidas_test.py
      algorithms/double_oracle_test.py
      algorithms/jpsro_test.py
      algorithms/lp_solver_test.py
      algorithms/response_graph_ucb_test.py
      algorithms/sequence_form_lp_test.py
      algorithms/value_iteration_test.py
      egt/alpharank_test.py
      egt/alpharank_visualizer_test.py
      egt/visualization_test.py
      games/kuhn_poker_test.py
      tests/matrix_game_utils_test.py
  )
endif()

# Tests that are excluded when running via the installed wheel.
# Some bundle data not shipped with the wheel (e.g. playthroughs)
set(WHEEL_EXCLUDED_PYTHON_TESTS
  algorithms/response_graph_ucb_test.py
  games/tic_tac_toe_test.py
  ../integration_tests/playthrough_test.py)

# Create a python test.
foreach(py_test_file IN LISTS PYTHON_TESTS)
  if (NOT (OPEN_SPIEL_BUILDING_WHEEL AND ${py_test_file} IN_LIST WHEEL_EXCLUDED_PYTHON_TESTS))
    add_test(NAME python/${py_test_file} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${py_test_file})

    # We need two elements in the python path: CURRENT_BINARY_DIR to pick up
    # pyspiel.so, and CURRENT_SOURCE_DIR for the Python source files. We use
    # CURRENT_SOURCE_DIR/../.. so that the Python module imports are of the form:
    #  from open_spiel.python import rl_environment.
    set_property(TEST python/${py_test_file}
        PROPERTY ENVIRONMENT
        PYTHONPATH=${OPEN_SPIEL_PYTHONPATH}
        TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
  endif()
endforeach(py_test_file)

# Additional tests (running examples as tests)
# We don't generate these automatically because we may want custom parameters.
if (OPEN_SPIEL_ENABLE_JAX AND NOT OPEN_SPIEL_BUILDING_WHEEL)
  add_test(NAME python_examples_bridge_supervised_learning
           COMMAND ${Python3_EXECUTABLE}
           ${CMAKE_CURRENT_SOURCE_DIR}/examples/bridge_supervised_learning.py
           --iterations 10
           --eval_every 5
           --data_path ${CMAKE_CURRENT_SOURCE_DIR}/examples/data/bridge)
  set_property(TEST python_examples_bridge_supervised_learning
      PROPERTY ENVIRONMENT
      PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../..;
      TEST_SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
