# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.algorithms.psro_v2.best_response_oracle."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
import pyspiel


class _ConstantPolicy(policy.Policy):

  def __init__(self, game, action):
    super().__init__(game, range(game.num_players()))
    self.action = action

  def action_probabilities(self, state, player_id=None):
    return {self.action: 1.0}


class BestResponseOracleTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      ("python_sequential", "py", True),
      ("cpp_sequential", "cpp", True),
      ("python_simultaneous", "py", False))
  def test_joint_best_response_preserves_opponent_correlation(
      self, backend, turn_based):
    # Player 0 prefers action 0 when the opponents coordinate, but action 1
    # when they independently choose between their two policies.
    payoff = np.array([[[1., 0.], [0., 1.]],
                       [[0.75, 0.75], [0.75, 0.75]]])
    game = pyspiel.create_tensor_game([payoff, np.zeros_like(payoff),
                                       np.zeros_like(payoff)])
    if turn_based:
      game = pyspiel.convert_to_turn_based(game)

    zero, one = _ConstantPolicy(game, 0), _ConstantPolicy(game, 1)
    populations = [[zero], [zero, one], [zero, one]]
    oracle = best_response_oracle.BestResponseOracle(
        game=game, best_response_backend=backend)

    for weights, expected_action in (
        (np.array([[[0.5, 0.0], [0.0, 0.5]]]), 0),
        (np.full((1, 2, 2), 0.25), 1)):
      params = [[{
          "total_policies": populations,
          "current_player": 0,
          "probabilities_of_playing_policies": weights,
      }], [], []]
      response = oracle(game, params, using_joint_strategies=True)[0][0]
      action_probabilities = response.action_probabilities(
          game.new_initial_state(), 0)
      self.assertEqual(action_probabilities.get(expected_action), 1.0)
      self.assertEqual(action_probabilities.get(1 - expected_action, 0.0), 0.0)
      if expected_action == 0:
        self.assertIsInstance(response,
                              best_response.JointMixtureBestResponsePolicy)
        self.assertAlmostEqual(response.value(), 1.0)
      elif backend == "cpp":
        self.assertIsInstance(response, best_response.CPPBestResponsePolicy)

  @parameterized.parameters("py", "cpp")
  def test_joint_best_response_on_three_player_kuhn_poker(self, backend):
    game = pyspiel.load_game("kuhn_poker(players=3)")

    zero, one = _ConstantPolicy(game, 0), _ConstantPolicy(game, 1)
    profiles = [[zero, zero, zero], [zero, one, one]]
    params = [[{
        "total_policies": [[zero], [zero, one], [zero, one]],
        "current_player": 0,
        "probabilities_of_playing_policies": np.array(
            [[[0.5, 0.0], [0.0, 0.5]]]),
    }], [], []]
    oracle = best_response_oracle.BestResponseOracle(
        game=game, best_response_backend=backend)
    response = oracle(game, params, using_joint_strategies=True)[0][0]

    profile_values = [expected_game_score.policy_value(
        game.new_initial_state(), [response, profile[1], profile[2]])[0]
                      for profile in profiles]
    self.assertAlmostEqual(np.mean(profile_values), 1.0)
    self.assertAlmostEqual(response.value(), 1.0)

  @parameterized.parameters(("kuhn_poker", 2), ("kuhn_poker", 3),
                            ("leduc_poker", 2))
  def test_cpp_python_best_response_oracle(self, game_name, num_players):
    # Tests that these best responses interface well with Best Response Oracle
    game = pyspiel.load_game(game_name, {"players": num_players})
    all_states, _ = best_response.compute_states_and_info_states_if_none(
        game, all_states=None, state_to_information_state=None)

    current_best = [
        [policy.TabularPolicy(game).__copy__()] for _ in range(num_players)
    ]
    probabilities_of_playing_policies = [[1.] for _ in range(num_players)]

    # Construct the python oracle
    py_oracle = best_response_oracle.BestResponseOracle(
        best_response_backend="py")

    # Construct the cpp oracle. Note that in this regime, BestResponseOracle
    # uses base_policy to construct and cache TabularBestResponse internally.
    cpp_oracle = best_response_oracle.BestResponseOracle(
        game=game, best_response_backend="cpp")

    # Prepare the computation of the best responses with each backend
    # pylint:disable=g-complex-comprehension
    training_params = [[{
        "total_policies": current_best,
        "current_player": i,
        "probabilities_of_playing_policies": probabilities_of_playing_policies
    }] for i in range(num_players)]
    # pylint:enable=g-complex-comprehension

    py_best_rep = py_oracle(game, training_params)

    cpp_best_rep = cpp_oracle(game, training_params)

    # Compare the policies
    for state in all_states.values():
      i_player = state.current_player()
      py_dict = py_best_rep[i_player][0].action_probabilities(state)
      cpp_dict = cpp_best_rep[i_player][0].action_probabilities(state)

      for action in py_dict.keys():
        self.assertEqual(py_dict.get(action, 0.0), cpp_dict.get(action, 0.0))
      for action in cpp_dict.keys():
        self.assertEqual(py_dict.get(action, 0.0), cpp_dict.get(action, 0.0))


if __name__ == "__main__":
  absltest.main()
