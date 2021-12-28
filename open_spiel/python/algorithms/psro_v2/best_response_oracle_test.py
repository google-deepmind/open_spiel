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

# Lint as: python3
"""Tests for open_spiel.python.algorithms.psro_v2.best_response_oracle."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
import pyspiel


class BestResponseOracleTest(parameterized.TestCase, absltest.TestCase):

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
