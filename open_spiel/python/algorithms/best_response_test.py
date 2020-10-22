# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for open_spiel.python.algorithms.best_response."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import get_all_states
import pyspiel


class BestResponseTest(parameterized.TestCase, absltest.TestCase):

  def test_best_response_is_a_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    test_policy = policy.UniformRandomPolicy(game)
    br = best_response.BestResponsePolicy(game, policy=test_policy, player_id=0)
    expected_policy = {
        "0": 1,  # Bet in case opponent folds when winning
        "1": 1,  # Bet in case opponent folds when winning
        "2": 0,  # Both equally good (we return the lowest action)
        # Some of these will never happen under the best-response policy,
        # but we have computed best-response actions anyway.
        "0pb": 0,  # Fold - we're losing
        "1pb": 1,  # Call - we're 50-50
        "2pb": 1,  # Call - we've won
    }
    self.assertEqual(
        expected_policy,
        {key: br.best_response_action(key) for key in expected_policy.keys()})

  @parameterized.parameters(["kuhn_poker", "leduc_poker"])
  def test_cpp_and_python_implementations_are_identical(self, game_name):
    game = pyspiel.load_game(game_name)

    python_policy = policy.UniformRandomPolicy(game)
    pyspiel_policy = pyspiel.UniformRandomPolicy(game)

    all_states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False,
        to_string=lambda s: s.information_state_string())

    for current_player in range(game.num_players()):
      python_br = best_response.BestResponsePolicy(game, current_player,
                                                   python_policy)
      cpp_br = pyspiel.TabularBestResponse(
          game, current_player, pyspiel_policy).get_best_response_policy()

      for state in all_states.values():
        if state.current_player() != current_player:
          continue

        # TODO(b/141737795): Decide what to do about this.
        self.assertEqual(
            python_br.action_probabilities(state), {
                a: prob
                for a, prob in cpp_br.action_probabilities(state).items()
                if prob != 0
            })

  @parameterized.parameters(("kuhn_poker", 2))
  def test_cpp_and_python_best_response_are_identical(self, game_name,
                                                      num_players):
    game = pyspiel.load_game(game_name,
                             {"players": pyspiel.GameParameter(num_players)})

    test_policy = policy.TabularPolicy(game)
    for i_player in range(num_players):
      best_resp_py_backend = best_response.BestResponsePolicy(
          game, i_player, test_policy)
      best_resp_cpp_backend = best_response.CPPBestResponsePolicy(
          game, i_player, test_policy)
      for state in best_resp_cpp_backend.all_states.values():
        if i_player == state.current_player():
          py_dict = best_resp_py_backend.action_probabilities(state)
          cpp_dict = best_resp_cpp_backend.action_probabilities(state)

          # We do check like this, because the actions associated to a 0. prob
          # do not necessarily appear
          for key, value in py_dict.items():
            self.assertEqual(value, cpp_dict.get(key, 0.))
          for key, value in cpp_dict.items():
            self.assertEqual(value, py_dict.get(key, 0.))

  @parameterized.parameters(("kuhn_poker", 2), ("kuhn_poker", 3))
  def test_cpp_and_python_value_are_identical(self, game_name, num_players):
    game = pyspiel.load_game(game_name,
                             {"players": pyspiel.GameParameter(num_players)})
    test_policy = policy.TabularPolicy(game)
    root_state = game.new_initial_state()
    for i_player in range(num_players):
      best_resp_py_backend = best_response.BestResponsePolicy(
          game, i_player, test_policy)
      best_resp_cpp_backend = best_response.CPPBestResponsePolicy(
          game, i_player, test_policy)

      value_py_backend = best_resp_py_backend.value(root_state)
      value_cpp_backend = best_resp_cpp_backend.value(root_state)

      self.assertTrue(np.allclose(value_py_backend, value_cpp_backend))


if __name__ == "__main__":
  absltest.main()
