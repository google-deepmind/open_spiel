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

"""Tests for best_response_value."""

from absl.testing import absltest

from open_spiel.python import policy
import open_spiel.python.games
import pyspiel


class PolicyTest(absltest.TestCase):

  def test_get_action_probabilities_list_simultaneous_node(self):
    """Checks if the value of a policy computation works."""
    game = pyspiel.load_game(
      "python_iterated_prisoners_dilemma(max_game_length=6)")
    uniform_policy = policy.UniformRandomPolicy(game)
    actions_list, probability_list = zip(*policy
      .get_action_probabilities_list_simultaneous_node(
        game.new_initial_state(), uniform_policy))
    self.assertEqual(actions_list, [[0, 0], [0, 1], [1, 0], [1, 1]])
    self.assertEqual(probability_list, [0.25, 0.25, 0.25, 0.25])


if __name__ == "__main__":
  absltest.main()
