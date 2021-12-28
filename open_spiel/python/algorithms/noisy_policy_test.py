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
"""Tests for open_spiel.python.algorithms.noisy_policy."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import noisy_policy
import pyspiel


class NoisyPolicyTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.parameters(["kuhn_poker", "leduc_poker"])
  def test_cpp_and_python_implementations_are_identical(self, game_name):
    game = pyspiel.load_game(game_name)

    policy = openspiel_policy.UniformRandomPolicy(game)

    all_states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False,
        to_string=lambda s: s.information_state_string())

    for current_player in range(game.num_players()):
      noise = noisy_policy.NoisyPolicy(
          policy, player_id=current_player, alpha=0.5, beta=10.)
      for state in all_states.values():
        if state.current_player() < 0:
          continue

        if state.current_player() != current_player:
          self.assertEqual(
              policy.action_probabilities(state),
              noise.action_probabilities(state))
        else:
          self.assertNotEqual(
              policy.action_probabilities(state),
              noise.action_probabilities(state))

  @parameterized.parameters(["python_iterated_prisoners_dilemma"])
  def test_simultaneous_game_noisy_policy(self, game_name):
    game = pyspiel.load_game(game_name)

    policy = openspiel_policy.UniformRandomPolicy(game)

    all_states = get_all_states.get_all_states(
        game,
        depth_limit=10,
        include_terminals=False,
        include_chance_states=False,
        to_string=lambda s: s.history_str())

    for current_player in range(game.num_players()):
      noise = noisy_policy.NoisyPolicy(
          policy, player_id=current_player, alpha=0.5, beta=10.)
      for state in all_states.values():
        if state.current_player() == pyspiel.PlayerId.SIMULTANEOUS:
          for player_id in range(game.num_players()):
            if player_id != current_player:
              self.assertEqual(
                  policy.action_probabilities(state, player_id),
                  noise.action_probabilities(state, player_id))
            else:
              self.assertNotEqual(
                  policy.action_probabilities(state, player_id),
                  noise.action_probabilities(state, player_id))


if __name__ == "__main__":
  absltest.main()
