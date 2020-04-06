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
"""Tests for open_spiel.python.algorithms.noisy_policy."""

from absl.testing import absltest
from absl.testing import parameterized

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
      noise = noisy_policy.NoisyPolicy(policy, 0, alpha=0.5, beta=10.)
      for state in all_states.values():
        if state.current_player() != current_player:
          continue

        # TODO(b/141737795): Decide what to do about this.
        self.assertNotEqual(
            policy.action_probabilities(state),
            noise.action_probabilities(state))

if __name__ == "__main__":
  absltest.main()
