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

"""Tests for open_spiel.python.algorithms.sample_some_states."""

from absl.testing import absltest

from open_spiel.python.algorithms import sample_some_states
import pyspiel


class SampleSomeStatesTest(absltest.TestCase):

  def test_simple_games_with_one_rollout(self):
    game = pyspiel.load_game_as_turn_based("matrix_mp")
    states = sample_some_states.sample_some_states(
        game, max_rollouts=1)
    self.assertLen(states, 3)

    states = sample_some_states.sample_some_states(
        game, max_rollouts=1, include_terminals=False)
    self.assertLen(states, 2)

    states = sample_some_states.sample_some_states(
        game, max_rollouts=1, depth_limit=0)
    self.assertLen(states, 1)

    states = sample_some_states.sample_some_states(
        game, max_rollouts=1, depth_limit=1)
    self.assertLen(states, 2)

    game = pyspiel.load_game_as_turn_based("coordinated_mp")
    states = sample_some_states.sample_some_states(
        game, max_rollouts=1)
    self.assertLen(states, 4)

    states = sample_some_states.sample_some_states(
        game, max_rollouts=1, include_chance_states=False)
    self.assertLen(states, 3)


if __name__ == "__main__":
  absltest.main()
