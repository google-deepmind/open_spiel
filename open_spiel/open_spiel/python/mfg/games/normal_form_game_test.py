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

# Lint as python3
"""Tests for Python Crowd Modelling game."""

from absl.testing import absltest
from open_spiel.python.mfg.games import normal_form_game
import pyspiel

MFG_STR_CONST = "_a"


class MFGNormalFormGameTest(absltest.TestCase):

  def test_load(self):
    game = pyspiel.load_game("mean_field_nfg")
    game.new_initial_state()

  def test_create(self):
    """Checks we can create the game and clone states."""
    game = normal_form_game.MFGNormalFormGame()
    self.assertEqual(
        game.get_type().dynamics, pyspiel.GameType.Dynamics.MEAN_FIELD
    )
    print("Num distinct actions:", game.num_distinct_actions())
    state = game.new_initial_state()
    clone = state.clone()
    print("Initial state:", state)
    print("Cloned initial state:", clone)

  def test_create_with_params(self):
    game = pyspiel.load_game("mean_field_nfg(num_actions=10)")
    self.assertEqual(game.num_actions, 10)

  def test_reward(self):
    game = normal_form_game.MFGNormalFormGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.DEFAULT_PLAYER_ID)

    state.apply_action(0)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.MEAN_FIELD)
    state.update_distribution([1.0, 0.0, 0.0])
    self.assertAlmostEqual(state.rewards()[0], 10.0)
    self.assertAlmostEqual(state.returns()[0], 10.0)

    state = game.new_initial_state()
    state.apply_action(0)
    state.update_distribution([0.0, 1.0, 0.0])
    self.assertAlmostEqual(state.rewards()[0], -20.0)
    self.assertAlmostEqual(state.returns()[0], -20.0)

    self.assertTrue(state.is_terminal())


if __name__ == "__main__":
  absltest.main()
