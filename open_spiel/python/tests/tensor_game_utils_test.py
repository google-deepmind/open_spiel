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

"""Tests the C++ matrix game utility methods exposed to Python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import pyspiel


class TensorGamesUtilsTest(absltest.TestCase):

  def test_extensive_to_tensor_game_type(self):
    game = pyspiel.extensive_to_tensor_game(
        pyspiel.load_game(
            "turn_based_simultaneous_game(game=blotto(players=3,coins=5))"))
    game_type = game.get_type()
    self.assertEqual(game_type.dynamics, pyspiel.GameType.Dynamics.SIMULTANEOUS)
    self.assertEqual(game_type.chance_mode,
                     pyspiel.GameType.ChanceMode.DETERMINISTIC)
    self.assertEqual(game_type.information,
                     pyspiel.GameType.Information.ONE_SHOT)
    self.assertEqual(game_type.utility, pyspiel.GameType.Utility.ZERO_SUM)

  def test_extensive_to_tensor_game_payoff_tensor(self):
    turn_based_game = pyspiel.load_game_as_turn_based(
        "blotto(players=3,coins=5)")
    tensor_game1 = pyspiel.extensive_to_tensor_game(turn_based_game)
    tensor_game2 = pyspiel.load_tensor_game("blotto(players=3,coins=5)")
    self.assertEqual(tensor_game1.shape(), tensor_game2.shape())
    s0 = turn_based_game.new_initial_state()
    self.assertEqual(tensor_game1.shape()[0], s0.num_distinct_actions())
    for a0 in range(s0.num_distinct_actions()):
      s1 = s0.child(a0)
      self.assertEqual(tensor_game1.shape()[1], s1.num_distinct_actions())
      for a1 in range(s1.num_distinct_actions()):
        s2 = s1.child(a1)
        self.assertEqual(tensor_game1.shape()[2], s2.num_distinct_actions())
        for a2 in range(s2.num_distinct_actions()):
          s3 = s2.child(a2)
          self.assertTrue(s3.is_terminal())
          for player in range(3):
            self.assertEqual(
                s3.returns()[player],
                tensor_game1.player_utility(player, (a0, a1, a2)))
            self.assertEqual(
                s3.returns()[player],
                tensor_game2.player_utility(player, (a0, a1, a2)))


if __name__ == "__main__":
  absltest.main()
