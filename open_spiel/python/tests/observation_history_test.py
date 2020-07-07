# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

"""Tests for OpenSpiel observation histories."""

from absl.testing import absltest

import pyspiel


class ObservationHistoryTest(absltest.TestCase):

  def test_kuhn_rollout(self):
    game = pyspiel.load_game("kuhn_poker")

    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    self.assertEqual(state.public_observation_history(), ["start game"])
    self.assertEqual(state.action_observation_history(0), ["start game"])
    self.assertEqual(state.action_observation_history(1), ["start game"])

    state.apply_action(2)
    self.assertTrue(state.is_chance_node())
    self.assertEqual(state.public_observation_history(),
                     ["start game", "Deal to player 0"])
    self.assertEqual(state.action_observation_history(0),
                     ["start game", "Received card 2"])
    self.assertEqual(state.action_observation_history(1),
                     ["start game", "Deal to player 0"])

    state.apply_action(1)
    self.assertTrue(state.is_player_node())
    self.assertEqual(state.public_observation_history(),
                     ["start game", "Deal to player 0", "Deal to player 1"])
    self.assertEqual(state.action_observation_history(0),
                     ["start game", "Received card 2", "Deal to player 1"])
    self.assertEqual(state.action_observation_history(1),
                     ["start game", "Deal to player 0", "Received card 1"])

    state.apply_action(0)
    self.assertTrue(state.is_player_node())
    self.assertEqual(state.public_observation_history(),
                     ["start game", "Deal to player 0", "Deal to player 1",
                      "Pass"])
    self.assertEqual(state.action_observation_history(0),
                     ["start game", "Received card 2", "Deal to player 1",
                      0, "Pass"])
    self.assertEqual(state.action_observation_history(1),
                     ["start game", "Deal to player 0", "Received card 1",
                      "Pass"])


    state.apply_action(1)
    self.assertTrue(state.is_player_node())
    self.assertEqual(state.public_observation_history(),
                     ["start game", "Deal to player 0", "Deal to player 1",
                      "Pass", "Bet"])
    self.assertEqual(state.action_observation_history(0),
                     ["start game", "Received card 2", "Deal to player 1",
                      0, "Pass", "Bet"])
    self.assertEqual(state.action_observation_history(1),
                     ["start game", "Deal to player 0", "Received card 1",
                      "Pass", 1, "Bet"])

    state.apply_action(1)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.public_observation_history(),
                     ["start game", "Deal to player 0", "Deal to player 1",
                      "Pass", "Bet", "Bet"])
    self.assertEqual(state.action_observation_history(0),
                     ["start game", "Received card 2", "Deal to player 1",
                      0, "Pass", "Bet", 1, "Bet"])
    self.assertEqual(state.action_observation_history(1),
                     ["start game", "Deal to player 0", "Received card 1",
                      "Pass", 1, "Bet", "Bet"])


if __name__ == "__main__":
  absltest.main()
