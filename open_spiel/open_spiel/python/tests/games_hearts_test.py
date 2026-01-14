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

"""Tests for Python Hearts game."""

import json
from absl.testing import absltest
import pyspiel


class HeartsGameTest(absltest.TestCase):

  def test_structs(self):
    game = pyspiel.load_game('hearts')
    state = game.new_initial_state()
    # Apply some actions to get a non-initial state.
    initial_actions = [
        1,  # Pass left
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  # deal p0
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  # deal p1
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,  # deal p2
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  # deal p3
    ]
    for action in initial_actions:
      state.apply_action(action)

    # state should be in Pass phase now.
    # p0 passes 3 cards: 0, 1, 2
    state.apply_action(0)
    state.apply_action(1)
    state.apply_action(2)
    # p1 passes 3 cards: 13, 14, 15
    state.apply_action(13)
    state.apply_action(14)
    state.apply_action(15)
    # p2 passes 3 cards: 26, 27, 28
    state.apply_action(26)
    state.apply_action(27)
    state.apply_action(28)
    # p3 passes 3 cards: 39, 40, 41
    state.apply_action(39)
    state.apply_action(40)
    state.apply_action(41)

    self.assertEqual(state.current_player(), 1)
    state_struct = state.to_struct()
    self.assertEqual(state_struct.to_json(), state.to_json())
    self.assertEqual(state_struct.phase, 'Play')
    self.assertEqual(state_struct.current_player, 'Player_1')
    self.assertEqual(state_struct.pass_direction, 'Left')
    self.assertEqual(state_struct.points[0], 0)
    self.assertLen(state_struct.hands[0], 13)
    self.assertLen(state_struct.passed_cards[0], 3)
    self.assertLen(state_struct.received_cards[0], 3)
    self.assertEmpty(state_struct.tricks)
    self.assertFalse(state.hearts_broken())
    self.assertFalse(state_struct.hearts_broken)

    state_json = state.to_json()
    self.assertEqual(
        json.dumps(json.loads(state_json), sort_keys=True),
        json.dumps(
            json.loads(pyspiel.hearts.HeartsStateStruct(state_json).to_json()),
            sort_keys=True)
    )

    obs_struct = state.to_observation_struct(0)
    self.assertEqual(obs_struct.phase, state_struct.phase)
    self.assertEqual(obs_struct.current_player,
                     state_struct.current_player)
    self.assertEqual(obs_struct.hands[0], state_struct.hands[0])
    self.assertEqual(obs_struct.passed_cards[0],
                     state_struct.passed_cards[0])
    self.assertEqual(obs_struct.received_cards[0],
                     state_struct.received_cards[0])
    self.assertEqual(obs_struct.points[0], state_struct.points[0])
    self.assertLen(obs_struct.hands[1], 13)
    self.assertEqual(obs_struct.hands[1], ['XX'] * 13)
    self.assertEqual(obs_struct.passed_cards[1], ['XX'] * 3)
    self.assertEqual(obs_struct.received_cards[1], ['XX'] * 3)
    self.assertEqual(obs_struct.observing_player, 0)

    obs_json = obs_struct.to_json()
    self.assertEqual(
        json.dumps(json.loads(obs_json), sort_keys=True),
        json.dumps(
            json.loads(pyspiel.hearts.HeartsObservationStruct(
                obs_json).to_json()),
            sort_keys=True)
    )

if __name__ == '__main__':
  absltest.main()
