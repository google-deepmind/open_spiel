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

"""Tests for the game-specific functions for go."""

import json

from absl.testing import absltest
import pyspiel

go = pyspiel.go


class GamesGoTest(absltest.TestCase):

  def test_json(self):
    game = pyspiel.load_game("go")
    state = game.new_initial_state()
    state_struct = state.to_struct()
    self.assertEqual(state_struct.current_player, "B")
    self.assertEqual(state_struct.board_grid[0][0]["a1"], "EMPTY")
    state.apply_action(0)
    state_json = state.to_json()
    state_dict = json.loads(state_json)
    state_struct = go.GoStateStruct(state_json)
    self.assertEqual(state_struct.current_player, "W")
    self.assertEqual(state_dict["current_player"], "W")
    self.assertEqual(state_struct.board_grid[0][0]["a1"], "B")
    self.assertEqual(state_struct.to_json(),
                     json.dumps(state_dict, separators=(",", ":")))


if __name__ == "__main__":
  absltest.main()
