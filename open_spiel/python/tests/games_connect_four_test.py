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

"""Tests for the game-specific functions for connect_four."""

from absl.testing import absltest
import pyspiel

connect_four = pyspiel.connect_four


class GamesConnectFourTest(absltest.TestCase):

  def test_json(self):
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    state.apply_action(4)
    state_struct = state.to_struct()
    self.assertEqual(
        state_struct.board[0],
        [".", ".", ".", ".", "x", ".", "."],
    )
    self.assertEqual(state_struct.current_player, "o")
    json_from_struct = state_struct.to_json()
    state_json = state.to_json()
    self.assertEqual(json_from_struct, state_json)
    state_struct = connect_four.ConnectFourStateStruct(state_json)
    self.assertEqual(state_struct.to_json(), state_json)


if __name__ == "__main__":
  absltest.main()
