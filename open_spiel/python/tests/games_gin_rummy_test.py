# Copyright 2022 DeepMind Technologies Limited
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

"""Tests for the game-specific functions for gin rummy."""


from absl.testing import absltest

import pyspiel


class GamesGinRummyTest(absltest.TestCase):

  def test_bindings(self):
    game = pyspiel.load_game('gin_rummy')
    self.assertFalse(game.oklahoma())
    self.assertEqual(game.knock_card(), 10)
    state = game.new_initial_state()
    self.assertEqual(state.current_phase(), state.Phase.DEAL)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertIsNone(state.upcard())
    self.assertEqual(state.stock_size(), 52)
    self.assertEqual(state.hands(), [[], []])
    self.assertEqual(state.discard_pile(), [])
    self.assertEqual(state.deadwood(), [0, 0])
    self.assertEqual(state.knocked(), [False, False])


if __name__ == '__main__':
  absltest.main()
