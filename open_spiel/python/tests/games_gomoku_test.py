# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for the game-specific functions for gomoku."""


from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import pyspiel
from open_spiel.python.utils import file_utils
import numpy as np


FLAGS = flags.FLAGS


class GamesGomokuTest(parameterized.TestCase):
  def test_gomoku_game_funs(self):
    game = pyspiel.load_game("gomoku")
    print("dims", game.dims())
    print("size", game.size())
    print("wrap", game.wrap())
    print("connect", game.connect())
    print("anti", game.anti())
    coord = [2, 3]
    action = game.move_to_action(coord=coord)
    print(action)
    move = game.action_to_move(action)
    print(move)

  def test_gomoku_hash(self):
    game = pyspiel.load_game("gomoku")
    state = game.new_initial_state()
    print("hash", state.hash_value())
    state.apply_action(1)
    print("hash", state.hash_value())

  def test_gommoku_game_sim(self):
      game = pyspiel.load_game("gomoku")
      for _ in range(10):
        state = game.new_initial_state()
        mc = 0
        while not state.is_terminal():
          legal_actions = state.legal_actions()
          action = np.random.choice(legal_actions)
          state.apply_action(action)
          mc += 1
        print("mc", mc)

if __name__ == "__main__":
  absltest.main()
