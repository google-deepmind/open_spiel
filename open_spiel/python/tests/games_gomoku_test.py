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
import numpy as np

import pyspiel

FLAGS = flags.FLAGS


class GamesGomokuTest(parameterized.TestCase):

  def test_gomoku_game_funs(self):
    # Default 15 x 15 grid, 5 in a row wins, no wrap.
    game = pyspiel.load_game("gomoku")
    dims = game.dims()
    self.assertEqual(dims, 2, f"Incorrect dims {dims}")
    size = game.size()
    self.assertEqual(size, 15, f"Incorrect size {size}")
    connect = game.connect()
    self.assertEqual(connect, 5, f"Incorrect connect {connect}")
    wrap = game.wrap()
    self.assertEqual(wrap, False, f"Incorrect wrap {wrap}")
    anti = game.anti()
    self.assertEqual(anti, False, f"Incorrect anti {anti}")

    coord = [2, 3]
    action = game.move_to_action(coord=coord)
    print("action", action)
    move = game.action_to_move(action)
    self.assertEqual(coord, move, f"Coord {coord} move {move}")

  def test_gomoku_hash(self):
    game = pyspiel.load_game("gomoku(size=3,connect=3)")
    state = game.new_initial_state()
    hash0 = state.hash_value()
    self.assertEqual(hash0, 0, f"Initial board hash {hash0}")
    state.apply_action(1)
    hash1 = state.hash_value()
    sym1 = state.symmetric_hash()
    # States related by symmetry shoud have different hashes
    # but the same symmetric hash.
    state = game.new_initial_state()
    state.apply_action(5)
    hash2 = state.hash_value()
    sym2 = state.symmetric_hash()
    self.assertNotEqual(hash1, hash2, f"Hash1 {hash1} Hash2 {hash2}")
    self.assertEqual(sym1, sym2, f"Hash1 {sym1} Hash2 {sym2}")

    # We can change symmetry policy during a game.
    # Change allow_reflections from false to true.
    state = game.new_initial_state()
    policy = state.get_symmetry_policy()
    print("policy", policy)
    self.assertEqual(policy.allow_reflections, False, "Wrong symmetry policy")
    # verify policy is correct here
    state.apply_action(0)
    state.apply_action(1)
    sym11 = state.symmetric_hash()
    # set symmetry policy here
    policy.allow_reflections = True
    policy = state.get_symmetry_policy()
    self.assertEqual(policy.allow_reflections, True, "Wrong symmetry policy")
    sym12 = state.symmetric_hash()

    state = game.new_initial_state()
    policy = state.get_symmetry_policy()
    # verify policy is correct here
    self.assertEqual(policy.allow_reflections, False, "Wrong symmetry policy")
    state.apply_action(2)
    state.apply_action(1)
    sym21 = state.symmetric_hash()
    # set symmetry policy here
    policy.allow_reflections = True
    sym22 = state.symmetric_hash()
    self.assertNotEqual(sym11, sym21, f"Hash1 {sym11} Hash2 {sym21}")
    self.assertEqual(sym12, sym22, f"Hash1 {sym12} Hash2 {sym22}")

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

  def test_winning_line(self):
    game = pyspiel.load_game("gomoku(size=3,connect=3)")
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    state.apply_action(4)
    state.apply_action(2)
    state.apply_action(8)
    win = state.winning_line()
    self.assertEqual(win, [[0, 0], [1, 1], [2, 2]], f"win {win}")
    returns = state.returns()
    self.assertEqual(returns, [1.0, -1.0], f"returns {returns}")

    # same "winning line", but with anti it is the losing line.
    game = pyspiel.load_game("gomoku(size=3,connect=3,anti=true)")
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    state.apply_action(4)
    state.apply_action(2)
    state.apply_action(8)
    win = state.winning_line()
    self.assertEqual(win, [[0, 0], [1, 1], [2, 2]], f"win {win}")
    returns = state.returns()
    self.assertEqual(returns, [-1.0, 1.0], f"returns {returns}")

    # wrap winning line
    game = pyspiel.load_game("gomoku(size=4,connect=3,wrap=true)")
    state = game.new_initial_state()
    state.apply_action(2)
    state.apply_action(3)
    state.apply_action(7)
    state.apply_action(6)
    state.apply_action(8)
    win = state.winning_line()
    self.assertEqual(win, [[0, 2], [1, 3], [2, 0]], f"win {win}")
    returns = state.returns()
    self.assertEqual(returns, [1.0, -1.0], f"returns {returns}")

  def test_consistent_hash(self):
    game = pyspiel.load_game("gomoku")
    state = game.new_initial_state()
    state.apply_action(0)
    hash1 = state.hash_value()
    game = pyspiel.load_game("gomoku")
    state = game.new_initial_state()
    state.apply_action(0)
    hash2 = state.hash_value()
    self.assertEqual(hash1, hash2, f"Hash1 {hash1} Hash2 {hash2}")


if __name__ == "__main__":
  absltest.main()
