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

"""Tests for open_spiel.python.algorithms.get_all_states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python.algorithms import value_iteration
import pyspiel


class ValueIterationTest(absltest.TestCase):

  def test_solve_tic_tac_toe(self):
    game = pyspiel.load_game("tic_tac_toe")
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=0.01)

    initial_state = "...\n...\n..."
    cross_win_state = "...\n...\n.ox"
    naught_win_state = "x..\noo.\nxx."
    self.assertEqual(values[initial_state], 0)
    self.assertEqual(values[cross_win_state], 1)
    self.assertEqual(values[naught_win_state], -1)

  def test_solve_small_goofspiel(self):
    # TODO(author5): This test fails with num_cards = 4 with a new version of
    # LAPACK (3.10.0), which is used by cvxopt. Might be a bug or bad assumption
    # about the handling of numerical error. Look into this.
    game = pyspiel.load_game("goofspiel", {"num_cards": 3})
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=1e-6)

    initial_state = game.new_initial_state()
    assert initial_state.is_chance_node()
    root_value = 0
    for action, action_prob in initial_state.chance_outcomes():
      next_state = initial_state.child(action)
      root_value += action_prob * values[str(next_state)]

    # Symmetric game: value is 0
    self.assertAlmostEqual(root_value, 0)

  def test_solve_small_oshi_zumo(self):
    # Oshi-Zumo(5, 2, 0)
    game = pyspiel.load_game("oshi_zumo", {"coins": 5, "size": 2})
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=1e-6, cyclic_game=True)

    initial_state = game.new_initial_state()
    # Symmetric game: value is 0
    self.assertAlmostEqual(values[str(initial_state)], 0)

    # Oshi-Zumo(5, 2, 1)
    game = pyspiel.load_game("oshi_zumo", {"coins": 5, "size": 2, "min_bid": 1})
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=1e-6, cyclic_game=False)

    initial_state = game.new_initial_state()
    # Symmetric game: value is 0
    self.assertAlmostEqual(values[str(initial_state)], 0)

  def test_solve_small_pig(self):
    game = pyspiel.load_game("pig", {"winscore": 20})
    values = value_iteration.value_iteration(
        game, depth_limit=-1, threshold=1e-6, cyclic_game=True)
    initial_state = game.new_initial_state()
    print("Value of Pig(20): ", values[str(initial_state)])


if __name__ == "__main__":
  absltest.main()
