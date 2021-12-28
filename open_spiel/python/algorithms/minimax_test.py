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

"""Tests for open_spiel.python.algorithms.minimax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python.algorithms import minimax
import pyspiel


class MinimaxTest(absltest.TestCase):

  def test_compute_game_value(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")

    game_score, _ = minimax.alpha_beta_search(tic_tac_toe)
    self.assertEqual(0., game_score)

  def test_compute_game_value_with_evaluation_function(self):
    # We only check it runs
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")

    game_score, _ = minimax.alpha_beta_search(
        tic_tac_toe, value_function=lambda x: 0, maximum_depth=1)
    self.assertEqual(0., game_score)

  def test_win(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    state = tic_tac_toe.new_initial_state()

    # Construct:
    # .o.
    # .x.
    # ...
    state.apply_action(4)
    state.apply_action(1)
    game_score, _ = minimax.alpha_beta_search(tic_tac_toe, state=state)
    self.assertEqual(1., game_score)

  def test_loss(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    state = tic_tac_toe.new_initial_state()

    # Construct:
    # ...
    # xox
    # ..o
    state.apply_action(5)
    state.apply_action(4)
    state.apply_action(3)
    state.apply_action(8)
    game_score, _ = minimax.alpha_beta_search(tic_tac_toe, state=state)
    self.assertEqual(-1., game_score)


if __name__ == "__main__":
  absltest.main()
