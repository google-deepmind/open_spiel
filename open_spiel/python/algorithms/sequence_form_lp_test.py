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

"""Tests for LP solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from open_spiel.python.algorithms import sequence_form_lp
import pyspiel


class SFLPTest(absltest.TestCase):

  def test_rock_paper_scissors(self):
    game = pyspiel.load_game_as_turn_based("matrix_rps")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    self.assertAlmostEqual(val1, 0)
    self.assertAlmostEqual(val2, 0)

  def test_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(val1, -1 / 18)
    self.assertAlmostEqual(val2, +1 / 18)

  def test_kuhn_poker_efg(self):
    game = pyspiel.load_efg_game(pyspiel.get_kuhn_poker_efg_data())
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(val1, -1 / 18)
    self.assertAlmostEqual(val2, +1 / 18)

  def test_leduc_poker(self):
    game = pyspiel.load_game("leduc_poker")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # values obtained from Appendix E.2 of Lanctot et al. 2017, A Unified
    # Game-Theoretic Approach to Multiagent Reinforcement Learning.
    # https://arxiv.org/abs/1711.00832
    self.assertAlmostEqual(val1, -0.085606424078, places=6)
    self.assertAlmostEqual(val2, 0.085606424078, places=6)

  def test_iigoofspiel4(self):
    game = pyspiel.load_game_as_turn_based("goofspiel", {
        "imp_info": True,
        "num_cards": 4,
        "points_order": "descending",
    })
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # symmetric game, should be 0
    self.assertAlmostEqual(val1, 0)
    self.assertAlmostEqual(val2, 0)

  # TODO(author5): currently does not work because TTT's information state is
  # not perfect recall. Enable this test when fixed.
  # def test_tictactoe(self):
  #   game = pyspiel.load_game("tic_tac_toe")
  #   val1, val2 = sequence_form_lp.solve_zero_sum_game(game)
  #   self.assertAlmostEqual(val1, 0)
  #   self.assertAlmostEqual(val2, 0)

  # This test takes too long for non-glpk solvers, and glpk solver is not
  # supported within google's internal cvxopt import. When solving via glpk,
  # (locally, outside of google's testing framework), the test takes >300
  # seconds, so it is disabled by default, but still left here for reference.
  # Note, value is taken from an independent implementation but also found in
  # Neller & Lanctot 2013, An Introduction to Counterfactual Regret Minimization
  # http://modelai.gettysburg.edu/2013/cfr/cfr.pdf
  #
  # def test_liars_dice(self):
  #  game = pyspiel.load_game("liars_dice")
  #  val1, val2 = sequence_form_lp.solve_zero_sum_game(game, solver="glpk")
  #  self.assertAlmostEqual(val1, -0.027131782945736)
  #  self.assertAlmostEqual(val2, 0.027131782945736)


if __name__ == "__main__":
  absltest.main()
