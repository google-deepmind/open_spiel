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
import numpy as np

from open_spiel.python.algorithms import lp_solver
import pyspiel


class LPSolversTest(absltest.TestCase):

  def test_rock_paper_scissors(self):
    p0_sol, p1_sol, p0_sol_val, p1_sol_val = (
        lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game(
                [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
                [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])))
    self.assertLen(p0_sol, 3)
    self.assertLen(p1_sol, 3)
    for i in range(3):
      self.assertAlmostEqual(p0_sol[i], 1.0 / 3.0)
      self.assertAlmostEqual(p1_sol[i], 1.0 / 3.0)
    self.assertAlmostEqual(p0_sol_val, 0.0)
    self.assertAlmostEqual(p1_sol_val, 0.0)

  def test_biased_rock_paper_scissors(self):
    # See sec 6.2 of Bosansky et al. 2016. Algorithms for Computing Strategies
    # in Two-Player Simultaneous Move Games
    # http://mlanctot.info/files/papers/aij-2psimmove.pdf
    p0_sol, p1_sol, p0_sol_val, p1_sol_val = (
        lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game(
                [[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],
                [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]])))
    self.assertLen(p0_sol, 3)
    self.assertLen(p1_sol, 3)
    self.assertAlmostEqual(p0_sol[0], 1.0 / 16.0, places=4)
    self.assertAlmostEqual(p1_sol[0], 1.0 / 16.0, places=4)
    self.assertAlmostEqual(p0_sol[1], 10.0 / 16.0, places=4)
    self.assertAlmostEqual(p1_sol[1], 10.0 / 16.0, places=4)
    self.assertAlmostEqual(p0_sol[2], 5.0 / 16.0, places=4)
    self.assertAlmostEqual(p1_sol[2], 5.0 / 16.0, places=4)
    self.assertAlmostEqual(p0_sol_val, 0.0)
    self.assertAlmostEqual(p1_sol_val, 0.0)

  def test_asymmetric_pure_nonzero_val(self):
    #        c0      c1       c2
    # r0 | 2, -2 |  1, -1 |  5, -5
    # r1 |-3,  3 | -4,  4 | -2,  2
    #
    # Pure eq (r0,c1) for a value of (1, -1)
    # 2nd row is dominated, and then second player chooses 2nd col.
    p0_sol, p1_sol, p0_sol_val, p1_sol_val = (
        lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game([[2.0, 1.0, 5.0], [-3.0, -4.0, -2.0]],
                                       [[-2.0, -1.0, -5.0], [3.0, 4.0, 2.0]])))
    self.assertLen(p0_sol, 2)
    self.assertLen(p1_sol, 3)
    self.assertAlmostEqual(p0_sol[0], 1.0)
    self.assertAlmostEqual(p0_sol[1], 0.0)
    self.assertAlmostEqual(p1_sol[0], 0.0)
    self.assertAlmostEqual(p1_sol[1], 1.0)
    self.assertAlmostEqual(p0_sol_val, 1.0)
    self.assertAlmostEqual(p1_sol_val, -1.0)

  def test_solve_blotto(self):
    blotto_matrix_game = pyspiel.load_matrix_game("blotto")
    p0_sol, p1_sol, p0_sol_val, p1_sol_val = (
        lp_solver.solve_zero_sum_matrix_game(blotto_matrix_game))
    self.assertLen(p0_sol, blotto_matrix_game.num_rows())
    self.assertLen(p1_sol, blotto_matrix_game.num_cols())
    # Symmetric game, must be zero
    self.assertAlmostEqual(p0_sol_val, 0.0)
    self.assertAlmostEqual(p1_sol_val, 0.0)

  def _assert_dominated(self, *args, **kwargs):
    self.assertTrue(lp_solver.is_dominated(*args, **kwargs))

  def _assert_undominated(self, *args, **kwargs):
    self.assertFalse(lp_solver.is_dominated(*args, **kwargs))

  def test_dominance(self):
    self._assert_undominated(0, [[1., 1.], [2., 0.], [0., 2.]], 0,
                             lp_solver.DOMINANCE_STRICT)
    self._assert_undominated(0, [[1., 1.], [2., 0.], [0., 2.]], 0,
                             lp_solver.DOMINANCE_WEAK)
    self._assert_dominated(0, [[1., 1.], [2.1, 0.], [0., 2.]], 0,
                           lp_solver.DOMINANCE_STRICT)

    self._assert_undominated(0, [[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]], 0,
                             lp_solver.DOMINANCE_STRICT)
    self._assert_dominated(0, [[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]], 0,
                           lp_solver.DOMINANCE_WEAK)
    self._assert_dominated(0, [[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]], 0,
                           lp_solver.DOMINANCE_VERY_WEAK)

    self._assert_dominated(0, [[1., 1., 1.], [2.1, 0., 1.], [0., 2., 2.]], 0,
                           lp_solver.DOMINANCE_STRICT)
    self._assert_dominated(0, [[1., 1., 1.], [2.1, 0., 1.], [0., 2., 2.]], 0,
                           lp_solver.DOMINANCE_WEAK)
    self._assert_dominated(0, [[1., 1., 1.], [2.1, 0., 1.], [0., 2., 2.]], 0,
                           lp_solver.DOMINANCE_VERY_WEAK)

    self._assert_undominated(0, [[1., 1., 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                             lp_solver.DOMINANCE_STRICT)
    self._assert_undominated(0, [[1., 1., 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                             lp_solver.DOMINANCE_WEAK)
    self._assert_dominated(0, [[1., 1., 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                           lp_solver.DOMINANCE_VERY_WEAK)

    self._assert_undominated(0, [[1., 1.1, 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                             lp_solver.DOMINANCE_STRICT)
    self._assert_undominated(0, [[1., 1.1, 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                             lp_solver.DOMINANCE_WEAK)
    self._assert_undominated(0, [[1., 1.1, 1.], [2., 0., 2.], [0., 2., 0.]], 0,
                             lp_solver.DOMINANCE_VERY_WEAK)

  def test_dominance_3player(self):
    self._assert_undominated(0,
                             [[[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]]] * 3,
                             1, lp_solver.DOMINANCE_STRICT)
    self._assert_dominated(0, [[[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]]] * 3,
                           1, lp_solver.DOMINANCE_WEAK)
    self._assert_dominated(0, [[[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]]] * 3,
                           1, lp_solver.DOMINANCE_VERY_WEAK)

  def test_dominance_prisoners_dilemma(self):
    self._assert_dominated(0, pyspiel.load_matrix_game("matrix_pd"), 1,
                           lp_solver.DOMINANCE_STRICT)
    self._assert_undominated(1, pyspiel.load_matrix_game("matrix_pd"), 1,
                             lp_solver.DOMINANCE_VERY_WEAK)

  def test_dominance_mixture(self):
    mixture = lp_solver.is_dominated(
        0, [[1., 1., 1.], [2., 0., 1.], [0., 2., 2.]],
        0,
        lp_solver.DOMINANCE_WEAK,
        return_mixture=True)
    self.assertAlmostEqual(mixture[0], 0)
    self.assertAlmostEqual(mixture[1], 0.5)
    self.assertAlmostEqual(mixture[2], 0.5)

  def _checked_iterated_dominance(self, *args, **kwargs):
    reduced_game, live_actions = lp_solver.iterated_dominance(*args, **kwargs)
    if isinstance(reduced_game, pyspiel.MatrixGame):
      payoffs_shape = [2, reduced_game.num_rows(), reduced_game.num_cols()]
    else:
      payoffs_shape = list(reduced_game.shape)
    self.assertLen(live_actions, payoffs_shape[0])
    self.assertListEqual(payoffs_shape[1:], [
        np.sum(live_actions_for_player)
        for live_actions_for_player in live_actions
    ])
    return reduced_game, live_actions

  def test_iterated_dominance_prisoners_dilemma(self):
    # find the strictly dominant (D, D) strategy
    pd = pyspiel.load_matrix_game("matrix_pd")
    pd_dom, pd_live = self._checked_iterated_dominance(
        pd, lp_solver.DOMINANCE_STRICT)
    self.assertEqual(pd_dom.num_rows(), 1)
    self.assertEqual(pd_dom.num_cols(), 1)
    self.assertEqual(pd_dom.row_action_name(0), "Defect")
    self.assertEqual(pd_dom.col_action_name(0), "Defect")
    self.assertListEqual(pd_live[0].tolist(), [False, True])
    self.assertListEqual(pd_live[1].tolist(), [False, True])

  def test_iterated_dominance_auction(self):
    # find a strategy through iterated dominance that's not strictly dominant
    auction = pyspiel.extensive_to_matrix_game(
        pyspiel.load_game("first_sealed_auction(max_value=3)"))
    auction_dom, auction_live = self._checked_iterated_dominance(
        auction, lp_solver.DOMINANCE_STRICT)
    # there's just one non-dominated action
    self.assertEqual(auction_dom.num_rows(), 1)
    self.assertEqual(auction_dom.num_cols(), 1)
    best_action = [
        auction.row_action_name(row) for row in range(auction.num_rows())
    ].index(auction_dom.row_action_name(0))
    self.assertTrue(auction_live[0][best_action])
    # other actions are all weakly but not all strictly dominated
    self.assertNotIn(False, [
        lp_solver.is_dominated(action, auction, 0, lp_solver.DOMINANCE_WEAK)
        for action in range(6)
        if action != best_action
    ])
    self.assertIn(False, [
        lp_solver.is_dominated(action, auction, 0, lp_solver.DOMINANCE_STRICT)
        for action in range(6)
        if action != best_action
    ])

  def test_iterated_dominance_ordering(self):
    for _ in range(100):
      game = np.random.randint(5, size=(2, 3, 3))
      unused_reduced_strict, live_strict = self._checked_iterated_dominance(
          game, lp_solver.DOMINANCE_STRICT)
      unused_reduced_weak, live_weak = self._checked_iterated_dominance(
          game, lp_solver.DOMINANCE_WEAK)
      unused_reduced_vweak, live_vweak = self._checked_iterated_dominance(
          game, lp_solver.DOMINANCE_VERY_WEAK)
      for player in range(2):
        self.assertTrue((live_strict[player] >= live_weak[player]).all())
        self.assertTrue((live_strict[player] >= live_vweak[player]).all())
        self.assertIn(True, live_vweak[player])

  def test_iterated_dominance_strict_invariance(self):
    for _ in range(100):
      game = np.random.randint(5, size=(3, 2, 2, 3))
      unused_reduced, live = self._checked_iterated_dominance(
          game, lp_solver.DOMINANCE_STRICT)
      perms = [np.random.permutation(size) for size in game.shape]
      game_perm = game[tuple(np.meshgrid(
          *perms, indexing="ij"))].transpose([0] + list(1 + perms[0]))
      unused_reduced_perm, live_perm = self._checked_iterated_dominance(
          game_perm, lp_solver.DOMINANCE_STRICT)
      for player in range(3):
        perm_player = perms[0][player]
        self.assertListEqual(live_perm[player].tolist(),
                             live[perm_player][perms[1 + perm_player]].tolist())


if __name__ == "__main__":
  absltest.main()
