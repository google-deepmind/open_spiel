"""Tests for open_spiel.python.egt.ssd."""

from absl.testing import absltest

import numpy as np
import pyspiel

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import ssd
from open_spiel.python.egt import utils


class SSDTest(absltest.TestCase):

  def _get_game_payoffs(self, name):
    game = pyspiel.load_matrix_game(name)
    payoff_tables = utils.game_payoffs_array(game)
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    return payoff_tables

  def test_matrix_game_distributions(self):
    test_games = [
        ("matrix_rps", None),
        ("matrix_pd", None),
        ("matrix_coordination", None),
        ("matrix_bos", None),
        ("matrix_brps", None),
        ("matrix_cd", None),
        ("matrix_mp", None),
        ("matrix_rpsw", None),
        ("matrix_sh", None),
        ("matrix_shapleys_game", None),
    ]
    for game_name, _ in test_games:
      with self.subTest(game=game_name):
        payoff_tables = self._get_game_payoffs(game_name)
        ssd_dist = ssd.compute_ssd(
            payoff_tables,
            payoffs_are_hpt_format=False,
            perturbation_strength=0.0001,
            use_sparse=False)
        _, _, alpharank_dist, _, _ = alpharank.compute(
            payoff_tables,
            m=20,
            alpha=0.2)
        print(f"Game: {game_name}")
        print("  SSD distribution:", ssd_dist)
        print("  AlphaRank distribution:", alpharank_dist)

  def test_reduce_logic(self):
    for use_sparse in [False, True]:
      with self.subTest(use_sparse=use_sparse):
        if use_sparse:
          matrix = ssd.SparsePolyMatrix((4, 4))
          matrix.set(1, 0, ssd._make_linear_poly(1.0, 0.0))
          matrix.set(2, 1, ssd._make_linear_poly(1.0, 0.0))
          matrix.set(0, 2, ssd._make_linear_poly(1.0, 0.0))
          matrix.set(0, 3, ssd._make_linear_poly(1.0, 0.0))

          matrix.set(3, 0, ssd._make_linear_poly(0.0, 1.0))  # eps
          matrix.set(1, 0, ssd._make_linear_poly(1.0, -1.0))

        else:
          matrix = np.zeros((4, 4), dtype=object)
          one = np.poly1d([1])
          eps = np.poly1d([1, 0])
          one_minus_eps = np.poly1d([-1, 1])

          matrix[1, 0] = one_minus_eps
          matrix[3, 0] = eps
          matrix[2, 1] = one
          matrix[0, 2] = one
          matrix[0, 3] = one

        dist = ssd._compute_ssd(matrix)

        expected = np.array([1/3, 1/3, 1/3, 0.0])
        np.testing.assert_allclose(dist, expected, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
