"""Tests for open_spiel.python.egt.ssd."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table
from open_spiel.python.egt import ssd
from open_spiel.python.egt import utils
import pyspiel


def _compute_ssd(num_states, transitions):
  """Returns the SSD given (source, target, cost, resistance) transitions."""
  chain = ssd.PerturbedMarkovChain(num_states)
  for source, target, cost, resistance in transitions:
    chain.add_transition(source, target, cost, resistance)
  return chain.stochastically_stable_distribution()


class SSDTest(parameterized.TestCase):

  def test_adaptive_learning_in_qwerty(self):
    """Tests the m = s = 1 adaptive learning example of Wicks & Greenwald."""
    # The states are dD, dQ, qD and qQ. One mistake has probability
    # (2 * eps - eps^2) / 4 and two mistakes have probability eps^2 / 4.
    transitions = [
        (0, 1, 0.5, 1), (0, 2, 0.5, 1), (0, 3, 0.25, 2),
        (1, 0, 0.5, 1), (1, 2, 1., 0), (1, 3, 0.5, 1),
        (2, 0, 0.5, 1), (2, 1, 1., 0), (2, 3, 0.5, 1),
        (3, 0, 0.25, 2), (3, 1, 0.5, 1), (3, 2, 0.5, 1),
    ]  # pyformat: disable
    np.testing.assert_allclose(_compute_ssd(4, transitions), 4 * [0.25])

  def test_numerically_unstable_example(self):
    """Eqn. 13 of Wicks & Greenwald"""
    transitions = [
        (0, 1, 0.5, 0),
        (1, 0, 0.5, 0), (1, 2, 1., 5),
        (2, 3, 0.5, 0),
        (3, 2, 0.5, 0), (3, 4, 1., 2),
        (4, 0, 1., 3), (4, 2, 0.5, 0),
    ]  # pyformat: disable
    np.testing.assert_allclose(
        _compute_ssd(5, transitions), [1 / 3, 1 / 3, 1 / 6, 1 / 6, 0],
        atol=1e-12)

  def test_transient_state_is_not_stochastically_stable(self):
    transitions = [(0, 1, 1., 0), (0, 3, 1., 1), (1, 2, 1., 0), (2, 0, 1., 0),
                   (3, 0, 1., 0)]
    np.testing.assert_allclose(
        _compute_ssd(4, transitions), [1 / 3, 1 / 3, 1 / 3, 0], atol=1e-12)

  def test_costs_break_ties_between_equal_resistances(self):
    transitions = [(0, 1, 0.3, 0.5), (1, 0, 0.1, 0.5)]
    np.testing.assert_allclose(_compute_ssd(2, transitions), [0.25, 0.75])
    transitions = [(0, 1, 0.3, 0.5), (1, 0, 0.1, 0.75)]
    np.testing.assert_allclose(_compute_ssd(2, transitions), [0., 1.])

  def test_matches_stationary_distribution_for_small_eps(self):
    rng = np.random.RandomState(0)
    num_states = 6
    eps = 1e-5
    for _ in range(20):
      chain = ssd.PerturbedMarkovChain(num_states)
      matrix = np.zeros((num_states, num_states))
      for source in range(num_states):
        targets = {(source + 1) % num_states, rng.randint(num_states)}
        for target in targets.difference([source]):
          cost, resistance = rng.uniform(0.1, 0.4), rng.randint(3)
          chain.add_transition(source, target, cost, resistance)
          matrix[source, target] = cost * eps**resistance
        matrix[source, source] = 1 - np.sum(matrix[source])
      pi = np.linalg.lstsq(
          np.vstack([matrix.T - np.eye(num_states), np.ones(num_states)]),
          np.append(np.zeros(num_states), 1.), rcond=None)[0]
      np.testing.assert_allclose(
          chain.stochastically_stable_distribution(), pi, atol=1e-3)

  def test_irregular_process_raises(self):
    with self.assertRaises(ValueError):
      _compute_ssd(3, [(0, 1, 1., 0), (1, 0, 1., 0)])

  @parameterized.parameters(
      "matrix_bos", "matrix_brps", "matrix_cd", "matrix_coordination",
      "matrix_mp", "matrix_pd", "matrix_rps", "matrix_rpsw", "matrix_sh",
      "matrix_shapleys_game")
  def test_is_limit_of_infinite_alpha_alpharank(self, game_name):
    game = pyspiel.load_matrix_game(game_name)
    payoff_tables = utils.game_payoffs_array(game)
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

    pi = alpharank.sweep_pi_vs_epsilon(payoff_tables)
    for use_sparse in [False, True]:
      np.testing.assert_allclose(
          ssd.compute_ssd(payoff_tables, use_sparse=use_sparse), pi, atol=1e-4)

  def test_heuristic_payoff_tables(self):
    game = pyspiel.load_matrix_game("matrix_sh")
    payoff_tables = utils.game_payoffs_array(game)
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    hpts = [heuristic_payoff_table.from_matrix_game(payoff_tables[0])]
    np.testing.assert_allclose(
        ssd.compute_ssd(hpts), ssd.compute_ssd(payoff_tables))

  def test_single_profile_game(self):
    np.testing.assert_allclose(
        ssd.compute_ssd([np.ones((1, 1)), np.ones((1, 1))]), [1.])


if __name__ == "__main__":
  absltest.main()
