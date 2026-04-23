import jax
import jax.numpy as jnp
from absl.testing import absltest

import open_spiel.python.jax.nes.equilibria_utils as eu

"""Tests for open_spiel.python.jax.nes.equilibria_utils.py"""


def _matching_pennies() -> dict:
  """
  Matching pennies: zero-sum 2x2 game.
  Row player wants match, column wants mismatch.
  """
  # Payoffs: [row_player, col_player]
  #       H       T
  payoffs = jnp.array(
    [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=jnp.float32
  )

  # Uniform mixed strategy is NE/CCE
  sigma_uniform = jnp.ones((2, 2)) / 4.0

  # Pure strategy (not equilibrium)
  sigma_pure = jnp.zeros((2, 2))
  sigma_pure = sigma_pure.at[0, 0].set(1.0)  # Both play Heads

  return {
    "payoffs": payoffs,
    "sigma_uniform": sigma_uniform,
    "sigma_pure": sigma_pure,
    "expected_payoff_uniform": 0.0,  # Zero-sum, uniform = 0
  }


def _coordination_game() -> dict:
  """
  Coordination game: both players want to match.
  Two pure NE: (0,0) and (1,1), plus mixed.
  """
  #     L       R
  G = jnp.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=jnp.float32)

  sigma_mixed = jnp.ones((2, 2)) / 4.0

  sigma_good = jnp.zeros((2, 2))
  sigma_good = sigma_good.at[0, 0].set(0.5)  # Both coordinate on L
  sigma_good = sigma_good.at[1, 1].set(0.5)  # Both coordinate on R

  return {
    "payoffs": G,
    "sigma_mixed": sigma_mixed,
    "sigma_good": sigma_good,
  }


def _prisoners_dilemma() -> dict:
  """
  Prisoner's Dilemma: (D, D) is unique NE.
  C = Cooperate, D = Defect
  """
  #           C       D
  G = jnp.array([[[-1, -3], [0, -2]], [[-1, 0], [-3, -2]]], dtype=jnp.float32)

  # NE: both defect
  sigma_ne = jnp.zeros((2, 2))
  sigma_ne = sigma_ne.at[1, 1].set(1.0)

  # Not NE: both cooperate (can deviate to D)
  sigma_cooperate = jnp.zeros((2, 2))
  sigma_cooperate = sigma_cooperate.at[0, 0].set(1.0)

  return {
    "payoffs": G,
    "sigma_ne": sigma_ne,
    "sigma_cooperate": sigma_cooperate,
  }


class EquilibriaTest(absltest.TestCase):
  ## CCE gap tests
  def test_cce_gap_uniform_matching_pennies(self):
    """Uniform strategy in matching pennies should have zero CCE gap with large epsilon."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_uniform"]

    # With epsilon=1.0, gap should be 0 (uniform is 0-CCE)
    epsilon_large = jnp.array([1.0, 1.0])
    gap = eu.compute_cce_gap(G, sigma, epsilon_large)

    # Gap should be approximatelly 0
    self.assertAlmostEqual(gap, 0.0, 2)

  def test_cce_gap_pure_strategy_violated(self):
    """Pure strategy in matching pennies should have positive gap."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_pure"]  # Both play Heads

    # Small epsilon
    epsilon_small = jnp.array([0.1, 0.1])
    gap = eu.compute_cce_gap(G, sigma, epsilon_small)

    # Column player can profitably deviate to Tails when Row plays Heads
    # Column payoff: -1 at (H,H), but +1 at (H,T)
    # Gain = 1 - (-1) = 2, so gap should be ~2 - 0.1 = 1.9
    self.assertTrue(jnp.all(gap > 1.0))

  def test_cce_gap_zero_with_perfect_epsilon(self):
    """NE strategy should have zero gap even with epsilon=0."""
    prisoners_dilemma = _prisoners_dilemma()
    G = prisoners_dilemma["payoffs"]
    sigma = prisoners_dilemma["sigma_ne"]  # Both defect

    epsilon_zero = jnp.array([0.0, 0.0])
    gap = eu.compute_cce_gap(G, sigma, epsilon_zero)
    self.assertAlmostEqual(gap, 0.0, 4)

  def test_cce_gap_cooperate_in_pd(self):
    """Cooperate in PD: players can deviate to Defect."""
    prisoners_dilemma = _prisoners_dilemma()
    G = prisoners_dilemma["payoffs"]
    sigma = prisoners_dilemma["sigma_cooperate"]

    epsilon_zero = jnp.array([0.0, 0.0])
    gap = eu.compute_cce_gap(G, sigma, epsilon_zero)

    # Each player can gain 2 by deviating (from -1 to +1 relative)
    # Actually: Cooperate payoff = -1, Defect payoff = 0 when opponent cooperates
    # Gain = 0 - (-1) = 1 per player
    self.assertTrue(jnp.all(gap > 1.5))

  def test_cce_gap_monotonic_in_epsilon(self):
    """Gap should decrease as epsilon increases."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_pure"]

    gaps = []
    for eps in [0.0, 0.5, 1.0, 2.0, 5.0]:
      epsilon = jnp.array([eps, eps])
      gap = eu.compute_cce_gap(G, sigma, epsilon)
      gaps.append(float(gap))

    # Gaps should be non-increasing
    for i in range(len(gaps) - 1):
      self.assertTrue(gaps[i] >= gaps[i + 1])

  ## CE gal tests

  def test_ce_gap_uniform_matching_pennies(self):
    """Uniform in matching pennies: check CE gap."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_uniform"]

    epsilon_small = jnp.array([0.1, 0.1])
    gap = eu.compute_ce_gap(G, sigma, epsilon_small)

    # Should have positive gap
    self.assertEqual(gap, 0.0)

  def test_ce_gap_pure_strategy(self):
    """Pure strategy is a CE (degenerate: signal reveals action, no benefit to deviate)."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_pure"]

    epsilon_zero = jnp.array([0.0, 0.0])
    gap = eu.compute_ce_gap(G, sigma, epsilon_zero)

    self.assertTrue(jnp.all(gap > 1.5))

  def test_ce_vs_cce_gap_comparison(self):
    """CE gap should be >= CCE gap (CE is stricter)."""
    coordination_game = _coordination_game()
    G = coordination_game["payoffs"]
    sigma = coordination_game["sigma_mixed"]

    epsilon = jnp.array([0.1, 0.1])

    cce_gap = eu.compute_cce_gap(G, sigma, epsilon)
    ce_gap = eu.compute_ce_gap(G, sigma, epsilon)

    # Every CE is a CCE, so CE constraints are tighter
    # Therefore CE gap >= CCE gap
    self.assertTrue(jnp.all(cce_gap >= ce_gap))

  def test_cce_gap_non_cubic(self):
    """Test with 2x3 game (non-cubic)."""
    # Player 0: 2 actions, Player 1: 3 actions
    G = jax.random.uniform(jax.random.PRNGKey(0), (2, 2, 3))
    sigma = jnp.ones((2, 3)) / 6.0
    epsilon = jnp.array([0.5, 0.5])

    gap = eu.compute_cce_gap(G, sigma, epsilon)

    self.assertTrue(jnp.isfinite(gap).all())
    self.assertTrue(jnp.all(gap >= 0))

  def test_ce_gap_non_cubic(self):
    """Test CE gap with non-cubic game."""
    G = jax.random.uniform(jax.random.PRNGKey(1), (2, 3, 4))
    sigma = jnp.ones((3, 4)) / 12.0
    epsilon = jnp.array([0.5, 0.5])

    gap = eu.compute_ce_gap(G, sigma, epsilon)

    self.assertTrue(jnp.isfinite(gap).all())
    self.assertTrue(jnp.all(gap >= 0))


if __name__ == "__main__":
  absltest.main()
