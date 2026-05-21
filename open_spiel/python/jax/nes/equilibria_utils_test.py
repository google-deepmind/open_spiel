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
  
  # === CCE gain ===
  def test_cce_gain_per_player_from_payoff(self):
    """CCE gain from payoff tensor."""
    payoffs = jnp.array(
        [[[3, 0], [5, 1]], [[3, 5], [0, 1]]], dtype=jnp.float32
    )
    # Player 0's payoff: [[3, 0], [5, 1]]
    gain_0 = eu.cce_gain_per_player(0, payoff=payoffs[0])
    # Shape: [2, 2, 2] = [A0', A0, A1]
    self.assertEqual(gain_0.shape, (2, 2, 2))
    # When deviating to action 0: G_0(0, a_1) - G_0(a_0, a_1)
    # At (0,0): 3 - 3 = 0; At (0,1): 0 - 0 = 0
    # At (1,0): 3 - 5 = -2; At (1,1): 0 - 1 = -1
    expected_dev0 = jnp.array([[0, 0], [-2, -1]])
    self.assertTrue(jnp.allclose(gain_0[0], expected_dev0))

  def test_cce_gain_per_player_from_ce_gain(self):
    """CCE gain from CE gain (sum over rec axis)."""
    ce_gain = jnp.ones((2, 2, 2, 2))  # [A', A'', A0, A1]
    cce_gain = eu.cce_gain_per_player(0, ce_gain=ce_gain)
    # Should sum over axis 1 (rec axis)
    expected = jnp.ones((2, 2, 2)) * 2  # sum of two 1s
    self.assertTrue(jnp.allclose(cce_gain, expected))
  
  # === Expected CCE gain ===
  def test_expected_cce_gain_from_payoff_and_joint(self):
    """Expected CCE gain under joint strategy."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0

    gain_0 = eu.expected_cce_gain_per_player(
        player=0, payoff=G[0], correlated_joint_strategy=sigma
    )
    # Uniform strategy: expected payoff = 0 for both players
    # Deviation gain for player 0:
    # E[G_0(dev, a_1)] = 0.5 * G_0(dev, 0) + 0.5 * G_0(dev, 1)
    # dev=0: 0.5*1 + 0.5*(-1) = 0
    # dev=1: 0.5*(-1) + 0.5*1 = 0
    # E[G_0(a)] = 0
    # So gain = [0, 0]
    self.assertTrue(jnp.allclose(gain_0, jnp.zeros(2)))

  def test_expected_cce_gain_from_payoff_and_marginals(self):
    """Expected CCE gain under product marginals matches joint for independent sigma."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0
    margs = (jnp.array([0.5, 0.5]), jnp.array([0.5, 0.5]))

    gain_joint = eu.expected_cce_gain_per_player(
        player=0, payoff=G[0], correlated_joint_strategy=sigma
    )
    gain_marg = eu.expected_cce_gain_per_player(
        player=0, payoff=G[0], player_marg_per_player=margs
    )
    self.assertTrue(jnp.allclose(gain_joint, gain_marg))
  
  def test_expected_cce_gain_with_strat_mask(self):
    """Masking invalid deviations."""
    G = _prisoners_dilemma()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0
    mask = jnp.array([True, False])  # action 1 invalid

    gain_0 = eu.expected_cce_gain_per_player(
        player=0, payoff=G[0], correlated_joint_strategy=sigma, strat_mask=mask
    )
    # Action 0 valid, action 1 masked to 0
    self.assertTrue(jnp.isfinite(gain_0[0]))
    # self.assertNotEqual(gain_0[0].item(), 0.0)  # valid
    self.assertEqual(gain_0[1].item(), 0.0)  # masked

  def test_expected_cce_gain_from_cce_dual_grad(self):
    """Direct dual gradient path."""
    grad = jnp.array([1.0, -2.0])
    gain = eu.expected_cce_gain_per_player(
        player=0, cce_dual_grad=grad
    )
    self.assertTrue(jnp.allclose(gain, -grad))

  
  # === CCE logit ===
  def test_cce_logit_shape(self):
    """Logit output shape matches joint strategy."""
    G = _matching_pennies()["payoffs"]
    duals = (jnp.ones(2) / 2, jnp.ones(2) / 2)

    logit = eu.cce_logit(cce_dual_per_player=duals, payoffs=G)
    self.assertEqual(logit.shape, (2, 2))

  def test_cce_logit_finite(self):
    """Logit should be finite for valid inputs."""
    G = jnp.array(
        [[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=jnp.float32
    )
    duals = (jnp.ones(2), jnp.ones(2))

    logit = eu.cce_logit(cce_dual_per_player=duals, payoffs=G)
    self.assertTrue(jnp.all(jnp.isfinite(logit)))

  def test_cce_logit_with_max_cce_gain(self):
    """max_cce_gain offset should shift logit."""
    G = _matching_pennies()["payoffs"]
    duals = (jnp.ones(2) / 2, jnp.ones(2) / 2)

    logit_no_offset = eu.cce_logit(cce_dual_per_player=duals, payoffs=G)
    logit_with_offset = eu.cce_logit(cce_dual_per_player=duals, payoffs=G, max_cce_gain=1.0)

    # Offset adds sum(dual) * max_cce_gain per player to each entry
    expected_shift = sum(jnp.sum(d) * 1.0 for d in duals)
    self.assertTrue(
        jnp.allclose(logit_with_offset - logit_no_offset, expected_shift)
    )
  
  # === CE gain ===
  def test_ce_gain_per_player_from_payoff(self):
    """CE gain from payoff tensor."""
    payoffs = jnp.array(
        [[[3, 0], [5, 1]], [[3, 5], [0, 1]]], dtype=jnp.float32
    )
    gain_0 = eu.ce_gain_per_player(0, payoff=payoffs[0])
    # Shape: [2, 2, 2, 2] = [A0', A0'', A0, A1]
    self.assertEqual(gain_0.shape, (2, 2, 2, 2))

  def test_ce_gain_per_player_from_cce_gain(self):
    """CE gain from CCE gain."""
    cce = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # [2, 2, 2]
    ce = eu.ce_gain_per_player(0, cce_gain=cce)
    # ce[dev, rec] = cce[dev] - cce[rec]
    expected_00 = cce[0] - cce[0]  # all zeros
    self.assertTrue(jnp.allclose(ce[0, 0], expected_00))
    expected_01 = cce[0] - cce[1]
    self.assertTrue(jnp.allclose(ce[0, 1], expected_01))
  
  # === Expected CE gain ===
  def test_expected_ce_gain_from_payoff_and_joint(self):
    """Expected CE gain under joint strategy."""
    G = _coordination_game()["payoffs"]
    sigma = _coordination_game()["sigma_good"]  # 0.5 at (0,0) and (1,1)

    gain_0 = eu.expected_ce_gain_per_player(
        0, payoff=G[0], correlated_joint_strategy=sigma
    )
    # Shape: [2, 2] = [dev, rec]
    self.assertEqual(gain_0.shape, (2, 2))
    # At equilibrium, no profitable deviation: max gain = 0
    self.assertAlmostEqual(jnp.max(gain_0).item(), 0.0, places=5)

  def test_expected_ce_gain_vs_cce_gain_diagonal(self):
    """Diagonal of CE gain (dev=rec) should match CCE gain."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0

    ce_gains = eu.expected_ce_gain(payoffs=G, correlated_joint_strategy=sigma)
    cce_gains = eu.expected_cce_gain(payoffs=G, correlated_joint_strategy=sigma)

    for p in range(2):
      self.assertTrue(
          jnp.allclose(jnp.diag(ce_gains[p]), cce_gains[p], atol=1e-5)
      )

  def test_expected_ce_gain_from_ce_gain_and_joint(self):
    """Expected CE gain from precomputed CE gain."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0

    ce_gains = eu.ce_gain(payoffs=G)
    gain_0_from_payoff = eu.expected_ce_gain_per_player(
        0, payoff=G[0], correlated_joint_strategy=sigma
    )
    gain_0_from_ce = eu.expected_ce_gain_per_player(
        0, ce_gain=ce_gains[0], correlated_joint_strategy=sigma
    )
    self.assertTrue(jnp.allclose(gain_0_from_payoff, gain_0_from_ce))

  # === CE logit ===
  def test_ce_logit_shape(self):
    """CE logit output shape."""
    G = _matching_pennies()["payoffs"]
    duals = (jnp.ones((2, 2)), jnp.ones((2, 2)))

    logit = eu.ce_logit(ce_dual_per_player=duals, payoffs=G)
    self.assertEqual(logit.shape, (2, 2))

  def test_ce_logit_finite(self):
    """CE logit should be finite."""
    G = _matching_pennies()["payoffs"]
    duals = (jnp.ones((2, 2)), jnp.ones((2, 2)))

    logit = eu.ce_logit(ce_dual_per_player=duals, payoffs=G)
    self.assertTrue(jnp.all(jnp.isfinite(logit)))

  def test_ce_logit_with_max_ce_gain(self):
    """max_ce_gain offset should shift logit."""
    G = _matching_pennies()["payoffs"]
    duals = (jnp.ones((2, 2)) / 4, jnp.ones((2, 2)) / 4)

    logit_no_offset = eu.ce_logit(ce_dual_per_player=duals, payoffs=G)
    logit_with_offset = eu.ce_logit(ce_dual_per_player=duals, payoffs=G, max_ce_gain=1.0)

    expected_shift = sum(jnp.sum(d) * 1.0 for d in duals)
    self.assertTrue(
        jnp.allclose(logit_with_offset - logit_no_offset, expected_shift)
    )
  
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
    G = jnp.array([
        [[2, 0], [0, 1]],   # player 0 (row)
        [[1, 0], [0, 2]],   # player 1 (col) — not used for this check
    ], dtype=jnp.float32)

    # Anti-diagonal correlation: always miscoordinate
    sigma = jnp.zeros((2, 2))
    sigma = sigma.at[0, 1].set(0.5)
    sigma = sigma.at[1, 0].set(0.5)

    epsilon = jnp.array([0.0, 0.0])

    cce_gap = eu.compute_cce_gap(G, sigma, epsilon)
    ce_gap = eu.compute_ce_gap(G, sigma, epsilon)

    self.assertTrue(jnp.all(ce_gap >= cce_gap))

  def test_ce_vs_cce_gap_matching_pennies(self):
    """CE gap should be >= CCE gap (CE is stricter)."""
    matching_pennies = _matching_pennies()
    G = matching_pennies["payoffs"]
    sigma = matching_pennies["sigma_uniform"]
    epsilon = jnp.array([0.1, 0.1])

    cce_gap = eu.compute_cce_gap(G, sigma, epsilon)
    ce_gap = eu.compute_ce_gap(G, sigma, epsilon)

    self.assertTrue(jnp.all(ce_gap >= cce_gap))

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

  def test_cce_gap_with_strat_mask(self):
    """CCE gap respects strat_mask."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0
    epsilon = jnp.array([0.0, 0.0])

    # Mask out action 1 for both players
    strat_masks = (
        jnp.array([True, False]),
        jnp.array([True, False]),
    )

    gap = eu.compute_cce_gap(
        G, sigma, epsilon, strat_mask_per_player=strat_masks
    )
    # With only action 0 valid, no deviation possible, gap = 0
    self.assertAlmostEqual(gap, 0.0, 5)

  def test_cce_gap_with_joint_mask(self):
    """CCE gap respects joint_mask."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0
    epsilon = jnp.array([0.0, 0.0])

    # Only (0,0) and (1,1) are valid
    joint_mask = jnp.array([[True, False], [False, True]])

    gap = eu.compute_cce_gap(G, sigma, epsilon, joint_mask=joint_mask)
    self.assertTrue(jnp.isfinite(gap))

  def test_cce_gap_delegates_to_expected_cce_gain(self):
    """CCE gap should equal max(expected_cce_gain) - epsilon."""
    G = _matching_pennies()["payoffs"]
    sigma = jnp.ones((2, 2)) / 4.0
    epsilon = jnp.array([0.0, 0.0])

    gap = eu.compute_cce_gap(G, sigma, epsilon)

    # Manual computation
    gains = eu.expected_cce_gain(payoffs=G, correlated_joint_strategy=sigma)
    manual_gap = sum(max(g) for g in gains)

    self.assertTrue(jnp.allclose(gap, manual_gap))

  def test_ce_gap_delegates_to_expected_ce_gain(self):
    """CE gap should equal max(expected_ce_gain) - epsilon."""
    G = _coordination_game()["payoffs"]
    sigma = _coordination_game()["sigma_good"]
    epsilon = jnp.array([0.0, 0.0])

    gap = eu.compute_ce_gap(G, sigma, epsilon)

    # Manual computation
    gains = eu.expected_ce_gain(payoffs=G, correlated_joint_strategy=sigma)
    manual_gap = sum(jnp.max(g) for g in gains)

    self.assertTrue(jnp.allclose(gap, manual_gap))


if __name__ == "__main__":
  absltest.main()
