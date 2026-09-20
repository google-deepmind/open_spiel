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

"""Tests for open_spiel.python.egt.alpharank."""

from unittest import mock

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
import matplotlib
matplotlib.use("agg")  # switch backend for testing

import numpy as np

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import heuristic_payoff_table
from open_spiel.python.egt import utils
import pyspiel


class AlphaRankTest(absltest.TestCase):

  def test_stationary_distribution(self):
    """Tests stationary distribution using payoffs from Han et al., 2013."""
    r = 1.
    t = 2.
    p = 0.
    s = -1.
    delta = 4.
    eps = 0.25
    payoff_tables = [
        np.asarray([[r - eps / 2., r - eps, 0, s + delta - eps, r - eps],
                    [r, r, s, s, s], [0, t, p, p, p], [t - delta, t, p, p, p],
                    [r, t, p, p, p]])
    ]

    m = 20
    alpha = 0.1
    expected_pi = np.asarray(
        [0.40966787, 0.07959841, 0.20506998, 0.08505983, 0.2206039])

    # Test payoffs in matrix format
    _, _, pi_matrix, _, _ = alpharank.compute(
        payoff_tables, m=m, alpha=alpha, use_local_selection_model=False)
    np.testing.assert_array_almost_equal(pi_matrix, expected_pi, decimal=4)

    # Test payoffs in HPT format
    hpts = [heuristic_payoff_table.from_matrix_game(payoff_tables[0])]
    _, _, pi_hpts, _, _ = alpharank.compute(
        hpts, m=m, alpha=alpha, use_local_selection_model=False)
    np.testing.assert_array_almost_equal(pi_hpts, expected_pi, decimal=4)

  def test_constant_sum_transition_matrix(self):
    """Tests closed-form transition matrix computation for constant-sum case."""

    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game)

    # Checks if the game is symmetric and runs single-population analysis if so
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

    m = 20
    alpha = 0.1

    # Case 1) General-sum game computation (slower)
    game_is_constant_sum = False
    use_local_selection_model = False
    payoff_sum = None
    c1, rhos1 = alpharank._get_singlepop_transition_matrix(
        payoff_tables[0], payoffs_are_hpt_format, m, alpha,
        game_is_constant_sum, use_local_selection_model, payoff_sum)

    # Case 2) Constant-sum closed-form computation (faster)
    game_is_constant_sum, payoff_sum = utils.check_is_constant_sum(
        payoff_tables[0], payoffs_are_hpt_format)
    c2, rhos2 = alpharank._get_singlepop_transition_matrix(
        payoff_tables[0], payoffs_are_hpt_format, m, alpha,
        game_is_constant_sum, use_local_selection_model, payoff_sum)

    # Ensure both cases match
    np.testing.assert_array_almost_equal(c1, c2)
    np.testing.assert_array_almost_equal(rhos1, rhos2)

  def test_sweep_pi_vs_alpha_recovers_from_failed_first_compute(self):
    """Sweep should retry rather than crash when the first compute() fails."""
    payoff_tables = [np.asarray([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])]
    calls = {"n": 0}

    def fake_compute(*args, **kwargs):
      calls["n"] += 1
      if calls["n"] == 1:
        # E.g. a reducible Markov chain at the warm-start alpha.
        raise ValueError("Expected 1 stationary distribution, but found 2")
      pi = np.asarray([1. / 3, 1. / 3, 1. / 3])
      return None, None, pi, None, None

    with mock.patch.object(alpharank, "compute", side_effect=fake_compute):
      pi = alpharank.sweep_pi_vs_alpha(payoff_tables, m=20)
    np.testing.assert_allclose(pi, [1. / 3, 1. / 3, 1. / 3])
    self.assertGreater(calls["n"], 1)

  def test_sweep_pi_vs_alpha_recovers_from_failure_after_one_success(self):
    """Sweep should not index past the collected pi history."""
    payoff_tables = [np.asarray([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])]
    calls = {"n": 0}

    def fake_compute(*args, **kwargs):
      calls["n"] += 1
      if calls["n"] == 2:
        # Second compute fails, when there is only one successful iteration.
        raise ValueError("array must not contain infs or NaNs")
      pi = np.asarray([1. / 3, 1. / 3, 1. / 3])
      return None, None, pi, None, None

    with mock.patch.object(alpharank, "compute", side_effect=fake_compute):
      pi = alpharank.sweep_pi_vs_alpha(
          payoff_tables, m=20, warm_start_alpha=100.0)
    np.testing.assert_allclose(pi, [1. / 3, 1. / 3, 1. / 3])
    self.assertGreater(calls["n"], 2)

  def test_sweep_pi_vs_alpha_gives_up_on_repeated_failures(self):
    """A compute() that never succeeds should raise, not shrink alpha forever."""
    payoff_tables = [np.asarray([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])]
    calls = {"n": 0}

    def fake_compute(*args, **kwargs):
      calls["n"] += 1
      # E.g. a payoff table with infs/NaNs in it: no alpha will fix this.
      raise ValueError("array must not contain infs or NaNs")

    with mock.patch.object(alpharank, "compute", side_effect=fake_compute):
      with self.assertRaises(RuntimeError):
        alpharank.sweep_pi_vs_alpha(
            payoff_tables, m=20, warm_start_alpha=100.0)
    self.assertLess(calls["n"], 1000)

  def test_sweep_pi_vs_alpha_gives_up_after_some_successes(self):
    """Same, but the failures start once a few iterations have gone through."""
    payoff_tables = [np.asarray([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])]
    calls = {"n": 0}

    def fake_compute(*args, **kwargs):
      calls["n"] += 1
      if calls["n"] > 2:
        raise ValueError("array must not contain infs or NaNs")
      # Not converged, so the sweep keeps going after these two.
      pi = np.asarray([0.2 + 0.01 * calls["n"], 0.3, 0.47])
      return None, None, pi, None, None

    with mock.patch.object(alpharank, "compute", side_effect=fake_compute):
      with self.assertRaises(RuntimeError):
        alpharank.sweep_pi_vs_alpha(
            payoff_tables, m=20, warm_start_alpha=10.0)
    self.assertLess(calls["n"], 1000)


if __name__ == "__main__":
  absltest.main()
