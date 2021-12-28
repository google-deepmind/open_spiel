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

# Lint as python3
"""Tests for Python Kuhn Poker."""

from absl.testing import absltest
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import sequence_form_lp
from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python.games import kuhn_poker  # pylint: disable=unused-import
from open_spiel.python.observation import make_observation
import pyspiel


class KuhnPokerTest(absltest.TestCase):

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_kuhn_poker")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_consistent(self):
    """Checks the Python and C++ game implementations are the same."""
    py_game = pyspiel.load_game("python_kuhn_poker")
    cc_game = pyspiel.load_game("kuhn_poker")
    obs_types = [None, pyspiel.IIGObservationType(perfect_recall=True)]
    py_observations = [make_observation(py_game, o) for o in obs_types]
    cc_observations = [make_observation(cc_game, o) for o in obs_types]
    py_states = get_all_states(py_game)
    cc_states = get_all_states(cc_game)
    self.assertCountEqual(list(cc_states), list(py_states))
    for key, cc_state in cc_states.items():
      py_state = py_states[key]
      np.testing.assert_array_equal(py_state.history(), cc_state.history())
      np.testing.assert_array_equal(py_state.returns(), cc_state.returns())
      for py_obs, cc_obs in zip(py_observations, cc_observations):
        for player in (0, 1):
          py_obs.set_from(py_state, player)
          cc_obs.set_from(cc_state, player)
          np.testing.assert_array_equal(py_obs.tensor, cc_obs.tensor)

  def test_nash_value_sequence_form_lp(self):
    """Checks Nash value using a Python sequence form LP solver."""
    game = pyspiel.load_game("python_kuhn_poker")
    val1, val2, _, _ = sequence_form_lp.solve_zero_sum_game(game)
    # value from Kuhn 1950 or https://en.wikipedia.org/wiki/Kuhn_poker
    self.assertAlmostEqual(val1, -1 / 18)
    self.assertAlmostEqual(val2, +1 / 18)

  def test_exploitability_uniform_random_py(self):
    """Checks the exploitability of the uniform random policy using Python."""
    # NashConv of uniform random test_policy from (found on Google books):
    # https://link.springer.com/chapter/10.1007/978-3-319-75931-9_5
    game = pyspiel.load_game("python_kuhn_poker")
    test_policy = policy.UniformRandomPolicy(game)
    expected_nash_conv = 11 / 12
    self.assertAlmostEqual(
        exploitability.exploitability(game, test_policy),
        expected_nash_conv / 2)

  def test_exploitability_uniform_random_cc(self):
    """Checks the exploitability of the uniform random policy using C++."""
    game = pyspiel.load_game("python_kuhn_poker")
    test_policy = pyspiel.UniformRandomPolicy(game)
    expected_nash_conv = 11 / 12
    self.assertAlmostEqual(
        pyspiel.exploitability(game, test_policy), expected_nash_conv / 2)


if __name__ == "__main__":
  absltest.main()
