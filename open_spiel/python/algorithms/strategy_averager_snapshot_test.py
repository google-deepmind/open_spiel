# Copyright 2022 DeepMind Technologies Limited
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

"""Sliding strategy averages must retain each submitted strategy value."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.algorithms import nfg_utils
from open_spiel.python.algorithms import regret_matching


class StrategyAveragerSnapshotTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 4, 10)
  def test_reused_arrays_match_saved_window_values(self, window):
    averager = nfg_utils.StrategyAverager(2, [2, 3], window)
    strategies = [np.zeros(2), np.zeros(3)]
    history = []
    for step in range(7):
      strategies[0][:] = [step / 7, 1 - step / 7]
      strategies[1][:] = np.array([step + 1.0, 2.0, 3.0]) / (step + 6)
      history.append([s.copy() for s in strategies])
      averager.append(strategies)
      for actual, player in zip(averager.average_strategies(), range(2)):
        expected = np.mean([s[player] for s in history[-window:]], axis=0)
        np.testing.assert_allclose(actual, expected)
      strategies[0][:] = [-100.0, 100.0]
      strategies[1][:] = 5.0
      for player, actual in enumerate(averager.average_strategies()):
        np.testing.assert_allclose(
            actual, np.mean([s[player] for s in history[-window:]], axis=0)
        )

  def test_replacing_outer_list_entries_does_not_replace_history(self):
    strategies = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    averager = nfg_utils.StrategyAverager(2, [2, 2], 3)
    averager.append(strategies)
    strategies[:] = [np.array([0.5, 0.5]), np.array([0.25, 0.75])]
    averager.append(strategies)
    first, second = averager.average_strategies()
    np.testing.assert_allclose(first, [0.75, 0.25])
    np.testing.assert_allclose(second, [0.125, 0.875])

  def test_views_do_not_alias_their_external_backing_buffer(self):
    backing = np.array([[1.0, 9.0, 0.0, 9.0], [0.0, 9.0, 1.0, 9.0]])
    views = (backing[0, ::2], backing[1, ::2])
    averager = nfg_utils.StrategyAverager(2, [2, 2], 2)
    averager.append(views)
    backing[:] = -4.0
    first, second = averager.average_strategies()
    np.testing.assert_array_equal(first, [1.0, 0.0])
    np.testing.assert_array_equal(second, [0.0, 1.0])

  def test_python_list_probabilities_are_snapshotted(self):
    strategies = [[0.2, 0.8]]
    averager = nfg_utils.StrategyAverager(1, [2], 2)
    averager.append(strategies)
    strategies[0][:] = [0.8, 0.2]
    averager.append(strategies)
    np.testing.assert_allclose(averager.average_strategies()[0], [0.5, 0.5])

  def test_window_and_full_history_agree_during_in_place_regret_updates(self):
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    strategies = [np.array([0.8, 0.2]), np.array([0.3, 0.7])]
    regrets = [np.zeros(2), np.zeros(2)]
    full = nfg_utils.StrategyAverager(2, [2, 2])
    window = nfg_utils.StrategyAverager(2, [2, 2], 20)
    full.append(strategies)
    window.append(strategies)
    for _ in range(10):
      updated = regret_matching._regret_matching_step(
          [payoff, -payoff], strategies, regrets, 0.01
      )
      for previous, new in zip(strategies, updated):
        previous[:] = new
      full.append(strategies)
      window.append(strategies)
    for actual, expected in zip(
        window.average_strategies(), full.average_strategies()
    ):
      np.testing.assert_allclose(actual, expected)

  @parameterized.parameters(None, 2)
  def test_returned_average_and_input_are_independent(self, window):
    original = [np.array([0.25, 0.75])]
    averager = nfg_utils.StrategyAverager(1, [2], window)
    averager.append(original)
    result = averager.average_strategies()
    result[0][:] = 0.0
    np.testing.assert_array_equal(original[0], [0.25, 0.75])
    np.testing.assert_array_equal(
        averager.average_strategies()[0], [0.25, 0.75]
    )


if __name__ == '__main__':
  absltest.main()
