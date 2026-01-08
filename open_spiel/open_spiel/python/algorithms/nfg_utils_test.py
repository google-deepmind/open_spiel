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

from absl.testing import absltest

import numpy as np
from open_spiel.python.algorithms import nfg_utils


class NfgUtilsTest(absltest.TestCase):

  def test_strategy_averager_len_smaller_than_window(self):
    averager = nfg_utils.StrategyAverager(2, [2, 2], window_size=50)
    averager.append([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    averager.append([np.array([0.0, 1.0]), np.array([1.0, 0.0])])
    avg_strategies = averager.average_strategies()
    self.assertLen(avg_strategies, 2)
    self.assertAlmostEqual(avg_strategies[0][0], 0.5)
    self.assertAlmostEqual(avg_strategies[0][1], 0.5)
    self.assertAlmostEqual(avg_strategies[1][0], 0.5)
    self.assertAlmostEqual(avg_strategies[1][1], 0.5)

  def test_strategy_averager(self):
    first_action_strat = np.array([1.0, 0.0])
    second_action_strat = np.array([0.0, 1.0])
    averager_full = nfg_utils.StrategyAverager(2, [2, 2])
    averager_window5 = nfg_utils.StrategyAverager(2, [2, 2], window_size=5)
    averager_window6 = nfg_utils.StrategyAverager(2, [2, 2], window_size=6)
    for _ in range(5):
      averager_full.append([first_action_strat, first_action_strat])
      averager_window5.append([first_action_strat, first_action_strat])
      averager_window6.append([first_action_strat, first_action_strat])
    for _ in range(5):
      averager_full.append([second_action_strat, second_action_strat])
      averager_window5.append([second_action_strat, second_action_strat])
      averager_window6.append([second_action_strat, second_action_strat])
    avg_full = averager_full.average_strategies()
    avg_window5 = averager_window5.average_strategies()
    avg_window6 = averager_window6.average_strategies()
    self.assertAlmostEqual(avg_full[0][1], 0.5)
    self.assertAlmostEqual(avg_window5[0][1], 5.0 / 5.0)
    self.assertAlmostEqual(avg_window6[0][1], 5.0 / 6.0)


if __name__ == '__main__':
  absltest.main()

