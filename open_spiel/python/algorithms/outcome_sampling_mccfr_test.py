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

"""Tests for open_spiel.python.algorithms.cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import outcome_sampling_mccfr
import pyspiel

# Convergence results change depending on
# the seed specified for running the tests.
# For this reason, test thresholds have been adapted
# taking the maximum Nash exploitability value obtained
# from multiple runs.
# For more details see https://github.com/deepmind/open_spiel/pull/458
SEED = 39823987


class OutcomeSamplingMCCFRTest(absltest.TestCase):

  def test_outcome_sampling_leduc_2p(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("leduc_poker")
    os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(10000):
      os_solver.iteration()
    conv = exploitability.nash_conv(game, os_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))

    self.assertLess(conv, 3.07)

  def test_outcome_sampling_kuhn_2p(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker")
    os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(10000):
      os_solver.iteration()
    conv = exploitability.nash_conv(game, os_solver.average_policy())
    print("Kuhn2P, conv = {}".format(conv))
    self.assertLess(conv, 0.17)
    # ensure that to_tabular() works on the returned policy
    # and the tabular policy is equivalent
    tabular_policy = os_solver.average_policy().to_tabular()
    conv2 = exploitability.nash_conv(game, tabular_policy)
    self.assertEqual(conv, conv2)

  def test_outcome_sampling_kuhn_3p(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker", {"players": 3})
    os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(10000):
      os_solver.iteration()
    conv = exploitability.nash_conv(game, os_solver.average_policy())
    print("Kuhn3P, conv = {}".format(conv))
    self.assertLess(conv, 0.22)


if __name__ == "__main__":
  absltest.main()
