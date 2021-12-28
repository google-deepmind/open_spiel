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
from open_spiel.python.algorithms import external_sampling_mccfr
import pyspiel

SEED = 39823987


class ExternalSamplingMCCFRTest(absltest.TestCase):

  def test_external_sampling_leduc_2p_simple(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))
    self.assertLess(conv, 5)
    # ensure that to_tabular() works on the returned policy and
    # the tabular policy is equivalent
    tabular_policy = es_solver.average_policy().to_tabular()
    conv2 = exploitability.nash_conv(game, tabular_policy)
    self.assertEqual(conv, conv2)

  def test_external_sampling_leduc_2p_full(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("leduc_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.FULL)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Leduc2P, conv = {}".format(conv))
    self.assertLess(conv, 5)

  def test_external_sampling_kuhn_2p_simple(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Kuhn2P, conv = {}".format(conv))
    self.assertLess(conv, 1)

  def test_external_sampling_kuhn_2p_full(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.FULL)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Kuhn2P, conv = {}".format(conv))
    self.assertLess(conv, 1)

  # Liar's dice takes too long, so disable this test. Leave code for reference.
  # pylint: disable=g-unreachable-test-method
  def disabled_test_external_sampling_liars_dice_2p_simple(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("liars_dice")
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(1):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Liar's dice, conv = {}".format(conv))
    self.assertLess(conv, 2)

  def test_external_sampling_kuhn_3p_simple(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker", {"players": 3})
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Kuhn3P, conv = {}".format(conv))
    self.assertLess(conv, 2)

  def test_external_sampling_kuhn_3p_full(self):
    np.random.seed(SEED)
    game = pyspiel.load_game("kuhn_poker", {"players": 3})
    es_solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.FULL)
    for _ in range(10):
      es_solver.iteration()
    conv = exploitability.nash_conv(game, es_solver.average_policy())
    print("Kuhn3P, conv = {}".format(conv))
    self.assertLess(conv, 2)


if __name__ == "__main__":
  absltest.main()
