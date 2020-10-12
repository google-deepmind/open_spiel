# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
import pyspiel

SEED = 39823987


class OutcomeSamplingMCCFRTest(absltest.TestCase):

    def test_outcome_sampling_leduc_2p_simple(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("leduc_poker")
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_simple')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Leduc2P, conv = {}".format(conv))
        self.assertLess(conv, 2.5)

    def test_outcome_sampling_leduc_2p_full(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("leduc_poker")
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_full')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Leduc2P, conv = {}".format(conv))
        self.assertLess(conv, 2.5)

    def test_outcome_sampling_kuhn_2p_simple(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("kuhn_poker")
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_simple')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Kuhn2P, conv = {}".format(conv))
        self.assertLess(conv, 0.05)

    def test_outcome_sampling_kuhn_2p_full(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("kuhn_poker")
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_full')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Kuhn2P, conv = {}".format(conv))
        self.assertLess(conv, 0.05)

    def test_outcome_sampling_kuhn_3p_simple(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("kuhn_poker",
                                 {"players": pyspiel.GameParameter(3)})
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_simple')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Kuhn3P, conv = {}".format(conv))
        self.assertLess(conv, 0.1)

    def test_outcome_sampling_kuhn_3p_full(self):
        np.random.seed(SEED)
        game = pyspiel.load_game("kuhn_poker",
                                 {"players": pyspiel.GameParameter(3)})
        os_solver = external_sampling_mccfr.ExternalSamplingSolver(game, 'k_full')
        for _ in range(1000):
            os_solver.iteration()
        conv = exploitability.nash_conv(
            game,
            policy.tabular_policy_from_callable(game,
                                                os_solver.callable_avg_policy()))
        print("Kuhn3P, conv = {}".format(conv))
        self.assertLess(conv, 0.1)


if __name__ == "__main__":
    absltest.main()
