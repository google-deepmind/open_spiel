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

"""Tests for open_spiel.python.algorithms.action_value_vs_best_response.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import action_value_vs_best_response
import pyspiel


class ActionValuesVsBestResponseTest(absltest.TestCase):

  def test_kuhn_poker_uniform(self):
    game = pyspiel.load_game("kuhn_poker")
    calc = action_value_vs_best_response.Calculator(game)
    (expl, avvbr, cfrp,
     player_reach_probs) = calc(0, policy.UniformRandomPolicy(game),
                                ["0", "1", "2", "0pb", "1pb", "2pb"])
    self.assertAlmostEqual(expl, 15 / 36)
    np.testing.assert_allclose(
        avvbr,
        [
            [-1.5, -2.0],  # 0 (better to pass)
            [-0.5, -0.5],  # 1 (same)
            [0.5, 1.5],  # 2 (better to bet)
            [-1.0, -2.0],  # 0pb - losing
            [-1.0, 0.0],  # 1pb - best response is bet always
            [-1.0, 2.0],  # 2pb - winning
        ])
    np.testing.assert_allclose(cfrp, [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose([1, 1, 1, 1 / 2, 1 / 2, 1 / 2],
                               player_reach_probs)

  def test_kuhn_poker_always_pass_p0(self):
    game = pyspiel.load_game("kuhn_poker")
    calc = action_value_vs_best_response.Calculator(game)
    (expl, avvbr, cfrp, player_reach_probs) = calc(
        0, policy.FirstActionPolicy(game),
        ["0", "1", "2", "0pb", "1pb", "2pb"])
    self.assertAlmostEqual(expl, 1.)
    np.testing.assert_allclose(
        avvbr,
        [
            # Opening bet. If we pass, we always lose (pass-pass with op's K,
            # otherwise pass-bet-pass).
            # If we bet, we always win (because op's best response is to pass,
            # because this is an unreachable state and we break ties in favour
            # of the lowest action).
            [-1, 1],
            [-1, 1],
            [-1, 1],
            # We pass, opp bets into us. This can be either J or Q (K will pass
            # because of the tie-break rules).
            # So we are guaranteed to be winning with Q or K.
            [-1, -2],  # 0pb
            [-1, 2],  # 1pb
            [-1, 2],  # 2pb
        ])
    np.testing.assert_allclose(cfrp, [1 / 3, 1 / 3, 1 / 3, 1 / 6, 1 / 6, 1 / 3])
    np.testing.assert_allclose([1., 1., 1., 1., 1., 1.], player_reach_probs)


if __name__ == "__main__":
  absltest.main()
