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

"""Tests for open_spiel.python.algorithms.policy_value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel


class PolicyValueTest(absltest.TestCase):

  def test_expected_game_score_uniform_random_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    uniform_policy = policy.UniformRandomPolicy(game)
    uniform_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [uniform_policy] * 2)
    self.assertTrue(np.allclose(uniform_policy_values, [1 / 8, -1 / 8]))

  def test_expected_game_score_uniform_random_iterated_prisoner_dilemma(self):
    game = pyspiel.load_game(
        "python_iterated_prisoners_dilemma(max_game_length=6)")
    pi = policy.UniformRandomPolicy(game)
    values = expected_game_score.policy_value(game.new_initial_state(), pi)
    # 4*(1-0.875**6)/0.125 = 17.6385498
    np.testing.assert_allclose(values, [17.6385498, 17.6385498])


if __name__ == "__main__":
  absltest.main()
