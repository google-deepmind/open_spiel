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

"""Tests for open_spiel.python.algorithms.policy_value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

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


if __name__ == "__main__":
  absltest.main()
