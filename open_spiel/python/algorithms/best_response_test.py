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

# Lint as: python3
"""Tests for google3.third_party.open_spiel.python.algorithms.best_response."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from open_spiel.python import policy
from open_spiel.python.algorithms import best_response as pyspiel_best_response
import pyspiel


class BestResponseTest(unittest.TestCase):

  def test_best_response_is_a_policy(self):
    game = pyspiel.load_game("kuhn_poker")
    test_policy = policy.UniformRandomPolicy(game)
    best_response = pyspiel_best_response.BestResponsePolicy(
        game, policy=test_policy, player_id=0)
    expected_policy = {
        "0": 1,  # Bet in case opponent folds when winning
        "1": 1,  # Bet in case opponent folds when winning
        "2": 0,  # Both equally good (we return the lowest action)
        # Some of these will never happen under the best-response policy,
        # but we have computed best-response actions anyway.
        "0pb": 0,  # Fold - we're losing
        "1pb": 1,  # Call - we're 50-50
        "2pb": 1,  # Call - we've won
    }
    self.assertEqual(
        expected_policy, {
            key: best_response.best_response_action(key)
            for key in expected_policy.keys()
        })


if __name__ == "__main__":
  unittest.main()
