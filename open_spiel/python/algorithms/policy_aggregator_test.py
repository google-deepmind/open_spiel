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

"""Tests for open_spiel.python.algorithms.policy_aggregator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_aggregator


class PolicyAggregatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "kuhn_poker",
          "game_name": "kuhn_poker"
      }, {
          "testcase_name": "leduc_poker",
          "game_name": "leduc_poker"
      })
  def test_policy_aggregation_random(self, game_name):
    env = rl_environment.Environment(game_name)

    policies = [[policy.UniformRandomPolicy(env.game)
                 for _ in range(2)]
                for _ in range(2)]
    probabilities = [
        list(np.ones(len(policies)) / len(policies)) for _ in range(2)
    ]

    pol_ag = policy_aggregator.PolicyAggregator(env.game)
    aggr_policy = pol_ag.aggregate([0], policies, probabilities)

    for item in aggr_policy.policy[0].items():
      _, probs = zip(*item[1].items())
      const_probs = tuple([probs[0]] * len(probs))
      self.assertEqual(probs, const_probs)

  @parameterized.named_parameters(
      {
          "testcase_name": "kuhn_poker",
          "game_name": "kuhn_poker"
      }, {
          "testcase_name": "leduc_poker",
          "game_name": "leduc_poker"
      })
  def test_policy_aggregation_tabular_randinit(self, game_name):
    env = rl_environment.Environment(game_name)

    mother_policy = policy.TabularPolicy(env.game).copy_with_noise(
        1, 10, np.random.RandomState(0))
    policies = [[mother_policy.__copy__() for _ in range(2)] for _ in range(2)]
    probabilities = [
        list(np.ones(len(policies)) / len(policies)) for _ in range(2)
    ]

    pol_ag = policy_aggregator.PolicyAggregator(env.game)
    aggr_policy = pol_ag.aggregate([0], policies, probabilities)

    for state, value in aggr_policy.policy[0].items():
      polici = mother_policy.policy_for_key(state)

      value_normal = {
          action: probability
          for action, probability in enumerate(polici)
          if probability > 0
      }
      for key in value_normal.keys():
        self.assertAlmostEqual(value[key], value_normal[key], 8)


if __name__ == "__main__":
  unittest.main()
