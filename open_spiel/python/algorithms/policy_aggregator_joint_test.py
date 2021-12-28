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

"""Tests for open_spiel.python.algorithms.policy_aggregator_joint."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_aggregator_joint


class JointPolicyAggregatorTest(parameterized.TestCase):

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
    num_players = 2
    num_joint_policies = 4

    joint_policies = [[
        policy.UniformRandomPolicy(env.game) for _ in range(num_players)
    ] for _ in range(num_joint_policies)]
    probabilities = np.ones(len(joint_policies))
    probabilities /= np.sum(probabilities)

    pol_ag = policy_aggregator_joint.JointPolicyAggregator(env.game)
    aggr_policy = pol_ag.aggregate([0, 1], joint_policies, probabilities)

    self.assertLen(aggr_policy.policies, num_players)
    for player in range(num_players):
      player_policy = aggr_policy.policies[player]
      self.assertNotEmpty(player_policy)
      for state_action_probs in player_policy.values():
        probs = list(state_action_probs.values())
        expected_prob = 1. / len(probs)
        for prob in probs:
          self.assertEqual(expected_prob, prob)


if __name__ == "__main__":
  absltest.main()
