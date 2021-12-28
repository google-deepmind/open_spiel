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

"""Example for policy_aggregator_example.

Example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator

FLAGS = flags.FLAGS
flags.DEFINE_string("game_name", "kuhn_poker", "Game name")


class TestPolicy(policy.Policy):

  def __init__(self, action_int):
    self._action_int = action_int

  def action_probabilities(self, state, player_id=None):
    return {self._action_int: 1.0}


def main(unused_argv):
  env = rl_environment.Environment(FLAGS.game_name)

  policies = [[
      policy.TabularPolicy(env.game).copy_with_noise(alpha=float(i), beta=1.0)
      for i in range(2)
  ] for _ in range(2)]  # pylint: disable=g-complex-comprehension

  probabilities = [
      list(np.ones(len(policies[i])) / len(policies[i])) for i in range(2)
  ]

  pol_ag = policy_aggregator.PolicyAggregator(env.game)
  aggr_policies = pol_ag.aggregate([0, 1], policies, probabilities)

  exploitabilities = exploitability.nash_conv(env.game, aggr_policies)
  print("Exploitability : {}".format(exploitabilities))

  print(policies[0][0].action_probability_array)
  print(policies[0][1].action_probability_array)
  print(aggr_policies.policy)

  print("\nCopy Example")

  mother_policy = policy.TabularPolicy(env.game).copy_with_noise(1, 10)
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
    for key in value.keys():
      print("State : {}. Key : {}. Aggregated : {}. Real : {}. Passed : {}"
            .format(state, key, value[key], value_normal[key],
                    np.abs(value[key] - value_normal[key]) < 1e-8))


if __name__ == "__main__":
  app.run(main)
