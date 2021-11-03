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

"""Tests for dqn_policies."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.jax import dqn
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import dqn_policies
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
import pyspiel


class DQNPoliciesTest(parameterized.TestCase):

  @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'),
                                  ('cpp', 'mfg_crowd_modelling'))
  def test_dqn_policies(self, name):
    """Check if DQNPolicies works as expected.

    Args:
      name: Name of the game.
    """
    game = pyspiel.load_game(name)
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)

    envs = [
          rl_environment.Environment(game, distribution=dist, mfg_population=p)
          for p in range(game.num_players())
      ]
      
    info_state_size = envs[0].observation_spec()["info_state"][0]
    num_actions = envs[0].action_spec()["num_actions"]
    hidden_layers_sizes = [128, 128]

    joint_avg_policy = dqn_policies.DQNPolicies(envs, info_state_size, num_actions, hidden_layers_sizes)
    dbr_value = policy_value.PolicyValue(game, dist, joint_avg_policy)

    dbr_val = dbr_value(game.new_initial_state())
    self.assertEqual(dbr_val, 26.135153373673624)


if __name__ == '__main__':
  absltest.main()
