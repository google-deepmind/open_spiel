# Copyright 2022 DeepMind Technologies Limited
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

"""Tests for Munchausen deep online mirror descent."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import munchausen_deep_mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
import pyspiel


class DeepOnlineMirrorDescentTest(parameterized.TestCase):

  @parameterized.named_parameters(('cpp', 'mfg_crowd_modelling'),
                                  ('python', 'python_mfg_crowd_modelling'))
  def test_train(self, name):
    """Checks that the training works."""
    game = pyspiel.load_game(name)
    assert game.num_players() == 1
    uniform_policy = policy.UniformRandomPolicy(game)
    uniform_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(
        game, mfg_distribution=uniform_dist, mfg_population=0)
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    np.random.seed(0)
    args = {
        'alpha': 0.9,
        'batch_size': 128,
        'discount_factor': 1.0,
        'epsilon_decay_duration': 20000000,
        'epsilon_end': 0.1,
        'epsilon_start': 0.1,
        'gradient_clipping': 40,
        'hidden_layers_sizes': [128, 128],
        'learn_every': 64,
        'learning_rate': 0.01,
        'loss': 'mse',
        'min_buffer_size_to_learn': 500,
        'optimizer': 'adam',
        'replay_buffer_capacity': 2000,
        'tau': 10,
        'update_target_network_every': 50
    }
    agent = munchausen_deep_mirror_descent.MunchausenDQN(
        0, info_state_size, num_actions, **args)
    md = munchausen_deep_mirror_descent.DeepOnlineMirrorDescent(
        game, [env], [agent], num_episodes_per_iteration=100)
    for _ in range(10):
      md.iteration()
    nash_conv_md = nash_conv.NashConv(game, md.policy)
    self.assertLessEqual(nash_conv_md.nash_conv(), 3)


if __name__ == '__main__':
  absltest.main()
