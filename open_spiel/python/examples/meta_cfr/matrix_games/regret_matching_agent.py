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

"""Regret matching."""
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS


class RegretMatchingAgent:
  """Regret matching agent."""

  def __init__(self, num_actions, data_loader):
    self.num_actions = num_actions
    # self.regret_sum = jax.numpy.array(np.zeros(self.num_actions))
    self.regret_sum = jax.numpy.array(
        np.zeros(shape=[FLAGS.batch_size, 1, self.num_actions]))
    self.data_loader = data_loader

  def train(self):
    pass

  def initial_policy(self):
    self.last_policy = self.regret_matching_policy(self.regret_sum)
    return self.last_policy

  def next_policy(self, last_values):
    value = jnp.matmul(self.last_policy, last_values)
    last_values = jnp.transpose(last_values, [0, 2, 1])
    current_regrets = last_values - value
    self.regret_sum += current_regrets
    self.last_policy = self.regret_matching_policy(self.regret_sum)
    return self.last_policy

  def regret_matching_policy(self, regret_sum):
    """Regret matching policy."""

    strategy = np.copy(regret_sum)
    strategy[strategy < 0] = 0
    strategy_sum = np.sum(strategy, axis=-1)
    for i in range(FLAGS.batch_size):
      if strategy_sum[i] > 0:
        strategy[i] /= strategy_sum[i]
      else:
        strategy[i] = np.repeat(1 / self.num_actions, self.num_actions)
    return strategy
