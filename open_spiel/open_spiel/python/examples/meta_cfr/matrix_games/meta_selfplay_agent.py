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

"""Meta-regret matching with self-play agents."""
from typing import List

from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python.examples.meta_cfr.matrix_games import utils

FLAGS = flags.FLAGS


def opponent_best_response_strategy(utility):
  opponent_action = jnp.argmin(utility, axis=-1)
  opponent_strategy = jax.nn.one_hot(opponent_action, FLAGS.num_actions)
  return opponent_strategy


def _mlp_forwards(mlp_hidden_sizes: List[int]) -> hk.Transformed:
  """Returns a haiku transformation of the MLP model to be used in optimizer.

  Args:
    mlp_hidden_sizes: List containing size of linear layers.

  Returns:
    Haiku transformation of the RNN network.
  """
  def forward_fn(inputs):
    mlp = hk.nets.MLP(mlp_hidden_sizes, activation=jax.nn.relu, name="mlp")
    return mlp(inputs)
  return hk.transform(forward_fn)


class OptimizerModel:
  """Optimizer model."""

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

    self.model = _mlp_forwards([64, 16, FLAGS.num_actions])

    self._net_init = self.model.init
    self.net_apply = self.model.apply

    self.opt_update, self.net_params, self.opt_state = None, None, None

  def lr_scheduler(self, init_value):
    schedule_fn = optax.polynomial_schedule(
        init_value=init_value, end_value=0.05, power=1., transition_steps=50)
    return schedule_fn

  def get_optimizer_model(self):
    schedule_fn = self.lr_scheduler(self.learning_rate)
    opt_init, self.opt_update = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn),
        optax.scale(-self.learning_rate))
    rng = jax.random.PRNGKey(10)
    dummy_input = np.random.normal(
        loc=0, scale=10., size=(FLAGS.batch_size, 1, FLAGS.num_actions))
    self.net_params = self._net_init(rng, dummy_input)
    self.opt_state = opt_init(self.net_params)


class MetaSelfplayAgent:
  """Meta player."""

  def __init__(self, repeats, training_epochs, data_loader):
    self.repeats = repeats
    self.training_epochs = training_epochs
    self.net_apply = None
    self.net_params = None
    self.regret_sum = None
    self.step = 0
    self.data_loader = data_loader

  def train(self):
    self.training_optimizer()
    self.regret_sum = jnp.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])

  def initial_policy(self):
    x = self.net_apply(self.net_params, None, self.regret_sum)
    self.last_policy = jax.nn.softmax(x)
    self.step += 1
    return self.last_policy

  def next_policy(self, last_values):
    value = jnp.matmul(self.last_policy, last_values)
    curren_regret = jnp.transpose(last_values, [0, 2, 1]) - value
    self.regret_sum += curren_regret

    x = self.net_apply(self.net_params, None, self.regret_sum / (self.step + 1))
    self.last_policy = jax.nn.softmax(x)
    self.step += 1
    return self.last_policy

  def training_optimizer(self):
    """Training optimizer."""

    optimizer = OptimizerModel(0.01)
    optimizer.get_optimizer_model()

    for _ in range(FLAGS.num_batches):
      batch_payoff = next(self.data_loader)
      # for _ in range(self.repeats):
      grads = jax.grad(
          utils.meta_loss,
          has_aux=False)(optimizer.net_params, optimizer.net_apply,
                         batch_payoff, self.training_epochs)

      updates, optimizer.opt_state = optimizer.opt_update(
          grads, optimizer.opt_state)
      optimizer.net_params = optax.apply_updates(optimizer.net_params, updates)

    self.net_apply = optimizer.net_apply
    self.net_params = optimizer.net_params
