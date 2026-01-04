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

"""RNN meta-regret matching with self-play agents."""

from typing import List

from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python.examples.meta_cfr.matrix_games.rnn_model import RNNModel

FLAGS = flags.FLAGS


def _make_network(lstm_hidden_sizes: List[int],
                  mlp_hidden_sizes: List[int],
                  output_dim: int) -> hk.RNNCore:
  """set up the network."""

  layers = []
  for k, hidden_size in enumerate(lstm_hidden_sizes):
    layers += [hk.LSTM(hidden_size, name=f'lstm_layer_{k}'), jax.nn.relu]
  layers += [hk.nets.MLP(mlp_hidden_sizes + [output_dim], name='mlp')]
  return RNNModel(layers)


def _make_forwards(lstm_hidden_sizes: List[int], mlp_hidden_sizes: List[int],
                   output_dim: int, batch_size: int) -> hk.Transformed:

  """Forward pass."""

  def forward_fn(inputs):
    rnn = _make_network(lstm_hidden_sizes, mlp_hidden_sizes, output_dim)
    initial_state = rnn.initial_state(batch_size=batch_size)
    outputs, _ = hk.dynamic_unroll(rnn, inputs, initial_state, time_major=False)
    return outputs

  network = hk.transform(forward_fn)
  return network


def meta_loss(opt_params, net_apply, payoff, steps, rng):
  """Meta loss function."""

  regret_sum_x = np.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])
  regret_sum_y = np.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])
  total_loss = 0

  @jax.jit
  def body_fun(s, total_loss):
    nonlocal regret_sum_x
    nonlocal regret_sum_y
    x = net_apply(opt_params, rng, regret_sum_x / (s + 1))
    y = net_apply(opt_params, rng, regret_sum_y / (s + 1))

    strategy_x = jax.nn.softmax(x)
    strategy_y = jnp.transpose(jax.nn.softmax(y), [0, 2, 1])

    values_x = jnp.matmul(payoff, strategy_y)
    values_y = -jnp.matmul(strategy_x, payoff)

    value_x = jnp.matmul(jnp.matmul(strategy_x, payoff), strategy_y)
    value_y = -value_x

    curren_regret_x = values_x - value_x
    curren_regret_y = values_y - value_y
    curren_regret_x = jnp.transpose(curren_regret_x, [0, 2, 1])

    regret_sum_x += curren_regret_x
    regret_sum_y += curren_regret_y

    current_loss = jnp.max(
        jax.numpy.concatenate([curren_regret_x, curren_regret_y], axis=2),
        axis=[1, 2])
    total_loss += current_loss
    return total_loss
  def fori_loop(lower, steps, body_fun, total_loss):
    val = total_loss
    for i in range(lower, steps):
      val = body_fun(i, total_loss)
    return val
  total_loss = fori_loop(0, steps, body_fun, total_loss)
  return jnp.mean(total_loss)


class OptimizerModel:
  """Optimizer model."""

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
    self.model = _make_forwards(
        lstm_hidden_sizes=[20],
        mlp_hidden_sizes=[],
        output_dim=3,
        batch_size=FLAGS.batch_size)
    self.net_apply = self.model.apply
    self.net_init = self.model.init
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
    self.net_params = self.net_init(rng, dummy_input)
    self.opt_state = opt_init(self.net_params)


class MetaSelfplayAgent:
  """Meta player agent."""

  def __init__(self, repeats, training_epochs, data_loader):
    self.repeats = repeats
    self.training_epochs = training_epochs
    self.net_apply = None
    self.net_params = None
    self.regret_sum = None
    self.step = 0
    self.data_loader = data_loader
    self._rng = hk.PRNGSequence(10)

  def train(self):
    self.training_optimizer()
    self.regret_sum = jnp.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])

  def initial_policy(self):
    x = self.net_apply(self.net_params, next(self._rng), self.regret_sum)
    self.last_policy = jax.nn.softmax(x)
    self.step += 1
    return self.last_policy

  def next_policy(self, last_values):
    value = jnp.matmul(self.last_policy, last_values)
    curren_regret = jnp.transpose(last_values, [0, 2, 1]) - value
    self.regret_sum += curren_regret

    x = self.net_apply(self.net_params, next(self._rng),
                       self.regret_sum / (self.step + 1))
    self.last_policy = jax.nn.softmax(x)
    self.step += 1
    return self.last_policy

  def training_optimizer(self):
    """Train optimizer."""

    optimizer = OptimizerModel(0.01)
    optimizer.get_optimizer_model()
    for _ in range(FLAGS.num_batches):
      batch_payoff = next(self.data_loader)
      for _ in range(self.repeats):
        grads = jax.grad(
            meta_loss, has_aux=False)(optimizer.net_params, optimizer.net_apply,
                                      batch_payoff, self.training_epochs,
                                      next(self._rng))

        updates, optimizer.opt_state = optimizer.opt_update(
            grads, optimizer.opt_state)
        optimizer.net_params = optax.apply_updates(optimizer.net_params,
                                                   updates)
    self.net_apply = optimizer.net_apply
    self.net_params = optimizer.net_params
