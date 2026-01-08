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

"""Utility functions for meta learning for regret minimization."""

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS


def meta_loss(opt_params, net_apply, payoff, steps):

  """Returns the meta learning loss value.

  Args:
    opt_params: Optimizer parameters.
    net_apply: Apply function.
    payoff: Payoff matrix.
    steps: Number of steps.

  Returns:
    Accumulated loss value over number of steps.

  """
  regret_sum_x = np.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])
  regret_sum_y = np.zeros(shape=[FLAGS.batch_size, 1, FLAGS.num_actions])
  total_loss = 0
  step = 0

  @jax.jit
  def scan_body(carry, x):
    nonlocal regret_sum_x
    nonlocal regret_sum_y
    regret_sum_x, regret_sum_y, current_step, total_loss = carry
    x = net_apply(opt_params, None, regret_sum_x / (current_step + 1))
    y = net_apply(opt_params, None, regret_sum_y / (current_step + 1))

    strategy_x = jax.nn.softmax(x)
    strategy_y = jnp.transpose(jax.nn.softmax(y), [0, 2, 1])

    values_x = jnp.matmul(payoff, strategy_y)  # val_x = payoff * st_y
    values_y = -jnp.matmul(strategy_x, payoff)  # val_y = -1 * payoff * st_x

    value_x = jnp.matmul(jnp.matmul(strategy_x, payoff), strategy_y)
    value_y = -value_x

    curren_regret_x = values_x - value_x
    curren_regret_y = values_y - value_y
    curren_regret_x = jnp.transpose(curren_regret_x, [0, 2, 1])

    regret_sum_x += curren_regret_x
    regret_sum_y += curren_regret_y

    current_loss = jnp.mean(jnp.max(
        jax.numpy.concatenate([curren_regret_x, curren_regret_y], axis=2),
        axis=[1, 2]), axis=-1)
    total_loss += current_loss
    current_step += 1
    return (regret_sum_x, regret_sum_y, current_step, total_loss), None

  (regret_sum_x, regret_sum_y, step, total_loss), _ = jax.lax.scan(
      scan_body,
      (regret_sum_x, regret_sum_y, step, total_loss),
      None,
      length=steps,
  )

  return total_loss
