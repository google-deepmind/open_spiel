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
