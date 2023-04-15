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

"""Evaluation."""

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS


@jax.jit
def compute_best_response_strategy(utility):
  actions_count = utility.shape[-1]
  opponent_action = jnp.argmin(utility, axis=-1)
  opponent_strategy = jax.nn.one_hot(opponent_action, actions_count)
  return opponent_strategy


@jax.jit
def compute_values_against_best_response(strategy, payoff):
  utility = jnp.matmul(strategy, payoff)
  br_strategy = compute_best_response_strategy(utility)
  return jnp.matmul(payoff, jnp.transpose(br_strategy))


def evaluate_against_best_response(agent, payoff_batch, steps_count):
  """Evaluation against best response agent.

  Args:
    agent: Agent model.
    payoff_batch: Payoff matrix.
    steps_count: Number of steps.
  """
  current_policy = agent.initial_policy()
  values = jax.vmap(compute_values_against_best_response)(current_policy,
                                                          payoff_batch)
  for step in range(steps_count):
    current_policy = agent.next_policy(values)
    values = jax.vmap(compute_values_against_best_response)(current_policy,
                                                            payoff_batch)
    values = jnp.transpose(values, [0, 1, 2])
    value = jnp.matmul(current_policy, values)

    for i in range(value.shape[0]):
      print(step, np.mean(np.asarray(value[i])))


def compute_regrets(payoff_batch, strategy_x, strategy_y):
  values_y = -jnp.matmul(strategy_x, payoff_batch)
  values_x = jnp.transpose(
      jnp.matmul(payoff_batch, jnp.transpose(strategy_y, [0, 2, 1])), [0, 2, 1])
  value_x = jnp.matmul(
      jnp.matmul(strategy_x, payoff_batch),
      jnp.transpose(strategy_y, [0, 2, 1]))
  value_y = -value_x
  regrets_x = values_x - value_x
  regrets_y = values_y - value_y
  return regrets_x, regrets_y


def evaluate_in_selfplay(agent_x, agent_y, payoff_batch, steps_count):
  """Evalute in selfplay.

  Args:
    agent_x: First agent.
    agent_y: Second agent.
    payoff_batch: Payoff matrix.
    steps_count: Number of steps.
  """
  payoff_batch_size = payoff_batch.shape[0]

  regret_sum_x = np.zeros(shape=[payoff_batch_size, 1, FLAGS.num_actions])
  regret_sum_y = np.zeros(shape=[payoff_batch_size, 1, FLAGS.num_actions])
  strategy_x = agent_x.initial_policy()
  strategy_y = agent_y.initial_policy()

  regrets_x, regrets_y = compute_regrets(payoff_batch, strategy_x, strategy_y)
  regret_sum_x += regrets_x
  regret_sum_y += regrets_y
  for s in range(steps_count):
    values_y = -jnp.matmul(strategy_x, payoff_batch)
    values_x = jnp.transpose(
        jnp.matmul(payoff_batch, jnp.transpose(strategy_y, [0, 2, 1])),
        [0, 2, 1])

    values_x = jnp.transpose(values_x, [0, 2, 1])
    values_y = jnp.transpose(values_y, [0, 2, 1])
    strategy_x = agent_x.next_policy(values_x)
    strategy_y = agent_y.next_policy(values_y)

    regrets_x, regrets_y = compute_regrets(payoff_batch, strategy_x, strategy_y)
    regret_sum_x += regrets_x
    regret_sum_y += regrets_y
    print(
        jnp.mean(
            jnp.max(
                jnp.concatenate([regret_sum_x, regret_sum_y], axis=2),
                axis=[1, 2]) / (s + 1)))
