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

"""Utilities for PPO: vectorized GAE, metrics tracking, and plotting.

This module provides:
  - compute_gae: Vectorized GAE (Generalized Advantage Estimation) using
    jax.lax.scan in reverse time order. Fully JIT-compatible with no Python
    loops.
  - TrainingMetrics: Dataclass for accumulating training and evaluation metrics.
  - plot_training_curves: Matplotlib-based plotting for training diagnostics.
"""

import dataclasses

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Compute GAE advantages and returns using jax.lax.scan.

  Performs a single reverse-order scan over the trajectory. The scan carry
  propagates the decaying advantage sum from the end of the trajectory to the
  beginning, implementing the recursive GAE formula:

    delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t       = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    return_t  = A_t + V(s_t)

  This function is pure and compatible with jax.jit.

  Args:
    rewards: per-step rewards, shape (T,).
    values: value estimates V(s_t), shape (T,).
    dones: 1.0 if step t is terminal, 0.0 otherwise, shape (T,).
    gamma: discount factor.
    gae_lambda: GAE lambda parameter.

  Returns:
    Tuple of (advantages, returns), each shape (T,).
  """
  next_values = jnp.concatenate([values[1:], jnp.zeros(1)])

  def _scan_step(last_gae, x):
    reward, value, next_value, done = x
    non_terminal = 1.0 - done
    delta = reward + gamma * next_value * non_terminal - value
    new_gae = delta + gamma * gae_lambda * non_terminal * last_gae
    return new_gae, new_gae

  _, advantages = jax.lax.scan(
      _scan_step,
      jnp.float32(0.0),
      (rewards, values, next_values, dones),
      reverse=True,
  )

  returns = advantages + values
  return advantages, returns


@dataclasses.dataclass
class TrainingMetrics:
  """Accumulates training and evaluation metrics for logging and plotting.

  Training metrics (policy_loss, value_loss, entropy, approx_kl) are recorded
  every iteration. Evaluation metrics (exploitability, avg_returns) are recorded
  at eval intervals only.
  """

  iterations: list = dataclasses.field(default_factory=list)
  policy_loss: list = dataclasses.field(default_factory=list)
  value_loss: list = dataclasses.field(default_factory=list)
  entropy: list = dataclasses.field(default_factory=list)
  approx_kl: list = dataclasses.field(default_factory=list)

  eval_iterations: list = dataclasses.field(default_factory=list)
  exploitability: list = dataclasses.field(default_factory=list)
  avg_returns: list = dataclasses.field(default_factory=list)

  def record_train(self, iteration, metrics):
    """Record per-iteration training metrics from agent.learn()."""
    self.iterations.append(iteration)
    self.policy_loss.append(metrics.get("policy_loss", 0.0))
    self.value_loss.append(metrics.get("value_loss", 0.0))
    self.entropy.append(metrics.get("entropy", 0.0))
    self.approx_kl.append(metrics.get("approx_kl", 0.0))

  def record_eval(self, iteration, expl, avg_ret):
    """Record periodic evaluation metrics."""
    self.eval_iterations.append(iteration)
    self.exploitability.append(expl)
    self.avg_returns.append(list(avg_ret))


def plot_training_curves(metrics, game_name, save_path=None):
  """Plot training loss curves, entropy, and exploitability.

  Args:
    metrics: a TrainingMetrics instance.
    game_name: string used in the plot title.
    save_path: if provided, save the figure to this file path instead of
      displaying it.
  """
  # pylint: disable=g-import-not-at-top
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  fig.suptitle(f"PPO Self-Play: {game_name}", fontsize=14)

  axes[0, 0].plot(metrics.iterations, metrics.policy_loss)
  axes[0, 0].set_title("Policy Loss")
  axes[0, 0].set_xlabel("Iteration")
  axes[0, 0].grid(True, alpha=0.3)

  axes[0, 1].plot(metrics.iterations, metrics.value_loss)
  axes[0, 1].set_title("Value Loss")
  axes[0, 1].set_xlabel("Iteration")
  axes[0, 1].grid(True, alpha=0.3)

  axes[1, 0].plot(metrics.iterations, metrics.entropy)
  axes[1, 0].set_title("Entropy")
  axes[1, 0].set_xlabel("Iteration")
  axes[1, 0].grid(True, alpha=0.3)

  if metrics.exploitability:
    axes[1, 1].plot(metrics.eval_iterations, metrics.exploitability)
    axes[1, 1].set_title("Exploitability")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(True, alpha=0.3)
  else:
    axes[1, 1].set_visible(False)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
