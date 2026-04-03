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

"""PPO (Proximal Policy Optimization) agent implemented in JAX.

This module implements the clipped PPO algorithm (Schulman et al., 2017) using
Flax NNX for the neural network and optax for optimization. The agent supports
self-play in multi-agent turn-based games: a single agent instance controls all
players by reading the current player from each time_step. Per-player
trajectories are tracked internally and GAE (Generalized Advantage Estimation)
is computed per-player at episode boundaries.

Supported games include any OpenSpiel sequential game accessible through
rl_environment.Environment, e.g. kuhn_poker and leduc_poker.

Note: PPO is a single-policy gradient method and does not have convergence
guarantees in imperfect-information games. For such games, algorithms like CFR
will achieve lower exploitability. This implementation is intended as a clean
reference example for policy gradient methods on OpenSpiel.

See open_spiel/python/examples/ppo_example_jax.py for a usage example.
"""

import collections
from typing import Iterable

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python import rl_agent

ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min


class ActorCriticNetwork(nn.Module):
  """Actor-critic network with a shared trunk, policy head, and value head.

  Architecture:
    trunk:  [Linear(hidden) -> tanh] * len(hidden_sizes)
    policy: Linear(num_actions), orthogonal init std=0.01
    value:  Linear(1), orthogonal init std=1.0
  """

  def __init__(
      self,
      input_size: int,
      num_actions: int,
      hidden_sizes: Iterable[int] = (64, 64),
      seed: int = 0,
  ):
    rngs = nn.Rngs(seed)
    ortho = nn.initializers.orthogonal

    layers = []
    in_size = input_size
    for h_size in hidden_sizes:
      layers.append(
          nn.Linear(
              in_size, h_size, kernel_init=ortho(jnp.sqrt(2)), rngs=rngs))
      layers.append(jax.nn.tanh)
      in_size = h_size
    self.trunk = nn.Sequential(*layers)

    self.policy_head = nn.Linear(
        in_size, num_actions, kernel_init=ortho(0.01), rngs=rngs)
    self.value_head = nn.Linear(
        in_size, 1, kernel_init=ortho(1.0), rngs=rngs)

  def __call__(self, x: chex.Array) -> tuple[chex.Array, chex.Array]:
    hidden = self.trunk(x)
    logits = self.policy_head(hidden)
    value = jnp.squeeze(self.value_head(hidden), axis=-1)
    return logits, value


class RolloutBuffer:
  """Stores processed PPO rollout data consumed during learning.

  Transitions are appended during episode collection (after GAE computation)
  and converted to JAX arrays for the PPO update.
  """

  def __init__(self):
    self.clear()

  def clear(self):
    """Reset all stored data."""
    self.observations = []
    self.actions = []
    self.log_probs = []
    self.values = []
    self.advantages = []
    self.returns = []
    self.legal_masks = []

  @property
  def size(self) -> int:
    return len(self.observations)

  def add(self, obs, action, log_prob, value, advantage, ret, legal_mask):
    """Append a single processed transition to the buffer."""
    self.observations.append(obs)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.values.append(value)
    self.advantages.append(advantage)
    self.returns.append(ret)
    self.legal_masks.append(legal_mask)

  def as_jnp(self) -> dict[str, chex.Array]:
    """Convert buffer contents to JAX arrays for training."""
    return {
        "observations": jnp.array(self.observations),
        "actions": jnp.array(self.actions, dtype=jnp.int32),
        "log_probs": jnp.array(self.log_probs, dtype=jnp.float32),
        "advantages": jnp.array(self.advantages, dtype=jnp.float32),
        "returns": jnp.array(self.returns, dtype=jnp.float32),
        "legal_masks": jnp.array(self.legal_masks),
    }


def compute_gae(transitions, gamma, gae_lambda):
  """Compute GAE advantages and returns for a single-player trajectory.

  Each player's sequence of decisions within one episode forms a trajectory.
  The last transition is terminal (bootstrapped value = 0).

  Args:
    transitions: list of dicts with keys 'value' and 'reward'.
    gamma: discount factor.
    gae_lambda: GAE lambda parameter.

  Returns:
    Tuple of (advantages, returns), each np.ndarray of shape (n,).
  """
  n = len(transitions)
  advantages = np.zeros(n, dtype=np.float32)
  last_gae = 0.0
  for t in reversed(range(n)):
    if t == n - 1:
      next_value = 0.0
      non_terminal = 0.0
    else:
      next_value = transitions[t + 1]["value"]
      non_terminal = 1.0
    delta = (transitions[t]["reward"]
             + gamma * next_value * non_terminal
             - transitions[t]["value"])
    last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
    advantages[t] = last_gae
  values = np.array([t["value"] for t in transitions], dtype=np.float32)
  returns = advantages + values
  return advantages, returns


class PPO(rl_agent.AbstractAgent):
  """PPO Agent implemented in JAX with Flax NNX.

  Supports self-play: a single agent instance controls all players in a
  turn-based game by reading the current player from each time_step.
  Per-player trajectories are tracked internally and GAE is computed
  per-player at episode boundaries.

  Typical usage (self-play):

    agent = PPO(player_id=0, info_state_size=11, num_actions=4)
    for _ in range(num_iterations):
        for _ in range(episodes_per_batch):
            time_step = env.reset()
            while not time_step.last():
                output = agent.step(time_step)
                time_step = env.step([output.action])
                agent.post_step(time_step)
            agent.step(time_step)
        metrics = agent.learn()
  """

  # pylint: disable=g-bare-generic

  def __init__(
      self,
      player_id: int,
      info_state_size: int,
      num_actions: int,
      hidden_sizes: Iterable[int] = (64, 64),
      update_epochs: int = 4,
      num_minibatches: int = 4,
      learning_rate: float = 2.5e-4,
      gamma: float = 0.99,
      gae_lambda: float = 0.95,
      clip_coef: float = 0.2,
      entropy_coef: float = 0.01,
      value_coef: float = 0.5,
      max_grad_norm: float = 0.5,
      normalize_advantages: bool = True,
      seed: int = 42,
  ):
    """Initialize the PPO agent.

    Args:
      player_id: integer player id (used by AbstractAgent interface).
      info_state_size: length of the flattened observation tensor.
      num_actions: total number of distinct actions in the game.
      hidden_sizes: widths of hidden layers in the actor-critic trunk.
      update_epochs: number of passes over the rollout data per learn() call.
      num_minibatches: number of minibatches to split the data into.
      learning_rate: Adam learning rate.
      gamma: discount factor.
      gae_lambda: GAE lambda parameter.
      clip_coef: PPO clipping coefficient (epsilon).
      entropy_coef: entropy bonus coefficient.
      value_coef: value loss coefficient.
      max_grad_norm: maximum gradient norm for clipping.
      normalize_advantages: whether to normalize advantages per minibatch.
      seed: random seed.
    """
    self.player_id = player_id
    self._num_actions = num_actions
    self._info_state_size = info_state_size

    self._gamma = gamma
    self._gae_lambda = gae_lambda
    self._clip_coef = clip_coef
    self._entropy_coef = entropy_coef
    self._value_coef = value_coef
    self._update_epochs = update_epochs
    self._num_minibatches = num_minibatches
    self._max_grad_norm = max_grad_norm
    self._normalize_advantages = normalize_advantages

    self._rng = jax.random.key(seed)

    self._network = ActorCriticNetwork(
        info_state_size, num_actions, tuple(hidden_sizes), seed=seed)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
    self._optimizer = nn.Optimizer(self._network, optimizer, wrt=nn.Param)

    self._graphdef = nn.graphdef((self._network, self._optimizer))
    self._jit_update = self._build_jit_update()

    self._buffer = RolloutBuffer()

    # Per-player episode tracking for self-play GAE computation.
    self._episode_data = collections.defaultdict(list)
    self._pending_rewards = collections.defaultdict(float)

  def _next_rng(self) -> chex.PRNGKey:
    """Split and return the next PRNG subkey."""
    self._rng, subkey = jax.random.split(self._rng)
    return subkey

  def step(self, time_step, is_evaluation=False) -> rl_agent.StepOutput:
    """Select an action given a time_step.

    Reads the current player from the time_step so a single agent can handle
    all players (self-play). At terminal states, processes the completed
    episode and returns a None action.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: if True, skips data collection.

    Returns:
      A StepOutput with the chosen action and action probabilities.
    """
    if time_step.last():
      if not is_evaluation:
        self._process_episode_end()
      return rl_agent.StepOutput(action=None, probs=[])

    current_player = time_step.current_player()
    obs = np.array(
        time_step.observations["info_state"][current_player], dtype=np.float32)
    legal_actions = time_step.observations["legal_actions"][current_player]

    legal_mask = np.zeros(self._num_actions, dtype=np.bool_)
    legal_mask[legal_actions] = True

    logits, value = self._network(jnp.array(obs))
    masked_logits = jnp.where(
        jnp.array(legal_mask), logits, ILLEGAL_ACTION_LOGITS_PENALTY)
    probs = jax.nn.softmax(masked_logits)
    log_probs_all = jax.nn.log_softmax(masked_logits)

    action = int(jax.random.categorical(self._next_rng(), masked_logits))

    if not is_evaluation:
      self._episode_data[current_player].append({
          "obs": obs,
          "action": action,
          "log_prob": float(log_probs_all[action]),
          "value": float(value),
          "legal_mask": legal_mask,
          "reward": self._pending_rewards[current_player],
      })
      self._pending_rewards[current_player] = 0.0

    return rl_agent.StepOutput(action=action, probs=np.array(probs))

  def post_step(self, time_step):
    """Record per-player rewards after an environment step.

    Must be called after env.step() and before the next agent.step().

    Args:
      time_step: the TimeStep returned by env.step().
    """
    if time_step.rewards is not None:
      for pid in range(len(time_step.rewards)):
        self._pending_rewards[pid] += time_step.rewards[pid]

  def _process_episode_end(self):
    """Compute per-player GAE and flush episode data into the buffer."""
    for pid, transitions in self._episode_data.items():
      if not transitions:
        continue
      transitions[-1]["reward"] += self._pending_rewards.get(pid, 0.0)

      advantages, returns = compute_gae(
          transitions, self._gamma, self._gae_lambda)

      for t, trans in enumerate(transitions):
        self._buffer.add(
            obs=trans["obs"],
            action=trans["action"],
            log_prob=trans["log_prob"],
            value=trans["value"],
            advantage=float(advantages[t]),
            ret=float(returns[t]),
            legal_mask=trans["legal_mask"],
        )

    self._episode_data.clear()
    self._pending_rewards.clear()

  def learn(self) -> dict[str, float]:
    """Perform PPO clipped update on collected rollout data.

    Returns:
      Dictionary of average training metrics (policy_loss, value_loss,
      entropy) over all minibatch updates.
    """
    if self._buffer.size == 0:
      return {}

    data = self._buffer.as_jnp()
    batch_size = self._buffer.size
    minibatch_size = max(1, batch_size // self._num_minibatches)

    state = nn.state((self._network, self._optimizer))

    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_ent_loss = 0.0
    num_updates = 0

    for _ in range(self._update_epochs):
      perm = np.random.permutation(batch_size)
      for start in range(0, batch_size - minibatch_size + 1, minibatch_size):
        mb_idx = jnp.array(perm[start:start + minibatch_size])

        mb_adv = data["advantages"][mb_idx]
        if self._normalize_advantages:
          mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        loss, (pg_loss, v_loss, ent_loss), state = self._jit_update(
            state,
            data["observations"][mb_idx],
            data["actions"][mb_idx],
            data["log_probs"][mb_idx],
            mb_adv,
            data["returns"][mb_idx],
            data["legal_masks"][mb_idx],
        )

        total_pg_loss += float(pg_loss)
        total_v_loss += float(v_loss)
        total_ent_loss += float(ent_loss)
        num_updates += 1

    nn.update((self._network, self._optimizer), state)
    self._buffer.clear()

    n = max(num_updates, 1)
    return {
        "policy_loss": total_pg_loss / n,
        "value_loss": total_v_loss / n,
        "entropy": total_ent_loss / n,
    }

  def _build_jit_update(self):
    """Build JIT-compiled PPO minibatch update function."""
    graphdef = self._graphdef
    clip_coef = self._clip_coef
    value_coef = self._value_coef
    entropy_coef = self._entropy_coef

    def _loss_fn(network, obs, actions, old_log_probs, advantages, returns,
                 legal_masks):
      logits, values = network(obs)
      masked_logits = jnp.where(
          legal_masks, logits, ILLEGAL_ACTION_LOGITS_PENALTY)
      log_probs_all = jax.nn.log_softmax(masked_logits)
      new_log_probs = log_probs_all[jnp.arange(actions.shape[0]), actions]

      probs = jax.nn.softmax(masked_logits)
      entropy = -jnp.sum(
          jnp.where(legal_masks, probs * log_probs_all, 0.0), axis=-1)

      ratio = jnp.exp(new_log_probs - old_log_probs)
      pg_loss1 = -advantages * ratio
      pg_loss2 = -advantages * jnp.clip(
          ratio, 1.0 - clip_coef, 1.0 + clip_coef)
      pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

      v_loss = 0.5 * jnp.mean((values - returns) ** 2)
      ent_loss = entropy.mean()

      total_loss = pg_loss + value_coef * v_loss - entropy_coef * ent_loss
      return total_loss, (pg_loss, v_loss, ent_loss)

    @jax.jit
    def update(state, obs, actions, old_log_probs, advantages, returns,
               legal_masks):
      network, optimizer = nn.merge(graphdef, state, copy=True)
      (loss, aux), grads = nn.value_and_grad(_loss_fn, has_aux=True)(
          network, obs, actions, old_log_probs, advantages, returns,
          legal_masks)
      optimizer.update(network, grads)
      return loss, aux, nn.state((network, optimizer))

    return update

  @property
  def network(self) -> ActorCriticNetwork:
    return self._network

  @property
  def buffer_size(self) -> int:
    return self._buffer.size

  @property
  def num_actions(self) -> int:
    return self._num_actions
