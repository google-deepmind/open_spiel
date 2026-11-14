# Copyright 2026 DeepMind Technologies Limited
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

"""Nash Policy Gradient (NashPG) agent implemented in JAX.

See the paper https://arxiv.org/abs/2510.18183 (Yu et al., 2026) for details.

NashPG is a policy-gradient Nash solver: each player runs an independent PPO
update with an extra penalty `magnet_coef * KL(pi || rho)` pulling its policy
toward a magnet (reference) policy `rho` that is periodically re-copied from
the current policy.

This mirrors `open_spiel.python.pytorch.nash_pg` (same constructor arguments
except `device`, same `step`, `update_magnet`, `loss` and `magnet_refreshes`).

Notes:
  - Each player's magnet refreshes on its own schedule. For a synchronized
    refresh, set `auto_update_magnet=False` and call `update_magnet()` yourself.
  - Supports both sequential and simultaneous-move games. However, dense
    rewards in a sequential game need the driver to step every agent at every
    time step. See the example.

Example: see `open_spiel/python/examples/nash_pg_jax_example.py`
"""

from typing import NamedTuple

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python import rl_agent

ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min


class Transition(NamedTuple):
  """One of this player's decisions and what followed it.

  `reward` and `done` cover the interval up to this player's next decision (or
  the end of the episode).
  """

  info_state: np.ndarray
  legal_actions_mask: np.ndarray
  action: int
  logprob: float
  value: float
  reward: float
  done: bool


class LossInfo(NamedTuple):
  """Loss terms of the last minibatch of the most recent update."""

  policy: float
  value: float
  entropy: float
  magnet_kl: float


class ActorCritic(nn.Module):
  """Separate tanh MLPs for the policy logits and the state value."""

  def __init__(
      self,
      info_state_size: int,
      num_actions: int,
      hidden_layers_sizes: tuple[int, ...],
      rngs: nn.Rngs,
  ):
    self.actor = self._mlp(
        info_state_size, hidden_layers_sizes, num_actions, 0.01, rngs
    )
    self.critic = self._mlp(info_state_size, hidden_layers_sizes, 1, 1.0, rngs)

  @staticmethod
  def _mlp(in_size, hidden_layers_sizes, out_size, out_scale, rngs):
    """Orthogonally initialized tanh MLP (the usual PPO initialization)."""
    layers = []
    for size in hidden_layers_sizes:
      layers += [
          nn.Linear(
              in_size,
              size,
              kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
              rngs=rngs,
          ),
          jnp.tanh,
      ]
      in_size = size
    layers.append(
        nn.Linear(
            in_size,
            out_size,
            kernel_init=nn.initializers.orthogonal(out_scale),
            rngs=rngs,
        )
    )
    return nn.Sequential(*layers)

  def __call__(self, info_state: chex.Array, legal_actions_mask: chex.Array):
    """Returns (masked log-probabilities, state value)."""
    logits = jnp.where(
        legal_actions_mask,
        self.actor(info_state),
        ILLEGAL_ACTION_LOGITS_PENALTY,
    )
    return jax.nn.log_softmax(logits, axis=-1), self.critic(info_state)[..., 0]


def masked_entropy(log_probs, legal_actions_mask):
  """Entropy of each row, summing only over legal actions (no 0 * -inf)."""
  p_log_p = jnp.where(legal_actions_mask, jnp.exp(log_probs) * log_probs, 0.0)
  return -p_log_p.sum(axis=-1)


def masked_kl(log_p, log_q, legal_actions_mask):
  """KL(p || q) of each row, summing only over legal actions."""
  terms = jnp.where(legal_actions_mask, jnp.exp(log_p) * (log_p - log_q), 0.0)
  return terms.sum(axis=-1)


def gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
  """GAE(lambda) over a time-ordered sequence of one player's decisions."""
  next_values = jnp.append(values[1:], bootstrap_value)
  not_done = 1.0 - dones
  deltas = rewards + gamma * next_values * not_done - values

  def body(carry, x):
    delta, nd = x
    carry = delta + gamma * gae_lambda * nd * carry
    return carry, carry

  _, advantages = jax.lax.scan(
      body, jnp.zeros_like(bootstrap_value), (deltas, not_done), reverse=True
  )
  return advantages


class NashPG(rl_agent.AbstractAgent):
  """Nash Policy Gradient agent.

  Note: Unlike the paper, which shares one network between players,
  each player here has its own policy/value network.
  """

  def __init__(
      self,
      player_id: int,
      info_state_size: int,
      num_actions: int,
      hidden_layers_sizes=(128, 128),
      batch_size: int = 1024,
      update_epochs: int = 4,
      num_minibatches: int = 4,
      learning_rate: float = 1e-3,
      magnet_coef: float = 0.2,
      magnet_update_period: int = 40,
      auto_update_magnet: bool = True,
      gamma: float = 1.0,
      gae_lambda: float = 0.95,
      clip_coef: float = 0.2,
      entropy_coef: float = 0.1,
      value_coef: float = 0.5,
      max_grad_norm: float = 0.5,
      seed: int = 42,
      name: str = "nash_pg",
  ):
    """Initializes the agent.

    Args:
      player_id: int, index of the player this learner controls.
      info_state_size: int, length of the flat information-state vector.
      num_actions: int, size of the action space.
      hidden_layers_sizes: iterable of int, hidden sizes of the MLPs.
      batch_size: int, number of this player's own decisions per update. Must
        be divisible by `num_minibatches`.
      update_epochs: int, optimization epochs per update.
      num_minibatches: int, minibatches per epoch.
      learning_rate: float, Adam learning rate.
      magnet_coef: float, weight of the KL-to-magnet penalty.
      magnet_update_period: int, updates between magnet refreshes.
      auto_update_magnet: bool, whether to refresh the magnet automatically.
      gamma: float, discount factor.
      gae_lambda: float, GAE(lambda) parameter.
      clip_coef: float, PPO clip range for the policy and value losses.
      entropy_coef: float, entropy bonus weight.
      value_coef: float, value loss weight.
      max_grad_norm: float, global gradient-norm clip.
      seed: int, seeds the network initialization, minibatch shuffling and
        action sampling.
      name: str, agent name.
    """
    if batch_size % num_minibatches:
      raise ValueError("batch_size must be divisible by num_minibatches.")
    self.player_id = player_id
    self._name = name
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._magnet_update_period = magnet_update_period
    self._auto_update_magnet = auto_update_magnet
    self._rng = np.random.RandomState(seed)
    self._rngkey = jax.random.key(seed)

    network = ActorCritic(
        info_state_size,
        num_actions,
        tuple(int(h) for h in hidden_layers_sizes),
        nn.Rngs(params=seed),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
    optimizer = nn.Optimizer(network, tx, wrt=nn.Param)
    # Training state (network and optimizer) for the update, plus a separate
    # handle on the network parameters for acting and for the magnet.
    self._graphdef, self._state = nn.split((network, optimizer))
    self._net_graphdef, self._params = nn.split(network, nn.Param)
    self._magnet_params = self._params

    self._infer = self._make_infer()
    self._update_fn = self._make_update(
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        magnet_coef=magnet_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
    )

    self._buffer = []
    self._pending = None
    self._pending_reward = 0.0
    self._total_updates = 0
    self._magnet_refreshes = 0
    self._last_loss = None

  def _make_infer(self):
    """Jitted (params, info_state, mask) -> (log_probs, value)."""
    net_graphdef = self._net_graphdef

    @jax.jit
    def infer(params, info_state, legal_actions_mask):
      network = nn.merge(net_graphdef, params, copy=True)
      return network(info_state, legal_actions_mask)

    return infer

  def _make_update(
      self,
      update_epochs,
      num_minibatches,
      magnet_coef,
      gamma,
      gae_lambda,
      clip_coef,
      entropy_coef,
      value_coef,
  ):
    """Builds the jitted PPO + magnet-KL update over one full batch."""
    graphdef = self._graphdef
    net_graphdef = self._net_graphdef
    batch_size = self._batch_size
    minibatch_size = batch_size // num_minibatches

    def loss_fn(network, mb):
      """Clipped PPO loss plus `magnet_coef * KL(pi || rho)` (paper Eq. 12)."""
      log_probs, new_value = network(mb["obs"], mb["mask"])
      logprob = log_probs[jnp.arange(minibatch_size), mb["action"]]
      ratio = jnp.exp(logprob - mb["old_logprob"])
      clipped = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
      adv = mb["advantage"]
      pg_loss = jnp.maximum(-adv * ratio, -adv * clipped).mean()

      value_clipped = mb["old_value"] + jnp.clip(
          new_value - mb["old_value"], -clip_coef, clip_coef
      )
      v_loss = (
          0.5
          * jnp.maximum(
              (new_value - mb["returns"]) ** 2,
              (value_clipped - mb["returns"]) ** 2,
          ).mean()
      )

      entropy = masked_entropy(log_probs, mb["mask"]).mean()
      kl = masked_kl(log_probs, mb["magnet_log_probs"], mb["mask"]).mean()
      loss = (
          pg_loss
          + value_coef * v_loss
          - entropy_coef * entropy
          + magnet_coef * kl
      )
      return loss, LossInfo(pg_loss, v_loss, entropy, kl)

    @jax.jit
    def update(state, magnet_params, batch, bootstrap_value, key):
      # The magnet is fixed during the update, so its log-probs are computed
      # once here, outside the differentiated loss.
      magnet = nn.merge(net_graphdef, magnet_params, copy=True)
      magnet_log_probs, _ = magnet(batch["obs"], batch["mask"])

      advantages = gae(
          batch["reward"],
          batch["old_value"],
          batch["done"],
          bootstrap_value,
          gamma,
          gae_lambda,
      )
      returns = advantages + batch["old_value"]
      if batch_size > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(ddof=1) + 1e-8
        )
      data = dict(
          obs=batch["obs"],
          mask=batch["mask"],
          action=batch["action"],
          old_logprob=batch["old_logprob"],
          old_value=batch["old_value"],
          advantage=advantages,
          returns=returns,
          magnet_log_probs=magnet_log_probs,
      )

      def minibatch_step(state, mb):
        network, optimizer = nn.merge(graphdef, state, copy=True)
        grads, info = nn.grad(loss_fn, has_aux=True)(network, mb)
        optimizer.update(network, grads)
        return nn.state((network, optimizer)), info

      def epoch_step(state, epoch_key):
        perm = jax.random.permutation(epoch_key, batch_size)
        shuffled = jax.tree.map(
            lambda x: x[perm].reshape(
                (num_minibatches, minibatch_size) + x.shape[1:]
            ),
            data,
        )
        return jax.lax.scan(minibatch_step, state, shuffled)

      keys = jax.random.split(key, update_epochs)
      state, infos = jax.lax.scan(epoch_step, state, keys)
      network, _ = nn.merge(graphdef, state, copy=True)
      params = nn.state(network, nn.Param)
      last_info = jax.tree.map(lambda x: x[-1, -1], infos)
      return state, params, last_info

    return update

  def step(self, time_step, is_evaluation=False):
    """Returns an action for the state and, unless evaluating, also learns."""
    my_turn = (not time_step.last()) and (
        time_step.is_simultaneous_move()
        or time_step.current_player() == self.player_id
    )

    action, probs = None, []
    if my_turn:
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      mask = np.zeros(self._num_actions, dtype=bool)
      mask[legal_actions] = True
      action, probs, logprob, value = self._act(info_state, mask)

    if is_evaluation:
      return rl_agent.StepOutput(action=action, probs=probs)

    # Credit the rewards seen since our last decision to that decision.
    if self._pending is not None and time_step.rewards is not None:
      self._pending_reward += time_step.rewards[self.player_id]
      if my_turn or time_step.last():
        self._buffer.append(
            self._pending._replace(
                reward=self._pending_reward, done=time_step.last()
            )
        )
        self._pending = None

    if time_step.last():
      self._pending, self._pending_reward = None, 0.0
      self._maybe_learn(bootstrap_value=0.0)
    elif my_turn:
      self._pending = Transition(
          info_state=np.asarray(info_state, dtype=np.float32),
          legal_actions_mask=mask,
          action=action,
          logprob=logprob,
          value=value,
          reward=0.0,
          done=False,
      )
      self._pending_reward = 0.0
      self._maybe_learn(bootstrap_value=value)

    return rl_agent.StepOutput(action=action, probs=probs)

  def update_magnet(self):
    """Copies the current policy into the magnet (reference) policy."""
    # JAX arrays are immutable, so keeping a reference is a true snapshot.
    self._magnet_params = self._params
    self._magnet_refreshes += 1

  @property
  def loss(self):
    """The most recent `LossInfo`, or None before the first update."""
    return self._last_loss

  @property
  def magnet_refreshes(self):
    """How many times the magnet has been refreshed so far."""
    return self._magnet_refreshes

  def _act(self, info_state, mask):
    """Samples an action, returns (action, full_probs, logprob, value)."""
    log_probs, value = self._infer(
        self._params, np.asarray(info_state, dtype=np.float32), mask
    )
    log_probs = np.asarray(log_probs)
    probs = np.exp(log_probs.astype(np.float64))
    probs /= probs.sum()
    action = int(self._rng.choice(self._num_actions, p=probs))
    return action, probs, float(log_probs[action]), float(value)

  def _next_rng_key(self):
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _maybe_learn(self, bootstrap_value):
    """Runs an update once `batch_size` transitions have accumulated."""
    if len(self._buffer) < self._batch_size:
      return
    cols = Transition(*zip(*self._buffer))
    batch = {
        "obs": np.stack(cols.info_state),
        "mask": np.stack(cols.legal_actions_mask),
        "action": np.asarray(cols.action, np.int32),
        "old_logprob": np.asarray(cols.logprob, np.float32),
        "old_value": np.asarray(cols.value, np.float32),
        "reward": np.asarray(cols.reward, np.float32),
        "done": np.asarray(cols.done, np.float32),
    }
    self._state, self._params, info = self._update_fn(
        self._state,
        self._magnet_params,
        batch,
        np.float32(bootstrap_value),
        self._next_rng_key(),
    )
    self._last_loss = LossInfo(*(float(x) for x in info))
    self._buffer = []
    self._total_updates += 1
    if (
        self._auto_update_magnet
        and self._total_updates % self._magnet_update_period == 0
    ):
      self.update_magnet()
