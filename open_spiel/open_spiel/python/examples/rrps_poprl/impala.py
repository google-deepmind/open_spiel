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
"""IMPALA agent implemented in JAX.

This is a basic IMPALA agent adapted from the example in the Haiku project:
https://github.com/deepmind/dm-haiku/tree/main/examples/impala
"""

import collections
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import chex
import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tree

from open_spiel.python import rl_agent
from open_spiel.python.examples.rrps_poprl import rl_environment


AgentOutput = collections.namedtuple(
    "AgentOutput", ["policy_logits", "values", "action", "prediction_logits"]
)
NetOutput = collections.namedtuple(
    "NetOutput", ["policy_logits", "value", "prediction_logits"]
)
Transition = collections.namedtuple(
    "Transition", ["timestep", "agent_out", "agent_state"]
)
NetFactory = Callable[[int], hk.RNNCore]
Nest = Any


# The IMPALA paper sums losses, rather than taking the mean.
# We wrap rlax to do so as well.
def policy_gradient_loss(logits, *args):
  """rlax.policy_gradient_loss, but with sum(loss) and [T, B, ...] inputs."""
  # jax.experimental.host_callback.id_print(logits.shape)
  # print(logits.shape)
  mean_per_batch = jax.vmap(rlax.policy_gradient_loss, in_axes=1)(logits, *args)
  total_loss_per_batch = mean_per_batch * logits.shape[0]
  return jnp.sum(total_loss_per_batch)


def entropy_loss(logits, *args):
  """rlax.entropy_loss, but with sum(loss) and [T, B, ...] inputs."""
  mean_per_batch = jax.vmap(rlax.entropy_loss, in_axes=1)(logits, *args)
  total_loss_per_batch = mean_per_batch * logits.shape[0]
  return jnp.sum(total_loss_per_batch)


def mean_pred_loss_without_batch(
    logits_t: chex.Array,
    labels: chex.Array,
) -> chex.Array:
  """Mean prediction loss without batch dimension."""
  chex.assert_rank([logits_t, labels], [2, 1])
  chex.assert_type([logits_t, labels], [float, int])
  labels_one_hot = jax.nn.one_hot(labels, logits_t.shape[-1])
  softmax_xent = -jnp.sum(labels_one_hot * jax.nn.log_softmax(logits_t))
  softmax_xent /= labels.shape[0]
  return softmax_xent


def prediction_loss(logits, labels):
  # print(logits.shape)  -> [T, B, num_preds]
  # print(labels.shape)  -> [T, B]
  mean_per_batch = jax.vmap(mean_pred_loss_without_batch, in_axes=1)(
      logits, labels
  )
  total_loss_per_batch = mean_per_batch * logits.shape[0]
  return jnp.sum(total_loss_per_batch)


def _preprocess_none(t) -> np.ndarray:
  if t is None:
    return np.array(0.0, dtype=np.float32)
  else:
    return np.asarray(t)


def preprocess_step(
    timestep: rl_environment.TimeStep, num_players
) -> rl_environment.TimeStep:
  # TODO(author5): fix for our time steps (should be multiple discounts)
  if timestep.discounts is None:
    timestep = timestep._replace(discounts=[1.0] * num_players)
  if timestep.rewards is None:
    timestep = timestep._replace(rewards=[0.0] * num_players)
  # print(timestep)
  return tree.map_structure(_preprocess_none, timestep)


# dm_env: return TimeStep(StepType.FIRST, None, None, observation)
# OpenSpiel: "observations", "rewards", "discounts", "step_type"
def restart(dummy_obs, num_players):
  all_obs = {
      "info_state": [dummy_obs.copy() for i in range(num_players)],
      "legal_actions": [np.zeros(3)],
      "prediction_label": 0,
  }
  return rl_environment.TimeStep(
      all_obs, [0.0], None, rl_environment.StepType.FIRST
  )


class BasicRNN(hk.RNNCore):
  """A simple recurrent neural network."""

  def __init__(
      self,
      player_id,
      num_actions,
      hidden_layer_sizes,
      num_predictions,
      name=None,
  ):
    super().__init__(name=name)
    self._player_id = player_id
    self._num_actions = num_actions
    self._num_predictions = num_predictions
    self._hidden_layer_sizes = hidden_layer_sizes
    if isinstance(hidden_layer_sizes, int):
      self._hidden_layer_sizes = [hidden_layer_sizes]
    elif isinstance(hidden_layer_sizes, tuple):
      self._hidden_layer_sizes = list(hidden_layer_sizes)
    self._core = hk.ResetCore(hk.LSTM(256))

  def initial_state(self, batch_size):
    return self._core.initial_state(batch_size)

  def __call__(self, x: rl_environment.TimeStep, state):
    x = jax.tree_util.tree_map(lambda t: t[None, ...], x)
    return self.unroll(x, state)

  def unroll(self, x, state):
    modules = [hk.Flatten()]
    for hsize in self._hidden_layer_sizes:
      modules.append(hk.Linear(hsize))
      modules.append(jax.nn.relu)
    torso_net = hk.Sequential(modules)
    torso_output = hk.BatchApply(torso_net)(
        x.observations["info_state"][self._player_id]
    )
    should_reset = jnp.equal(x.step_type, int(rl_environment.StepType.FIRST))
    core_input = (torso_output, should_reset)
    core_output, state = hk.dynamic_unroll(self._core, core_input, state)
    policy_logits = hk.Linear(self._num_actions)(core_output)
    prediction_logits = hk.Linear(self._num_predictions)(core_output)
    value = hk.Linear(1)(core_output)
    value = jnp.squeeze(value, axis=-1)
    return (
        NetOutput(
            policy_logits=policy_logits,
            value=value,
            prediction_logits=prediction_logits,
        ),
        state,
    )
    # torso_output = torso_net(x.observations["info_state"][self._player_id])
    # policy_logits = hk.Linear(self._num_actions)(torso_output)
    # prediction_logits = hk.Linear(self._num_predictions)(torso_output)
    # value = hk.Linear(1)(torso_output)
    # value = jnp.squeeze(value, axis=-1)
    # return NetOutput(policy_logits=policy_logits,
    #                  value=value,
    #                  prediction_logits=prediction_logits), state


class IMPALA(rl_agent.AbstractAgent):
  """IMPALA agent implementation in JAX."""

  def __init__(
      self,
      player_id,
      state_representation_size,
      num_actions,
      num_players,
      unroll_len,
      net_factory: NetFactory,
      rng_key,
      max_abs_reward,
      learning_rate=0.0001,
      entropy=0.01,
      discount_factor=0.99,
      hidden_layers_sizes=128,
      batch_size=16,
      num_predictions=10,
      prediction_weight=0.01,
      max_global_gradient_norm=None,
  ):
    self._player_id = player_id
    self._state_representation_size = state_representation_size
    self._num_actions = num_actions
    self._num_players = num_players
    self._unroll_len = unroll_len
    self._rng_key = rng_key
    self._max_abs_reward = max_abs_reward
    self._learning_rate = learning_rate
    self._batch_size = batch_size
    self._discount_factor = discount_factor
    self._entropy = entropy
    self._hidden_layer_sizes = hidden_layers_sizes
    self._num_predictions = num_predictions
    self._prediction_weight = prediction_weight
    self._dummy_obs = np.zeros(
        shape=state_representation_size, dtype=np.float32
    )

    # pylint: disable=too-many-function-args
    net_factory = functools.partial(
        net_factory,
        player_id,
        num_actions,
        hidden_layers_sizes,
        num_predictions,
    )
    # Instantiate two hk.transforms() - one for getting the initial state of the
    # agent, another for actually initializing and running the agent.
    _, self._initial_state_apply_fn = hk.without_apply_rng(
        hk.transform(lambda batch_size: net_factory().initial_state(batch_size))
    )

    self._init_fn, self._apply_fn = hk.without_apply_rng(
        hk.transform(lambda obs, state: net_factory().unroll(obs, state))
    )

    # Learner components
    # self._opt = optax.rmsprop(5e-3, decay=0.99, eps=1e-7)
    self._opt = optax.rmsprop(self._learning_rate, decay=0.99, eps=1e-7)
    # self._opt = optax.sgd(self._learning_rate)

    # Prepare parameters and initial state
    self._rng_key, subkey = jax.random.split(self._rng_key)
    init_params = self.initial_params(subkey)
    self._frame_count_and_params = (0, jax.device_get(init_params))
    (_, params) = self._frame_count_and_params
    self._opt_state = self._opt.init(params)
    self._rng_key, _ = jax.random.split(self._rng_key)
    self._agent_state = self._agent_state = self.initial_state(None)
    self._traj = []
    self._batch = []
    self._last_policy = None
    self._last_predictions = None

  @functools.partial(jax.jit, static_argnums=0)
  def initial_params(self, rng_key):
    """Initializes the agent params given the RNG key."""

    dummy_inputs = jax.tree_util.tree_map(
        lambda t: np.zeros(t.shape, t.dtype), self._dummy_obs
    )
    dummy_inputs = preprocess_step(
        restart(dummy_inputs, self._num_players), self._num_players
    )
    # Add time and batch dimensions
    dummy_inputs = jax.tree_util.tree_map(
        lambda t: t[None, None, ...], dummy_inputs
    )
    # print(dummy_inputs)
    return self._init_fn(rng_key, dummy_inputs, self.initial_state(1))

  @functools.partial(jax.jit, static_argnums=(0, 1))
  def initial_state(self, batch_size: Optional[int]):
    """Returns agent initial state."""
    # We expect that generating the initial_state does not require parameters.
    return self._initial_state_apply_fn(None, batch_size)

  @functools.partial(jax.jit, static_argnums=(0,))
  def internal_step(
      self,
      rng_key,
      params: hk.Params,
      timestep: rl_environment.TimeStep,
      state: Nest,
  ) -> Tuple[AgentOutput, Nest]:
    """For a given single-step, unbatched timestep, output the chosen action."""
    # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.
    # print("calling internal_step")
    timestep = jax.tree_util.tree_map(lambda t: t[None, None, ...], timestep)
    state = jax.tree_util.tree_map(lambda t: t[None, ...], state)
    net_out, next_state = self._apply_fn(params, timestep, state)
    # print(timestep)
    # Remove the padding from above.
    net_out = jax.tree_util.tree_map(
        lambda t: jnp.squeeze(t, axis=(0, 1)), net_out
    )
    next_state = jax.tree_util.tree_map(
        lambda t: jnp.squeeze(t, axis=0), next_state
    )
    # Sample an action and return.
    action = hk.multinomial(rng_key, net_out.policy_logits, num_samples=1)
    action = jnp.squeeze(action, axis=-1)
    return (
        AgentOutput(
            net_out.policy_logits,
            net_out.value,
            action,
            net_out.prediction_logits,
        ),
        next_state,
    )

  def unroll(
      self,
      params: hk.Params,
      trajectory: rl_environment.TimeStep,
      state: Nest,
  ) -> AgentOutput:
    """Unroll the agent along trajectory."""
    net_out, _ = self._apply_fn(params, trajectory, state)
    return AgentOutput(
        net_out.policy_logits,
        net_out.value,
        action=[],
        prediction_logits=net_out.prediction_logits,
    )

  def _loss(
      self,
      theta: hk.Params,
      trajectories: Transition,
  ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute vtrace-based actor-critic loss."""
    # All the individual components are vectorized to be [T, B, ...]
    # print(trajectories)
    # Transition(timestep=TimeStep(observations={
    # 'current_player': array([[-2, -2, -2, -2, -2],
    #   [-2, -2, -2, -2, -2],
    #   [-2, -2, -2, -2, -2],
    #   [-2, -2, -2, -2, -2],
    # Since prediction_label is a scalar, it ends up being [T, B]

    initial_state = jax.tree_util.tree_map(
        lambda t: t[0], trajectories.agent_state
    )
    learner_outputs = self.unroll(theta, trajectories.timestep, initial_state)
    v_t = learner_outputs.values[1:]
    # Remove bootstrap timestep from non-timesteps.
    _, actor_out, _ = jax.tree_util.tree_map(lambda t: t[:-1], trajectories)
    learner_outputs = jax.tree_util.tree_map(lambda t: t[:-1], learner_outputs)
    v_tm1 = learner_outputs.values

    # Get the discount, reward, step_type from the *next* timestep.
    timestep = jax.tree_util.tree_map(lambda t: t[1:], trajectories.timestep)
    discounts = timestep.discounts[self._player_id] * self._discount_factor
    rewards = timestep.rewards[self._player_id]
    if self._max_abs_reward > 0:
      rewards = jnp.clip(rewards, -self._max_abs_reward, self._max_abs_reward)

    # The step is uninteresting if we transitioned LAST -> FIRST.
    # timestep corresponds to the *next* time step, so we filter for FIRST.
    mask = jnp.not_equal(timestep.step_type, int(rl_environment.StepType.FIRST))
    mask = mask.astype(jnp.float32)

    rhos = rlax.categorical_importance_sampling_ratios(
        learner_outputs.policy_logits, actor_out.policy_logits, actor_out.action
    )
    # vmap vtrace_td_error_and_advantage to take/return [T, B, ...].
    vtrace_td_error_and_advantage = jax.vmap(
        rlax.vtrace_td_error_and_advantage, in_axes=1, out_axes=1
    )

    vtrace_returns = vtrace_td_error_and_advantage(
        v_tm1, v_t, rewards, discounts, rhos
    )
    pg_advs = vtrace_returns.pg_advantage
    # print(learner_outputs.policy_logits.shape)
    # jax.experimental.host_callback.id_print(learner_outputs.policy_logits.shape)
    pg_loss = policy_gradient_loss(
        learner_outputs.policy_logits, actor_out.action, pg_advs, mask
    )

    baseline_loss = 0.5 * jnp.sum(jnp.square(vtrace_returns.errors) * mask)
    ent_loss = entropy_loss(learner_outputs.policy_logits, mask)

    pred_loss = 0
    if self._prediction_weight > 0:
      pred_loss = prediction_loss(
          learner_outputs.prediction_logits,
          trajectories.timestep.observations["prediction_label"][:-1],
      )

    total_loss = pg_loss
    total_loss += 0.5 * baseline_loss
    total_loss += self._entropy * ent_loss
    total_loss += self._prediction_weight * pred_loss

    logs = {}
    logs["PG_loss"] = pg_loss
    logs["baseline_loss"] = baseline_loss
    logs["entropy_loss"] = ent_loss
    logs["prediction_loss"] = pred_loss
    logs["total_loss"] = total_loss
    return total_loss, logs

  @functools.partial(jax.jit, static_argnums=0)
  def update(self, params, opt_state, batch: Transition):
    """The actual update function."""
    (_, logs), grads = jax.value_and_grad(self._loss, has_aux=True)(
        params, batch
    )

    grad_norm_unclipped = optimizers.l2_norm(grads)
    updates, updated_opt_state = self._opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    weight_norm = optimizers.l2_norm(params)
    logs.update({
        "grad_norm_unclipped": grad_norm_unclipped,
        "weight_norm": weight_norm,
    })
    return params, updated_opt_state, logs

  def _learning_step(self):
    # print("Learning!!!")
    # Prepare for consumption, then put batch onto device.
    # stacked_batch = jax.tree_multimap(lambda *xs: np.stack(xs, axis=1),
    #                                 *self._batch)
    stacked_batch = jax.tree_util.tree_map(
        lambda *xs: np.stack(xs, axis=1), *self._batch
    )
    # self._device_q.put(jax.device_put(stacked_batch))
    jax.device_put(stacked_batch)
    num_frames, params = self._frame_count_and_params
    params, self._opt_state, _ = self.update(
        params, self._opt_state, stacked_batch
    )
    self._frame_count_and_params = (num_frames + 1, params)
    self._batch = []

  def last_policy(self):
    return self._last_policy

  def last_predictions(self):
    return self._last_predictions

  def step(self, time_step, is_evaluation=False):
    # Hack to run with environments that include the serialized state: simply
    # remove it.
    if "serialized_state" in time_step.observations:
      del time_step.observations["serialized_state"]
    # OpenSpiel time steps have lists of floats. First convert to numpy.
    for p in range(self._num_players):
      time_step.observations["info_state"][p] = np.asarray(
          time_step.observations["info_state"][p]
      )

    # print(time_step)
    # TODO(author5): the arrays need to be the same shape, so when the
    # legal actions are empty vs full, this is a problem.
    # Fix later. for now, replace with a constant
    time_step.observations["legal_actions"] = [np.ones(3)]

    agent_state = self._agent_state
    (_, params) = self._frame_count_and_params
    jax.device_put(params)
    time_step = preprocess_step(time_step, self._num_players)
    self._rng_key, subkey = jax.random.split(self._rng_key)
    agent_out, next_state = self.internal_step(
        subkey, params, time_step, agent_state
    )

    self._last_policy = jax.nn.softmax(agent_out.policy_logits).copy()
    self._last_predictions = jax.nn.softmax(agent_out.prediction_logits).copy()

    transition = Transition(
        timestep=time_step, agent_out=agent_out, agent_state=agent_state
    )
    self._agent_state = next_state

    # Do not add to trajectory or check for learning during evaluation.
    if not is_evaluation:
      self._traj.append(transition)
      # Check for learning step.
      if len(self._traj) >= self._unroll_len:
        trajectory = jax.device_get(self._traj)
        # trajectory = jax.tree_multimap(lambda *xs: np.stack(xs), *trajectory)
        trajectory = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs), *trajectory
        )
        self._batch.append(trajectory)
        self._traj = self._traj[-1:]
        if len(self._batch) >= self._batch_size:
          self._learning_step()

    if time_step.last():
      return None

    assert 0 <= agent_out.action < self._num_actions
    # TODO(author5): get probs from the policy in internal_step
    probs = np.zeros(self._num_actions, dtype=np.float32)
    probs[agent_out.action] = 1.0
    return rl_agent.StepOutput(action=agent_out.action, probs=probs)
