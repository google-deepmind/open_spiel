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

"""Policy gradient methods implemented in Jax."""

import collections
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

import collections
import numpy as np


from open_spiel.python import rl_agent

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")


class PolicyGradient(rl_agent.AbstractAgent):
  """Policy Gradient Agent implementation in Jax.
  """

  def __init__(self,
               player_id,
               info_state_size,
               num_actions,
               loss_str="a2c",
               loss_class=None,
               hidden_layers_sizes=(128,),
               batch_size=16,
               critic_learning_rate=0.01,
               pi_learning_rate=0.001,
               entropy_cost=0.01,
               num_critic_before_pi=8,
               additional_discount_factor=1.0,
               max_global_gradient_norm=None,
               optimizer_str="sgd",
               seed=42):
    """Initialize the PolicyGradient agent.
    Args:
      player_id: int, player identifier. Usually its position in the game.
      info_state_size: int, info_state vector size.
      num_actions: int, number of actions per info state.
      loss_str: string or None. If string, must be one of ["rpg", "qpg", "rm",
        "a2c"] and defined in `_get_loss_class`. If None, a loss class must be
        passed through `loss_class`. Defaults to "a2c".
      loss_class: Class or None. If Class, it must define the policy gradient
        loss. If None a loss class in a string format must be passed through
        `loss_str`. Defaults to None.
      hidden_layers_sizes: iterable, defines the neural network layers. Defaults
          to (128,), which produces a NN: [INPUT] -> [128] -> ReLU -> [OUTPUT].
      batch_size: int, batch size to use for Q and Pi learning. Defaults to 128.
      critic_learning_rate: float, learning rate used for Critic (Q or V).
        Defaults to 0.001.
      pi_learning_rate: float, learning rate used for Pi. Defaults to 0.001.
      entropy_cost: float, entropy cost used to multiply the entropy loss. Can
        be set to None to skip entropy computation. Defaults to 0.001.
      num_critic_before_pi: int, number of Critic (Q or V) updates before each
        Pi update. Defaults to 8 (every 8th critic learning step, Pi also
        learns).
      additional_discount_factor: float, additional discount to compute returns.
        Defaults to 1.0, in which case, no extra discount is applied.  None that
        users must provide *only one of* `loss_str` or `loss_class`.
      max_global_gradient_norm: float or None, maximum global norm of a gradient
        to which the gradient is shrunk if its value is larger.
      optimizer_str: String defining which optimizer to use. Supported values
        are {sgd, adam}
      seed: random seed
    """
    assert bool(loss_str) ^ bool(loss_class), "Please provide only one option."
    self._kwargs = locals()
    loss_class = loss_class if loss_class else self._get_loss_class(loss_str)
    self._loss_class = loss_class

    self.player_id = player_id
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._extra_discount = additional_discount_factor
    self._num_critic_before_pi = num_critic_before_pi
    self._max_global_gradient_norm = max_global_gradient_norm

    self._episode_data = []
    self._dataset = collections.defaultdict(list)
    self._prev_time_step = None
    self._prev_action = None

    # Step counters
    self._step_counter = 0
    self._episode_counter = 0
    self._num_learn_steps = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Network
    # activate final as we plug logit and qvalue heads afterwards.

    def net_torso(x):
      mlp = hk.nets.MLP(self._layer_sizes)
      return mlp(x)

    self.hk_net_torso = hk.without_apply_rng(hk.transform(net_torso))
    self.hk_net_torso_apply = jax.jit(self.hk_net_torso.apply)
    self.rng = jax.random.PRNGKey(seed)
    x = jnp.ones((1, info_state_size))
    self.hk_net_torso_params = self.hk_net_torso.init(self.rng, x=x)

    def policy_logits_layer(x):
      mlp = hk.nets.MLP([num_actions])
      return mlp(x)

    self.hk_policy_logits_layer = hk.without_apply_rng(
        hk.transform(policy_logits_layer))
    self.hk_policy_logits_layer_apply = jax.jit(
        self.hk_policy_logits_layer.apply)

    torso = self.hk_net_torso_apply(self.hk_net_torso_params, x)
    self.rng, _ = jax.random.split(self.rng)
    self.hk_policy_logits_layer_params = self.hk_policy_logits_layer.init(
        self.rng, x=torso)

    self._savers = []

    # Add baseline (V) head for A2C (or Q-head for QPG / RPG / RMPG)

    if loss_class.__name__ == "policy_gradient_loss":
      self._pi_loss = self._a2c_pi_loss

      def baseline_layer(x):
        mlp = hk.nets.MLP([1])
        return mlp(x)

      self.hk_baseline_layer = hk.without_apply_rng(
          hk.transform(baseline_layer))
      self.hk_baseline_layer_apply = jax.jit(self.hk_baseline_layer.apply)
      self.rng, _ = jax.random.split(self.rng)
      self.hk_baseline_layer_params = self.hk_baseline_layer.init(
          self.rng, x=torso)

#       def critic_network(x, torso_params, baseline_params):
#         torso_head = self.hk_net_torso_apply(torso_params, x)
#         return self.hk_baseline_layer_apply(baseline_params, x=torso_head)

    else:
      self._pi_loss = self._pg_loss

      def q_network(x):
        mlp = hk.nets.MLP([num_actions])
        return mlp(x)

      self.hk_q_network = hk.without_apply_rng(hk.transform(q_network))
      self.hk_q_network_apply = jax.jit(self.hk_q_network.apply)
      self.rng, _ = jax.random.split(self.rng)
      self.hk_q_network_params = self.hk_q_network.init(self.rng, x=torso)

#       def critic_network(x, torso_params, q_params):
#         torso_head = self.hk_net_torso_apply(torso_params, x)
#         return self.hk_q_network_apply(q_params, x=torso_head)

    # Pi loss
    self._entropy_cost = entropy_cost

    self._loss_str = loss_str

    if optimizer_str == "adam":
      self._torso_critic_optimizer = optax.adam(critic_learning_rate)
      self._value_optimizer = optax.adam(critic_learning_rate)
      self._torso_pi_optimizer = optax.adam(pi_learning_rate)
      self._pi_optimizer = optax.adam(pi_learning_rate)

    elif optimizer_str == "sgd":
      self._torso_critic_optimizer = optax.sgd(critic_learning_rate)
      self._value_optimizer = optax.sgd(critic_learning_rate)
      self._torso_pi_optimizer = optax.sgd(pi_learning_rate)
      self._pi_optimizer = optax.sgd(pi_learning_rate)

    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    if max_global_gradient_norm:
      self._torso_critic_optimizer = optax.chain(self._torso_critic_optimizer,
                                                 optax.clip_by_global_norm(max_global_gradient_norm))
      self._torso_pi_optimizer = optax.chain(self._torso_pi_optimizer,
                                             optax.clip_by_global_norm(max_global_gradient_norm))
      self._pi_optimizer = optax.chain(self._pi_optimizer,
                                       optax.clip_by_global_norm(max_global_gradient_norm))
      self._value_optimizer = optax.chain(self._value_optimizer,
                                          optax.clip_by_global_norm(max_global_gradient_norm))

    torso_critic_opt_init, torso_critic_opt_update = self._torso_critic_optimizer.init, self._torso_critic_optimizer.update
    torso_pi_opt_init, torso_pi_opt_update = self._torso_pi_optimizer.init, self._torso_pi_optimizer.update
    pi_opt_init, pi_opt_update = self._pi_optimizer.init, self._pi_optimizer.update
    value_opt_init, value_opt_update = self._value_optimizer.init, self._value_optimizer.update

    self._torso_critic_opt_update_fn = self._get_update_func(
        torso_critic_opt_update)
    self._torso_pi_opt_update_fn = self._get_update_func(torso_pi_opt_update)
    self._pi_opt_update_fn = self._get_update_func(pi_opt_update)
    self._value_opt_update_fn = self._get_update_func(value_opt_update)

    self._torso_critic_opt_state = torso_critic_opt_init(
        self.hk_net_torso_params)
    self._torso_pi_opt_state = torso_pi_opt_init(self.hk_net_torso_params)
    self._pi_opt_state = pi_opt_init(self.hk_policy_logits_layer_params)

    if loss_class.__name__ == "policy_gradient_loss":
      self._critic_loss_and_grad = jax.value_and_grad(
          self._a2c_critic_loss, has_aux=False, argnums=(0, 1))
      self._value_opt_state = value_opt_init(self.hk_baseline_layer_params)
    else:
      self._critic_loss_and_grad = jax.value_and_grad(
          self._critic_loss, has_aux=False, argnums=(0, 1))
      self._value_opt_state = value_opt_init(self.hk_q_network_params)

    # self._critic_loss = lambda x, y: jnp.mean((x-y)**2)

    self._pi_loss_and_grad = jax.value_and_grad(
        self._pi_loss, has_aux=False, argnums=(0, 1))
    self._jit_pi_update = jax.jit(self.get_update(
        self._torso_pi_opt_update_fn, self._pi_opt_update_fn, self._pi_loss_and_grad))
    self._jit_critic_update = jax.jit(self.get_update(
        self._torso_critic_opt_update_fn, self._value_opt_update_fn, self._critic_loss_and_grad))

  def _a2c_pi_loss(self, torso_params, policy_logits_layer_params, value_network_params, info_states,
                   actions, returns):

    torso_out = self.hk_net_torso_apply(torso_params, info_states)

    baselines = jnp.squeeze(self.hk_baseline_layer_apply(
        value_network_params, torso_out), axis=1)
    advantages = returns - baselines
    policy_logits = self.hk_policy_logits_layer_apply(
        policy_logits_layer_params, torso_out)
    chex.assert_equal_shape([returns, baselines, actions, advantages])

    pi_loss = self._loss_class(
        logits_t=policy_logits,
        a_t=actions,
        adv_t=advantages,
        w_t=jnp.ones(returns.shape))
    ent_loss = rlax.entropy_loss(
        logits_t=policy_logits, w_t=jnp.ones(returns.shape))
    return pi_loss + self._entropy_cost * ent_loss

  def _a2c_critic_loss(self, torso_params, value_network_params, policy_logits_layer_params, info_states,
                       actions, returns):
    torso_out = self.hk_net_torso_apply(torso_params, info_states)

    baselines = jnp.squeeze(self.hk_baseline_layer_apply(
        value_network_params, torso_out), axis=1)
    chex.assert_equal_shape([returns, baselines, actions])

    return jnp.mean(jnp.square(baselines-returns))

  def _pg_loss(self, torso_params, policy_logits_layer_params, value_network_params, info_states,
               actions, returns):

    torso_out = self.hk_net_torso_apply(torso_params, info_states)
    policy_logits = self.hk_policy_logits_layer_apply(
        policy_logits_layer_params, torso_out)
    q_values = self.hk_q_network_apply(value_network_params, torso_out)

    chex.assert_equal_shape([policy_logits, q_values])

    pi_loss = self._loss_class(logits_t=policy_logits, q_t=q_values)
    ent_loss = rlax.entropy_loss(
        logits_t=policy_logits, w_t=jnp.ones(returns.shape))
    return pi_loss + self._entropy_cost * ent_loss

  def _critic_loss(self, torso_params, value_network_params, policy_logits_layer_params, info_states,
                   actions, returns):
    torso_out = self.hk_net_torso_apply(torso_params, info_states)
    q_values = self.hk_q_network_apply(value_network_params, torso_out)

    action_indices = jnp.stack(
        [jnp.arange(q_values.shape[0]), actions], axis=0)
    value_predictions = q_values[tuple(action_indices)]
    chex.assert_equal_shape([value_predictions, returns])
    return jnp.mean(jnp.square(value_predictions-returns))

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rlax.rpg_loss
    elif loss_str == "qpg":
      return rlax.qpg_loss
    elif loss_str == "rm":
      return rlax.rm_loss
    elif loss_str == "a2c":
      return rlax.policy_gradient_loss

  def _get_update_func(self, opt_update):

    def update(params, opt_state, gradient):
      """Learning rule (stochastic gradient descent)."""
      updates, opt_state = opt_update(gradient, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state

    return update

  def get_update(self, torso_opt_update_fn, other_opt_update_fn, loss_fn):
    """A unified funtional interface for both critic and pi loss updating."""

    def update(torso_params, other_layer_params, no_grad_other_layer_params, torso_opt_state, other_opt_state, info_states, actions, returns):
      loss_val, (torso_grad_val, other_layer_grad_val) = loss_fn(
          torso_params, other_layer_params, no_grad_other_layer_params, info_states, actions, returns)

      new_torso_params, new_torso_opt_state = torso_opt_update_fn(
          torso_params, torso_opt_state, torso_grad_val)
      new_other_layer_params, new_other_layer_state = other_opt_update_fn(
          other_layer_params, other_opt_state, other_layer_grad_val)
      return new_torso_params, new_other_layer_params, new_torso_opt_state, new_other_layer_state, loss_val

    return update

  def _act(self, info_state, legal_actions):
    # Make a singleton batch for NN compatibility: [1, info_state_size]
    info_state = jnp.reshape(np.array(info_state), [1, -1])
    torso_out = self.hk_net_torso_apply(self.hk_net_torso_params, info_state)
    self._policy_logits = self.hk_policy_logits_layer_apply(
        self.hk_policy_logits_layer_params, torso_out)

    policy_probs = jax.nn.softmax(self._policy_logits, axis=1)

    # Remove illegal actions, re-normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = policy_probs[0, legal_actions]
    if sum(probs) != 0:
      probs /= sum(probs)
    else:
      probs[legal_actions] = 1 / len(legal_actions)
    action = np.random.choice(len(probs), p=probs)
    return action, probs

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.
    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
            time_step.is_simultaneous_move() or
            self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action, probs = self._act(info_state, legal_actions)
    else:
      action = None
      probs = []

    if not is_evaluation:
      self._step_counter += 1

      # Add data points to current episode buffer.
      if self._prev_time_step:
        self._add_transition(time_step)

      # Episode done, add to dataset and maybe learn.
      if time_step.last():
        self._add_episode_data_to_dataset()
        self._episode_counter += 1

        if len(self._dataset["returns"]) >= self._batch_size:
          self._critic_update()
          self._num_learn_steps += 1
          if self._num_learn_steps % self._num_critic_before_pi == 0:
            self._pi_update()
          self._dataset = collections.defaultdict(list)

        self._prev_time_step = None
        self._prev_action = None
        return
      else:
        self._prev_time_step = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  @property
  def loss(self):
    return (self._last_critic_loss_value, self._last_pi_loss_value)

  def _add_episode_data_to_dataset(self):
    """Add episode data to the buffer."""
    info_states = [data.info_state for data in self._episode_data]
    rewards = [data.reward for data in self._episode_data]
    discount = [data.discount for data in self._episode_data]
    actions = [data.action for data in self._episode_data]

    # Calculate returns
    returns = np.array(rewards)
    for idx in reversed(range(len(rewards[:-1]))):
      returns[idx] = (
          rewards[idx] +
          discount[idx] * returns[idx + 1] * self._extra_discount)

    # Add flattened data points to dataset
    self._dataset["actions"].extend(actions)
    self._dataset["returns"].extend(returns)
    self._dataset["info_states"].extend(info_states)
    self._episode_data = []

  def _add_transition(self, time_step):
    """Adds intra-episode transition to the `_episode_data` buffer.
    Adds the transition from `self._prev_time_step` to `time_step`.
    Args:
      time_step: an instance of rl_environment.TimeStep.
    """
    assert self._prev_time_step is not None
    legal_actions = (
        self._prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            self._prev_time_step.observations["info_state"][self.player_id][:]),
        action=self._prev_action,
        reward=time_step.rewards[self.player_id],
        discount=time_step.discounts[self.player_id],
        legal_actions_mask=legal_actions_mask)

    self._episode_data.append(transition)

  def _critic_update(self):
    """Compute the Critic loss on sampled transitions & perform a critic update.
    Returns:
      The average Critic loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_states = jnp.asarray(self._dataset["info_states"])
    actions = jnp.asarray(self._dataset["actions"])
    returns = jnp.asarray(self._dataset["returns"])
    if self._loss_class.__name__ == "policy_gradient_loss":
      self.hk_net_torso_params, self.hk_baseline_layer_params, self._torso_critic_opt_state, self._value_opt_state, self._last_critic_loss_value = \
          self._jit_critic_update(self.hk_net_torso_params,
                                  self.hk_baseline_layer_params,
                                  self.hk_policy_logits_layer_params,
                                  self._torso_critic_opt_state,
                                  self._value_opt_state,
                                  info_states,
                                  actions,
                                  returns)
    else:
      self.hk_net_torso_params, self.hk_q_network_params, self._torso_critic_opt_state, self._value_opt_state, self._last_critic_loss_value = \
          self._jit_critic_update(self.hk_net_torso_params,
                                  self.hk_q_network_params,
                                  self.hk_policy_logits_layer_params,
                                  self._torso_critic_opt_state,
                                  self._value_opt_state,
                                  info_states,
                                  actions,
                                  returns)

    return self._last_critic_loss_value

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.
    Returns:
      The average Pi loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    info_states = jnp.asarray(self._dataset["info_states"])
    actions = jnp.asarray(self._dataset["actions"])
    returns = jnp.asarray(self._dataset["returns"])

    if self._loss_class.__name__ == "policy_gradient_loss":
      self.hk_net_torso_params, self.hk_policy_logits_layer_params, self._torso_pi_opt_state, self._pi_opt_state, self._last_pi_loss_value = \
          self._jit_pi_update(self.hk_net_torso_params,
                              self.hk_policy_logits_layer_params,
                              self.hk_baseline_layer_params,
                              self._torso_pi_opt_state,
                              self._pi_opt_state,
                              info_states,
                              actions,
                              returns)
    else:
      self.hk_net_torso_params, self.hk_policy_logits_layer_params, self._torso_pi_opt_state, self._pi_opt_state, self._last_pi_loss_value = \
          self._jit_pi_update(self.hk_net_torso_params,
                              self.hk_policy_logits_layer_params,
                              self.hk_q_network_params,
                              self._torso_pi_opt_state,
                              self._pi_opt_state,
                              info_states,
                              actions,
                              returns)
    return self._last_pi_loss_value

  def get_weights(self):
    weights = {}
    weights['net_torso_params'] = self.hk_net_torso_params
    weights['policy_logits_layer_params'] = self.hk_policy_logits_layer_params
    if self._loss_class.__name__ == "policy_gradient_loss":
      weights['value_layer'] = self.hk_baseline_layer_params
    else:
      weights['value_layer'] = self.hk_q_network_params
    return weights

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates its network's weights with noise.
    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.
    Returns:
      Perturbated copy of the model.
    """
    _ = self._kwargs.pop("self", None)
    copied_object = PolicyGradient(**self._kwargs)

    net_torso = getattr(copied_object, "hk_net_torso_params")
    self.rng, _ = jax.random.split(self.rng)
    copied_object.hk_net_torso_params = jax.tree_util.tree_map(lambda x: x.copy(
    ) * (1+sigma * jax.random.normal(self.rng, shape=x.shape)), net_torso)

    policy_logits_layer = getattr(
        copied_object, "hk_policy_logits_layer_params")
    self.rng, _ = jax.random.split(self.rng)
    copied_object.hk_policy_logits_layer_params = jax.tree_util.tree_map(lambda x: x.copy(
    ) * (1+sigma * jax.random.normal(self.rng, shape=x.shape)), policy_logits_layer)
    if hasattr(copied_object, "hk_q_network_params"):
      q_values_layer = getattr(copied_object, "hk_q_network_params")
      self.rng, _ = jax.random.split(self.rng)
      copied_object.hk_q_network_params = jax.tree_util.tree_map(lambda x: x.copy(
      ) * (1+sigma * jax.random.normal(self.rng, shape=x.shape)), q_values_layer)
    if hasattr(copied_object, "hk_baseline_layer_params"):
      baseline_layer = getattr(copied_object, "hk_baseline_layer_params")
      self.rng, _ = jax.random.split(self.rng)
      copied_object.hk_baseline_layer_params = jax.tree_util.tree_map(lambda x: x.copy(
      ) * (1+sigma * jax.random.normal(self.rng, shape=x.shape)), baseline_layer)

    return copied_object
