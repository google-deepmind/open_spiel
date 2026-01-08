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
"""Policy gradient methods implemented in JAX."""

import collections
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from open_spiel.python import rl_agent

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward discount legal_actions_mask next_info_state")


class NetA2C(hk.Module):
  """A simple network with a policy head and a baseline value head."""

  def __init__(self, num_actions, hidden_layers_sizes):
    super().__init__()
    self._num_actions = num_actions
    self._hidden_layers_sizes = hidden_layers_sizes

  def __call__(self, info_state):
    """Process a batch of observations."""
    torso = hk.nets.MLP(self._hidden_layers_sizes, activate_final=True)
    hidden = torso(info_state)
    policy_logits = hk.Linear(self._num_actions)(hidden)
    baseline = hk.Linear(1)(hidden)
    return policy_logits, baseline


class NetPG(hk.Module):
  """A simple network with a policy head and an action-value head."""

  def __init__(self, num_actions, hidden_layers_sizes):
    super().__init__()
    self._num_actions = num_actions
    self._hidden_layers_sizes = hidden_layers_sizes

  def __call__(self, info_state):
    """Process a batch of observations."""
    torso = hk.nets.MLP(self._hidden_layers_sizes, activate_final=True)
    hidden = torso(info_state)
    policy_logits = hk.Linear(self._num_actions)(hidden)
    q_values = hk.Linear(self._num_actions)(hidden)
    return policy_logits, q_values


def generate_a2c_pi_loss(net_apply, loss_class, entropy_cost, l2_actor_weight,
                         lambda_):
  """A function generator generates loss function."""

  def _a2c_pi_loss(net_params, batch):
    info_states, actions, rewards, discounts = batch["info_states"], batch[
        "actions"], batch["rewards"], batch["discounts"]
    policy_logits, baselines = net_apply(net_params, info_states)
    policy_logits = policy_logits[:-1]

    baselines = jnp.squeeze(baselines, axis=1)
    baselines = jnp.concatenate([baselines[:-1], jnp.zeros(1)])
    td_returns = rlax.lambda_returns(
        rewards,
        discounts,
        baselines[1:],
        lambda_=lambda_,
        stop_target_gradients=True)
    advantages = td_returns - baselines[:-1]
    chex.assert_equal_shape([td_returns, actions, advantages])
    pi_loss = loss_class(
        logits_t=policy_logits,
        a_t=actions,
        adv_t=advantages,
        w_t=jnp.ones(td_returns.shape))
    ent_loss = rlax.entropy_loss(
        logits_t=policy_logits, w_t=jnp.ones(td_returns.shape))
    l2_loss = jnp.sum(jnp.square(jax.flatten_util.ravel_pytree(net_params)[0]))
    return pi_loss + entropy_cost * ent_loss + l2_actor_weight * l2_loss

  return _a2c_pi_loss


def generate_a2c_critic_loss(net_apply, l2_critic_weight, lambda_):
  """A function generator generates loss function."""

  def _a2c_critic_loss(net_params, batch):
    info_states, rewards, discounts = batch["info_states"], batch[
        "rewards"], batch["discounts"]
    _, baselines = net_apply(net_params, info_states)
    baselines = jnp.squeeze(baselines, axis=1)
    baselines = jnp.concatenate([baselines[:-1], jnp.zeros(1)])

    td_lambda = rlax.td_lambda(
        v_tm1=baselines[:-1],
        r_t=rewards,
        discount_t=discounts,
        v_t=baselines[1:],
        lambda_=lambda_,
        stop_target_gradients=True)
    l2_loss = jnp.sum(jnp.square(jax.flatten_util.ravel_pytree(net_params)[0]))
    return jnp.mean(jnp.square(td_lambda)) + l2_critic_weight * l2_loss

  return _a2c_critic_loss


def generate_pg_pi_loss(net_apply, loss_class, entropy_cost, l2_actor_weight):
  """A function generator generates loss function."""

  def _pg_loss(net_params, batch):
    info_states = batch["info_states"]
    policy_logits, q_values = net_apply(net_params, info_states)
    chex.assert_equal_shape([policy_logits, q_values])
    pi_loss = loss_class(logits_t=policy_logits, q_t=q_values)
    ent_loss = rlax.entropy_loss(
        logits_t=policy_logits, w_t=jnp.ones(policy_logits.shape[:1]))
    l2_loss = jnp.sum(jnp.square(jax.flatten_util.ravel_pytree(net_params)[0]))
    return pi_loss + entropy_cost * ent_loss + l2_actor_weight * l2_loss

  return _pg_loss


def generate_pg_critic_loss(net_apply, l2_critic_weight, lambda_):
  """A function generator generates loss function."""

  def _critic_loss(net_params, batch):
    info_states, actions, rewards, discounts = batch["info_states"], batch[
        "actions"], batch["rewards"], batch["discounts"]
    _, q_values = net_apply(net_params, info_states)
    q_values = q_values[:-1]
    q_values = jnp.concatenate(
        [q_values, jnp.zeros(q_values[-1].reshape(1, -1).shape)])

    actions = jnp.concatenate([actions, jnp.zeros(1, dtype=int)])
    sarsa_lambda = rlax.sarsa_lambda(
        q_tm1=q_values[:-1],
        a_tm1=actions[:-1],
        r_t=rewards,
        discount_t=discounts,
        q_t=q_values[1:],
        a_t=actions[1:],
        lambda_=lambda_,
        stop_target_gradients=True)
    l2_loss = jnp.sum(jnp.square(jax.flatten_util.ravel_pytree(net_params)[0]))
    return jnp.mean(jnp.square(sarsa_lambda)) + l2_critic_weight * l2_loss

  return _critic_loss


def generate_act_func(net_apply):
  """A function generator generates act function."""

  def _act(net_params, info_state, action_mask, rng):
    info_state = jnp.reshape(info_state, [1, -1])
    policy_logits, _ = net_apply(net_params, info_state)
    policy_probs = jax.nn.softmax(policy_logits, axis=1)

    # Remove illegal actions, re-normalize probs
    probs = policy_probs[0] * action_mask

    probs /= jnp.sum(probs)
    action = jax.random.choice(rng, len(probs), p=probs)
    return action, probs

  return _act


class PolicyGradient(rl_agent.AbstractAgent):
  """Policy Gradient Agent implementation in JAX."""

  def __init__(self,
               player_id,
               info_state_size,
               num_actions,
               loss_str="a2c",
               loss_class=None,
               hidden_layers_sizes=(128,),
               lambda_=1.0,
               critic_learning_rate=0.01,
               pi_learning_rate=0.001,
               entropy_cost=0.01,
               l2_weight_actor=0.0,
               l2_weight_critic=0.0,
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
      lambda_: float, lambda in TD(lambda) or SARSA(lambda). Defaults to 1.0.
      critic_learning_rate: float, learning rate used for Critic (Q or V).
        Defaults to 0.001.
      pi_learning_rate: float, learning rate used for Pi. Defaults to 0.001.
      entropy_cost: float, entropy cost used to multiply the entropy loss. Can
        be set to None to skip entropy computation. Defaults to 0.001.
      l2_weight_actor: l2 penaly weight for actor network. Defaults to 0.0.
      l2_weight_critic: l2 penalty weight for critic network. Defaults to
        0.0.
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

    self.player_id = player_id
    self._num_actions = num_actions
    self._extra_discount = additional_discount_factor
    self._num_critic_before_pi = num_critic_before_pi

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

    self._loss_str = loss_str

    # Network
    # activate final as we plug logit and qvalue heads afterwards.
    net_class = NetA2C if loss_str == "a2c" else NetPG

    def net_func(info_input):
      net = net_class(num_actions, hidden_layers_sizes)
      return net(info_input)

    hk_net = hk.without_apply_rng(hk.transform(net_func))

    hk_net_apply = hk_net.apply
    self.rng = jax.random.PRNGKey(seed)
    init_inputs = jnp.ones((1, info_state_size))
    self.hk_net_params = hk_net.init(self.rng, init_inputs)

    self._act = jax.jit(generate_act_func(hk_net_apply))

    if optimizer_str == "adam":
      critic_optimizer = optax.adam(critic_learning_rate)
      pi_optimizer = optax.adam(pi_learning_rate)

    elif optimizer_str == "sgd":
      critic_optimizer = optax.sgd(critic_learning_rate)
      pi_optimizer = optax.sgd(pi_learning_rate)

    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    if max_global_gradient_norm:
      pi_optimizer = optax.chain(
          pi_optimizer, optax.clip_by_global_norm(max_global_gradient_norm))
      critic_optimizer = optax.chain(
          critic_optimizer, optax.clip_by_global_norm(max_global_gradient_norm))

    pi_opt_init, pi_opt_update = pi_optimizer.init, pi_optimizer.update
    critic_opt_init, critic_opt_update = critic_optimizer.init, critic_optimizer.update

    self._pi_opt_state = pi_opt_init(self.hk_net_params)

    if loss_str == "a2c":
      pi_loss_and_grad = jax.value_and_grad(
          generate_a2c_pi_loss(hk_net_apply, loss_class, entropy_cost,
                               l2_weight_actor, lambda_))
      critic_loss_and_grad = jax.value_and_grad(
          generate_a2c_critic_loss(hk_net_apply, l2_weight_critic, lambda_))
      self._critic_opt_state = critic_opt_init(self.hk_net_params)
    else:
      pi_loss_and_grad = jax.value_and_grad(
          generate_pg_pi_loss(hk_net_apply, loss_class, entropy_cost,
                              l2_weight_actor))
      critic_loss_and_grad = jax.value_and_grad(
          generate_pg_critic_loss(hk_net_apply, l2_weight_critic, lambda_))
      self._critic_opt_state = critic_opt_init(self.hk_net_params)

    self._jit_pi_update = jax.jit(
        self._get_update(pi_opt_update, pi_loss_and_grad))
    self._jit_critic_update = jax.jit(
        self._get_update(critic_opt_update, critic_loss_and_grad))

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rlax.rpg_loss
    elif loss_str == "qpg":
      return rlax.qpg_loss
    elif loss_str == "rm":
      return rlax.rm_loss
    elif loss_str == "a2c":
      return rlax.policy_gradient_loss

  def _get_update(self, opt_update, loss_fn):

    def update(net_params, opt_state, batch):
      loss_val, grad_val = loss_fn(net_params, batch)
      updates, new_opt_state = opt_update(grad_val, opt_state)
      new_net_params = optax.apply_updates(net_params, updates)
      return new_net_params, new_opt_state, loss_val

    return update

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (time_step.is_simultaneous_move() or
                                   self.player_id
                                   == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      action_mask = np.zeros(self._num_actions)
      action_mask[legal_actions] = 1
      self.rng, _ = jax.random.split(self.rng)
      action, probs = self._act(self.hk_net_params, np.asarray(info_state),
                                action_mask, self.rng)
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
        self._episode_counter += 1

        self._critic_update()
        self._num_learn_steps += 1
        if self._num_learn_steps % self._num_critic_before_pi == 0:
          self._pi_update()
        self._episode_data = []

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
        legal_actions_mask=legal_actions_mask,
        next_info_state=(
            time_step.observations["info_state"][self.player_id][:]))

    self._episode_data.append(transition)

  def _critic_update(self):
    """Compute the Critic loss on sampled transitions & perform a critic update.

    Returns:
      The average Critic loss obtained on this batch.
    """
    batch = {}
    batch["info_states"] = jnp.asarray(
        [transition.info_state for transition in self._episode_data] +
        [self._episode_data[-1].next_info_state])
    batch["rewards"] = jnp.asarray(
        [transition.reward for transition in self._episode_data])
    batch["discounts"] = jnp.asarray(
        [transition.discount for transition in self._episode_data])
    if self._loss_str != "a2c":
      batch["actions"] = jnp.asarray(
          [transition.action for transition in self._episode_data])

    self.hk_net_params, self._critic_opt_state, self._last_critic_loss_value = self._jit_critic_update(
        self.hk_net_params, self._critic_opt_state, batch)
    return self._last_critic_loss_value

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.

    Returns:
      The average Pi loss obtained on this batch.
    """
    batch = {}
    batch["info_states"] = jnp.asarray(
        [transition.info_state for transition in self._episode_data] +
        [self._episode_data[-1].next_info_state])

    if self._loss_str == "a2c":
      batch["discounts"] = jnp.asarray(
          [transition.discount for transition in self._episode_data])
      batch["actions"] = jnp.asarray(
          [transition.action for transition in self._episode_data])
      batch["rewards"] = jnp.asarray(
          [transition.reward for transition in self._episode_data])
    self.hk_net_params, self._pi_opt_state, self._last_pi_loss_value = self._jit_pi_update(
        self.hk_net_params, self._pi_opt_state, batch)
    return self._last_pi_loss_value
