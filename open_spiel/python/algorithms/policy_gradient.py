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

# Lint as python3.
r"""Policy Gradient based agents implemented in TensorFlow.

This class is composed of three policy gradient (PG) algorithms:

- Q-based Policy Gradient (QPG): an "all-actions" advantage actor-critic
algorithm differing from A2C in that all action values are used to estimate the
policy gradient (as opposed to only using the action taken into account):

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * (Q_a - baseline)

where (Q_a - baseline) is the usual advantage. QPG is also known as Mean
Actor-Critic (https://arxiv.org/abs/1709.00503).


- Regret policy gradient (RPG): a PG algorithm inspired by counterfactual regret
minimization (CFR). Unlike standard actor-critic methods (e.g. A2C), the loss is
defined purely in terms of thresholded regrets as follows:

    baseline = \sum_a pi_a * Q_a
    loss = regret = \sum_a relu(Q_a - baseline)

where gradients only flow through the action value (Q_a) part and are blocked on
the baseline part (which is trained separately by usual MSE loss).
The lack of negative sign in the front of the loss represents a switch from
gradient ascent on the score to descent on the loss.


- Regret Matching Policy Gradient (RMPG): inspired by regret-matching, the
policy gradient is by weighted by the thresholded regret:

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * relu(Q_a - baseline)


These algorithms were published in NeurIPS 2018. Paper title: "Actor-Critic
Policy Optimization in Partially Observable Multiagent Environment", the paper
is available at: https://arxiv.org/abs/1810.09026.

- Advantage Actor Critic (A2C): The popular advantage actor critic (A2C)
algorithm. The algorithm uses the baseline (Value function) as a control variate
to reduce variance of the policy gradient. The loss is only computed for the
actions actually taken in the episode as opposed to a loss computed for all
actions in the variants above.

  advantages = returns - baseline
  loss = -log(pi_a) * advantages

The algorithm can be found in the textbook:
https://incompleteideas.net/book/RLbook2018.pdf under the chapter on
`Policy Gradients`.

See  open_spiel/python/algorithms/losses/rl_losses_test.py for an example of the
loss computation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.algorithms.losses import rl_losses

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")


class PolicyGradient(rl_agent.AbstractAgent):
  """RPG Agent implementation in TensorFlow.

  See open_spiel/python/examples/single_agent_catch.py for an usage example.
  """

  def __init__(self,
               session,
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
               optimizer_str="sgd"):
    """Initialize the PolicyGradient agent.

    Args:
      session: Tensorflow session.
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
    """
    assert bool(loss_str) ^ bool(loss_class), "Please provide only one option."
    self._kwargs = locals()
    loss_class = loss_class if loss_class else self._get_loss_class(loss_str)
    self._loss_class = loss_class

    self.player_id = player_id
    self._session = session
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
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

    # Placeholders
    self._info_state_ph = tf.placeholder(
        shape=[None, info_state_size], dtype=tf.float32, name="info_state_ph")
    self._action_ph = tf.placeholder(
        shape=[None], dtype=tf.int32, name="action_ph")
    self._return_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="return_ph")

    # Network
    # activate final as we plug logit and qvalue heads afterwards.
    self._net_torso = simple_nets.MLPTorso(info_state_size, self._layer_sizes)
    torso_out = self._net_torso(self._info_state_ph)
    torso_out_size = self._layer_sizes[-1]
    self._policy_logits_layer = simple_nets.Linear(
        torso_out_size,
        self._num_actions,
        activate_relu=False,
        name="policy_head")
    # Do not remove policy_logits_network. Even if it's not used directly here,
    # other code outside this file refers to it.
    self.policy_logits_network = simple_nets.Sequential(
        [self._net_torso, self._policy_logits_layer])
    self._policy_logits = self._policy_logits_layer(torso_out)
    self._policy_probs = tf.nn.softmax(self._policy_logits)

    self._savers = []

    # Add baseline (V) head for A2C (or Q-head for QPG / RPG / RMPG)
    if loss_class.__name__ == "BatchA2CLoss":
      self._baseline_layer = simple_nets.Linear(
          torso_out_size, 1, activate_relu=False, name="baseline")
      self._baseline = tf.squeeze(self._baseline_layer(torso_out), axis=1)
    else:
      self._q_values_layer = simple_nets.Linear(
          torso_out_size,
          self._num_actions,
          activate_relu=False,
          name="q_values_head")
      self._q_values = self._q_values_layer(torso_out)

    # Critic loss
    # Baseline loss in case of A2C
    if loss_class.__name__ == "BatchA2CLoss":
      self._critic_loss = tf.reduce_mean(
          tf.losses.mean_squared_error(
              labels=self._return_ph, predictions=self._baseline))
    else:
      # Q-loss otherwise.
      action_indices = tf.stack(
          [tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
      value_predictions = tf.gather_nd(self._q_values, action_indices)
      self._critic_loss = tf.reduce_mean(
          tf.losses.mean_squared_error(
              labels=self._return_ph, predictions=value_predictions))
    if optimizer_str == "adam":
      self._critic_optimizer = tf.train.AdamOptimizer(
          learning_rate=critic_learning_rate)
    elif optimizer_str == "sgd":
      self._critic_optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=critic_learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    def minimize_with_clipping(optimizer, loss):
      grads_and_vars = optimizer.compute_gradients(loss)
      if max_global_gradient_norm is not None:
        grads, variables = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, max_global_gradient_norm)
        grads_and_vars = list(zip(grads, variables))

      return optimizer.apply_gradients(grads_and_vars)

    self._critic_learn_step = minimize_with_clipping(self._critic_optimizer,
                                                     self._critic_loss)

    # Pi loss
    pg_class = loss_class(entropy_cost=entropy_cost)
    if loss_class.__name__ == "BatchA2CLoss":
      self._pi_loss = pg_class.loss(
          policy_logits=self._policy_logits,
          baseline=self._baseline,
          actions=self._action_ph,
          returns=self._return_ph)
    else:
      self._pi_loss = pg_class.loss(
          policy_logits=self._policy_logits, action_values=self._q_values)
    if optimizer_str == "adam":
      self._pi_optimizer = tf.train.AdamOptimizer(
          learning_rate=pi_learning_rate)
    elif optimizer_str == "sgd":
      self._pi_optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=pi_learning_rate)

    self._pi_learn_step = minimize_with_clipping(self._pi_optimizer,
                                                 self._pi_loss)
    self._loss_str = loss_str
    self._initialize()

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rl_losses.BatchRPGLoss
    elif loss_str == "qpg":
      return rl_losses.BatchQPGLoss
    elif loss_str == "rm":
      return rl_losses.BatchRMLoss
    elif loss_str == "a2c":
      return rl_losses.BatchA2CLoss

  def _act(self, info_state, legal_actions):
    # Make a singleton batch for NN compatibility: [1, info_state_size]
    info_state = np.reshape(info_state, [1, -1])
    policy_probs = self._session.run(
        self._policy_probs, feed_dict={self._info_state_ph: info_state})

    # Remove illegal actions, re-normalize probs
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = policy_probs[0][legal_actions]
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

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    for name, saver in self._savers:
      path = saver.save(
          self._session,
          self._full_checkpoint_name(checkpoint_dir, name),
          latest_filename=self._latest_checkpoint_filename(name))
      logging.info("saved to path: %s", path)

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  def restore(self, checkpoint_dir):
    for name, saver in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      saver.restore(self._session, full_checkpoint_dir)

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
    critic_loss, _ = self._session.run(
        [self._critic_loss, self._critic_learn_step],
        feed_dict={
            self._info_state_ph: self._dataset["info_states"],
            self._action_ph: self._dataset["actions"],
            self._return_ph: self._dataset["returns"],
        })
    self._last_critic_loss_value = critic_loss
    return critic_loss

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.

    Returns:
      The average Pi loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    pi_loss, _ = self._session.run(
        [self._pi_loss, self._pi_learn_step],
        feed_dict={
            self._info_state_ph: self._dataset["info_states"],
            self._action_ph: self._dataset["actions"],
            self._return_ph: self._dataset["returns"],
        })
    self._last_pi_loss_value = pi_loss
    return pi_loss

  def get_weights(self):
    variables = [self._session.run(self._net_torso.variables)]
    variables.append(self._session.run(self._policy_logits_layer.variables))
    if self._loss_class.__name__ == "BatchA2CLoss":
      variables.append(self._session.run(self._baseline_layer.variables))
    else:
      variables.append(self._session.run(self._q_values_layer.variables))
    return variables

  def _initialize(self):
    initialization_torso = tf.group(
        *[var.initializer for var in self._net_torso.variables])
    initialization_logit = tf.group(
        *[var.initializer for var in self._policy_logits_layer.variables])
    if self._loss_class.__name__ == "BatchA2CLoss":
      initialization_baseline_or_q_val = tf.group(
          *[var.initializer for var in self._baseline_layer.variables])
    else:
      initialization_baseline_or_q_val = tf.group(
          *[var.initializer for var in self._q_values_layer.variables])
    initialization_crit_opt = tf.group(
        *[var.initializer for var in self._critic_optimizer.variables()])
    initialization_pi_opt = tf.group(
        *[var.initializer for var in self._pi_optimizer.variables()])

    self._session.run(
        tf.group(*[
            initialization_torso, initialization_logit,
            initialization_baseline_or_q_val, initialization_crit_opt,
            initialization_pi_opt
        ]))
    self._savers = [("torso", tf.train.Saver(self._net_torso.variables)),
                    ("policy_head",
                     tf.train.Saver(self._policy_logits_layer.variables))]
    if self._loss_class.__name__ == "BatchA2CLoss":
      self._savers.append(
          ("baseline", tf.train.Saver(self._baseline_layer.variables)))
    else:
      self._savers.append(
          ("q_head", tf.train.Saver(self._q_values_layer.variables)))

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

    net_torso = getattr(copied_object, "_net_torso")
    policy_logits_layer = getattr(copied_object, "_policy_logits_layer")
    if hasattr(copied_object, "_q_values_layer"):
      q_values_layer = getattr(copied_object, "_q_values_layer")
    if hasattr(copied_object, "_baseline_layer"):
      baseline_layer = getattr(copied_object, "_baseline_layer")

    if copy_weights:
      copy_mlp_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(net_torso.variables, self._net_torso.variables)
      ])
      self._session.run(copy_mlp_weights)

      copy_logit_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(policy_logits_layer.variables,
                            self._policy_logits_layer.variables)
      ])
      self._session.run(copy_logit_weights)
      if hasattr(copied_object, "_q_values_layer"):
        copy_q_value_weights = tf.group(*[
            va.assign(vb * (1 + sigma * tf.random.normal(vb.shape))) for va, vb
            in zip(q_values_layer.variables, self._q_values_layer.variables)
        ])
        self._session.run(copy_q_value_weights)
      if hasattr(copied_object, "_baseline_layer"):
        copy_baseline_weights = tf.group(*[
            va.assign(vb * (1 + sigma * tf.random.normal(vb.shape))) for va, vb
            in zip(baseline_layer.variables, self._baseline_layer.variables)
        ])
        self._session.run(copy_baseline_weights)

    for var in getattr(copied_object, "_critic_optimizer").variables():
      self._session.run(var.initializer)
    for var in getattr(copied_object, "_pi_optimizer").variables():
      self._session.run(var.initializer)

    return copied_object
