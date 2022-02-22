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

"""DQN agent implemented in TensorFlow."""

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
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in TensorFlow.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(self,
               session,
               player_id,
               state_representation_size,
               num_actions,
               hidden_layers_sizes=128,
               replay_buffer_capacity=10000,
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=0.01,
               update_target_network_every=1000,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e6),
               optimizer_str="sgd",
               loss_str="mse"):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()

    self.player_id = player_id
    self._session = session
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration

    # TODO(author6) Allow for optional replay buffer config.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create required TensorFlow placeholders to perform the Q-network updates.
    self._info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="info_state_ph")
    self._action_ph = tf.placeholder(
        shape=[None], dtype=tf.int32, name="action_ph")
    self._reward_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="reward_ph")
    self._is_final_step_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="is_final_step_ph")
    self._next_info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="next_info_state_ph")
    self._legal_actions_mask_ph = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="legal_actions_mask_ph")

    self._q_network = simple_nets.MLP(state_representation_size,
                                      self._layer_sizes, num_actions)
    self._q_values = self._q_network(self._info_state_ph)

    self._target_q_network = simple_nets.MLP(state_representation_size,
                                             self._layer_sizes, num_actions)
    self._target_q_values = self._target_q_network(self._next_info_state_ph)

    # Stop gradient to prevent updates to the target network while learning
    self._target_q_values = tf.stop_gradient(self._target_q_values)

    self._update_target_network = self._create_target_network_update_op(
        self._q_network, self._target_q_network)

    # Create the loss operations.
    # Sum a large negative constant to illegal action logits before taking the
    # max. This prevents illegal action values from being considered as target.
    illegal_actions = 1 - self._legal_actions_mask_ph
    illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY
    max_next_q = tf.reduce_max(
        tf.math.add(tf.stop_gradient(self._target_q_values), illegal_logits),
        axis=-1)
    target = (
        self._reward_ph +
        (1 - self._is_final_step_ph) * self._discount_factor * max_next_q)

    action_indices = tf.stack(
        [tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
    predictions = tf.gather_nd(self._q_values, action_indices)

    self._savers = [("q_network", tf.train.Saver(self._q_network.variables)),
                    ("target_q_network",
                     tf.train.Saver(self._target_q_network.variables))]

    if loss_str == "mse":
      loss_class = tf.losses.mean_squared_error
    elif loss_str == "huber":
      loss_class = tf.losses.huber_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    self._loss = tf.reduce_mean(
        loss_class(labels=target, predictions=predictions))

    if optimizer_str == "adam":
      self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    self._learn_step = self._optimizer.minimize(self._loss)
    self._initialize()

  def get_step_counter(self):
    return self._step_counter

  def step(self, time_step, is_evaluation=False, add_transition_record=True):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      epsilon = self._get_epsilon(is_evaluation)
      action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._step_counter += 1

      if self._step_counter % self._learn_every == 0:
        self._last_loss_value = self.learn()

      if self._step_counter % self._update_target_network_every == 0:
        self._session.run(self._update_target_network)

      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        self.add_transition(self._prev_timestep, self._prev_action, time_step)

      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    legal_actions = (time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask)
    self._replay_buffer.add(transition)

  def _create_target_network_update_op(self, q_network, target_q_network):
    """Create TF ops copying the params of the Q-network to the target network.

    Args:
      q_network: A q-network object that implements provides the `variables`
                 property representing the TF variable list.
      target_q_network: A target q-net object that provides the `variables`
                        property representing the TF variable list.

    Returns:
      A `tf.Operation` that updates the variables of the target.
    """
    self._variables = q_network.variables[:]
    self._target_variables = target_q_network.variables[:]
    assert self._variables
    assert len(self._variables) == len(self._target_variables)
    return tf.group([
        tf.assign(target_v, v)
        for (target_v, v) in zip(self._target_variables, self._variables)
    ])

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      info_state = np.reshape(info_state, [1, -1])
      q_values = self._session.run(
          self._q_values, feed_dict={self._info_state_ph: info_state})[0]
      legal_q_values = q_values[legal_actions]
      action = legal_actions[np.argmax(legal_q_values)]
      probs[action] = 1.0
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer.sample(self._batch_size)
    info_states = [t.info_state for t in transitions]
    actions = [t.action for t in transitions]
    rewards = [t.reward for t in transitions]
    next_info_states = [t.next_info_state for t in transitions]
    are_final_steps = [t.is_final_step for t in transitions]
    legal_actions_mask = [t.legal_actions_mask for t in transitions]
    loss, _ = self._session.run(
        [self._loss, self._learn_step],
        feed_dict={
            self._info_state_ph: info_states,
            self._action_ph: actions,
            self._reward_ph: rewards,
            self._is_final_step_ph: are_final_steps,
            self._next_info_state_ph: next_info_states,
            self._legal_actions_mask_ph: legal_actions_mask,
        })
    return loss

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    """Saves the q network and the target q-network.

    Note that this does not save the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory where checkpoints will be saved.
    """
    for name, saver in self._savers:
      path = saver.save(
          self._session,
          self._full_checkpoint_name(checkpoint_dir, name),
          latest_filename=self._latest_checkpoint_filename(name))
      logging.info("Saved to path: %s", path)

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  def restore(self, checkpoint_dir):
    """Restores the q network and the target q-network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory from which checkpoints will be restored.
    """
    for name, saver in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      saver.restore(self._session, full_checkpoint_dir)

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def info_state_ph(self):
    return self._info_state_ph

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter

  def _initialize(self):
    initialization_weights = tf.group(
        *[var.initializer for var in self._variables])
    initialization_target_weights = tf.group(
        *[var.initializer for var in self._target_variables])
    initialization_opt = tf.group(
        *[var.initializer for var in self._optimizer.variables()])

    self._session.run(
        tf.group(*[
            initialization_weights, initialization_target_weights,
            initialization_opt,
        ]))

  def get_weights(self):
    variables = [self._session.run(self._q_network.variables)]
    variables.append(self._session.run(self._target_q_network.variables))
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates it with noise.

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
    copied_object = DQN(**self._kwargs)

    q_network = getattr(copied_object, "_q_network")
    target_q_network = getattr(copied_object, "_target_q_network")

    if copy_weights:
      copy_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(q_network.variables, self._q_network.variables)
      ])
      self._session.run(copy_weights)

      copy_target_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(target_q_network.variables,
                            self._target_q_network.variables)
      ])
      self._session.run(copy_target_weights)
    return copied_object
