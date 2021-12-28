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

"""Implements an Ephemeral Value Adjustment Agent.

See https://arxiv.org/abs/1810.08163.
The algorithm queries trajectories from a replay buffer based on similarities
to embedding representations and uses a parametric model to compute values for
counterfactual state-action pairs when integrating across those trajectories.
Finally, a weighted average between the parametric (DQN in this case) and the
non-parametric model is used to compute the policy.
"""

# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import numpy as np
import torch

from open_spiel.python import rl_agent
from open_spiel.python.pytorch import dqn

MEM_KEY_NAME = "embedding"

ValueBufferElement = collections.namedtuple("ValueElement", "embedding value")

ReplayBufferElement = collections.namedtuple(
    "ReplayElement", "embedding info_state action reward next_info_state "
    "is_final_step legal_actions_mask")


# TODO(author3) Refactor into data structures lib.
class QueryableFixedSizeRingBuffer(dqn.ReplayBuffer):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.  This extends the DQN replay
  buffer by allowing the contents to be fetched by L2 proximity to a query
  value.
  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def knn(self, key, key_name, k, trajectory_len=1):
    """Computes top-k neighbours based on L2 distance.

    Args:
      key: (np.array) key value to query memory.
      key_name:  (str) attribute name of key in memory elements.
      k: (int) number of neighbours to fetch.
      trajectory_len: (int) length of trajectory to fetch from replay buffer.

    Returns:
      List of tuples (L2 negative distance, BufferElement) sorted in increasing
      order by the negative L2 distqances  from the key.
    """
    distances = [(np.linalg.norm(getattr(sample, key_name) - key, 2,
                                 axis=0), sample) for sample in self._data]
    return sorted(distances, key=lambda v: -v[0])[:k]


class EVAAgent(object):
  """Implements a solver for Ephemeral VAlue Adjustment.

  See https://arxiv.org/abs/1810.08163.
  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.
  """

  def __init__(self,
               game,
               player_id,
               state_size,
               num_actions,
               embedding_network_layers=(128,),
               embedding_size=16,
               dqn_hidden_layers=(128, 128),
               batch_size=16,
               trajectory_len=10,
               num_neighbours=5,
               learning_rate=1e-4,
               mixing_parameter=0.9,
               memory_capacity=int(1e6),
               discount_factor=1.0,
               update_target_network_every=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e4),
               embedding_as_parametric_input=False):
    """Initialize the Ephemeral VAlue Adjustment algorithm.

    Args:
      game: (rl_environment.Environment) Open Spiel game.
      player_id: (int) Player id for this player.
      state_size: (int) Size of info state vector.
      num_actions: (int) number of actions.
      embedding_network_layers: (list[int]) Layer sizes of strategy net MLP.
      embedding_size: (int) Size of memory embeddings.
      dqn_hidden_layers: (list(int)) MLP layer sizes of DQN network.
      batch_size: (int) Size of batches for DQN learning steps.
      trajectory_len: (int) Length of trajectories from replay buffer.
      num_neighbours: (int) Number of neighbours to fetch from replay buffer.
      learning_rate: (float) Learning rate.
      mixing_parameter: (float) Value mixing parameter between 0 and 1.
      memory_capacity: Number af samples that can be stored in memory.
      discount_factor: (float) Discount factor for Q-Learning.
      update_target_network_every: How often to update DQN target network.
      epsilon_start: (float) Starting epsilon-greedy value.
      epsilon_end: (float) Final epsilon-greedy value.
      epsilon_decay_duration: (float) Number of steps over which epsilon decays.
      embedding_as_parametric_input: (bool) Whether we use embeddings as input
        to the parametric model.
    """
    assert (mixing_parameter >= 0 and mixing_parameter <= 1)
    self._game = game
    self.player_id = player_id
    self._env = game
    self._num_actions = num_actions
    self._info_state_size = state_size
    self._embedding_size = embedding_size
    self._lambda = mixing_parameter
    self._trajectory_len = trajectory_len
    self._num_neighbours = num_neighbours
    self._discount = discount_factor
    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration
    self._last_time_step = None
    self._last_action = None
    self._embedding_as_parametric_input = embedding_as_parametric_input

    self._embedding_network = dqn.MLP(self._info_state_size,
                                      list(embedding_network_layers),
                                      embedding_size)

    # The DQN agent requires this be an integer.
    if not isinstance(memory_capacity, int):
      raise ValueError("Memory capacity not an integer.")

    # Initialize the parametric & non-parametric Q-networks.
    self._agent = dqn.DQN(
        player_id,
        state_representation_size=self._info_state_size,
        num_actions=self._num_actions,
        hidden_layers_sizes=list(dqn_hidden_layers),
        replay_buffer_capacity=memory_capacity,
        replay_buffer_class=QueryableFixedSizeRingBuffer,
        batch_size=batch_size,
        learning_rate=learning_rate,
        update_target_network_every=update_target_network_every,
        learn_every=batch_size,
        discount_factor=1.0,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_duration=int(1e6))
    # Initialize Value Buffers - Fetch Replay buffers from agents.
    self._value_buffer = QueryableFixedSizeRingBuffer(memory_capacity)
    self._replay_buffer = self._agent.replay_buffer

    # Initialize non-parametric & EVA Q-values.
    self._v_np = collections.defaultdict(float)
    self._q_np = collections.defaultdict(lambda: [0] * self._num_actions)
    self._q_eva = collections.defaultdict(lambda: [0] * self._num_actions)

  @property
  def env(self):
    return self._env

  @property
  def loss(self):
    return self._agent.loss

  def _add_transition_value(self, infostate_embedding, value):
    """Adds the embedding and value to the ValueBuffer.

    Args:
      infostate_embedding: (np.array) embeddig vector.
      value: (float) Value associated with state embeding.
    """
    transition = ValueBufferElement(embedding=infostate_embedding, value=value)
    self._value_buffer.add(transition)

  def _add_transition_replay(self, infostate_embedding, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.
    Args:
      infostate_embedding: embeddig vector.
      time_step: an instance of rl_environment.TimeStep.
    """
    prev_timestep = self._last_time_step
    assert prev_timestep is not None
    legal_actions = (
        prev_timestep.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    reward = time_step.rewards[self.player_id] if time_step.rewards else 0.0
    transition = ReplayBufferElement(
        embedding=infostate_embedding,
        info_state=(prev_timestep.observations["info_state"][self.player_id]),
        action=self._last_action,
        reward=reward,
        next_info_state=time_step.observations["info_state"][self.player_id],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask)
    self._replay_buffer.add(transition)

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the value functions.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states.
    if not time_step.last():
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      epsilon = self._get_epsilon(self._agent.step_counter, is_evaluation)

      # Sample an action from EVA via epsilon greedy policy.
      action, probs = self._epsilon_greedy(self._q_eva[tuple(info_state)],
                                           legal_actions, epsilon)

    # Update Step: Only with transitions and not when evaluating.
    if (not is_evaluation and self._last_time_step is not None):
      info_state = self._last_time_step.observations["info_state"][
          self.player_id]
      legal_actions = self._last_time_step.observations["legal_actions"][
          self.player_id]
      epsilon = self._get_epsilon(self._agent.step_counter, is_evaluation)

      # Get embedding.
      self._info_state = torch.Tensor(np.expand_dims(info_state, axis=0))
      infostate_embedding = self._embedding_network(
          self._info_state).detach()[0]

      neighbours_value = self._value_buffer.knn(infostate_embedding,
                                                MEM_KEY_NAME,
                                                self._num_neighbours, 1)
      # collect trace values of knn from L (value buffer) .. Q_np(s_k)
      neighbours_replay = self._replay_buffer.knn(infostate_embedding,
                                                  MEM_KEY_NAME,
                                                  self._num_neighbours,
                                                  self._trajectory_len)

      # Take a step with the parametric model and get q-values. Use embedding as
      # input to the parametric meodel.
      # TODO(author6) Recompute embeddings for buffers on learning steps.
      if self._embedding_as_parametric_input:
        last_time_step_copy = copy.deepcopy(self._last_time_step)
        last_time_step_copy.observations["info_state"][
            self.player_id] = infostate_embedding
        self._agent.step(last_time_step_copy, add_transition_record=False)
      else:
        self._agent.step(self._last_time_step, add_transition_record=False)
      q_values = self._agent._q_network(self._info_state).detach()[0]
      # Update EVA: Q_eva = lambda q_theta(s_t) + (1-lambda) sum(Q_np(s_k, .))/K
      for a in legal_actions:
        q_theta = q_values[a]
        self._q_eva[tuple(info_state)][a] = (
            self._lambda * q_theta + (1 - self._lambda) *
            sum([elem[1].value
                 for elem in neighbours_value]) / self._num_neighbours)

      # Append (e,s,a,r,s') to Replay Buffer
      self._add_transition_replay(infostate_embedding, time_step)

      # update Q_np with Traces using TCP
      self._trajectory_centric_planning(neighbours_replay)

      # Append Q_np(s, a) to Value Buffer
      self._add_transition_value(
          infostate_embedding, self._q_np[tuple(info_state)][self._last_action])

    # Prepare for the next episode.
    if time_step.last():
      self._last_time_step = None
      self._last_action = None
      return

    self._last_time_step = time_step
    self._last_action = action
    return rl_agent.StepOutput(action=action, probs=probs)

  def _trajectory_centric_planning(self, trajectories):
    """Performs trajectory centric planning.

    Uses trajectories from the replay buffer to update the non-parametric values
    while supplying counter-factual values with the parametric model.
    Args:
      trajectories: Current OpenSpiel game state.
    """
    # Calculate non-parametric values over the trajectories.
    # Iterate backward through trajectories
    for t in range(len(trajectories) - 1, 0, -1):
      elem = trajectories[t][1]
      s_tp1 = tuple(elem.next_info_state)
      s_t = tuple(elem.info_state)
      a_t = elem.action
      r_t = elem.reward
      legal_actions = elem.legal_actions_mask
      if t < len(trajectories) - 1:
        for action in range(len(legal_actions)):
          if not legal_actions[action]:
            continue
          if action == elem.action:
            self._q_np[s_t][a_t] = (r_t + self._discount * self._v_np[s_tp1])
          else:
            self._agent.info_state = torch.Tensor(
                np.expand_dims(elem.info_state, axis=0))
            q_values_parametric = self._agent._q_network(
                self._agent.info_state).detach().numpy()
            self._q_np[s_t][a_t] = q_values_parametric[0][action]

      # Set V(s_t)
      if t == len(trajectories) - 1:
        # Sample from the parametric model.
        self._agent.info_state = torch.Tensor(
            np.expand_dims(elem.info_state, axis=0))
        q_values_parametric = self._agent._q_network(
            self._agent.info_state).detach().numpy()
        self._v_np[s_t] = np.max(q_values_parametric)
      else:
        self._v_np[s_t] = max(self._q_np[s_t])

  def _epsilon_greedy(self, q_values, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.
    Args:
      q_values: list of Q-values by action.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    q_values = np.array(q_values)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      legal_q_values = q_values[legal_actions]
      action = legal_actions[np.argmax(legal_q_values)]
      # Reduce max_q for numerical stability. Result is the same.
      max_q = np.max(legal_q_values)
      e_x = np.exp(legal_q_values - max_q)
      probs[legal_actions] = e_x / e_x.sum(axis=0)
    return action, probs

  def _get_epsilon(self, step_counter, is_evaluation):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration))
    return decayed_epsilon

  def action_probabilities(self, state):
    """Returns action probabilites dict for a single batch."""
    # TODO(author3, author6): Refactor this to expect pre-normalized form.
    if hasattr(state, "information_state_tensor"):
      state_rep = tuple(state.information_state_tensor(self.player_id))
    elif hasattr(state, "observation_tensor"):
      state_rep = tuple(state.observation_tensor(self.player_id))
    else:
      raise AttributeError("Unable to extract normalized state vector.")
    legal_actions = state.legal_actions(self.player_id)
    if legal_actions:
      _, probs = self._epsilon_greedy(
          self._q_eva[state_rep], legal_actions, epsilon=0.0)
      return {a: probs[a] for a in range(self._num_actions)}
    else:
      raise ValueError("Node has no legal actions to take.")
