# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AlphaZero Bot implemented in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel


class AlphaZero(object):
  """AlphaZero implementation.

  Follows the pseudocode AlphaZero implementation given in the paper
  DOI:10.1126/science.aar6404.
  """

  def __init__(self,
               game,
               bot,
               model,
               replay_buffer_capacity=int(1e6),
               action_selection_transition=30):
    """AlphaZero constructor.

    Args:
      game: a pyspiel.Game object
      bot: an MCTSBot object.
      model: A Model.
      replay_buffer_capacity: the size of the replay buffer in which the results
        of self-play games are stored.
      action_selection_transition: an integer representing the move number in a
        game of self-play when greedy action selection is used. Before this,
        actions are sampled from the MCTS policy.

    Raises:
      ValueError: if incorrect inputs are supplied.
    """

    game_info = game.get_type()
    if game.num_players() != 2:
      raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
      raise ValueError("The game must be a Deterministic one, not {}".format(
          game.chance_mode))
    if (game_info.information !=
        pyspiel.GameType.Information.PERFECT_INFORMATION):
      raise ValueError(
          "The game must be a perfect information one, not {}".format(
              game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("The game must be turn-based, not {}".format(
          game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
      raise ValueError("The game must be 0-sum, not {}".format(game.utility))
    if game.num_players() != 2:
      raise ValueError("Game must have exactly 2 players.")

    self.game = game
    self.bot = bot
    self.model = model
    self.replay_buffer = dqn.ReplayBuffer(replay_buffer_capacity)
    self.action_selection_transition = action_selection_transition

  def update(self, num_training_epochs=10, batch_size=128, verbose=False):
    """Trains the neural net.

    Randomly sampls data from the replay buffer. An update resets the optimizer
    state.

    Args:
      num_training_epochs: An epoch represents one pass over the training data.
        The total number training iterations this corresponds to is
        num_training_epochs * len(replay_buffer)/batch_size.
      batch_size: the number of examples sampled from the replay buffer and
        used for each net training iteration.
      verbose: whether to print training metrics during training.

    Returns:
      A list of length num_training_epochs. Each element of this list is
        a Losses tuples, averaged per epoch.
    """
    num_epoch_iters = math.ceil(len(self.replay_buffer) / float(batch_size))
    losses = []
    for epoch in range(num_training_epochs):
      epoch_losses = []
      for _ in range(num_epoch_iters):
        train_data = self.replay_buffer.sample(batch_size)
        epoch_losses.append(self.model.update(train_data))

      epoch_losses = (sum(epoch_losses, model_lib.Losses(0, 0, 0)) /
                      len(epoch_losses))
      losses.append(epoch_losses)
      if verbose:
        print("Epoch {}: {}".format(epoch, epoch_losses))

    return losses

  def self_play(self, num_self_play_games=5000):
    """Uses the current state of the net with MCTS to play full games against.

    Args:
      num_self_play_games: the number of self-play games to play using the
        current net and MCTS.
    """
    for _ in range(num_self_play_games):
      self._self_play_single()

  def _self_play_single(self):
    """Play a single game and add it to the replay buffer."""
    state = self.game.new_initial_state()
    trajectory = []

    while not state.is_terminal():
      root = self.bot.mcts_search(state)
      target_policy = np.zeros(self.game.num_distinct_actions(),
                               dtype=np.float32)
      for child in root.children:
        target_policy[child.action] = child.explore_count
      target_policy /= sum(target_policy)

      trajectory.append(model_lib.TrainInput(
          state.observation_tensor(), state.legal_actions_mask(),
          target_policy, root.total_reward / root.explore_count))

      action = self._select_action(root.children, len(trajectory))
      state.apply_action(action)

    terminal_rewards = state.rewards()
    for state in trajectory:
      self.replay_buffer.add(
          model_lib.TrainInput(state.observation, state.legals_mask,
                               state.policy, terminal_rewards[0]))

  def _select_action(self, children, game_history_len):
    explore_counts = [(child.explore_count, child.action) for child in children]
    if game_history_len < self.action_selection_transition:
      probs = np_softmax(np.array([i[0] for i in explore_counts]))
      action_index = np.random.choice(range(len(probs)), p=probs)
      action = explore_counts[action_index][1]
    else:
      _, action = max(explore_counts)
    return action


def np_softmax(logits):
  max_logit = np.amax(logits, axis=-1, keepdims=True)
  exp_logit = np.exp(logits - max_logit)
  return exp_logit / np.sum(exp_logit, axis=-1, keepdims=True)

