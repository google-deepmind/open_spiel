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

"""Mirror Descent (https://arxiv.org/pdf/2103.00623.pdf)."""
from typing import Optional

import numpy as np

from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import distribution
import pyspiel


def softmax_projection(logits):
  max_l = max(logits)
  exp_l = [np.exp(l - max_l) for l in logits]
  norm_exp = sum(exp_l)
  return [l / norm_exp for l in exp_l]


class ProjectedPolicy(policy_std.Policy):
  """Project values on the policy simplex."""

  def __init__(self, game, player_ids,
               cumulative_state_value: value.ValueFunction):
    """Initializes the projected policy.

    Args:
      game: The game to analyze.
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
      cumulative_state_value: The cumulative state value to project.
    """
    super(ProjectedPolicy, self).__init__(game, player_ids)
    self._cumulative_state_value = cumulative_state_value

  def cumulative_value(self, state, action=None):
    if action is None:
      return self._cumulative_state_value(
          state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))
    else:
      new_state = state.child(action)
      return state.rewards()[0] + self._cumulative_state_value(
          new_state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID))

  def action_probabilities(self, state, player_id=None):
    action_logit = [(a, self.cumulative_value(state, action=a))
                    for a in state.legal_actions()]
    action, logit = zip(*action_logit)
    prob = softmax_projection(logit)
    action_prob = zip(action, prob)
    return dict(action_prob)


class MirrorDescent(object):
  """The mirror descent algorithm."""

  def __init__(self,
               game,
               state_value: Optional[value.ValueFunction] = None,
               lr=0.01,
               root_state=None):
    """Initializes mirror descent.

    Args:
      game: The game,
      state_value: A state value function. Default to TabularValueFunction.
      lr: The learning rate of mirror descent,
      root_state: The state of the game at which to start. If `None`, the game
        root state is used.
    """
    self._game = game
    if root_state is None:
      self._root_states = game.new_initial_states()
    else:
      self._root_states = [root_state]
    self._policy = policy_std.UniformRandomPolicy(game)
    self._distribution = distribution.DistributionPolicy(game, self._policy)
    self._md_step = 0
    self._lr = lr

    self._state_value = (state_value if state_value
                         else value.TabularValueFunction(game))
    self._cumulative_state_value = value.TabularValueFunction(game)

  def eval_state(self, state, learning_rate):
    """Evaluate the value of a state and update the cumulative sum."""
    state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
    if self._state_value.has(state_str):
      return self._state_value(state_str)
    elif state.is_terminal():
      self._state_value.set_value(
          state_str,
          state.rewards()[state.mean_field_population()])
      self._cumulative_state_value.add_value(
          state_str, learning_rate * self._state_value(state_str))
      return self._state_value(state_str)
    elif state.current_player() == pyspiel.PlayerId.CHANCE:
      self._state_value.set_value(state_str, 0.0)
      for action, prob in state.chance_outcomes():
        new_state = state.child(action)
        self._state_value.add_value(
            state_str, prob * self.eval_state(new_state, learning_rate))
      self._cumulative_state_value.add_value(
          state_str, learning_rate * self._state_value(state_str))
      return self._state_value(state_str)
    elif state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
      dist_to_register = state.distribution_support()
      dist = [
          self._distribution.value_str(str_state, 0.0)
          for str_state in dist_to_register
      ]
      new_state = state.clone()
      new_state.update_distribution(dist)
      self._state_value.set_value(
          state_str,
          state.rewards()[state.mean_field_population()] +
          self.eval_state(new_state, learning_rate))
      self._cumulative_state_value.add_value(
          state_str, learning_rate * self._state_value(state_str))
      return self._state_value(state_str)
    else:
      assert int(state.current_player()) >= 0, "The player id should be >= 0"
      v = 0.0
      for action, prob in self._policy.action_probabilities(state).items():
        new_state = state.child(action)
        v += prob * self.eval_state(new_state, learning_rate)
      self._state_value.set_value(
          state_str,
          state.rewards()[state.mean_field_population()] + v)
      self._cumulative_state_value.add_value(
          state_str, learning_rate * self._state_value(state_str))
      return self._state_value(state_str)

  def iteration(self, learning_rate=None):
    """an iteration of Mirror Descent."""
    self._md_step += 1
    # TODO(sertan): Fix me.
    self._state_value = value.TabularValueFunction(self._game)
    for state in self._root_states:
      self.eval_state(state, learning_rate if learning_rate else self._lr)
    self._policy = ProjectedPolicy(self._game,
                                   list(range(self._game.num_players())),
                                   self._cumulative_state_value)
    self._distribution = distribution.DistributionPolicy(
        self._game, self._policy)

  def get_policy(self):
    return self._policy
