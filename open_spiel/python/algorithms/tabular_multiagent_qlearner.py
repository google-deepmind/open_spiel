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

"""Tabular Multiagent Q-learning agent.

Currently implementations include:
Nash-Q: https://www.jmlr.org/papers/volume4/hu03a/hu03a.pdf
Correlated-Q: https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf, where both CE-Q and CCE-Q are supported
"""

import abc
import collections
import itertools
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms.jpsro import _mgce, _mgcce
from open_spiel.python.algorithms.matrix_nash import lemke_howson_solve


def valuedict():
  return collections.defaultdict(float)


class JointActionSolver:
  @abc.abstractmethod
  def __call__(self, payoffs_array):
    """Find a joint action mixture and values for the current one-step game

      Args:
        payoffs_array: a `numpy.ndarray` of utilities of a game.

      Returns:
        res_mixtures: a list of mixed strategies for each agent
        res_values: a list of expected utilities for each agent
    """


class TwoPlayerNashSolver(JointActionSolver):
  """A joint action solver solving for Nash for two-player games 

  uses python.algorithms.matrix_nash.lemke_howson_solve
  """

  def __call__(self, payoffs_array):
    assert len(payoffs_array) == 2

    row_payoffs, col_payoffs = payoffs_array[0], payoffs_array[1]
    A0, A1 = payoffs_array.shape[1:]

    best_value = float('-inf')
    res_mixtures, res_values = None, None

    for (row_mixture, col_mixture) in lemke_howson_solve(row_payoffs, col_payoffs):
      # TO-DO: handle the case where the LH solver gave ineligible answer
      if np.sum(np.isnan(row_mixture)) or np.sum(np.isnan(col_mixture)):
        continue
      row_mixture_, col_mixture_ = row_mixture.reshape(
          (-1, 1)), col_mixture.reshape((-1, 1))
      row_value, col_value = (row_mixture_.T.dot(row_payoffs).dot(col_mixture_)).item(
      ), (row_mixture_.T.dot(col_payoffs).dot(col_mixture_)).item()
      # Currently using maximizing social welfare for equilibrium selection
      if(row_value + col_value > best_value):
        best_value = row_value + col_value
        res_mixtures = [row_mixture, col_mixture]
        res_values = [row_value, col_value]

    # If no plauisble nash found, use uniform mixed strategies
    if not res_mixtures:
      res_mixtures = [np.ones(A0)/A0, np.ones(A1)/A1]
      row_mixture_, col_mixture_ = res_mixtures[0].reshape(
          (-1, 1)), res_mixtures[1].reshape((-1, 1))
      res_values = [(row_mixture_.T.dot(row_payoffs).dot(col_mixture_)).item(
      ), (row_mixture_.T.dot(col_payoffs).dot(col_mixture_)).item()]

    return res_mixtures, res_values


class CorrelatedEqSolver(JointActionSolver):

  """A joint action solver solving for correlated equilibrium

  uses python.algorithms.jspro._mgce and _mgcce for solving (coarse) correlated equilibrium
  """

  def __init__(self, is_CCE=False):
    self._is_CCE = is_CCE

  def __call__(self, payoffs_array):
    assert len(payoffs_array) > 0
    N = len(payoffs_array)
    mixture, _ = _mgcce(payoffs_array, [
                        1] * N, ignore_repeats=True) if self._is_CCE else _mgce(payoffs_array, [1] * N, ignore_repeats=True)
    mixtures, values = [], []
    for n in range(N):
      values.append(np.sum(payoffs_array[n] * mixture))
      mixtures.append(np.sum(mixture, axis=tuple(
          [n_ for n_ in range(N) if n_ != n])))
    return mixtures, values


class MAQLearner(rl_agent.AbstractAgent):
  def __init__(self,
               player_id,
               num_players,
               num_actions,
               joint_action_solver,
               step_size=0.1,
               epsilon_schedule=rl_tools.ConstantSchedule(0.2),
               discount_factor=1.0):
    """Initialize the Multiagent joint-action Q-Learning agent.

    The joint_action_solver solves for one-step matrix game defined by Q-tables
    """
    self._player_id = player_id
    self._num_players = num_players
    self._num_actions = num_actions
    self._joint_action_solver = joint_action_solver
    self._step_size = step_size
    self._epsilon_schedule = epsilon_schedule
    self._epsilon = epsilon_schedule.value
    self._discount_factor = discount_factor
    self._q_values = [collections.defaultdict(
        valuedict) for _ in range(num_players)]
    self._prev_info_state = None
    self._last_loss_value = None

  def _get_payoffs_array(self, info_state):
    payoffs_array = np.zeros(
        (self._num_players,) + tuple(self._num_actions))
    for joint_action in itertools.product(*[range(dim) for dim in self._num_actions]):
      for n in range(self._num_players):
        payoffs_array[(
            n, ) + joint_action] = self._q_values[n][info_state][joint_action]
    return payoffs_array

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    If the agent has not been to `info_state`, a valid random action is chosen.
    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float, prob of taking an exploratory action.
    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions[self._player_id])
    joint_actions_dims = [len(legal_actions[n])
                          for n in range(self._num_players)]

    state_probs, _ = self._joint_action_solver(
        self._get_payoffs_array(info_state))

    probs[legal_actions[self._player_id]] = epsilon / \
        len(legal_actions[self._player_id])
    probs += (1 - epsilon) * state_probs[self._player_id]
    action = np.random.choice(
        range(self._num_actions[self._player_id]), p=probs)
    return action, probs

  def step(self, time_step, actions=None, is_evaluation=False):
    """Returns the action to be taken and updates the Q-values if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    info_state = str(time_step.observations["info_state"])

    legal_actions = time_step.observations["legal_actions"]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    # Act step: don't act at terminal states.
    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon

      # select according to the joint action solver
      action, probs = self._epsilon_greedy(
          info_state, legal_actions, epsilon=epsilon)

    # Learn step: don't learn during evaluation or at first agent steps.

    actions = tuple(actions)

    if self._prev_info_state and not is_evaluation:
      _, next_state_values = \
          self._joint_action_solver(self._get_payoffs_array(info_state))
      # update Q values for every agent
      for n in range(self._num_players):
        target = time_step.rewards[n]
        if not time_step.last():  # Q values are zero for terminal.
          target += self._discount_factor * next_state_values[n]

        prev_q_value = self._q_values[n][self._prev_info_state][actions]

        self._q_values[n][self._prev_info_state][actions] += (
            self._step_size * (target - prev_q_value))

      # Decay epsilon, if necessary.
      self._epsilon = self._epsilon_schedule.step()

      if time_step.last():  # prepare for the next episode.
        self._prev_info_state = None
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._prev_info_state = info_state

    return rl_agent.StepOutput(action=action, probs=probs)
