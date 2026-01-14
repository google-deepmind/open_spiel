# Copyright 2023 DeepMind Technologies Limited
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

"""This module implements a generic environment for iterated normal form games.

It does so wuth automatic vectorization. Along with the environment, it also
provides pre-defined factory functions for common games such as the iterated
prisoners dilemma and the iterated matching pennies.
"""

import numpy as np
from pyspiel import PlayerId

from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import StepType
from open_spiel.python.rl_environment import TimeStep


class IteratedMatrixGame(Environment):
  """Environment for iterated normal form games.

  Supports automatic vectorization.
  """

  def __init__(
      self,
      payoff_matrix: np.ndarray,
      iterations: int,
      batch_size=1,
      include_remaining_iterations=True,
  ):
    # pylint: disable=super-init-not-called
    self._payoff_matrix = np.array(payoff_matrix, dtype=np.float32)
    self._iterations = iterations
    self._num_players = payoff_matrix.ndim - 1
    self._batch_size = batch_size
    self._include_remaining_iterations = include_remaining_iterations
    self._t = 0
    self._actions = np.arange(
        np.prod(self.action_spec()['num_actions'])
    ).reshape(*[payoff_matrix.shape[p] for p in range(self._num_players)])

  def one_hot(self, x, n):
    return np.eye(n)[x]

  @property
  def num_players(self):
    return self._num_players

  def observation_spec(self):
    info_state_spec, legal_actions_spec = [], []
    for i in range(self._num_players):
      num_actions = np.prod(self._payoff_matrix.shape[:-1]) + 1
      if self._include_remaining_iterations:
        num_actions += 1
      info_state_spec.append([num_actions])
      legal_actions_spec.append(self._payoff_matrix.shape[i])
    return {
        'info_state': tuple(info_state_spec),
        'legal_actions': tuple(legal_actions_spec),
        'current_player': (),
    }

  def action_spec(self):
    num_actions, mins, maxs = [], [], []
    for i in range(self._num_players):
      num_actions.append(self._payoff_matrix.shape[i])
      mins.append(0)
      maxs.append(self._payoff_matrix.shape[i] - 1)

    return {
        'num_actions': tuple(num_actions),
        'min': tuple(mins),
        'max': tuple(maxs),
        'dtype': int,
    }

  def step(self, actions: np.ndarray):
    if actions.ndim == 1:
      actions = actions[None, :]
    payoffs = self._payoff_matrix[tuple(actions.T)]
    s1 = self.one_hot(
        self._actions[tuple(actions.T)] + 1, n=np.max(self._actions) + 2
    )
    s2 = self.one_hot(
        self._actions[tuple(actions[..., ::-1].T)] + 1,
        n=np.max(self._actions) + 2,
    )
    rewards = [
        np.squeeze(p)
        for p in np.split(
            payoffs, indices_or_sections=self._num_players, axis=1
        )
    ]
    discounts = [np.ones_like(r) for r in rewards]
    if self._t == self._iterations - 1:
      step_type = StepType.LAST
    else:
      step_type = StepType.MID
    self._t += 1
    remaining_iters = float((self._iterations - self._t)) / self._iterations

    info_state = [s1, s2]
    if self._include_remaining_iterations:
      info_state = np.concatenate(
          [
              info_state,
              np.full((self._batch_size, 1), fill_value=remaining_iters),
          ],
          axis=-1,
      )

    legal_actions = self._get_legal_actions()
    return TimeStep(
        observations={
            'info_state': info_state,
            'legal_actions': legal_actions,
            'batch_size': actions.shape[0],
            'current_player': PlayerId.SIMULTANEOUS,
        },
        rewards=rewards,
        discounts=discounts,
        step_type=step_type,
    )

  def _get_legal_actions(self):
    legal_actions = []
    for p in range(self.num_players):
      actions = np.arange(self.action_spec()['num_actions'][p])
      legal_actions.append([actions] * self._batch_size)
    return np.array(legal_actions)

  def reset(self):
    self._t = 0
    info_state = np.zeros((
        self.num_players,
        self._batch_size,
        *self.observation_spec()['info_state'][0],
    ))
    info_state[..., 0] = 1.0
    if self._include_remaining_iterations:
      info_state[..., -1] = 1.0
    rewards = np.squeeze(np.zeros((self.num_players, self._batch_size)))
    discounts = np.squeeze(np.ones((self.num_players, self._batch_size)))
    return TimeStep(
        observations={
            'info_state': [
                np.squeeze(s).astype(np.float32) for s in info_state
            ],
            'legal_actions': self._get_legal_actions(),
            'batch_size': self._batch_size,
            'current_player': PlayerId.SIMULTANEOUS,
        },
        rewards=[np.squeeze(a).astype(np.float32) for a in rewards],
        discounts=[np.squeeze(a).astype(np.float32) for a in discounts],
        step_type=StepType.FIRST,
    )


def IteratedPrisonersDilemma(iterations: int, batch_size=1):
  return IteratedMatrixGame(
      payoff_matrix=np.array([[[-1, -1], [-3, 0]], [[0, -3], [-2, -2]]]),
      iterations=iterations,
      batch_size=batch_size,
      include_remaining_iterations=False,
  )


def IteratedMatchingPennies(iterations: int, batch_size=1):
  return IteratedMatrixGame(
      payoff_matrix=np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]),
      iterations=iterations,
      batch_size=batch_size,
      include_remaining_iterations=False,
  )
