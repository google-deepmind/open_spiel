# Copyright 2022 DeepMind Technologies Limited
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
"""Boltzmann Q learning agent.

This algorithm is a variation of Q learning that uses action selection
based on boltzmann probability interpretation of Q-values.

For more details, see equation (2) page 2 in
   https://arxiv.org/pdf/1109.1528.pdf
"""

import numpy as np

from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner


class BoltzmannQLearner(tabular_qlearner.QLearner):
  """Tabular Boltzmann Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.

  The tic_tac_toe example uses the standard Qlearner. Using the
  BoltzmannQlearner is
  identical and only differs in the initialization of the agents.
  """

  def __init__(self,
               player_id,
               num_actions,
               step_size=0.1,
               discount_factor=1.0,
               temperature_schedule=rl_tools.ConstantSchedule(.5),
               centralized=False):
    super().__init__(
        player_id,
        num_actions,
        step_size=step_size,
        discount_factor=discount_factor,
        epsilon_schedule=temperature_schedule,
        centralized=centralized)

  def _softmax(self, info_state, legal_actions, temperature):
    """Action selection based on boltzmann probability interpretation of Q-values.

    For more details, see equation (2) page 2 in
    https://arxiv.org/pdf/1109.1528.pdf

    Args:
        info_state: hashable representation of the information state.
        legal_actions: list of actions at `info_state`.
        temperature: temperature used for softmax.

    Returns:
        A valid soft-max selected action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)

    if temperature > 0.0:
      probs += [
          np.exp((1 / temperature) * self._q_values[info_state][i])
          for i in range(self._num_actions)
      ]
      probs /= np.sum(probs)
    else:
      # Temperature = 0 causes normal greedy action selection
      greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
      greedy_actions = [
          a for a in legal_actions if self._q_values[info_state][a] == greedy_q
      ]

      probs[greedy_actions] += 1 / len(greedy_actions)

    action = np.random.choice(range(self._num_actions), p=probs)
    return action, probs

  def _get_action_probs(self, info_state, legal_actions, epsilon):
    """Returns a selected action and the probabilities of legal actions."""
    return self._softmax(info_state, legal_actions, temperature=epsilon)
