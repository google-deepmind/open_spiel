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

"""Python base module for the implementations of Monte Carlo Counterfactual Regret Minimization."""

import numpy as np
from open_spiel.python import policy

REGRET_INDEX = 0
AVG_POLICY_INDEX = 1


class AveragePolicy(policy.Policy):
  """A policy object representing the average policy for MCCFR algorithms."""

  def __init__(self, game, player_ids, infostates):
    # Do not create a copy of the dictionary
    # but work on the same object
    super().__init__(game, player_ids)
    self._infostates = infostates

  def action_probabilities(self, state, player_id=None):
    """Returns the MCCFR average policy for a player in a state.

    If the policy is not defined for the provided state, a uniform
    random policy is returned.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for which we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state. If the policy is defined for the state, this
      will contain the average MCCFR strategy defined for that state.
      Otherwise, it will contain all legal actions, each with the same
      probability, equal to 1 / num_legal_actions.
    """
    if player_id is None:
      player_id = state.current_player()
    legal_actions = state.legal_actions()
    info_state_key = state.information_state_string(player_id)
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is None:
      return {a: 1 / len(legal_actions) for a in legal_actions}
    avstrat = (
        retrieved_infostate[AVG_POLICY_INDEX] /
        retrieved_infostate[AVG_POLICY_INDEX].sum())
    return {legal_actions[i]: avstrat[i] for i in range(len(legal_actions))}


class MCCFRSolverBase(object):
  """A base class for both outcome MCCFR and external MCCFR."""

  def __init__(self, game):
    self._game = game
    self._infostates = {}  # infostate keys -> [regrets, avg strat]
    self._num_players = game.num_players()

  def _lookup_infostate_info(self, info_state_key, num_legal_actions):
    """Looks up an information set table for the given key.

    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.

    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape
        [num_legal_actions].
          The average is weighted using `my_reach`
    """
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is not None:
      return retrieved_infostate

    # Start with a small amount of regret and total accumulation, to give a
    # uniform policy: this will get erased fast.
    self._infostates[info_state_key] = [
        np.ones(num_legal_actions, dtype=np.float64) / 1e6,
        np.ones(num_legal_actions, dtype=np.float64) / 1e6,
    ]
    return self._infostates[info_state_key]

  def _add_regret(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][REGRET_INDEX][action_idx] += amount

  def _add_avstrat(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][AVG_POLICY_INDEX][action_idx] += amount

  def average_policy(self):
    """Computes the average policy, containing the policy for all players.

    Returns:
      An average policy instance that should only be used during
      the lifetime of solver object.
    """
    return AveragePolicy(self._game, list(range(self._num_players)),
                         self._infostates)

  def _regret_matching(self, regrets, num_legal_actions):
    """Applies regret matching to get a policy.

    Args:
      regrets: numpy array of regrets for each action.
      num_legal_actions: number of legal actions at this state.

    Returns:
      numpy array of the policy indexed by the index of legal action in the
      list.
    """
    positive_regrets = np.maximum(regrets,
                                  np.zeros(num_legal_actions, dtype=np.float64))
    sum_pos_regret = positive_regrets.sum()
    if sum_pos_regret <= 0:
      return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
    else:
      return positive_regrets / sum_pos_regret
