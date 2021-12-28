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

"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import open_spiel.python.algorithms.mccfr as mccfr
import pyspiel


class OutcomeSamplingSolver(mccfr.MCCFRSolverBase):
  """An implementation of outcome sampling MCCFR."""

  def __init__(self, game):
    super().__init__(game)
    # This is the epsilon exploration factor. When sampling episodes, the
    # updating player will sampling according to expl * uniform + (1 - expl) *
    # current_policy.
    self._expl = 0.6

    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "MCCFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

  def iteration(self):
    """Performs one iteration of outcome sampling.

    An iteration consists of one episode for each player as the update
    player.
    """
    for update_player in range(self._num_players):
      state = self._game.new_initial_state()
      self._episode(
          state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

  def _baseline(self, state, info_state, aidx):  # pylint: disable=unused-argument
    # Default to vanilla outcome sampling
    return 0

  def _baseline_corrected_child_value(self, state, info_state, sampled_aidx,
                                      aidx, child_value, sample_prob):
    # Applies Eq. 9 of Schmid et al. '19
    baseline = self._baseline(state, info_state, aidx)
    if aidx == sampled_aidx:
      return baseline + (child_value - baseline) / sample_prob
    else:
      return baseline

  def _episode(self, state, update_player, my_reach, opp_reach, sample_reach):
    """Runs an episode of outcome sampling.

    Args:
      state: the open spiel state to run from (will be modified in-place).
      update_player: the player to update regrets for (the other players
        update average strategies)
      my_reach: reach probability of the update player
      opp_reach: reach probability of all the opponents (including chance)
      sample_reach: reach probability of the sampling (behavior) policy

    Returns:
      util is a real value representing the utility of the update player
    """
    if state.is_terminal():
      return state.player_return(update_player)

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      aidx = np.random.choice(range(len(outcomes)), p=probs)
      state.apply_action(outcomes[aidx])
      return self._episode(state, update_player, my_reach,
                           probs[aidx] * opp_reach, probs[aidx] * sample_reach)

    cur_player = state.current_player()
    info_state_key = state.information_state_string(cur_player)
    legal_actions = state.legal_actions()
    num_legal_actions = len(legal_actions)
    infostate_info = self._lookup_infostate_info(info_state_key,
                                                 num_legal_actions)
    policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                   num_legal_actions)
    if cur_player == update_player:
      uniform_policy = (
          np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
      sample_policy = self._expl * uniform_policy + (1.0 - self._expl) * policy
    else:
      sample_policy = policy
    sampled_aidx = np.random.choice(range(num_legal_actions), p=sample_policy)
    state.apply_action(legal_actions[sampled_aidx])
    if cur_player == update_player:
      new_my_reach = my_reach * policy[sampled_aidx]
      new_opp_reach = opp_reach
    else:
      new_my_reach = my_reach
      new_opp_reach = opp_reach * policy[sampled_aidx]
    new_sample_reach = sample_reach * sample_policy[sampled_aidx]
    child_value = self._episode(state, update_player, new_my_reach,
                                new_opp_reach, new_sample_reach)

    # Compute each of the child estimated values.
    child_values = np.zeros(num_legal_actions, dtype=np.float64)
    for aidx in range(num_legal_actions):
      child_values[aidx] = self._baseline_corrected_child_value(
          state, infostate_info, sampled_aidx, aidx, child_value,
          sample_policy[aidx])
    value_estimate = 0
    for aidx in range(num_legal_actions):
      value_estimate += policy[sampled_aidx] * child_values[aidx]

    if cur_player == update_player:
      # Now the regret and avg strategy updates.
      policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                     num_legal_actions)

      # Estimate for the counterfactual value of the policy.
      cf_value = value_estimate * opp_reach / sample_reach

      # Update regrets.
      #
      # Note: different from Chapter 4 of Lanctot '13 thesis, the utilities
      # coming back from the recursion are already multiplied by the players'
      # tail reaches and divided by the sample tail reach. So when adding
      # regrets to the table, we need only multiply by the opponent reach and
      # divide by the sample reach to this point.
      for aidx in range(num_legal_actions):
        # Estimate for the counterfactual value of the policy replaced by always
        # choosing sampled_aidx at this information state.
        cf_action_value = child_values[aidx] * opp_reach / sample_reach
        self._add_regret(info_state_key, aidx, cf_action_value - cf_value)

      # update the average policy
      for aidx in range(num_legal_actions):
        increment = my_reach * policy[aidx] / sample_reach
        self._add_avstrat(info_state_key, aidx, increment)

    return value_estimate
