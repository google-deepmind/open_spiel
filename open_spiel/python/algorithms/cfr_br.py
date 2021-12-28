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

"""Python implementation of the CFR-BR algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

# pylint: disable=protected-access
_CFRSolverBase = cfr._CFRSolverBase
_update_current_policy = cfr._update_current_policy
_apply_regret_matching_plus_reset = cfr._apply_regret_matching_plus_reset
# pylint: enable=protected-access


class CFRBRSolver(_CFRSolverBase):
  """Implements the Counterfactual Regret Minimization (CFR-BR) algorithm.

  This is Counterfactual Regret Minimization against Best Response, from
  Michael Johanson and al., 2012, Finding Optimal Abstract Strategies in
  Extensive-Form Games,
  https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf).

  The algorithm
  computes an approximate Nash policy for n-player zero-sum games, but the
  implementation is currently restricted to 2-player.

  It uses an exact Best Response and full tree traversal.

  One iteration for a n-player game consists of the following:

  - Compute the BR of each player against the rest of the players.
  - Then, for each player p sequentially (from player 0 to N-1):
    - Compute the conterfactual reach probabilities and action values for player
      p, playing against the set of the BR for all other players.
    - Update the player `p` policy using these values.

  CFR-BR should converge with high probability (see the paper), but we can also
  compute the time-averaged strategy.

  The implementation reuses the `action_values_vs_best_response` module and
  thus uses TabularPolicies. This will run only for smallish games.
  """

  def __init__(self, game, linear_averaging=False, regret_matching_plus=False):
    # pyformat: disable
    """Initializer.

    Args:
      game: The `pyspiel.Game` to run on.
      linear_averaging: Whether to use linear averaging, i.e.
        cumulative_policy[info_state][action] += (
          iteration_number * reach_prob * action_prob)

        or not:

        cumulative_policy[info_state][action] += reach_prob * action_prob
      regret_matching_plus: Whether to use Regret Matching+:
        cumulative_regrets = max(cumulative_regrets + regrets, 0)
        or simply regret matching:
        cumulative_regrets = cumulative_regrets + regrets
    """
    # pyformat: enable
    if game.num_players() != 2:
      raise ValueError("Game {} does not have {} players.".format(game, 2))

    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "CFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

    super(CFRBRSolver, self).__init__(
        game,
        alternating_updates=True,
        linear_averaging=linear_averaging,
        regret_matching_plus=regret_matching_plus)

    self._best_responses = {i: None for i in range(game.num_players())}

  def _compute_best_responses(self):
    """Computes each player best-response against the pool of other players."""

    def policy_fn(state):
      key = state.information_state_string()
      return self._get_infostate_policy(key)

    current_policy = policy.tabular_policy_from_callable(self._game, policy_fn)

    for player_id in range(self._game.num_players()):
      self._best_responses[player_id] = exploitability.best_response(
          self._game, current_policy, player_id)

  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    self._iteration += 1

    self._compute_best_responses()

    for player in range(self._num_players):
      # We do not use policies, to not have to call `state.information_state`
      # several times (in here and within policy).
      policies = []
      for p in range(self._num_players):
        # pylint: disable=g-long-lambda
        policies.append(
            lambda infostate_str, p=p:
            {self._best_responses[p]["best_response_action"][infostate_str]: 1})
        # pylint: enable=g-long-lambda
      policies[player] = self._get_infostate_policy

      self._compute_counterfactual_regret_for_player(
          state=self._root_node,
          policies=policies,
          reach_probabilities=np.ones(self._num_players + 1),
          player=player)

      if self._regret_matching_plus:
        _apply_regret_matching_plus_reset(self._info_state_nodes)
    _update_current_policy(self._current_policy, self._info_state_nodes)
