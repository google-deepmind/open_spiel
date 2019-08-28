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

"""Implementations of Rectified Nash Response.

One iteration of the algorithm consists in :
1) Compute the nash probability vector for current list of strategies
2) From every strategy used (For which nash probability > 0 in rectified nash
setting), generate a new best response strategy against the
nash-probability-weighted mixture of strategies using oracle ; perhaps only
considering agents in the mixture that are beaten (Rectified nash setting).
3) Update meta game matrix with new game results.

See Balduzzi et Al., 2019, https://arxiv.org/pdf/1901.08106.pdf for clearer
explanation, especially for 2).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer
from open_spiel.python.policy import TabularPolicy

# Constant, specifying the threshold below which probabilities are considered
# 0 in the Rectified Nash Response setting.
EPSILON_MIN_POSITIVE_PROBA = 1e-6


class RNRSolver(abstract_meta_trainer.AbstractMetaTrainer):
  """An implementation of (Rectified) Nash Response (RNR).

  RNR is Algorithm 4 in (Balduzzi, 2019, "Open-ended Learning in Symmetric
  Zero-sum Games"). NR, Nash response, is algorithm 3.
  Refer to the paper for details:
  Balduzzi et Al., 2019, https://arxiv.org/pdf/1901.08106.pdf
  """

  def __init__(self,
               game,
               oracle,
               sims_per_entry,
               initial_policies=None,
               restrict_training=True,
               rectify_training=True,
               meta_strategy_computation_method=None,
               **kwargs):
    """Initialize the RNR solver.

    Arguments:
      game: The open_spiel game object. Must be a two player, zero sum,
        symmetric game.
      oracle: Callable that takes as input: - game - policy - policies played -
        array representing the probability of playing policy i - other kwargs
        and returns a new best response.
      sims_per_entry: Number of simulations to run to estimate each element of
        the game outcome matrix.
      initial_policies: An initial policy, from which the optimization process
        will start.
      restrict_training: A boolean, specifying whether to omit policies whose
        selection probability is 0 from training (True) or to train on them as
        well (False).
      rectify_training: A boolean, specifying whether to train only against
        opponents we beat (True), or against all opponents, including those who
        beat us (False).
      meta_strategy_computation_method: Callable taking a GenPSROSolver object
        and returning a list of meta strategies (One list entry per player),
        or string selecting pre-implemented methods. String value can be:
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      **kwargs: kwargs for meta strategy computation.
    To use Rectified Nash Response: rectify_nash = True nash_opponent_selection
      = True
    To use Nash Response: rectify_nash = False nash_opponent_selection = True
    To use Uniform Response: rectify_nash = False nash_opponent_selection =
      False
    """
    self._sims_per_entry = sims_per_entry
    self._restrict_training = restrict_training
    self._rectify_training = rectify_training

    assert (game.num_players() <= 2), ("Error : RNR implementation doesn't work"
                                       " for games with more than two players. "
                                       "Please use generalized_psro instead.")
    super(RNRSolver, self).__init__(game, oracle, initial_policies,
                                    meta_strategy_computation_method, **kwargs)

  def _initialize_policy(self, initial_policies):
    # A list of policy.Policy instances representing the strategies available
    # to players.
    self._policies = [
        initial_policies if initial_policies else TabularPolicy(self._game)
    ]

  def _initialize_game_state(self):
    self._meta_games = np.zeros((1, 1))

  def update_agents(self):
    """Updates each agent using the oracle."""
    # If rectifying Nash, only update agents with probability
    # of being selected by Nash. This step selects them.
    if self._restrict_training:
      used_policies = [
          self._policies[i] for i in range(len(self._policies)) if
          self._meta_strategy_probabilities[0][i] > EPSILON_MIN_POSITIVE_PROBA
      ]
    else:
      used_policies = self._policies

    # Generate new policies via oracle function, and put them in a new list.
    self._new_policies = []
    for pol in used_policies:
      new_policy = self._oracle(
          self._game,
          pol, [None, self._policies],
          0, [None, self._meta_strategy_probabilities[0]],
          rectify_training=self._rectify_training)
      self._new_policies.append(new_policy)

  def update_empirical_gamestate(self, seed=None):
    """Given new agents in _new_policies, update meta_games through simulations.

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    # Concatenate both lists.
    updated_policies = self._policies + self._new_policies

    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    total_number_policies = len(updated_policies)
    num_older_policies = len(self._policies)
    number_new_policies = len(self._new_policies)

    # Initializing the matrix with nans to recognize unestimated states.
    meta_games = np.full((total_number_policies, total_number_policies), np.nan)

    # Filling the matrix with already-known values.
    meta_games[:num_older_policies, :num_older_policies] = self._meta_games

    # Filling the matrix for newly added policies.
    for i, j in itertools.product(
        range(number_new_policies), range(total_number_policies)):
      if i + num_older_policies == j:
        meta_games[j, j] = 0
      elif np.isnan(meta_games[i + num_older_policies, j]):
        utility_estimate = self.sample_episodes(
            (self._new_policies[i], updated_policies[j]),
            self._sims_per_entry)[0]
        meta_games[i + num_older_policies, j] = utility_estimate
        # 0 sum game
        meta_games[j, i + num_older_policies] = -utility_estimate

    self._meta_games = meta_games
    self._policies = updated_policies
    return meta_games
