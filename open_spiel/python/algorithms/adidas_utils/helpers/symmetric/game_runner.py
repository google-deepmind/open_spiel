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

"""Utils for computing gradient information: run games and record payoffs.
"""

from absl import logging  # pylint:disable=unused-import

import numpy as np


def construct_game_queries(base_profile, num_checkpts):
  """Constructs a list of checkpoint selection tuples to query value function.

  Each query tuple (p1's selected checkpt, ..., p7's selected checkpt)
  fixes the players in the game of diplomacy to be played. It may be necessary
  to play several games with the same players to form an accurate estimate of
  the value or payoff for each player as checkpts contain stochastic policies.

  Args:
    base_profile: list of selected checkpts for each player, i.e.,
      a sample from the player strategy profile ([x_i ~ p(x_i)])
    num_checkpts: number of checkpts available to each player
  Returns:
    Set of query tuples containing a selected checkpoint index for each player.
  """
  new_queries = set([])

  pi, pj = 0, 1
  new_profile = list(base_profile)
  for ai in range(num_checkpts):
    new_profile[pi] = ai
    for aj in range(num_checkpts):
      new_profile[pj] = aj
      query = tuple(new_profile)
      new_queries.update([query])

  return new_queries


def construct_game_queries_for_exp(base_profile, num_checkpts):
  """Constructs a list of checkpoint selection tuples to query value function.

  Each query tuple (p1's selected checkpt, ..., p7's selected checkpt)
  fixes the players in the game of diplomacy to be played. It may be necessary
  to play several games with the same players to form an accurate estimate of
  the value or payoff for each player as checkpts contain stochastic policies.

  Args:
    base_profile: list of selected checkpts for each player, i.e.,
      a sample from the player strategy profile ([x_i ~ p(x_i)])
    num_checkpts: number of checkpts available to each player
  Returns:
    Set of query tuples containing a selected checkpoint index for each player.
  """
  new_queries = set([])

  pi = 0
  new_profile = list(base_profile)
  for ai in range(num_checkpts):
    new_profile[pi] = ai
    query = tuple(new_profile)
    new_queries.update([query])

  return new_queries


def run_games_and_record_payoffs(game_queries, evaluate_game, ckpt_to_policy):
  """Simulate games according to game queries and return results.

  Args:
    game_queries: set of tuples containing indices specifying each players strat
    evaluate_game: callable function that takes a list of policies as argument
    ckpt_to_policy: maps a strat (or checkpoint) to a policy
  Returns:
    dictionary: key=query, value=np.array of payoffs (1 for each player)
  """
  game_results = {}
  for query in game_queries:
    policies = [ckpt_to_policy[ckpt] for ckpt in query]
    payoffs = evaluate_game(policies)
    game_results.update({query: payoffs})
  return game_results


def form_payoff_matrices(game_results, num_checkpts):
  """Packages dictionary of game results into a payoff tensor.

  Args:
    game_results: dictionary of payoffs for each game evaluated
    num_checkpts: int, number of strats (or ckpts) per player
  Returns:
    payoff_matrices: np.array (2 x num_checkpts x num_checkpts) with payoffs for
      two players (assumes symmetric game and only info for 2 players is needed
      for stochastic gradients)
  """
  payoff_matrices = np.zeros((2, num_checkpts, num_checkpts))
  for profile, payoffs in game_results.items():
    i, j = profile[:2]
    payoff_matrices[:, i, j] = payoffs[:2]
  return payoff_matrices
