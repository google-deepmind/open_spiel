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

import itertools

from absl import logging  # pylint:disable=unused-import

import numpy as np


def construct_game_queries(base_profile, num_checkpts):
  """Constructs a list of checkpoint selection tuples to query value function.

  Each query tuple (key, query) where key = (pi, pj) and query is
  (p1's selected checkpt, ..., p7's selected checkpt) fixes the players in the
  game of diplomacy to be played. It may be necessary to play several games with
  the same players to form an accurate estimate of the value or payoff for each
  player as checkpts contain stochastic policies.

  Args:
    base_profile: list of selected checkpts for each player, i.e.,
      a sample from the player strategy profile ([x_i ~ p(x_i)])
    num_checkpts: list of ints, number of strats (or ckpts) per player
  Returns:
    Set of query tuples containing a selected checkpoint index for each player.
  """
  new_queries = set([])

  num_players = len(base_profile)
  for pi, pj in itertools.combinations(range(num_players), 2):
    new_profile = list(base_profile)
    for ai in range(num_checkpts[pi]):
      new_profile[pi] = ai
      for aj in range(num_checkpts[pj]):
        new_profile[pj] = aj
        query = tuple(new_profile)
        pair = (pi, pj)
        new_queries.update([(pair, query)])

  return new_queries


def construct_game_queries_for_exp(base_profile, num_checkpts):
  """Constructs a list of checkpoint selection tuples to query value function.

  Each query tuple (key, query) where key = (pi,) and query is
  (p1's selected checkpt, ..., p7's selected checkpt) fixes the players in the
  game of diplomacy to be played. It may be necessary to play several games with
  the same players to form an accurate estimate of the value or payoff for each
  player as checkpts contain stochastic policies.

  Args:
    base_profile: list of selected checkpts for each player, i.e.,
      a sample from the player strategy profile ([x_i ~ p(x_i)])
    num_checkpts: list of ints, number of strats (or ckpts) per player
  Returns:
    Set of query tuples containing a selected checkpoint index for each player.
  """
  new_queries = set([])

  num_players = len(base_profile)
  for pi in range(num_players):
    new_profile = list(base_profile)
    for ai in range(num_checkpts[pi]):
      new_profile[pi] = ai
      query = tuple(new_profile)
      new_queries.update([(pi, query)])

  return new_queries


def run_games_and_record_payoffs(game_queries, evaluate_game, ckpt_to_policy):
  """Simulate games according to game queries and return results.

  Args:
    game_queries: set of tuples containing indices specifying each players strat
      key_query = (agent_tuple, profile_tuple) format
    evaluate_game: callable function that takes a list of policies as argument
    ckpt_to_policy: list of maps from strat (or checkpoint) to a policy, one
      map for each player
  Returns:
    dictionary: key=key_query, value=np.array of payoffs (1 for each player)
  """
  game_results = {}
  for key_query in game_queries:
    _, query = key_query
    policies = [ckpt_to_policy[pi][ckpt_i] for pi, ckpt_i in enumerate(query)]
    payoffs = evaluate_game(policies)
    game_results.update({key_query: payoffs})
  return game_results


def form_payoff_matrices(game_results, num_checkpts):
  """Packages dictionary of game results into a payoff tensor.

  Args:
    game_results: dictionary of payoffs for each game evaluated, keys are
      (pair, profile) where pair is a tuple of the two agents played against
      each other and profile indicates pure joint action played by all agents
    num_checkpts: list of ints, number of strats (or ckpts) per player
  Returns:
    payoff_matrices: dict of np.arrays (2 x num_checkpts x num_checkpts) with
      payoffs for two players. keys are pairs above with lowest index agent
      first
  """
  num_players = len(num_checkpts)
  payoff_matrices = {}
  for pi, pj in itertools.combinations(range(num_players), 2):
    key = (pi, pj)
    payoff_matrices[key] = np.zeros((2, num_checkpts[pi], num_checkpts[pj]))
  for key_profile, payoffs in game_results.items():
    key, profile = key_profile
    i, j = key
    ai = profile[i]
    aj = profile[j]
    payoff_matrices[key][0, ai, aj] = payoffs[i]
    payoff_matrices[key][1, ai, aj] = payoffs[j]
  return payoff_matrices
