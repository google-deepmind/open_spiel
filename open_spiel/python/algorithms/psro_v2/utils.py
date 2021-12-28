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

# Lint as: python3
"""Various general utility functions."""

import random
import numpy as np

from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms import policy_aggregator_joint
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import utils as alpharank_utils


def empty_list_generator(number_dimensions):
  result = []
  for _ in range(number_dimensions - 1):
    result = [result]
  return result


def random_choice(outcomes, probabilities):
  """Samples from discrete probability distribution.

  `numpy.choice` does not seem optimized for repeated calls, this code
  had higher performance.

  Args:
    outcomes: List of categorical outcomes.
    probabilities: Discrete probability distribtuion as list of floats.

  Returns:
    Entry of `outcomes` sampled according to the distribution.
  """
  cumsum = np.cumsum(probabilities)
  return outcomes[np.searchsorted(cumsum/cumsum[-1], random.random())]


def sample_strategy(total_policies,
                    probabilities_of_playing_policies,
                    probs_are_marginal=True):
  """Samples strategies given probabilities.

  Uses independent sampling if probs_are_marginal, and joint sampling otherwise.

  Args:
    total_policies: if probs_are_marginal, this is a list, each element a list
      of each player's policies. If not, this is a list of joint policies. In
      both cases the policy orders must match that of
      probabilities_of_playing_policies.
    probabilities_of_playing_policies: if probs_are_marginal, this is a list,
      with the k-th element also a list specifying the play probabilities of the
      k-th player's policies. If not, this is a list of play probabilities of
      the joint policies specified by total_policies.
    probs_are_marginal: a boolean indicating if player-wise marginal
      probabilities are provided in probabilities_of_playing_policies. If False,
      then play_probabilities is assumed to specify joint distribution.

  Returns:
    sampled_policies: A list specifying a single sampled joint strategy.
  """

  if probs_are_marginal:
    return sample_strategy_marginal(total_policies,
                                    probabilities_of_playing_policies)
  else:
    return sample_strategy_joint(total_policies,
                                 probabilities_of_playing_policies)


def sample_strategy_marginal(total_policies, probabilities_of_playing_policies):
  """Samples strategies given marginal probabilities.

  Uses independent sampling if probs_are_marginal, and joint sampling otherwise.

  Args:
    total_policies: A list, each element a list of each player's policies.
    probabilities_of_playing_policies: This is a list, with the k-th element
      also a list specifying the play probabilities of the k-th player's
      policies.

  Returns:
    sampled_policies: A list specifying a single sampled joint strategy.
  """

  num_players = len(total_policies)
  sampled_policies = []
  for k in range(num_players):
    current_policies = total_policies[k]
    current_probabilities = probabilities_of_playing_policies[k]
    sampled_policy_k = random_choice(current_policies, current_probabilities)
    sampled_policies.append(sampled_policy_k)
  return sampled_policies


def sample_random_tensor_index(probabilities_of_index_tensor):
  shape = probabilities_of_index_tensor.shape
  reshaped_probas = probabilities_of_index_tensor.reshape(-1)

  strat_list = list(range(len(reshaped_probas)))
  chosen_index = random_choice(strat_list, reshaped_probas)
  return np.unravel_index(chosen_index, shape)


def sample_strategy_joint(total_policies, probabilities_of_playing_policies):
  """Samples strategies given joint probabilities.

  Uses independent sampling if probs_are_marginal, and joint sampling otherwise.

  Args:
    total_policies: A list, each element a list of each player's policies.
    probabilities_of_playing_policies: This is a list of play probabilities of
      the joint policies specified by total_policies.

  Returns:
    sampled_policies: A list specifying a single sampled joint strategy.
  """

  sampled_index = sample_random_tensor_index(probabilities_of_playing_policies)
  sampled_policies = []
  for player in range(len(sampled_index)):
    ind = sampled_index[player]
    sampled_policies.append(total_policies[player][ind])
  return sampled_policies


def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))


def round_maintain_sum(x):
  """Returns element-wise rounded version y of a vector x, with sum(x)==sum(y).

  E.g., if x = array([3.37625333, 2.27920304, 4.34454364]), note sum(x) == 10.
  However, naively doing y = np.round(x) yields sum(y) == 9. In this function,
  however, the rounded counterpart y will have sum(y) == 10.

  Args:
    x: a vector.
  """
  y = np.floor(x)
  sum_diff = round(sum(x)) - sum(y)  # Difference of original vs. floored sum
  indices = np.argsort(y - x)[:int(sum_diff)]  # Indices with highest difference
  y[indices] += 1  # Add the missing mass to the elements with the most missing
  return y


def get_alpharank_marginals(payoff_tables, pi):
  """Returns marginal strategy rankings for each player given joint rankings pi.

  Args:
    payoff_tables: List of meta-game payoff tables for a K-player game, where
      each table has dim [n_strategies_player_1 x ... x n_strategies_player_K].
      These payoff tables may be asymmetric.
    pi: The vector of joint rankings as computed by alpharank. Each element i
      corresponds to a unique integer ID representing a given strategy profile,
      with profile_to_id mappings provided by
      alpharank_utils.get_id_from_strat_profile().

  Returns:
    pi_marginals: List of np.arrays of player-wise marginal strategy masses,
      where the k-th player's np.array has shape [n_strategies_player_k].
  """
  num_populations = len(payoff_tables)

  if num_populations == 1:
    return pi
  else:
    num_strats_per_population = alpharank_utils.get_num_strats_per_population(
        payoff_tables, payoffs_are_hpt_format=False)
    num_profiles = alpharank_utils.get_num_profiles(num_strats_per_population)
    pi_marginals = [np.zeros(n) for n in num_strats_per_population]
    for i_strat in range(num_profiles):
      strat_profile = (
          alpharank_utils.get_strat_profile_from_id(num_strats_per_population,
                                                    i_strat))
      for i_player in range(num_populations):
        pi_marginals[i_player][strat_profile[i_player]] += pi[i_strat]
    return pi_marginals


def remove_epsilon_negative_probs(probs, epsilon=1e-9):
  """Removes negative probabilities that occur due to precision errors."""
  if len(probs[probs < 0]) > 0:  # pylint: disable=g-explicit-length-test
    # Ensures these negative probabilities aren't large in magnitude, as that is
    # unexpected and likely not due to numerical precision issues
    print("Probabilities received were: {}".format(probs[probs < 0]))
    assert np.alltrue(np.min(probs[probs < 0]) > -1.*epsilon), (
        "Negative Probabilities received were: {}".format(probs[probs < 0]))

    probs[probs < 0] = 0
    probs = probs / np.sum(probs)
  return probs


def get_joint_strategy_from_marginals(probabilities):
  """Returns a joint strategy tensor from a list of marginals.

  Args:
    probabilities: list of list of probabilities, one for each player.

  Returns:
    A joint strategy from a list of marginals.
  """
  probas = []
  for i in range(len(probabilities)):
    probas_shapes = [1] * len(probabilities)
    probas_shapes[i] = -1
    probas.append(np.array(probabilities[i]).reshape(probas_shapes))
  return np.product(probas)


def alpharank_strategy(solver, return_joint=False, **unused_kwargs):
  """Returns AlphaRank distribution on meta game matrix.

  This method works for general games.

  Args:
    solver: GenPSROSolver instance.
    return_joint: a boolean specifying whether to return player-wise
      marginals.

  Returns:
    marginals: a list, specifying for each player the alpharank marginal
      distributions on their strategies.
    joint_distr: a list, specifying the joint alpharank distributions for all
      strategy profiles.
  """
  meta_games = solver.get_meta_game()
  meta_games = [np.asarray(x) for x in meta_games]

  if solver.symmetric_game:
    meta_games = [meta_games[0]]

    # Get alpharank distribution via alpha-sweep
    joint_distr = alpharank.sweep_pi_vs_epsilon(
        meta_games)
    joint_distr = remove_epsilon_negative_probs(joint_distr)

    marginals = 2 * [joint_distr]
    joint_distr = get_joint_strategy_from_marginals(marginals)
    if return_joint:
      return marginals, joint_distr
    else:
      return joint_distr

  else:
    joint_distr = alpharank.sweep_pi_vs_epsilon(meta_games)
    joint_distr = remove_epsilon_negative_probs(joint_distr)

    if return_joint:
      marginals = get_alpharank_marginals(meta_games, joint_distr)
      return marginals, joint_distr
    else:
      return joint_distr


def get_strategy_profile_ids(payoff_tables):
  num_strats_per_population = (
      alpharank_utils.get_num_strats_per_population(
          payoff_tables, payoffs_are_hpt_format=False))
  return range(alpharank_utils.get_num_profiles(num_strats_per_population))


def get_joint_policies_from_id_list(payoff_tables, policies, profile_id_list):
  """Returns a list of joint policies, given a list of integer IDs.

  Args:
    payoff_tables: List of payoff tables, one per player.
    policies: A list of policies, one per player.
    profile_id_list: list of integer IDs, each corresponding to a joint policy.
      These integers correspond to those in get_strategy_profile_ids().

  Returns:
    selected_joint_policies: A list, with each element being a joint policy
      instance (i.e., a list of policies, one per player).
  """
  num_strats_per_population = (
      alpharank_utils.get_num_strats_per_population(
          payoff_tables, payoffs_are_hpt_format=False))
  np.testing.assert_array_equal(num_strats_per_population,
                                [len(p) for p in policies])
  num_players = len(policies)

  selected_joint_policies = []
  for profile_id in profile_id_list:
    # Compute the profile associated with the integer profile_id
    policy_profile = alpharank_utils.get_strat_profile_from_id(
        num_strats_per_population, profile_id)
    # Append the joint policy corresponding to policy_profile
    selected_joint_policies.append(
        [policies[k][policy_profile[k]] for k in range(num_players)])
  return selected_joint_policies


def compute_states_and_info_states_if_none(game,
                                           all_states=None,
                                           state_to_information_state=None):
  """Returns all_states and/or state_to_information_state for the game.

  To recompute everything, pass in None for both all_states and
  state_to_information_state. Otherwise, this function will use the passed in
  values to reconstruct either of them.

  Args:
    game: The open_spiel game.
    all_states: The result of calling get_all_states.get_all_states. Cached for
      improved performance.
    state_to_information_state: A dict mapping str(state) to
      state.information_state for every state in the game. Cached for improved
      performance.
  """
  if all_states is None:
    all_states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)

  if state_to_information_state is None:
    state_to_information_state = {
        state: all_states[state].information_state_string()
        for state in all_states
    }

  return all_states, state_to_information_state


def aggregate_policies(game, total_policies, probabilities_of_playing_policies):
  """Aggregate the players' policies.

  Specifically, returns a single callable policy object that is
  realization-equivalent to playing total_policies with
  probabilities_of_playing_policies. I.e., aggr_policy is a joint policy that
  can be called at any information state [via
  action_probabilities(state, player_id)].

  Args:
    game: The open_spiel game.
    total_policies: A list of list of all policy.Policy strategies used for
      training, where the n-th entry of the main list is a list of policies
      available to the n-th player.
    probabilities_of_playing_policies: A list of arrays representing, per
      player, the probabilities of playing each policy in total_policies for the
      same player.

  Returns:
    A callable object representing the policy.
  """
  aggregator = policy_aggregator.PolicyAggregator(game)

  return aggregator.aggregate(
      range(len(probabilities_of_playing_policies)), total_policies,
      probabilities_of_playing_policies)


def marginal_to_joint(policies):
  """Marginal policies to joint policies.

  Args:
    policies: List of list of policies, one list per player.

  Returns:
    Joint policies in the right order (np.reshape compatible).
  """
  shape = tuple([len(a) for a in policies])
  num_players = len(shape)
  total_length = np.prod(shape)
  indexes = np.array(list(range(total_length)))
  joint_indexes = np.unravel_index(indexes, shape)

  joint_policies = []
  for joint_index in zip(*joint_indexes):
    joint_policies.append([
        policies[player][joint_index[player]] for player in range(num_players)
    ])
  return joint_policies


def aggregate_joint_policies(game, total_policies,
                             probabilities_of_playing_policies):
  """Aggregate the players' joint policies.

  Specifically, returns a single callable policy object that is
  realization-equivalent to playing total_policies with
  probabilities_of_playing_policies. I.e., aggr_policy is a joint policy that
  can be called at any information state [via
  action_probabilities(state, player_id)].

  Args:
    game: The open_spiel game.
    total_policies: A list of list of all policy.Policy strategies used for
      training, where the n-th entry of the main list is a list of policies, one
      entry for each player.
    probabilities_of_playing_policies: A list of floats representing the
      probabilities of playing each joint strategy in total_policies.

  Returns:
    A callable object representing the policy.
  """
  aggregator = policy_aggregator_joint.JointPolicyAggregator(game)

  return aggregator.aggregate(
      range(len(total_policies[0])), total_policies,
      probabilities_of_playing_policies)
