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

"""Utils for evolutionary game theoretic analysis of games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np

import pyspiel


def n_choose_k(n, k):
  """Returns the combination choose k among n items."""
  f = math.factorial
  return int(f(n) / f(k) / f(n - k))


def grid_simplex(step=.1, boundary=False):
  """Generator for regular 'lattice' on the 2-simplex.

  Args:
    step: Defines spacing along one dimension.
    boundary: Include points on the boundary/face of the simplex.

  Yields:
    Next point on the grid.
  """
  eps = 1e-8
  start = 0. if boundary else step
  stop = 1. + eps if boundary else 1. - step + eps
  for a in np.arange(start, stop, step, dtype=np.double):
    for b in np.arange(start, stop - a, step, dtype=np.double):
      yield [a, b, 1. - a - b]


def sample_from_simplex(n, dim=3, vmin=0.):
  """Samples random points from a k-simplex.

  See Donald B. Rubin (1981) "The Bayesian Bootstrap", page 131.

  Args:
    n: Number of points that are sampled.
    dim: Dimension of the points to be sampled, e.g. dim=3 samples points from
      the 2-simplex.
    vmin: Minimum value of any coordinate of the resulting points, e.g. set
      vmin>0. to exclude points on the faces of the simplex.

  Returns:
    `ndarray(shape=(k, dim))` of uniformly random points on the (num-1)-simplex.
  """
  assert vmin >= 0.
  p = np.random.rand(n, dim - 1)
  p = np.sort(p, axis=1)
  p = np.hstack((np.zeros((n, 1)), p, np.ones((n, 1))))
  return (p[:, 1:] - p[:, 0:-1]) * (1 - 2 * vmin) + vmin


def game_payoffs_array(game):
  """Returns a `numpy.ndarray` of utilities for a game.

  NOTE: if the game is not a MatrixGame or a TensorGame then this may be costly.

  Args:
    game: A game.

  Returns:
    `numpy.ndarray` of dimension `num_players` + 1.
    First dimension is the player, followed by the actions of all players, e.g.
    a 3x3 game (2 players) has dimension [2,3,3].
  """
  if isinstance(game, pyspiel.MatrixGame):
    return np.stack([game.row_utilities(), game.col_utilities()])

  if not isinstance(game, pyspiel.TensorGame):
    game = pyspiel.extensive_to_tensor_game(game)
  return np.stack(
      [game.player_utilities(player) for player in range(game.num_players())])


def distribute(num_items, num_slots, normalize=False):
  """Yields all ways of distributing `num_items` items over `num_slots` slots.

  We assume that the ordering of the slots doesn't matter.

  Args:
    num_items: The number of items to distribute.
    num_slots: The number of slots.
    normalize: Normalizes the yielded tuple to contain floats in [0, 1] summing
      to 1.

  Yields:
    A tuple T containing `num_slots` positive integers such that
    `np.sum(T) == num_items` if `normalize == False` or `np.sum(T) == 1` if
    `normalize == True'.
  """
  normalization = 1
  if normalize:
    normalization = num_items
  # This is just the standard "bars and stars" problem.
  # See https://stackoverflow.com/questions/28965734/general-bars-and-stars.
  for c in itertools.combinations(
      range(num_items + num_slots - 1), num_slots - 1):
    # The combinations give you the indices of the internal bars.
    # pylint: disable=g-complex-comprehension
    yield tuple((b - a - 1) / normalization
                for (a, b) in zip([
                    -1,
                ] + list(c),
                                  list(c) + [num_items + num_slots - 1]))


def assert_is_1d_numpy_array(array):
  if not isinstance(array, np.ndarray):
    raise ValueError("The argument must be a numpy array, not a {}.".format(
        type(array)))

  if len(array.shape) != 1:
    raise ValueError(
        "The argument must be 1-dimensional, not of shape {}.".format(
            array.shape))


def assert_probabilities(array):
  if not all([item >= 0 for item in array]):
    raise ValueError("The vector must have all elements >= 0 items, not"
                     "{}".format(array))
  sum_ = np.sum(array)
  if not np.isclose(1, sum_):
    raise ValueError(
        "The sum of the probabilities  must be 1, not {}".format(sum_))


def sort_rows_lexicographically(array):
  """Returns a numpy array with lexicographic-ordered rows.

  This function can be used to check that 2 Heuristic Payoff Tables are equal,
  by normalizing them using a fixed ordering of the rows.

  Args:
    array: The 2D numpy array to sort by rows.
  """
  return np.array(sorted(array.tolist()))


def get_valid_next_profiles(num_strats_per_population, cur_profile):
  """Generates monomorphic strategy profile transitions given cur_profile.

  Given a current strategy profile, cur_profile, this generates all follow-up
  profiles that involve only a single other population changing its current
  monomorphic strategy to some other monomorphic strategy. Note that
  self-transitions from cur_profile to cur_profile are not included here, as
  they are a special case in our Markov chain.

  Args:
    num_strats_per_population: List of strategy sizes for each population.
    cur_profile: Current strategy profile.

  Yields:
    The next valid strategy profile transition.
  """
  num_populations = len(num_strats_per_population)

  for i_population_to_change in range(num_populations):
    for new_strat in range(num_strats_per_population[i_population_to_change]):
      # Ensure a transition will actually happen
      if new_strat != cur_profile[i_population_to_change]:
        next_profile = cur_profile.copy()
        next_profile[i_population_to_change] = new_strat
        yield i_population_to_change, next_profile


def get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format):
  """Returns a [num_populations] array of the num.

  of strategies per population.

  E.g., for a 3 population game, this returns
    [num_strats_population1, num_strats_population2, num_strats_population3]

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a 2D numpy array, or a _PayoffTableInterface
      object.
    payoffs_are_hpt_format: True indicates HPT format (i.e.
      _PayoffTableInterface object, False indicates 2D numpy array.
  """

  if payoffs_are_hpt_format:
    return np.asarray(
        [payoff_table.num_strategies for payoff_table in payoff_tables])
  else:
    # Non-HPT payoffs are matrices, so can directly return the payoff size
    return np.asarray(np.shape(payoff_tables[0]))


def get_num_profiles(num_strats_per_population):
  """Returns the total number of pure strategy profiles.

  Args:
    num_strats_per_population: A list of size `num_populations` of the number of
      strategies per population.

  Returns:
    The total number of pure strategy profiles.
  """
  return np.prod(num_strats_per_population)


def get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format):
  """Returns strategy labels corresponding to a payoff_table.

  Namely, for games where strategies have no human-understandable labels
  available, this function returns a labels object corresponding to the
  strategy profiles.

  Examples:
    Generated labels for a single-population game with 3 strategies:
      ['0','1','2'].
    Generated labels for a 3-population game with 2 strategies per population:
      {0: ['0','1'], 1: ['0','1'], 2: ['0','1']}

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a 2D numpy array, or a _PayoffTableInterface
      object.
    payoffs_are_hpt_format: Boolean indicating whether each payoff table in
      payoff_tables is a 2D numpy array, or a _PayoffTableInterface object (AKA
      Heuristic Payoff Table or HPT). True indicates HPT format, False indicates
      2D numpy array.

  Returns:
    Strategy labels.
  """

  num_populations = len(payoff_tables)

  if num_populations == 1:
    num_strats_per_population = get_num_strats_per_population(
        payoff_tables, payoffs_are_hpt_format)
    labels = [str(x) for x in range(num_strats_per_population[0])]
  else:
    num_strats_per_population = get_num_strats_per_population(
        payoff_tables, payoffs_are_hpt_format)
    labels = dict()
    label_text = []
    # Construct a list of strategy labels for each population
    for num_strats in num_strats_per_population:
      label_text.append([str(i_strat) for i_strat in range(num_strats)])
    population_ids = range(num_populations)
    labels = dict(zip(population_ids, label_text))

  return labels


def get_strat_profile_from_id(num_strats_per_population, profile_id):
  """Returns the strategy profile corresponding to a requested strategy ID.

  This is the inverse of the function get_id_from_strat_profile(). See that
  function for the indexing mechanism.

  Args:
    num_strats_per_population: List of strategy sizes for each population.
    profile_id: Integer ID of desired strategy profile, in
      {0,...,get_num_profiles-1}.

  Returns:
    The strategy profile whose ID was looked up.
  """

  num_populations = len(num_strats_per_population)
  strat_profile = np.zeros(num_populations, dtype=np.int32)

  for i_population in range(num_populations - 1, -1, -1):
    strat_profile[i_population] = (
        profile_id % num_strats_per_population[i_population])
    profile_id = profile_id // num_strats_per_population[i_population]

  return strat_profile


def get_label_from_strat_profile(num_populations, strat_profile, strat_labels):
  """Returns a human-readable label corresponding to the strategy profile.

  E.g., for Rock-Paper-Scissors, strategies 0,1,2 have labels "R","P","S".
  For strat_profile (1,2,0,1), this returns "(P,S,R,P)". If strat_profile is a
  single strategy (e.g., 0) this returns just its label (e.g., "R").

  Args:
    num_populations: Number of populations.
    strat_profile: Strategy profile of interest.
    strat_labels: Strategy labels.

  Returns:
    Human-readable label string.
  """
  if num_populations == 1:
    return strat_labels[strat_profile]
  else:
    label = "("
    for i_population, i_strat in enumerate(strat_profile):
      label += strat_labels[i_population][i_strat]
      if i_population < len(strat_profile) - 1:
        label += ","
    label += ")"
    return label


def get_id_from_strat_profile(num_strats_per_population, strat_profile):
  """Returns a unique integer ID representing the requested strategy profile.

  Map any `strat_profile` (there are `np.prod(num_strats_per_population)` such
  profiles) to {0,..., num_strat_profiles - 1}.

  The mapping is done using a usual counting strategy: With
  num_strats_per_population = [a1, ..., a_n]
  strat_profile = [b1, ..., b_n]

  we have

  id = b_1 + a1 * (b2 + a_2 * (b3 + a_3 *...))


  This is helpful for querying the element of our finite-population Markov
  transition matrix that corresponds to a transition between a specific pair of
  strategy profiles.

  Args:
    num_strats_per_population: List of strategy sizes for each population.
    strat_profile: The strategy profile (list of integers corresponding to the
      strategy of each agent) whose ID is requested.

  Returns:
    Unique ID of strat_profile.
  """

  if len(strat_profile) == 1:
    return strat_profile[0]

  return strat_profile[-1] + (num_strats_per_population[-1] *
                              get_id_from_strat_profile(
                                  num_strats_per_population[:-1],
                                  strat_profile[:-1]))


def compute_payoff(row_profile, col_profile, row_payoff_table):
  """Returns row's expected payoff in a bimatrix game.

  Args:
    row_profile: Row's strategy profile.
    col_profile: Column's strategy profile.
    row_payoff_table: Row's payoff table.
  """

  return np.dot(np.dot(row_profile.T, row_payoff_table), col_profile)


def check_is_constant_sum(payoff_table, payoffs_are_hpt_format):
  """Checks if single-population matrix game is constant-sum.

  Args:
    payoff_table: Either a 2D numpy array, or a _PayoffTableInterface object.
    payoffs_are_hpt_format: Boolean indicating whether payoff table is a
      _PayoffTableInterface object (AKA Heuristic Payoff Table or HPT), or a 2D
      numpy array. True indicates HPT, and False indicates numpy array.

  Returns:
    is_constant_sum: Boolean, True if constant-sum game.
    payoff_sum: Payoff sum if game is constant-sum, or None if not.
  """

  if payoffs_are_hpt_format:
    payoff_sum_table = np.asarray(payoff_table._payoffs).sum(axis=1)  # pylint: disable=protected-access
    is_constant_sum = np.isclose(
        payoff_sum_table, payoff_sum_table[0], atol=1e-14).all()
    payoff_sum = payoff_sum_table[0] if is_constant_sum else None
  else:
    payoff_sum_table = payoff_table + payoff_table.T
    is_constant_sum = np.isclose(
        payoff_sum_table, payoff_sum_table[0, 0], atol=1e-14).all()
    payoff_sum = payoff_sum_table[0, 0] if is_constant_sum else None
  return is_constant_sum, payoff_sum


def cluster_strats(pi, matching_decimals=4):
  """Clusters strategies using stationary distribution (pi) masses.

  Args:
    pi: stationary distribution.
    matching_decimals: the number of stationary distribution decimals that
      should match for strategies to be considered in the same cluster.

  Returns:
    Dictionary that maps unique stationary distribution masses to strategies.
  """

  rounded_masses = pi.round(decimals=matching_decimals)
  masses_to_strats = {}
  for i in np.unique(rounded_masses):
    masses_to_strats[i] = np.where(rounded_masses == i)[0]
  return masses_to_strats


def print_rankings_table(payoff_tables,
                         pi,
                         strat_labels,
                         num_top_strats_to_print=8):
  """Prints nicely-formatted table of strategy rankings.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a 2D numpy array, or a _PayoffTableInterface
      object.
    pi: Finite-population Markov chain stationary distribution.
    strat_labels: Strategy labels.
    num_top_strats_to_print: Number of top strategies to print.
  """

  num_populations = len(payoff_tables)
  payoffs_are_hpt_format = check_payoffs_are_hpt(payoff_tables)
  num_strats_per_population = get_num_strats_per_population(
      payoff_tables, payoffs_are_hpt_format)

  # More than total number of strats requested for printing, compute top and
  # use an extra row to indicate additional strategies not shown.
  row_for_lowrank_strats = True
  if num_top_strats_to_print >= len(pi):
    num_top_strats_to_print = len(pi)
    row_for_lowrank_strats = False

  # Cluster strategies according to stationary distr. (in case of tied ranks)
  masses_to_strats = cluster_strats(pi)

  def print_3col(col1, col2, col3):
    print("%-12s %-12s %-12s" % (col1, col2, col3))

  print_3col("Agent", "Rank", "Score")
  print_3col("-----", "----", "-----")

  rank = 1
  num_strats_printed = 0
  # Print a table of strategy rankings from highest to lowest mass
  for _, strats in sorted(masses_to_strats.items(), reverse=True):
    for strat in strats:
      if num_strats_printed >= num_top_strats_to_print:
        break
      rounded_pi = np.round(pi[strat], decimals=2)
      if num_populations == 1:
        strat_profile = strat
      else:
        strat_profile = get_strat_profile_from_id(num_strats_per_population,
                                                  strat)
      label = get_label_from_strat_profile(num_populations, strat_profile,
                                           strat_labels)
      print_3col(label, str(rank), str(np.abs(rounded_pi)))
      num_strats_printed += 1
    rank += 1
    if num_strats_printed >= num_top_strats_to_print:
      break

  # Ellipses to signify additional low-rank strategies are not printed
  if row_for_lowrank_strats:
    print_3col("...", "...", "...")


def is_symmetric_matrix_game(payoff_tables):
  """Checks if payoff_tables corresponds to a symmetric matrix game."""
  payoffs_are_hpt_format = check_payoffs_are_hpt(payoff_tables)

  if len(payoff_tables) == 2:
    if payoffs_are_hpt_format and np.array_equal(payoff_tables[0](),
                                                 payoff_tables[1]()):
      return True, [payoff_tables[0]]
    elif ~payoffs_are_hpt_format and np.array_equal(payoff_tables[0],
                                                    payoff_tables[1].T):
      return True, [payoff_tables[0]]
  return False, payoff_tables


def check_payoffs_are_hpt(payoff_tables):
  """Returns True if payoffs are in HPT format."""
  if isinstance(payoff_tables[0], np.ndarray):
    return False
  elif hasattr(payoff_tables[0], "is_hpt") and payoff_tables[0].is_hpt:
    return True
  else:
    raise TypeError("payoff_tables should be a list of payoff matrices/hpts.")
