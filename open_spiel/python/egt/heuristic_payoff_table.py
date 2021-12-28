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

"""An object to store the heuristic payoff table for a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import numpy as np

from open_spiel.python.egt import utils


def _inc_average(count, average, value):
  """Computes the incremental average, `a_n = ((n - 1)a_{n-1} + v_n) / n`."""
  count += 1
  average = ((count - 1) * average + value) / count
  return (count, average)


def from_match_results(df, consider_agents):
  """Builds a heuristic payoff table from average win probabilities.

  Args:
    df: a Pandas dataframe of match results. Must contain a column "agents"
      consisting of tuples of agent names, and a column "scores" consisting of
      the score for each agent in the match.
    consider_agents: a list of agent names. Will only consider matches in which
      exclusively these agents appeared.

  Returns:
    A PayoffTable object.

  Raises:
    ValueError: if dataframe is empty, or columns 'agents' and 'scores' not
    specified, or games have zero players.
  """
  if df.empty:
    raise ValueError("Please provide a non-empty dataframe.")
  if "agents" not in df.columns:
    raise ValueError("Dataframe must contain a column 'agents'.")
  if "scores" not in df.columns:
    raise ValueError("Dataframe must contain a column 'scores'.")

  num_strategies = len(consider_agents)
  num_players = len(df["agents"][0])

  if num_players == 0:
    raise ValueError("Games must have > 0 players.")

  count_per_distribution = {}
  win_prob_per_distribution = {}

  for i, row in df.iterrows():
    print("Parsing row {} / {} ...".format(i, len(df)), end="\r")
    agents = row["agents"]
    scores = row["scores"]
    assert len(agents) == len(scores) == num_players

    if not set(agents).issubset(set(consider_agents)):
      # Ignore agents outside those we are supposed to consider.
      continue
    elif len(set(agents)) == 1:
      # Special case of self-play: deal with separately.
      continue

    # Find winner(s): In each match one must determine a winning strategy. One
    # way of doing this is to average over the returns for each strategy and
    # then say that the one with the greatest returns is the winner.

    # Get unique score per agent by averaging.
    count_per_agent = collections.defaultdict(int)
    average_score_per_agent = collections.defaultdict(int)
    for agent, score in zip(agents, scores):
      count_per_agent[agent], average_score_per_agent[agent] = _inc_average(
          count_per_agent[agent], average_score_per_agent[agent], score)

    winner_score = max(average_score_per_agent.values())
    winner_agents = [
        k for k, v in average_score_per_agent.items() if v == winner_score
    ]
    winner_strategy_idxs = [
        consider_agents.index(winner) for winner in winner_agents
    ]

    # Select the winner as the one maximizing the selected statistics.
    win_probabilities = np.zeros(num_strategies)
    for winner_strategy_idx in winner_strategy_idxs:
      win_probabilities[winner_strategy_idx] = 1 / len(winner_strategy_idxs)

    distribution = np.zeros(num_strategies)
    for agent, count in count_per_agent.items():
      strategy_idx = consider_agents.index(agent)
      distribution[strategy_idx] = count

    distribution = tuple(distribution)

    if distribution not in count_per_distribution:
      count_per_distribution[distribution] = 1
      win_prob_per_distribution[distribution] = win_probabilities
      continue
    (count_per_distribution[distribution],
     win_prob_per_distribution[distribution]) = _inc_average(
         count_per_distribution[distribution],
         win_prob_per_distribution[distribution], win_probabilities)

  # Populate self-play case (strategy both wins and loses).
  for idx, agent in enumerate(consider_agents):
    distribution = np.zeros(num_strategies)
    distribution[idx] = num_players
    distribution = tuple(distribution)
    win_prob = np.zeros(num_strategies)
    win_prob[idx] = 0.5
    win_prob_per_distribution[distribution] = win_prob

  # Create empty (nan) payoff table.
  table = PayoffTable(num_players, num_strategies)

  # Populate with win probabilities.
  for distribution, payoff in win_prob_per_distribution.items():
    table[distribution] = payoff

  return table


def from_matrix_game(matrix_game):
  """Returns a PayOffTable given a symmetric 2-player matrix game.

  Args:
    matrix_game: The payoff matrix corresponding to a 2-player symmetric game.
  """

  if not isinstance(matrix_game, np.ndarray):
    raise ValueError("The matrix game should be a numpy array, not a {}".format(
        type(matrix_game)))
  num_strats_per_population = (
      utils.get_num_strats_per_population(
          payoff_tables=[matrix_game], payoffs_are_hpt_format=False))
  assert len(num_strats_per_population) == 2
  assert num_strats_per_population[0] == num_strats_per_population[1]
  num_strategies = num_strats_per_population[0]

  num_profiles = utils.get_num_profiles(num_strats_per_population)
  table = PayoffTable(num_players=2, num_strategies=num_strategies)

  # Construct the HPT by filling in the corresponding payoffs for each profile
  for id_profile in range(num_profiles):
    strat_profile = utils.get_strat_profile_from_id(num_strats_per_population,
                                                    id_profile)
    distribution = table.get_distribution_from_profile(strat_profile)
    # For symmetric matrix games, multiple strategy profiles correspond to the
    # same distribution and payoffs. Thus, ensure the table entry has not
    # already been filled by a previous strategy profile.
    if table.item_is_uninitialized(tuple(distribution)):
      payoffs = np.zeros(num_strategies)
      payoffs[strat_profile[0]] = matrix_game[strat_profile[0],
                                              strat_profile[1]]
      payoffs[strat_profile[1]] = matrix_game[strat_profile[1],
                                              strat_profile[0]]
      table[tuple(distribution)] = payoffs

  return table


def from_heuristic_payoff_table(hpt):
  """Returns a `PayoffTable` instance from a numpy 2D HPT."""
  [num_rows, num_columns] = hpt.shape
  assert num_columns % 2 == 0
  num_strategies = int(num_columns / 2)
  num_players = np.sum(hpt[0, :num_strategies])
  obj = PayoffTable(num_players, num_strategies, initialize_payoff_table=False)

  # pylint: disable=protected-access
  for row in hpt:
    payoff_row = np.array(row[num_strategies:])
    obj._payoff_table[tuple(row[:num_strategies])] = payoff_row

  assert len(obj._payoff_table) == num_rows
  # pylint: enable=protected-access
  return obj


def _compute_win_probability_from_elo(rating_1, rating_2):
  """Computes the win probability of 1 vs 2 based on the provided Elo ratings.

  Args:
    rating_1: The Elo rating of player 1.
    rating_2: The Elo rating of player 2.

  Returns:
    The win probability of player 1, when playing against 2.
  """
  m = max(rating_1, rating_2)  # We subtract the max for numerical stability.

  m1 = 10**((rating_1 - m) / 400)
  m2 = 10**((rating_2 - m) / 400)

  return m1 / (m1 + m2)


def from_elo_scores(elo_ratings, num_agents=2):
  """Computes the Elo win probability payoff matrix `X` from the Elo scores.

  Args:
    elo_ratings: The elo scores vector of length [num_strategies].
    num_agents: The number of agents. Only 2 agents are supported for now.

  Returns:
    The HPT associated to the Elo win probability payoff matrix `X`. The score
    for a given agent is given by its win probability given its Elo score.

  Raises:
    ValueError: If `num_agents != 2`.
  """
  if num_agents != 2:
    raise ValueError("Only 2 agents are supported, because we need to compute "
                     "the win probability and that can only be computed with "
                     "2 players.")
  num_strategies = len(elo_ratings)

  hpt_rows = []

  possible_teams = utils.distribute(num_agents, num_strategies, normalize=False)

  for distribution_row in possible_teams:
    payoff_row = np.zeros([num_strategies])
    non_zero_index = np.nonzero(distribution_row)[0]  # Why [0]?
    assert len(non_zero_index.shape) == 1

    if len(non_zero_index) > 1:
      index_first_player, index_second_player = non_zero_index
      prob = _compute_win_probability_from_elo(elo_ratings[index_first_player],
                                               elo_ratings[index_second_player])
      payoff_row[index_first_player] = prob
      payoff_row[index_second_player] = 1 - prob
    elif len(non_zero_index) == 1:
      payoff_row[non_zero_index[0]] = 0.5
    else:
      assert False, "Impossible case, we have at least one strategy used."

    hpt_rows.append(np.hstack([distribution_row, payoff_row]))

  return NumpyPayoffTable(np.vstack(hpt_rows))


class _PayoffTableInterface(metaclass=abc.ABCMeta):
  """An interface for the PayoffTable classes."""

  @abc.abstractmethod
  def __call__(self):
    """Returns a view of the table as a np.array."""

  @abc.abstractproperty
  def num_strategies(self):
    pass

  @abc.abstractproperty
  def num_players(self):
    pass

  @abc.abstractproperty
  def num_rows(self):
    pass

  def expected_payoff(self, strategy):
    """The expected payoff of each pure strategy against the mixed strategy.

    We define the expected payoff of a strategy A as the expected payoff of
    that strategy over the space of 2 randomly sampled

    The mixed strategy is equivalently the composition of an infinitely large
    population. To find the expected payoff, we:
    1. Compute the probabilities of sampling each player distribution in the
       heuristic payoff table from the population.
    2. Compute the expected payoff of pure strategy against the mixed
       strategy by averaging over the payoff rows with these probabilities.

    For each pure strategy we must normalize by the probability that it appeared
    in the player distribution at all; otherwise we would be undercounting.

    For more details, see https://arxiv.org/pdf/1803.06376.pdf.

    Args:
      strategy: an `np.array(shape=self._num_strategies)` of probabilities.

    Returns:
      An `np.array(shape=self._num_strategies)` of payoffs for pure strategies.

    Raises:
      ValueError: if the provided strategy probabilities do not define a valid
        distribution over `self._num_strategies` strategies.
    """
    if strategy.shape != (self.num_strategies,):
      raise ValueError("The strategy probabilities should be of shape "
                       "({},), not {}".format(self.num_strategies,
                                              strategy.shape))
    if np.around(np.sum(strategy), decimals=3) != 1.0:
      raise ValueError("The strategy probabilities should sum to 1.")
    if not all([p >= 0 for p in strategy]):
      raise ValueError("The strategy probabilities should all be >= 0.")

    distributions = self._distributions
    # Multinomial coefficients (one per distribution).
    coefficients = _multinomial_coefficients(distributions)
    # Probabilities of sampling each distribution given population composition.
    probabilities = _row_probabilities(coefficients, distributions, strategy)

    return _expected_payoff(probabilities, self._payoffs, strategy,
                            self._num_players)

  @property
  def _payoffs(self):
    """Returns an np.array containing the payoffs."""
    return self()[:, self.num_strategies:]

  @property
  def _distributions(self):
    """Returns an np.array containing the distribution over pure strategies."""
    return self()[:, :self.num_strategies]


class NumpyPayoffTable(object):
  """An object wrapping a Numpy array heuristic payoff table for a metagame.

  NOTE: We assume the number of players to be equal to the number of
  replicators.

  """

  def __init__(self, payoff_table, writeable=False):
    """Initializes an immutable payoff table.

    Let p be the number of players, k be the number of strategies. Then, there
    are Combinations(p + k - 1, k - 1) distinct configurations for the
    strategies of the p players.

    The payoff table is of shape [(p + k - 1)! / (p! * (k - 1)!), 2 * k].

    The first k columns encode the number of players playing each strategies.

    The second k columns encode the average payoff of each strategy in that
    game.

    Args:
      payoff_table: A numpy heuristic payoff table, which is assumed to be
        correctly constructed.
      writeable: Whether the numpy array payoff_table should be writeable. See
        https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ndarray.flags.html.
          However, locking a base object does not lock any views that already
          reference it,
    """
    self._writeable = writeable
    self._payoff_table = payoff_table

    [self._num_rows, num_columns] = self._payoff_table.shape
    assert num_columns % 2 == 0
    self._num_strategies = int(num_columns / 2)
    self._num_players = np.sum(self._payoff_table[0, :self._num_strategies])

  def __call__(self):
    """Returns a view of the table as a np.array.

    The mutability of the object is controlled by `writeable`.
    """
    if self._writeable:
      return self._payoff_table
    else:
      return np.copy(self._payoff_table)

  @property
  def writeable(self):
    return self._writeable

  @writeable.setter
  def writeable(self, writeable):
    self._writeable = writeable

  @property
  def num_strategies(self):
    return self._num_strategies

  @property
  def num_players(self):
    return self._num_players

  @property
  def num_rows(self):
    return self._num_rows


class PayoffTable(_PayoffTableInterface):
  """A mutable object to store the heuristic payoff table for a metagame."""

  def __init__(self, num_players, num_strategies, initialize_payoff_table=True):
    """A heuristic payoff table encodes payoffs from various strategy profiles.

    See `NumpyPayoffTable` for the description of the heuristic payoff table.

    Internally, this is represented as an OrderedDict {distribution: payoff}.

    Args:
      num_players: The number of players in the game.
      num_strategies: The number of strategies an individual could play.
      initialize_payoff_table: If `True`, nan entries will be created for all
        rows. If `False`, no rows are created at all.
    """
    super(PayoffTable, self).__init__()
    self.is_hpt = True
    self._num_players = num_players
    self._num_strategies = num_strategies
    self._payoff_table = collections.OrderedDict()

    if initialize_payoff_table:
      # Populate empty (nan) payoff table.
      player_distributions = utils.distribute(self._num_players,
                                              self._num_strategies)
      for d in player_distributions:
        self._payoff_table[d] = np.full(self._num_strategies, np.nan)

  def __call__(self):
    """Returns a view of the table as a np.array."""
    return np.concatenate((self._distributions, self._payoffs), axis=1)

  @property
  def _payoffs(self):
    """Returns an np.array containing the payoffs."""
    return np.array(list(self._payoff_table.values()))

  @property
  def _distributions(self):
    """Returns an np.array containing the distribution over pure strategies."""
    return np.array(list(self._payoff_table))

  @property
  def num_strategies(self):
    return self._num_strategies

  @property
  def num_players(self):
    return self._num_players

  @property
  def num_rows(self):
    return len(self._payoff_table)

  def __setitem__(self, distribution, payoff):
    assert distribution in self._payoff_table
    assert len(payoff) == self._num_strategies
    self._payoff_table[distribution] = payoff

  def __getitem__(self, distribution):
    """Returns the payoff profile for a given strategy distribution.

    Args:
      distribution: strategy profile tuple.

    Returns:
      Payoff profile for the corresponding strategy distribution.
    """
    return self._payoff_table[distribution]

  def item_is_uninitialized(self, distribution):
    return np.isnan(np.sum(self._payoff_table[distribution]))

  def get_distribution_from_profile(self, strat_profile):
    distribution = [0] * self.num_strategies
    for s in strat_profile:
      distribution[s] += 1
    return distribution


# The following provides utility functions to compute the expected payoff of
# a given strategy profile.
# See https://arxiv.org/pdf/1803.06376.pdf, page 3, left column.
#
# Usage:
#
# coefficients = _multinomial_coefficients(distributions, strategies):
# row_probabilities = _row_probabilities(coefficients, distributions, strategy)
# expected_payoff = _expected_payoff(row_probabilities, payoffs, composition,
#                                    num_players)
#
#
def _multinomial_coefficients(distributions):
  """Returns the multinomial coefficients.

  Args:
    distributions: The distributions table [num_rows, num_strategies].
  """
  v_factorial = np.vectorize(np.math.factorial)
  # Multinomial coefficients (one per distribution Ni).
  # (         P         )
  # ( Ni1, Ni1, ... Nik )
  coefficients = (
      v_factorial(np.sum(distributions, axis=1)) /
      np.prod(v_factorial(distributions), axis=1))

  return coefficients


def _row_probabilities(coefficients, distributions, strategy):
  """Returns the row probabilities [num_rows].

  Args:
    coefficients: The multinomial coefficients [num_rows].
    distributions: The distributions table [num_rows, num_strategies].
    strategy: The strategy array [num_strategies].
  """
  row_probabilities = coefficients * np.prod(
      np.power(strategy, distributions), axis=1)
  return row_probabilities


def _expected_payoff(row_probabilities, payoffs, strategy, num_players):
  # pylint: disable=g-doc-args
  r"""Returns the expected payoff.

  Computes (with p=num_players):

  r_j = \sum_i row_probabilities[i] * payoffs[i, j] / (1 - (1-strategy[j])^p)
  """
  # pylint: enable=g-doc-args
  [num_rows] = row_probabilities.shape
  [num_rows_2, num_strategies] = payoffs.shape
  [num_strategies_2] = strategy.shape
  assert num_rows == num_rows_2
  assert num_strategies == num_strategies_2

  # One per pure strategy.
  numerators = np.dot(np.transpose(payoffs), row_probabilities)
  # One per pure strategy.
  denominators = 1 - np.power(1 - strategy, num_players)
  return numerators / denominators
