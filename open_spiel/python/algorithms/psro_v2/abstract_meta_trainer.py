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
"""Abstract class for meta trainers (Generalized PSRO, RNR, ...)

Meta-algorithm with modular behaviour, allowing implementation of PSRO, RNR, and
other variations.
"""

import numpy as np
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils

_DEFAULT_STRATEGY_SELECTION_METHOD = "probabilistic"
_DEFAULT_META_STRATEGY_METHOD = "prd"


def _process_string_or_callable(string_or_callable, dictionary):
  """Process a callable or a string representing a callable.

  Args:
    string_or_callable: Either a string or a callable
    dictionary: Dictionary of shape {string_reference: callable}

  Returns:
    string_or_callable if string_or_callable is a callable ; otherwise,
    dictionary[string_or_callable]

  Raises:
    NotImplementedError: If string_or_callable is of the wrong type, or has an
      unexpected value (Not present in dictionary).
  """
  if callable(string_or_callable):
    return string_or_callable

  try:
    return dictionary[string_or_callable]
  except KeyError:
    raise NotImplementedError("Input type / value not supported. Accepted types"
                              ": string, callable. Acceptable string values : "
                              "{}. Input provided : {}".format(
                                  list(dictionary.keys()), string_or_callable))


def sample_episode(state, policies):
  """Samples an episode using policies, starting from state.

  Args:
    state: Pyspiel state representing the current state.
    policies: List of policy representing the policy executed by each player.

  Returns:
    The result of the call to returns() of the final state in the episode.
        Meant to be a win/loss integer.
  """
  if state.is_terminal():
    return np.array(state.returns(), dtype=np.float32)

  if state.is_simultaneous_node():
    actions = [None] * state.num_players()
    for player in range(state.num_players()):
      state_policy = policies[player](state, player)
      outcomes, probs = zip(*state_policy.items())
      actions[player] = utils.random_choice(outcomes, probs)
    state.apply_actions(actions)
    return sample_episode(state, policies)

  if state.is_chance_node():
    outcomes, probs = zip(*state.chance_outcomes())
  else:
    player = state.current_player()
    state_policy = policies[player](state)
    outcomes, probs = zip(*state_policy.items())

  state.apply_action(utils.random_choice(outcomes, probs))
  return sample_episode(state, policies)


class AbstractMetaTrainer(object):
  """Abstract class implementing meta trainers.

  If a trainer is something that computes a best response to given environment &
  agents, a meta trainer will compute which best responses to compute (Against
  what, how, etc)
  This class can support PBT, Hyperparameter Evolution, etc.
  """

  # pylint:disable=dangerous-default-value
  def __init__(self,
               game,
               oracle,
               initial_policies=None,
               meta_strategy_method=_DEFAULT_META_STRATEGY_METHOD,
               training_strategy_selector=_DEFAULT_STRATEGY_SELECTION_METHOD,
               symmetric_game=False,
               number_policies_selected=1,
               **kwargs):
    """Abstract Initialization for meta trainers.

    Args:
      game: A pyspiel game object.
      oracle: An oracle object, from an implementation of the AbstractOracle
        class.
      initial_policies: A list of initial policies, to set up a default for
        training. Resorts to tabular policies if not set.
      meta_strategy_method: String, or callable taking a MetaTrainer object and
        returning a list of meta strategies (One list entry per player).
        String value can be:
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      training_strategy_selector: A callable or a string. If a callable, takes
        as arguments: - An instance of `PSROSolver`, - a
          `number_policies_selected` integer. and returning a list of
          `num_players` lists of selected policies to train from.
        When a string, supported values are:
              - "top_k_probabilites": selects the first
                'number_policies_selected' policies with highest selection
                probabilities.
              - "probabilistic": randomly selects 'number_policies_selected'
                with probabilities determined by the meta strategies.
              - "exhaustive": selects every policy of every player.
              - "rectified": only selects strategies that have nonzero chance of
                being selected.
              - "uniform": randomly selects 'number_policies_selected' policies
                with uniform probabilities.
      symmetric_game: Whether to consider the current game as symmetric (True)
        game or not (False).
      number_policies_selected: Maximum number of new policies to train for each
        player at each PSRO iteration.
      **kwargs: kwargs for meta strategy computation and training strategy
        selection
    """
    self._iterations = 0
    self._game = game
    self._oracle = oracle
    self._num_players = self._game.num_players()

    self.symmetric_game = symmetric_game
    self._game_num_players = self._num_players
    self._num_players = 1 if symmetric_game else self._num_players

    self._number_policies_selected = number_policies_selected

    meta_strategy_method = _process_string_or_callable(
        meta_strategy_method, meta_strategies.META_STRATEGY_METHODS)
    print("Using {} as strategy method.".format(meta_strategy_method))

    self._training_strategy_selector = _process_string_or_callable(
        training_strategy_selector,
        strategy_selectors.TRAINING_STRATEGY_SELECTORS)
    print("Using {} as training strategy selector.".format(
        self._training_strategy_selector))

    self._meta_strategy_method = meta_strategy_method
    self._kwargs = kwargs

    self._initialize_policy(initial_policies)
    self._initialize_game_state()
    self.update_meta_strategies()

  def _initialize_policy(self, initial_policies):
    return NotImplementedError(
        "initialize_policy not implemented. Initial policies passed as"
        " arguments : {}".format(initial_policies))

  def _initialize_game_state(self):
    return NotImplementedError("initialize_game_state not implemented.")

  def iteration(self, seed=None):
    """Main trainer loop.

    Args:
      seed: Seed for random BR noise generation.
    """
    self._iterations += 1
    self.update_agents()  # Generate new, Best Response agents via oracle.
    self.update_empirical_gamestate(seed=seed)  # Update gamestate matrix.
    self.update_meta_strategies()  # Compute meta strategy (e.g. Nash)

  def update_meta_strategies(self):
    self._meta_strategy_probabilities = self._meta_strategy_method(self)
    if self.symmetric_game:
      self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

  def update_agents(self):
    return NotImplementedError("update_agents not implemented.")

  def update_empirical_gamestate(self, seed=None):
    return NotImplementedError("update_empirical_gamestate not implemented."
                               " Seed passed as argument : {}".format(seed))

  def sample_episodes(self, policies, num_episodes):
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num_episodes: Number of episodes to execute to estimate average return of
        policies.

    Returns:
      Average episode return over num episodes.
    """
    totals = np.zeros(self._num_players)
    for _ in range(num_episodes):
      totals += sample_episode(self._game.new_initial_state(),
                               policies).reshape(-1)
    return totals / num_episodes

  def get_meta_strategies(self):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    meta_strategy_probabilities = self._meta_strategy_probabilities
    if self.symmetric_game:
      meta_strategy_probabilities = self._game_num_players * meta_strategy_probabilities
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_meta_game(self):
    """Returns the meta game matrix."""
    meta_games = self._meta_games
    if self.symmetric_game:
      meta_games = self._game_num_players * meta_games
    return [np.copy(a) for a in meta_games]

  def get_policies(self):
    """Returns the players' policies."""
    policies = self._policies
    if self.symmetric_game:
      policies = self._game_num_players * policies
    return policies

  def get_kwargs(self):
    return self._kwargs
