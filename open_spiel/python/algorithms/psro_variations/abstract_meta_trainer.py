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

"""Abstract class for meta trainers (Generalized PSRO, RNR, ...)

Meta-algorithm with modular behaviour, allowing implementation of PSRO, RNR, and
other variations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import projected_replicator_dynamics
import pyspiel


# TODO(author4): Use C++ interface for (~10x) speedups in computing trajectories
def sample_episode(state, policies):
  """Samples an episode according to the policies, starting from state.

  Args:
    state: Pyspiel state representing the current state.
    policies: List of policy representing the policy executed by each player.

  Returns:
    The result of the call to returns() of the final state in the episode.
        Meant to be a win/loss integer.
  """
  if state.is_terminal():
    return np.array(state.returns(), dtype=np.float32)

  if state.is_chance_node():
    outcomes, probs = zip(*state.chance_outcomes())
  else:
    player = state.current_player()
    state_policy = policies[player](state)
    outcomes, probs = zip(*state_policy.items())

  state.apply_action(np.random.choice(outcomes, p=probs))
  return sample_episode(state, policies)


def uniform_strategy(solver):
  """Returns a Random Uniform distribution on policies.

  Args:
    solver: GenPSROSolver instance.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies
  if not isinstance(policies[0], list):
    policies = [policies]
  policy_lengths = [len(pol) for pol in policies]
  return [np.ones(pol_len) / pol_len for pol_len in policy_lengths]


def renormalize(probabilities):
  """Replaces all non-zero entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities


def nash_strategy(solver):
  """Returns nash distribution on meta game matrix.

  This method only works for two player zero-sum games.

  Args:
    solver: GenPSROSolver instance.

  Returns:
    Nash distribution on strategies.
  """
  meta_games = solver.get_meta_game
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  meta_games = [x.tolist() for x in meta_games]
  nash_prob_1, nash_prob_2, _, _ = (
      lp_solver.solve_zero_sum_matrix_game(
          pyspiel.create_matrix_game(*meta_games)))
  return [
      renormalize(np.array(nash_prob_1).reshape(-1)),
      renormalize(np.array(nash_prob_2).reshape(-1))
  ]


def prd_strategy(solver):
  """Computes Projected Replicator Dynamics strategies.

  Args:
    solver: GenPSROSolver instance.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs
  return projected_replicator_dynamics.projected_replicator_dynamics(
      meta_games, **kwargs)


META_STRATEGY_METHODS = {
    "uniform": uniform_strategy,
    "nash": nash_strategy,
    "prd": prd_strategy
}

DEFAULT_META_STRATEGY_METHOD = "prd"


class AbstractMetaTrainer(object):
  """Abstract class implementing meta trainers.

  If a trainer is something that computes a best response to given environment &
  agents, a meta trainer will compute which best responses to compute (Against
  what, how, etc)
  This class can support PBT, Hyperparameter Evolution, etc.
  """

  def __init__(self,
               game,
               oracle,
               initial_policies=None,
               meta_strategy_method=None,
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
      **kwargs: kwargs for meta strategy computation and training strategy
        selection
    """
    self._iterations = 0
    self._game = game
    self._oracle = oracle
    self._num_players = self._game.num_players()

    meta_strategy_method = meta_strategy_method or DEFAULT_META_STRATEGY_METHOD
    if isinstance(meta_strategy_method, str):
      self._meta_strategy_method = META_STRATEGY_METHODS[meta_strategy_method]
    elif callable(meta_strategy_method):
      self._meta_strategy_method = meta_strategy_method
    else:
      raise NotImplementedError(
          "Input type for strategy computation method not supported."
          " Accepted types : string, callable.")

    self._kwargs = kwargs

    self._initialize_policy(initial_policies)
    self._initialize_game_state()

  def _initialize_policy(self, initial_policies):
    return NotImplementedError(
        "initialize_policy not implemented. Initial policies passed as"
        " arguments : {}".format(initial_policies))

  def _initialize_game_state(self):
    return NotImplementedError("initialize_game_state not implemented.")

  def iteration(self, seed=None):
    self._iterations += 1
    self.update_meta_strategies()  # Compute nash equilibrium.
    self.update_agents()  # Generate new, Best Response agents via oracle.
    self.update_empirical_gamestate(seed=seed)  # Update gamestate matrix.

  def update_meta_strategies(self):
    self._meta_strategy_probabilities = self._meta_strategy_method(self)

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

  def get_and_update_meta_strategies(self, update=True):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    if update:
      self.update_meta_strategies()
    return self._meta_strategy_probabilities

  @property
  def get_meta_game(self):
    """Returns the meta game matrix."""
    return self._meta_games

  @property
  def get_policies(self):
    """Returns the players' policies."""
    return self._policies

  @property
  def get_kwargs(self):
    return self._kwargs
