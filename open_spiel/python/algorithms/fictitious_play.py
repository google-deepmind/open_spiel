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

# Lint as python3
"""Implementations of classical fictitious play.

See https://en.wikipedia.org/wiki/Fictitious_play.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability


def _uniform_policy(state):
  legal_actions = state.legal_actions()
  return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


def _callable_tabular_policy(tabular_policy):
  """Turns a tabular policy into a callable.

  Args:
    tabular_policy: A dictionary mapping information state key to a dictionary
      of action probabilities (action -> prob).

  Returns:
    A function `state` -> list of (action, prob)
  """

  def wrap(state):
    infostate_key = state.information_state_string(state.current_player())
    assert infostate_key in tabular_policy
    ap_list = []
    for action in state.legal_actions():
      assert action in tabular_policy[infostate_key]
      ap_list.append((action, tabular_policy[infostate_key][action]))
    return ap_list

  return wrap


class JointPolicy(policy.Policy):
  """A policy for all players in the game."""

  def __init__(self, game, policies):
    """Initializes a joint policy from a table of callables.

    Args:
       game: The game being played.
       policies: A dictionary mapping player number to a function `state` ->
         list of (action, prob).
    """
    super().__init__(game, list(range(game.num_players())))
    self.policies = policies

  def action_probabilities(self, state, player_id=None):
    return dict(self.policies[player_id or state.current_player()](state))


def _full_best_response_policy(br_infoset_dict):
  """Turns a dictionary of best response action selections into a full policy.

  Args:
    br_infoset_dict: A dictionary mapping information state to a best response
      action.

  Returns:
    A function `state` -> list of (action, prob)
  """

  def wrap(state):
    infostate_key = state.information_state_string(state.current_player())
    br_action = br_infoset_dict[infostate_key]
    ap_list = []
    for action in state.legal_actions():
      ap_list.append((action, 1.0 if action == br_action else 0.0))
    return ap_list

  return wrap


def _policy_dict_at_state(callable_policy, state):
  """Turns a policy function into a dictionary at a specific state.

  Args:
    callable_policy: A function from `state` -> lis of (action, prob),
    state: the specific state to extract the policy from.

  Returns:
    A dictionary of action -> prob at this state.
  """

  infostate_policy_list = callable_policy(state)
  infostate_policy = {}
  for ap in infostate_policy_list:
    infostate_policy[ap[0]] = ap[1]
  return infostate_policy


class XFPSolver(object):
  """An implementation of extensive-form fictitious play (XFP).

  XFP is Algorithm 1 in (Heinrich, Lanctot, and Silver, 2015, "Fictitious
  Self-Play in Extensive-Form Games"). Refer to the paper for details:
  http://mlanctot.info/files/papers/icml15-fsp.pdf.
  """

  def __init__(self, game, save_oracles=False):
    """Initialize the XFP solver.

    Arguments:
      game: the open_spiel game object.
      save_oracles: a boolean, indicating whether or not to save all the BR
        policies along the way (including the initial uniform policy). This
        could take up some space, and is only used when generating the meta-game
        for analysis.
    """

    self._game = game
    self._num_players = self._game.num_players()

    # A set of callables that take in a state and return a list of
    # (action, probability) tuples.
    self._oracles = [] if save_oracles else None

    # A set of callables that take in a state and return a list of
    # (action, probability) tuples.
    self._policies = []
    for _ in range(self._num_players):
      self._policies.append(_uniform_policy)
      if save_oracles:
        self._oracles.append([_uniform_policy])

    self._best_responses = [None] * self._num_players
    self._iterations = 0
    self._delta_tolerance = 1e-10
    self._average_policy_tables = []

  def average_policy_tables(self):
    """Returns a dictionary of information state -> dict of action -> prob.

    This is a joint policy (policy for all players).
    """
    return self._average_policy_tables

  def average_policy(self):
    """Returns the current average joint policy (policy for all players)."""
    return JointPolicy(self._game, self._policies)

  def iteration(self):
    self._iterations += 1
    self.compute_best_responses()
    self.update_average_policies()

  def compute_best_responses(self):
    """Updates self._oracles to hold best responses for each player."""
    for i in range(self._num_players):
      # Compute a best response policy to pi_{-i}.
      # First, construct pi_{-i}.
      joint_policy = self.average_policy()
      br_info = exploitability.best_response(self._game,
                                             joint_policy.to_tabular(), i)
      full_br_policy = _full_best_response_policy(
          br_info["best_response_action"])
      self._best_responses[i] = full_br_policy
      if self._oracles is not None:
        self._oracles[i].append(full_br_policy)

  def update_average_policies(self):
    """Update the average policies given the newly computed best response."""

    br_reach_probs = np.ones(self._num_players)
    avg_reach_probs = np.ones(self._num_players)
    self._average_policy_tables = [{} for _ in range(self._num_players)]
    self._recursively_update_average_policies(self._game.new_initial_state(),
                                              avg_reach_probs, br_reach_probs)
    for i in range(self._num_players):
      self._policies[i] = _callable_tabular_policy(
          self._average_policy_tables[i])

  def _recursively_update_average_policies(self, state, avg_reach_probs,
                                           br_reach_probs):
    """Recursive implementation of the average strategy update."""

    if state.is_terminal():
      return
    elif state.is_chance_node():
      for action, _ in state.chance_outcomes():
        new_state = state.clone()
        new_state.apply_action(action)
        self._recursively_update_average_policies(new_state, avg_reach_probs,
                                                  br_reach_probs)
    else:
      player = state.current_player()
      avg_policy = _policy_dict_at_state(self._policies[player], state)
      br_policy = _policy_dict_at_state(self._best_responses[player], state)
      legal_actions = state.legal_actions()
      infostate_key = state.information_state_string(player)
      # First traverse the subtrees.
      for action in legal_actions:
        assert action in br_policy
        assert action in avg_policy
        new_state = state.clone()
        new_state.apply_action(action)
        new_avg_reach = np.copy(avg_reach_probs)
        new_avg_reach[player] *= avg_policy[action]
        new_br_reach = np.copy(br_reach_probs)
        new_br_reach[player] *= br_policy[action]
        self._recursively_update_average_policies(new_state, new_avg_reach,
                                                  new_br_reach)
      # Now, do the updates.
      if infostate_key not in self._average_policy_tables[player]:
        alpha = 1 / (self._iterations + 1)
        self._average_policy_tables[player][infostate_key] = {}
        pr_sum = 0.0
        for action in legal_actions:
          pr = (
              avg_policy[action] + (alpha * br_reach_probs[player] *
                                    (br_policy[action] - avg_policy[action])) /
              ((1.0 - alpha) * avg_reach_probs[player] +
               alpha * br_reach_probs[player]))
          self._average_policy_tables[player][infostate_key][action] = pr
          pr_sum += pr
        assert (1.0 - self._delta_tolerance <= pr_sum <=
                1.0 + self._delta_tolerance)

  def sample_episode(self, state, policies):
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
    elif state.is_chance_node():
      outcomes = []
      probs = []
      for action, prob in state.chance_outcomes():
        outcomes.append(action)
        probs.append(prob)
      outcome = np.random.choice(outcomes, p=probs)
      state.apply_action(outcome)
      return self.sample_episode(state, policies)
    else:
      player = state.current_player()
      state_policy = _policy_dict_at_state(policies[player], state)
      actions = []
      probs = []
      for action in state_policy:
        actions.append(action)
        probs.append(state_policy[action])
      action = np.random.choice(actions, p=probs)
      state.apply_action(action)
      return self.sample_episode(state, policies)

  def sample_episodes(self, policies, num):
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num: Number of episodes to execute to estimate average return of policies.

    Returns:
      Average episode return over num episodes.
    """

    totals = np.zeros(self._num_players)
    for _ in range(num):
      totals += self.sample_episode(self._game.new_initial_state(), policies)
    return totals / num

  def get_empirical_metagame(self, sims_per_entry, seed=None):
    """Gets a meta-game tensor of utilities from episode samples.

    The tensor is a cross-table of all the saved oracles and initial uniform
    policy.

    Args:
        sims_per_entry: number of simulations (episodes) to perform per entry in
          the tables, i.e. each is a crude Monte Carlo estimate
        seed: the seed to set for random sampling, for reproducibility

    Returns:
        the K^n (KxKx...K, with dimension n) meta-game tensor where n is the
        number of players and K is the number of strategies (one more than the
        number of iterations of fictitious play since the initial uniform
        policy is included).
    """

    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracles is not None
    num_strategies = len(self._oracles[0])
    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    meta_games = []
    for _ in range(self._num_players):
      shape = [num_strategies] * self._num_players
      meta_game = np.ndarray(shape=shape, dtype=np.float32)
      meta_games.append(meta_game)
    for coord in itertools.product(
        range(num_strategies), repeat=self._num_players):
      policies = []
      for i in range(self._num_players):
        iteration = coord[i]
        policies.append(self._oracles[i][iteration])
      utility_estimates = self.sample_episodes(policies, sims_per_entry)
      for i in range(self._num_players):
        meta_games[i][coord] = utility_estimates[i]
    return meta_games
