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

"""Python implementation of the counterfactual regret minimization algorithm.

One iteration of CFR consists of:
1) Compute current strategy from regrets (e.g. using Regret Matching).
2) Compute values using the current strategy
3) Compute regrets from these values

The average policy is what converges to a Nash Equilibrium.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import attr
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

_INITIAL_POSITIVE_VALUE = 1e-15


@attr.s
class _InfoStateNode(object):
  """An object wrapping values associated to an information state."""
  # The list of the legal actions.
  legal_actions = attr.ib()
  # Map from information states string representations and actions to the
  # counterfactual regrets, accumulated over the policy iterations
  cumulative_regret = attr.ib(factory=lambda: collections.defaultdict(float))
  # Same as above for the cumulative of the policy probabilities computed
  # during the policy iterations
  cumulative_policy = attr.ib(factory=lambda: collections.defaultdict(float))


def _initialize_info_state_nodes(state, info_state_nodes,
                                 initial_positive_value):
  """Initializes info_state_nodes.

  Set `cumulative_regret` to _INITIAL_POSITIVE_VALUE
  for all (info_state, action). Also set the the legal_actions list.

  Args:
    state: The current state in the tree walk. This should be the root node when
      we call this function from a CFR solver.
    info_state_nodes: The dictionary `info_state_str` to `_InfoStateNode` to
      fill in-place.
    initial_positive_value: The initial value to use for both the cumulative
      regret and cumulative policy for all state-actions.
  """
  if state.is_terminal():
    return

  if state.is_chance_node():
    for action, unused_action_prob in state.chance_outcomes():
      _initialize_info_state_nodes(
          state.child(action), info_state_nodes, initial_positive_value)
    return

  current_player = state.current_player()
  info_state = state.information_state(current_player)

  info_state_node = info_state_nodes.get(info_state)
  if info_state_node is None:
    legal_actions = state.legal_actions(current_player)
    info_state_node = _InfoStateNode(legal_actions=legal_actions)
    info_state_nodes[info_state] = info_state_node

  for action in info_state_node.legal_actions:
    info_state_node.cumulative_regret[action] = initial_positive_value
    info_state_node.cumulative_policy[action] = 0
    _initialize_info_state_nodes(
        state.child(action), info_state_nodes, initial_positive_value)


def _apply_regret_matching_plus_reset(info_state_nodes):
  """Resets negative cumulative regrets to 0.

  Regret Matching+ corresponds to the following cumulative regrets update:
  cumulative_regrets = max(cumulative_regrets + regrets, 0)

  This must be done at the level of the information set, and thus cannot be
  done during the tree traversal (which is done on histories). It is thus
  performed as an additional step.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state_node in info_state_nodes.values():
    action_to_cum_regret = info_state_node.cumulative_regret
    for action, cumulative_regret in action_to_cum_regret.items():
      if cumulative_regret < 0:
        action_to_cum_regret[action] = 0


def _update_current_policy(current_policy, info_state_nodes):
  """Updates in place `current_policy` from the cumulative regrets.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    current_policy: A `policy.TabularPolicy` to be updated in-place.
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state, info_state_node in info_state_nodes.items():
    state_policy = current_policy.policy_for_key(info_state)

    for action, value in _regret_matching(
        info_state_node.cumulative_regret,
        info_state_node.legal_actions).items():
      state_policy[action] = value


def _update_average_policy(average_policy, info_state_nodes):
  """Updates in place `average_policy` to the average of all policies iterated.

  This function is a module level function to be reused by both CFRSolver and
  CFRBRSolver.

  Args:
    average_policy: A `policy.TabularPolicy` to be updated in-place.
    info_state_nodes: A dictionary {`info_state_str` -> `_InfoStateNode`}.
  """
  for info_state, info_state_node in info_state_nodes.items():
    info_state_policies_sum = info_state_node.cumulative_policy
    state_policy = average_policy.policy_for_key(info_state)
    probabilities_sum = sum(info_state_policies_sum.values())
    if probabilities_sum == 0:
      num_actions = len(info_state_node.legal_actions)
      for action in info_state_node.legal_actions:
        state_policy[action] = 1 / num_actions
    else:
      for action, action_prob_sum in info_state_policies_sum.items():
        state_policy[action] = action_prob_sum / probabilities_sum


class _CFRSolver(object):
  """Implements the Counterfactual Regret Minimization (CFR) algorithm.

  The algorithm computes an approximate Nash policy for 2 player zero-sum games.

  CFR can be view as a policy iteration algorithm. Importantly, the policies
  themselves do not converge to a Nash policy, but their average does.

  The main iteration loop is implemented in `evaluate_and_update_policy`:

  ```python
      game = pyspiel.load_game("game_name")
      initial_state = game.new_initial_state()

      cfr_solver = CFRSolver(game)

      for i in range(num_iterations):
        cfr.evaluate_and_update_policy()
  ```

  Once the policy has converged, the average policy (which converges to the Nash
  policy) can be computed:
  ```python
        average_policy = cfr_solver.ComputeAveragePolicy()
  ```
  """

  def __init__(self, game, initialize_cumulative_values, alternating_updates,
               linear_averaging, regret_matching_plus):
    # pyformat: disable
    """Initializer.

    Args:
      game: The `pyspiel.Game` to run on.
      initialize_cumulative_values: Whether to initialize the average policy to
        the uniform policy (and the initial cumulative regret to an epsilon
        value). This is independent of the first CFR iteration, which, when the
        policy is fixed during traversal and we perform non alternating updates,
        will also compute the uniform policy and add it to the average of
        policies.
      alternating_updates: If `True`, alternating updates are performed: for
        each player, we compute and update the cumulative regrets and policies.
        In that case, and when the policy is frozen during tree traversal, the
        cache is reset after each update for one player.
        Otherwise, the update is simultaneous.
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
    self._game = game
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()

    if initialize_cumulative_values:
      initial_positive_value = _INITIAL_POSITIVE_VALUE
    else:
      initial_positive_value = 0
    self._info_state_nodes = {}
    _initialize_info_state_nodes(
        self._root_node,
        info_state_nodes=self._info_state_nodes,
        initial_positive_value=initial_positive_value)

    self._policy_cache = {}

    self._root_node = self._game.new_initial_state()

    # This is for returning the current policy and average policy to a caller
    self._current_policy = policy.TabularPolicy(game)
    self._average_policy = self._current_policy.__copy__()

    self._linear_averaging = linear_averaging
    self._iteration = 0  # For possible linear-averaging.

    self._alternating_updates = alternating_updates
    self._regret_matching_plus = regret_matching_plus

  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    self._iteration += 1
    if self._alternating_updates:
      for player in range(self._game.num_players()):
        self._compute_counterfactual_regret_for_player(
            self._root_node, np.ones(self._game.num_players() + 1), player)
        if self._regret_matching_plus:
          _apply_regret_matching_plus_reset(self._info_state_nodes)
        self._policy_cache.clear()
    else:
      self._compute_counterfactual_regret_for_player(
          self._root_node, np.ones(self._game.num_players() + 1), player=None)
      if self._regret_matching_plus:
        _apply_regret_matching_plus_reset(self._info_state_nodes)

    self._policy_cache.clear()

  def policy(self):
    """Returns the current policy (TabularPolicy) (no convergence guarantees).

    This function exists to have access to the policy, but one should use
    `average_policy` for a Nash-Equilibrium converging sequence.
    """
    _update_current_policy(self._current_policy, self._info_state_nodes)
    return self._current_policy

  def average_policy(self):
    """Returns the average of all policies iterated.

    This average policy converges to a Nash policy as the number of iterations
    increases.

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `policy.TabularPolicy` object, giving the policy for both players.
    """
    _update_average_policy(self._average_policy, self._info_state_nodes)
    return self._average_policy

  def _compute_counterfactual_regret_for_player(self, state,
                                                reach_probabilities, player):
    """Increments the cumulative regrets and policy for `player`.

    Args:
      state: The initial game state to analyze from.
      reach_probabilities: The probability for each player of reaching `state`
        as a numpy array [prob for player 0, for player 1,..., for chance].
        `player_reach_probabilities[player]` will work in all cases.
      player: The 0-indexed player to update the values for. If `None`, the
        update for all players will be performed.

    Returns:
      The utility of `state` for all players, assuming all players follow the
      current policy defined by `self.Policy`.
    """
    if state.is_terminal():
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob
        state_value += action_prob * self._compute_counterfactual_regret_for_player(
            new_state, new_reach_probabilities, player)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state(current_player)
    legal_actions = state.legal_actions(current_player)

    # No need to continue on this history branch as no update will be performed
    # for any player.
    # The value we return here is not used in practice. If the conditional
    # statement is True, then the last taken action has probability 0 of
    # occurring, so the returned value is not impacting the parent node value.
    if all(reach_probabilities[:-1] == 0):
      return np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    # The utilities of the children states are computed recursively. As the
    # regrets are added to the information state regrets for each state in that
    # information state, the recursive call can only be made once per child
    # state. Therefore, the utilities are cached.
    children_utilities = {}

    info_state_policy = self._compute_policy_or_get_it_from_cache(
        info_state, legal_actions)
    for action, action_prob in info_state_policy.items():
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      child_utility = self._compute_counterfactual_regret_for_player(
          new_state, reach_probabilities=new_reach_probabilities, player=player)

      state_value += action_prob * child_utility
      children_utilities[action] = child_utility

    # If we are performing alternating updates, and the current player is not
    # the current_player, we skip the cumulative values update.
    # If we are performing simultaneous updates, we do update the cumulative
    # values.
    simulatenous_updates = player is None
    if not simulatenous_updates and current_player != player:
      return state_value

    reach_prob = reach_probabilities[current_player]
    counterfactual_reach_prob = (
        np.prod(reach_probabilities[:current_player]) *
        np.prod(reach_probabilities[current_player + 1:]))
    state_value_for_player = state_value[current_player]

    for action, action_prob in info_state_policy.items():
      cfr_regret = counterfactual_reach_prob * (
          children_utilities[action][current_player] - state_value_for_player)

      info_state_node = self._info_state_nodes[info_state]
      info_state_node.cumulative_regret[action] += cfr_regret
      if self._linear_averaging:
        info_state_node.cumulative_policy[
            action] += self._iteration * reach_prob * action_prob
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value

  def _compute_policy_or_get_it_from_cache(self, info_state, legal_actions):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    retrieved_state = self._policy_cache.get(info_state)

    if retrieved_state is not None:
      return self._policy_cache[info_state]

    policy_for_state = _regret_matching(
        self._info_state_nodes[info_state].cumulative_regret, legal_actions)
    self._policy_cache[info_state] = policy_for_state
    return policy_for_state


def _regret_matching(cumulative_regrets, legal_actions):
  """Returns an info state policy by applying regret-matching.

  Args:
    cumulative_regrets: A {action: cumulative_regret} dictionary.
    legal_actions: the list of legal actions at this state.

  Returns:
    A dict of action -> prob for all legal actions.
  """
  regrets = cumulative_regrets.values()
  sum_positive_regrets = sum((regret for regret in regrets if regret > 0))

  info_state_policy = {}
  if sum_positive_regrets > 0:
    for action in legal_actions:
      positive_action_regret = max(0.0, cumulative_regrets[action])
      info_state_policy[action] = (
          positive_action_regret / sum_positive_regrets)
  else:
    for action in legal_actions:
      info_state_policy[action] = 1.0 / len(legal_actions)
  return info_state_policy


class CFRPlusSolver(_CFRSolver):
  """CFR+ implementation.

  The algorithm computes an approximate Nash policy for 2 player zero-sum games.
  More generally, it should approach a no-regret set, which corresponds to the
  set of coarse-correlated equilibria. See https://arxiv.org/abs/1305.0034

  CFR can be view as a policy iteration algorithm. Importantly, the policies
  themselves do not converge to a Nash policy, but their average does.

  See https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf

  CFR+ is CFR with the following modifications:
  - use Regret Matching+ instead of Regret Matching.
  - use alternating updates instead of simultaneous updates.
  - use linear averaging.

  Usage:

  ```python
      game = pyspiel.load_game("game_name")
      initial_state = game.new_initial_state()

      cfr_solver = CFRSolver(game)

      for i in range(num_iterations):
        cfr.evaluate_and_update_policy()
  ```

  Once the policy has converged, the average policy (which converges to the Nash
  policy) can be computed:
  ```python
        average_policy = cfr_solver.ComputeAveragePolicy()
  ```
  """

  def __init__(self, game):
    super(CFRPlusSolver, self).__init__(
        game,
        initialize_cumulative_values=False,
        regret_matching_plus=True,
        alternating_updates=True,
        linear_averaging=True)


class CFRSolver(_CFRSolver):
  """Implements the Counterfactual Regret Minimization (CFR) algorithm.

  See https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf

  NOTE: We use alternating updates (which was not the case in the original
  paper) because it has been proved to be far more efficient.
  """

  def __init__(self, game):
    super(CFRSolver, self).__init__(
        game,
        initialize_cumulative_values=False,
        regret_matching_plus=False,
        alternating_updates=True,
        linear_averaging=False)


class CFRBRSolver(object):
  """Implements the Counterfactual Regret Minimization (CFR-BR) algorithm.

  This is Counterfactual Regret Minimization against Best Response, from
  Michael Johanson and al., 2012, Finding Optimal Abstract Strategies in
  Extensive-Form Games,
  https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf).

  The algorithm
  computes an approximate Nash policy for n-player zero-sum games, but the
  implementation is currently restricted to 2-player.

  It uses an exact Best Response and full tree traversal.

  One iteration for a n-player game consist of the following:

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

  def __init__(self,
               game,
               initialize_cumulative_values=False,
               linear_averaging=True,
               regret_matching_plus=True):
    # pyformat: disable
    """Initializer.

    Args:
      game: The `pyspiel.Game` to run on.
      initialize_cumulative_values: Whether to initialize the average policy to
        the uniform policy (and the initial cumulative regret to an epsilon
        value). This is independent of the first CFR iteration, which, when the
        policy is fixed during traversal and we perform non alternating updates,
        will also compute the uniform policy and add it to the average of
        policies.
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

    self._game = game
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()

    if initialize_cumulative_values:
      initial_positive_value = _INITIAL_POSITIVE_VALUE
    else:
      initial_positive_value = 0
    self._info_state_nodes = {}
    _initialize_info_state_nodes(
        self._root_node,
        info_state_nodes=self._info_state_nodes,
        initial_positive_value=initial_positive_value)
    self._policy_cache = {}

    self._root_node = self._game.new_initial_state()

    # This is for returning the current policy and average policy to a caller
    self._current_policy = policy.TabularPolicy(game)
    self._average_policy = self._current_policy.__copy__()

    self._linear_averaging = linear_averaging
    self._iteration = 0  # For possible linear-averaging.

    self._regret_matching_plus = regret_matching_plus

    self._best_responses = {i: None for i in range(game.num_players())}

  def _compute_best_responses(self):
    """Computes each player best-response against the pool of other players."""
    # pylint: disable=g-long-lambda
    current_policy = policy.PolicyFromCallable(
        self._game, lambda state: self._compute_policy_or_get_it_from_cache(
            state.information_state(), state.legal_actions()))
    # pylint: disable=g-long-lambda

    for player_id in range(self._game.num_players()):
      self._best_responses[player_id] = exploitability.best_response(
          self._game, current_policy, player_id)

  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    self._iteration += 1
    num_players = self._game.num_players()

    self._compute_best_responses()

    for player in range(num_players):
      # We do not use policies, to not have to call `state.information_state`
      # several times (in here and within policy).
      policies = []
      for p in range(num_players):
        # pylint: disable=g-long-lambda
        policies.append(
            lambda infostate_str, legal_actions, p=p:
            {self._best_responses[p]["best_response_action"][infostate_str]: 1})
        # pylint: enable=g-long-lambda
      policies[player] = self._compute_policy_or_get_it_from_cache

      self._compute_counterfactual_regret_for_player(
          self._root_node,
          policies,
          reach_probabilities=np.ones(num_players + 1),
          player=player)

      if self._regret_matching_plus:
        _apply_regret_matching_plus_reset(self._info_state_nodes)
    self._policy_cache.clear()

  def policy(self):
    """Returns the current policy as a `policy.TabularPolicy` object."""
    _update_current_policy(self._current_policy, self._info_state_nodes)
    return self._current_policy

  def average_policy(self):
    """Returns the time average of all policies iterated.

    This average policy converges to a Nash policy as the number of iterations
    increases.

    The policy is computed using the accumulated policy probabilities computed
    using `evaluate_and_update_policy`.

    Returns:
      A `policy.TabularPolicy` object, giving the policy for both players.
    """
    _update_average_policy(self._average_policy, self._info_state_nodes)
    return self._average_policy

  def _compute_counterfactual_regret_for_player(self, state, policies,
                                                reach_probabilities, player):
    """Increments the cumulative regrets and policy for `player`.

    Args:
      state: The initial game state to analyze from.
      policies: A list of `num_players` callables taking as input
        `info_state_str` and `legal_actions` and returning a {action: prob}
          dictionary.
      reach_probabilities: The probability for each player of reaching `state`
        as a numpy array [prob for player 0, for player 1,..., for chance].
        `player_reach_probabilities[player]` will work in all cases.
      player: The 0-indexed player to update the values for. If `None`, the
        update for all players will be performed.

    Returns:
      The utility of `state` for all players, assuming all players follow the
      current policy defined by `self.Policy`.
    """
    if state.is_terminal():
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob
        state_value += (
            action_prob * self._compute_counterfactual_regret_for_player(
                new_state, policies, new_reach_probabilities, player))
      return state_value

    current_player = state.current_player()
    info_state = state.information_state(current_player)
    legal_actions = state.legal_actions(current_player)

    # No need to continue on this history branch as no update will be performed
    # for any player.
    # The value we return here is not used in practice. If the conditional
    # statement is True, then the last taken action has probability 0 of
    # occurring, so the returned value is not impacting the parent node value.
    if all(reach_probabilities[:-1] == 0):
      return np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    # The utilities of the children states are computed recursively. As the
    # regrets are added to the information state regrets for each state in that
    # information state, the recursive call can only be made once per child
    # state. Therefore, the utilities are cached.
    children_utilities = {}

    info_state_policy = policies[current_player](info_state, legal_actions)
    for action, action_prob in info_state_policy.items():
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      child_utility = self._compute_counterfactual_regret_for_player(
          new_state,
          policies,
          reach_probabilities=new_reach_probabilities,
          player=player)

      state_value += action_prob * child_utility
      children_utilities[action] = child_utility

    # If we are performing alternating updates, and the current player is not
    # the current_player, we skip the cumulative values update.
    # If we are performing simultaneous updates, we do update the cumulative
    # values.
    simulatenous_updates = player is None
    if not simulatenous_updates and current_player != player:
      return state_value

    reach_prob = reach_probabilities[current_player]
    counterfactual_reach_prob = (
        np.prod(reach_probabilities[:current_player]) *
        np.prod(reach_probabilities[current_player + 1:]))
    state_value_for_player = state_value[current_player]

    for action, action_prob in info_state_policy.items():
      cfr_regret = counterfactual_reach_prob * (
          children_utilities[action][current_player] - state_value_for_player)

      info_state_node = self._info_state_nodes[info_state]
      info_state_node.cumulative_regret[action] += cfr_regret
      if self._linear_averaging:
        info_state_node.cumulative_policy[
            action] += self._iteration * reach_prob * action_prob
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value

  def _compute_policy_or_get_it_from_cache(self, info_state, legal_actions):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    retrieved_state = self._policy_cache.get(info_state)

    if retrieved_state is not None:
      return self._policy_cache[info_state]

    policy_for_state = _regret_matching(
        self._info_state_nodes[info_state].cumulative_regret, legal_actions)
    self._policy_cache[info_state] = policy_for_state
    return policy_for_state
