from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from open_spiel.python.algorithms import cfr
import numpy as np

class _DCFRSolver(cfr._CFRSolver):
  def __init__(self, game, initialize_cumulative_values, alternating_updates,
               linear_averaging, regret_matching_plus, alpha, beta, gamma):
    super(_DCFRSolver,self).__init__(game, initialize_cumulative_values, alternating_updates,linear_averaging, regret_matching_plus)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

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
    # Multiplying accumulative positive and negative regret with different alpha and beta
      if info_state_node.cumulative_regret[action] >= 0:
        info_state_node.cumulative_regret[action] *= (self._iteration ** self.alpha / (self._iteration ** self.alpha + 1))
      else:
        info_state_node.cumulative_regret[action] *= (self._iteration ** self.beta / (self._iteration ** self.beta + 1))

      if self._linear_averaging:
        info_state_node.cumulative_policy[
          action] += reach_prob * action_prob
    # Applying different weights of contribution to average strategy
        info_state_node.cumulative_policy[
          action] *= ((self._iteration / (self._iteration + 1)) ** self.gamma)
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value

class DCFRSolver(_DCFRSolver):
  def __init__(self, game, alpha=3/2, beta=0, gamma=2):
    super(DCFRSolver, self).__init__(
      game,
      initialize_cumulative_values=True,
      regret_matching_plus=False,
      alternating_updates=True,
      linear_averaging=True,
      alpha=alpha,
      beta=beta,
      gamma=gamma)

class LCFRSolver(_DCFRSolver):
  def __init__(self, game):
    super(LCFRSolver, self).__init__(
        game,
        initialize_cumulative_values=True,
        regret_matching_plus=False,
        alternating_updates=True,
        linear_averaging=True,
        alpha=1,
        beta=1,
        gamma=1)

