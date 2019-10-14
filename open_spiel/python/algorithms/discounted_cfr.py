"""Discounted CFR and Linear CFR algorithms.

This implements Discounted CFR and Linear CFR, from Noam Brown and Tuomas
Sandholm, 2019, "Solving Imperfect-Information Games via Discounted Regret
Minimization".
See https://arxiv.org/abs/1809.04040.

Linear CFR (LCFR), is identical to CFR, except on iteration `t` the updates to
the regrets and average strategies are given weight `t`.

Discounted CFR(alpha, beta, gamma) is defined by, at iteration `t`:
- multiplying the positive accumulated regrets by 1(t^alpha / (t^aplha + 1))
- multiplying the negative accumulated regrets by 1(t^beta / (t^beta + 1))
- multiplying the contribution to the average strategt by (t / (t + 1))^gamma

WARNING: This was contributed on Github, and the OpenSpiel team is not aware it
has been verified we can reproduce the paper results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from open_spiel.python.algorithms import cfr


class _DCFRSolver(cfr._CFRSolver):  # pylint: disable=protected-access
  """Discounted CFR."""

  def __init__(self, game, alternating_updates, linear_averaging,
               regret_matching_plus, alpha, beta, gamma):
    super(_DCFRSolver, self).__init__(game, alternating_updates,
                                      linear_averaging, regret_matching_plus)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def _compute_counterfactual_regret_for_player(self, state, policies,
                                                reach_probabilities, player):
    """Increments the cumulative regrets and policy for `player`.

    Args:
      state: The initial game state to analyze from.
      policies: Unused. To be compatible with the `_CFRSolver` signature.
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
            new_state, policies, new_reach_probabilities, player)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state(current_player)

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

    info_state_node = self._info_state_nodes[info_state]
    if policies is None:
      info_state_policy = self._get_infostate_policy(info_state)
    else:
      info_state_policy = policies[current_player](info_state)
    for action in state.legal_actions():
      action_prob = info_state_policy.get(action, 0.)
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      child_utility = self._compute_counterfactual_regret_for_player(
          new_state,
          policies=policies,
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
        info_state_node.cumulative_policy[action] += (reach_prob * action_prob * (
                self._iteration **self.gamma))
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value


  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    #print("run")
    self._iteration += 1
    if self._alternating_updates:
      for player in range(self._game.num_players()):
        self._compute_counterfactual_regret_for_player(
            self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            player=player)
        for info_key, info_state in self._info_state_nodes.items():
            # 16 is the index of the current player for the info set, This is used to do update for the current player
           if int(info_key[16]) == player:
               for action in info_state.cumulative_regret.keys():
                   if info_state.cumulative_regret[action] >= 0:
                       info_state.cumulative_regret[action] *= (self._iteration**self.alpha / (self._iteration**self.alpha + 1))
                   else:
                       info_state.cumulative_regret[action] *= (self._iteration**self.beta / (self._iteration**self.beta + 1))
        cfr._update_current_policy(self._current_policy, self._info_state_nodes)



class DCFRSolver(_DCFRSolver):

  def __init__(self, game, alpha=3 / 2, beta=0, gamma=2):
    super(DCFRSolver, self).__init__(
        game,
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
        regret_matching_plus=False,
        alternating_updates=True,
        linear_averaging=True,
        alpha=1,
        beta=1,
        gamma=1)