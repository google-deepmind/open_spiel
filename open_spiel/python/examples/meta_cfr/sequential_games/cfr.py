# Copyright 2022 DeepMind Technologies Limited
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

"""Counterfactual Regret Minimization."""

import copy
import enum
from typing import List, Tuple

from open_spiel.python.examples.meta_cfr.sequential_games.typing import GameTree
from open_spiel.python.examples.meta_cfr.sequential_games.typing import HistoryNode
from open_spiel.python.examples.meta_cfr.sequential_games.typing import InfostateMapping
from open_spiel.python.examples.meta_cfr.sequential_games.typing import InfostateNode


class Players(enum.IntEnum):
  CHANCE_PLAYER = 0
  PLAYER_1 = 1
  PLAYER_2 = 2


def compute_reach_probabilities(
    history_tree_node: HistoryNode,
    all_infostates_map: List[InfostateMapping]) -> None:
  """Computes reach probabilities for game tree information states.

  This function initializes counterfactual_reach_prob and player_reach_prob for
  all information states in the game tree, and then these values will be
  calculated in compute_reach_probability_dfs.

  Args:
    history_tree_node: Game tree HistoryTreeNode which is the root of the game
      tree.
    all_infostates_map: List of dictionaries (mapping from information state
      string representation to information state object) for each players
      (including chance player). This list will be empty when this function is
      called fot the first time and it'll be population during DFS tree
      traversal.
  """

  for infostate in (list(all_infostates_map[Players.PLAYER_1].values()) +
                    list(all_infostates_map[Players.PLAYER_2].values())):
    infostate.counterfactual_reach_prob = 0.
    infostate.player_reach_prob = 0.
  compute_reach_probability_dfs(history_tree_node, all_infostates_map)


def compute_reach_probability_dfs(
    history_tree_node: HistoryNode,
    all_infostates_map: List[InfostateMapping]) -> None:
  """Calculate reach probability values in dfs tree.

  This function is initially called by compute_reach_probabilities and it
  computes reach probabilities for all information state nodes in the tree by
  traversing the tree using DFS.

  Args:
    history_tree_node: Game tree HistoryTreeNode which is the root of the game
      tree.
    all_infostates_map: List of dictionaries (mapping from information state
      string representation to information state object) for each players
      (including chance player). This list will be empty when this function is
      called fot the first time and it'll be population during DFS tree
      traversal.
  """

  world_state = history_tree_node.world_state
  infostate_p1 = all_infostates_map[Players.PLAYER_1][
      world_state.get_infostate_string(Players.PLAYER_1)]
  infostate_p2 = all_infostates_map[Players.PLAYER_2][
      world_state.get_infostate_string(Players.PLAYER_2)]
  infostate_p1.counterfactual_reach_prob += history_tree_node.reach_probs[
      0] * history_tree_node.reach_probs[Players.PLAYER_2]
  infostate_p2.counterfactual_reach_prob += history_tree_node.reach_probs[
      0] * history_tree_node.reach_probs[Players.PLAYER_1]

  if infostate_p1.player_reach_prob != 0.:
    assert (infostate_p1.player_reach_prob == history_tree_node.reach_probs[
        Players.PLAYER_1])

  if infostate_p2.player_reach_prob != 0.:
    assert (infostate_p2.player_reach_prob == history_tree_node.reach_probs[
        Players.PLAYER_2])

  infostate_p1.player_reach_prob = history_tree_node.reach_probs[
      Players.PLAYER_1]
  infostate_p2.player_reach_prob = history_tree_node.reach_probs[
      Players.PLAYER_2]

  policy_p1 = infostate_p1.policy
  policy_p2 = infostate_p2.policy
  policy_chance = world_state.chance_policy
  actions_chance, actions_p1, actions_p2 = world_state.get_actions()
  for action_chance in actions_chance:
    for action_p1 in actions_p1:
      for action_p2 in actions_p2:
        history_tree_node.action_probs[(
            action_chance, action_p1, action_p2)] = policy_chance[
                action_chance] * policy_p1[action_p1] * policy_p2[action_p2]
        child_node = history_tree_node.get_child(
            (action_chance, action_p1, action_p2))
        child_node.reach_probs[
            Players.CHANCE_PLAYER] = history_tree_node.reach_probs[
                Players.CHANCE_PLAYER] * policy_chance[action_chance]
        child_node.reach_probs[
            Players.PLAYER_1] = history_tree_node.reach_probs[
                Players.PLAYER_1] * policy_p1[action_p1]
        child_node.reach_probs[
            Players.PLAYER_2] = history_tree_node.reach_probs[
                Players.PLAYER_2] * policy_p2[action_p2]
        compute_reach_probability_dfs(child_node, all_infostates_map)


def _get_opponent(player: int) -> int:
  return -1 * player + 3


def compute_best_response_values(infostate: InfostateNode) -> float:
  """Returns best response value for an infostate.

  Args:
    infostate: Information state.

  Returns:
    Best response value, which is the maximum action value chosen among all
    actions values of possible actions from infostate. If information state is a
    terminal node in the game tree, this value is calculated from history nodes
    reach probability for player and opponent, and game utility of terminal
    node. If infostate is not terminal, this value will be calculated in a
    recursive way.
  """
  if infostate.is_terminal():
    terminal_utility = 0
    for history_node in infostate.history_nodes:
      terminal_utility += history_node.reach_probs[
          0] * history_node.reach_probs[_get_opponent(
              infostate.player)] * history_node.world_state.get_utility(
                  infostate.player)
    return terminal_utility
  action_values = {action: 0 for action in infostate.get_actions()}
  infostate_actions = infostate.get_actions()
  for action in infostate_actions:
    action_values[action] = 0
    for child in infostate.children[action].values():
      action_values[action] += compute_best_response_values(child)
  return max(action_values.values())


def compute_best_response_policy(infostate: InfostateNode) -> float:
  """Calculate best response policy and returns best response value of infostate.

  Args:
    infostate: Information state.

  Returns:
    Best response value similar to what compute_best_response_values returns.
  """
  if infostate.is_terminal():
    terminal_utility = 0
    for history_node in infostate.history_nodes:
      terminal_utility += history_node.reach_probs[
          0] * history_node.reach_probs[_get_opponent(
              infostate.player)] * history_node.world_state.get_utility(
                  infostate.player)
    return terminal_utility
  action_values = {action: 0 for action in infostate.get_actions()}
  infostate_actions = infostate.get_actions()
  for action in infostate_actions:
    action_values[action] = 0
    for child in infostate.children[action].values():
      action_values[action] += compute_best_response_policy(child)

  infostate.policy = {action: 0 for action in infostate.get_actions()}
  max_action_value = max(action_values.values())
  for action in infostate_actions:
    if action_values[action] == max_action_value:
      infostate.policy[action] = 1
      break
  return max_action_value


def compute_counterfactual_values(infostate: InfostateNode) -> float:
  """Returns cfr value for an infostate.

  Args:
    infostate: Information state.

  Returns:
    Counterfactual value for infostate. This value is calculated from action
    value and policy of all legal actions of infostate information state.
  """
  if infostate.is_terminal():
    terminal_utility = 0
    for history_node in infostate.history_nodes:
      terminal_utility += history_node.reach_probs[
          0] * history_node.reach_probs[_get_opponent(
              infostate.player)] * history_node.world_state.get_utility(
                  infostate.player)
    return terminal_utility
  infostate_actions = infostate.get_actions()
  action_values = {action: 0 for action in infostate_actions}
  for action in infostate_actions:
    for child in infostate.children[action].values():
      action_values[action] += compute_counterfactual_values(child)
  infostate.counterfactual_action_values = action_values
  counterfactual_value = 0
  for action in infostate_actions:
    counterfactual_value += infostate.policy[action] * action_values[action]
  infostate.counterfactual_value = counterfactual_value
  return counterfactual_value


def update_regrets(infostates: List[InfostateNode]) -> None:
  """Updates regret value for each infostate in infostates.

  Args:
    infostates: List of information states
  """
  for infostate in infostates:
    for action in infostate.get_actions():
      current_regret = infostate.counterfactual_action_values[
          action] - infostate.counterfactual_value
      infostate.regret[action] += current_regret


def compute_next_policy(infostates: List[InfostateNode],
                        cfr_plus: bool = False) -> None:
  """Computes policy of next iteration for each infostate in infostates.

  Args:
    infostates: List of information states.
    cfr_plus: A flag which specifies if we update policy according to CFR or
      CFR-plus algorithm. True if we use CFR-plus, otherwise we use CFR.
  """
  for infostate in infostates:
    infostate_actions = infostate.get_actions()
    if cfr_plus:
      for action in infostate_actions:
        infostate.regret[action] = max(infostate.regret[action], 0.0)

    positive_regret_sum = 0
    for action in infostate_actions:
      if infostate.regret[action] > 0:
        positive_regret_sum += infostate.regret[action]

    actions_count = len(infostate_actions)
    next_policy = {a: 1.0 / actions_count for a in infostate_actions}

    if positive_regret_sum > 0:
      for action in infostate_actions:
        next_policy[action] = max(infostate.regret[action],
                                  0) / positive_regret_sum
    infostate.policy = next_policy


def cumulate_average_policy(infostates: List[InfostateNode],
                            weight: int = 1) -> None:
  """Cumulates policy values of each infostate in infostates.

  For each infostate, we update average policy and the sum of weighted average
  policy.

  Args:
    infostates: List of information states.
    weight: The weight we use to update policy and sum of weighted average
      policy. For CFR algorithm, weight is 1.
  """
  for infostate in infostates:
    for action in infostate.get_actions():
      infostate.average_policy[
          action] += infostate.player_reach_prob * infostate.policy[
              action] * weight
    infostate.average_policy_weight_sum += infostate.player_reach_prob * weight


def normalize_average_policy(infostates) -> None:
  """Updates infostate policy by normalizing average policy.

  Args:
    infostates: List of information states that their policies will be updated.
  """
  for infostate in infostates:
    for action in infostate.get_actions():
      infostate.policy[action] = infostate.average_policy[
          action] / infostate.average_policy_weight_sum


def best_response_counterfactual_regret_minimization_iteration(
    history_tree_node: HistoryNode,
    infostate_nodes: List[InfostateNode],
    all_infostates_map: List[InfostateMapping]) -> None:
  """Calculates CFRBR values.

  Args:
    history_tree_node: Game tree HistoryTreeNode which is the root of the game
      tree.
    infostate_nodes: List of all information state nodes.
    all_infostates_map: List of dictionaries (mapping from information state
      string representation to information state object) for each players
      (including chance player). This list will be empty when this function is
      called fot the first time and it'll be population during DFS tree
      traversal.
  """
  compute_next_policy(list(all_infostates_map[Players.PLAYER_1].values()))

  compute_reach_probabilities(history_tree_node, all_infostates_map)
  cumulate_average_policy(list(all_infostates_map[Players.PLAYER_1].values()))

  compute_best_response_policy(infostate_nodes[Players.PLAYER_2])
  compute_reach_probabilities(history_tree_node, all_infostates_map)
  compute_counterfactual_values(infostate_nodes[Players.PLAYER_1])

  update_regrets(list(all_infostates_map[Players.PLAYER_1].values()))


def counterfactual_regret_minimization_iteration(
    cfr_game_tree: GameTree,
    alternating_updates: bool,
    cfr_plus: bool,
    weight: int = 1) -> None:
  """Performs one iteration of CFR or CFR-plus.

  Args:
    cfr_game_tree: Game tree for an imperfect information game. This game tree
      is game tree of an openspiel game.
    alternating_updates: Boolean flag to do alternative update for players
      policies or not. If True, alternative updates will be performed (meaning
      we first calculate average policy, counterfactual values, regrets and next
      policy for player 1 first and then calculate all of these for player 2),
      otherwise both players average policies, counterfactual values and regrets
      will be updated right after each other (meaning, for example we calculate
      next_policy of player 1, and then next policy of player 2. Then, we
      calculate average policy for player 1 and then average policy for player
      2, and so on).
    cfr_plus: Boolean flag indicating if we perform CFR algorithm or CFR-plus.
      If True, we perform CFR-plus algorithm, otherwise we perform CFR
      algorithm.
    weight: The weight we use to update policy and sum of weighted average
      policy.
  """
  if alternating_updates:
    compute_reach_probabilities(cfr_game_tree.first_history_node,
                                cfr_game_tree.all_infostates_map)
    cumulate_average_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()),
        weight)
    compute_counterfactual_values(
        cfr_game_tree.infostate_nodes[Players.PLAYER_1])
    update_regrets(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()))
    compute_next_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()),
        cfr_plus)

    compute_reach_probabilities(cfr_game_tree.first_history_node,
                                cfr_game_tree.all_infostates_map)
    cumulate_average_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()),
        weight)
    compute_counterfactual_values(
        cfr_game_tree.infostate_nodes[Players.PLAYER_2])
    update_regrets(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()))
    compute_next_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()),
        cfr_plus)
  else:
    compute_next_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()),
        cfr_plus)
    compute_next_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()),
        cfr_plus)

    compute_reach_probabilities(cfr_game_tree.first_history_node,
                                cfr_game_tree.all_infostates_map)
    cumulate_average_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()),
        weight)
    cumulate_average_policy(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()),
        weight)

    compute_counterfactual_values(
        cfr_game_tree.infostate_nodes[Players.PLAYER_1])
    compute_counterfactual_values(
        cfr_game_tree.infostate_nodes[Players.PLAYER_2])

    update_regrets(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_1].values()))
    update_regrets(
        list(cfr_game_tree.all_infostates_map[Players.PLAYER_2].values()))


def compute_cfr_plus_values(cfr_game_tree: GameTree,
                            steps: int) -> Tuple[List[float], List[float]]:
  """Performs CFR-plus algorithm for a given number of steps.

  Args:
    cfr_game_tree: Game tree for an imperfect information game. This game tree
      is game tree of an openspiel game.
    steps: Number of CFR-plus steps.

  Returns:
    best_response_values_p1: List of best response values for player 1. The
    length of this list is equal to the number of steps.
    best_response_values_p2: List of best response values for player 2. The
    length of this list is equal to the number of steps.
  """
  best_response_values_p1 = []
  best_response_values_p2 = []
  for i in range(steps):
    counterfactual_regret_minimization_iteration(
        cfr_game_tree=cfr_game_tree,
        alternating_updates=True,
        cfr_plus=True,
        weight=i + 1)

    game_tree_copy = copy.deepcopy(cfr_game_tree)
    normalize_average_policy(
        game_tree_copy.all_infostates_map[Players.PLAYER_1].values())
    normalize_average_policy(
        game_tree_copy.all_infostates_map[Players.PLAYER_2].values())
    compute_reach_probabilities(game_tree_copy.first_history_node,
                                game_tree_copy.all_infostates_map)

    best_response_values_p1.append(
        compute_best_response_values(
            game_tree_copy.infostate_nodes[Players.PLAYER_1]))
    best_response_values_p2.append(
        compute_best_response_values(
            game_tree_copy.infostate_nodes[Players.PLAYER_2]))

  return best_response_values_p1, best_response_values_p2


def compute_cfr_values(cfr_game_tree: GameTree,
                       steps: int) -> Tuple[List[float], List[float]]:
  """Performs CFR algorithm for a given number of steps.

  Args:
    cfr_game_tree: Game tree for an imperfect information game. This game tree
      is game tree of an openspiel game.
    steps: Number of CFR-plus steps.

  Returns:
    best_response_values_p1: List of best response values for player 1. The
    length of this list is equal to the number of steps.
    best_response_values_p2: List of best response values for player 2. The
    length of this list is equal to the number of steps.
  """
  best_response_values_p1 = []
  best_response_values_p2 = []
  for _ in range(steps):
    counterfactual_regret_minimization_iteration(
        cfr_game_tree=cfr_game_tree, alternating_updates=False, cfr_plus=False)

    normalize_average_policy(
        cfr_game_tree.all_infostates_map[Players.PLAYER_1].values())
    normalize_average_policy(
        cfr_game_tree.all_infostates_map[Players.PLAYER_2].values())
    compute_reach_probabilities(cfr_game_tree.first_history_node,
                                cfr_game_tree.all_infostates_map)
    best_response_values_p1.append(
        compute_best_response_values(
            cfr_game_tree.infostate_nodes[Players.PLAYER_1]))
    best_response_values_p2.append(
        compute_best_response_values(
            cfr_game_tree.infostate_nodes[Players.PLAYER_2]))

  return best_response_values_p1, best_response_values_p2
