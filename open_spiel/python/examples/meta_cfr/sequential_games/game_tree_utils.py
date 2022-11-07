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

"""Game tree structure for imperfect information games."""

import copy
from typing import Any, Dict, List, Text, Tuple

from open_spiel.python.examples.meta_cfr.sequential_games import cfr
from open_spiel.python.examples.meta_cfr.sequential_games import openspiel_api


class HistoryTreeNode:
  """Tree node to build game tree in cfr and do DFS traverse on game tree.

  Attributes:
    world_state: Current world state representation.
    reach_probs: Reach probability of tree node for each player. We consider
      reach probability for chance player, player 1 and player 2.
    action_probs: Probability of actions taken by each player. We consider
      actions taken by chance player, player 1 and player 2. Keys of this
      dictionary are tuples of (action_chance, action_player_1,
      action_player_2).
    children: A dictionary from a taken action from this node to the
      HistoryTreeNode of the child we derive in the game tree by taking an
      action.
  """

  def __init__(self, world_state: openspiel_api.WorldState):
    self.world_state = world_state
    self.reach_probs = [1.0, 1.0, 1.0]
    self.action_probs = {}
    self._value_p1 = 0
    self.children = {}

  def add_child(self, child_world_state: 'HistoryTreeNode',
                actions: Tuple[int, int, int]) -> None:
    """Adds the child world state to dictionary of children of this node."""
    self.children[actions] = child_world_state

  def get_child(self, actions: Tuple[int, int, int]) -> 'HistoryTreeNode':
    """Returns a child world state that can be derived from an action."""
    return self.children[actions]


class InfoState:
  """Information state class.

  Attributes:
    history_nodes: History of game as players play.
    player: Index of current player.
    infostate_string: String representation of current informantion state.
    world_state: Current game world state.
    children: Children nodes of information states. The keys are actions, and
      values are dictionary from information state string to information state
      node.
    counterfactual_reach_prob: Counterfactural values of reach probability for
      the current information state.
    player_reach_prob: Reach probability of information state for the acting
      player.
    counterfactual_action_values: Counterfactual values for each action in this
      information state. This is a dictionary from action to counterfactual
      value of this action in this information state.
    counterfactual_value: Counterfactual value of this information state.
    regret: Regret of each action for all player's actions in this information
      state.
    policy: Policy of player in this information state.
    average_policy: Average policy for all player's actions in this information
      state.
    average_policy_weight_sum: Sum of weighted average policy. This is used to
      normalize average policy and derive policy in this information state.
  """

  def __init__(self, world_state: openspiel_api.WorldState, player: int,
               infostate_string: Text):
    self.history_nodes = []
    self.player = player
    self.infostate_string = infostate_string
    self.world_state = world_state
    self._actions = world_state.get_actions()
    self.children = {a: {} for a in self._actions[player]}
    self.counterfactual_reach_prob = 0.
    self.player_reach_prob = 0.
    self.counterfactual_action_values = {}
    self.counterfactual_value = 0
    self.regret = {a: 0. for a in self._actions[player]}

    actions_count = len(self._actions[player])
    self.policy = {
        a: 1.0 / actions_count for a in world_state.get_actions()[player]
    }

    self.average_policy = {a: 0. for a in self._actions[player]}
    self.average_policy_weight_sum = 0.

  def add_history_node(self, history_node: HistoryTreeNode) -> None:
    """Updates history nodes with a given(last) history node."""
    self.history_nodes.append(history_node)

  def add_child_infostate(self, action: int,
                          infostate_child: Any) -> None:
    """Adds child infostate derived from taking an action to self.children."""
    self.children[action][infostate_child.infostate_string] = infostate_child

  def get_actions(self) -> List[int]:
    """Returns legal actions in current information state for current player."""
    return self.history_nodes[0].world_state.get_actions()[self.player]

  def is_terminal(self) -> bool:
    """Returns True if information state is terminal, False otherwise."""
    return self.history_nodes[0].world_state.is_terminal()


class GameTree:
  """Game tree class to build for CFR-based algorithms.

  Attributes:
    first_history_node: Root node of game tree.
    infostate_nodes: List of information state nodes for each player (including
      chance player).
    all_infostates_map: List of dictionaries (mapping from information state
      string representation to information state object) for each players
      (including chance player).
  """

  def __init__(self, first_history_node: HistoryTreeNode,
               infostate_nodes: List[InfoState],
               all_infostates_map: List[Dict[str, InfoState]]):
    self.first_history_node = first_history_node
    self.infostate_nodes = infostate_nodes
    self.all_infostates_map = all_infostates_map


def build_tree_dfs(
    world_state: openspiel_api.WorldState,
    all_infostates_map: List[Dict[str, InfoState]]
) -> Tuple[HistoryTreeNode, List[InfoState]]:
  """Builds the game tree by DFS traversal.

  Args:
    world_state: An openspiel game world state representation that will be the
      root of game tree.
    all_infostates_map: List of dictionaries (mapping from information state
      string representation to information state object) for each players
      (including chance player). This list will be empty when this function is
      called and it'll be population during DFS tree traversal.

  Returns:
    tree_node: Root of the game tree built in DFS traversal.
    infostate_nodes: List of information state (root) tree node for each player
    (including chance player).
  """
  tree_node = HistoryTreeNode(world_state)

  infostate_nodes = [
      InfoState(world_state, 1, world_state.get_infostate_string(1)),
      InfoState(world_state, 1, world_state.get_infostate_string(1)),
      InfoState(world_state, 2, world_state.get_infostate_string(2))
  ]
  for p in [cfr.Players.PLAYER_1, cfr.Players.PLAYER_2]:
    infostate_string = world_state.get_infostate_string(p)
    if infostate_string not in all_infostates_map[p]:
      all_infostates_map[p][infostate_string] = InfoState(
          world_state, p, infostate_string)

    infostate = all_infostates_map[p][infostate_string]
    infostate.add_history_node(tree_node)

    infostate_nodes[p] = infostate
  actions = world_state.get_actions()
  actions_chance, actions_p1, actions_p2 = actions

  for action_chance in actions_chance:
    for action_p1 in actions_p1:
      for action_p2 in actions_p2:
        child_state = copy.deepcopy(world_state)
        child_state.apply_actions((action_chance, action_p1, action_p2))
        child_tree_node, child_infostates = build_tree_dfs(
            child_state, all_infostates_map)

        tree_node.add_child(child_tree_node,
                            (action_chance, action_p1, action_p2))
        infostate_nodes[1].add_child_infostate(action_p1, child_infostates[1])
        infostate_nodes[2].add_child_infostate(action_p2, child_infostates[2])

  return tree_node, infostate_nodes


def build_game_tree(world_state: openspiel_api.WorldState) -> GameTree:
  """Builds game tree for CFR-based algorithms.

  Args:
    world_state: An openspiel game world state representation that will be the
      root of game tree.

  Returns:
    Calls GameTree function which returns the following:
    tree_node: Root of the game tree built in DFS traversal.
    infostate_nodes: List of information state (root) tree node for each player
    (including chance player).
  """
  all_infostates_map = [{}, {}, {}]
  first_history_node, infostate_nodes = build_tree_dfs(world_state,
                                                       all_infostates_map)
  return GameTree(first_history_node, infostate_nodes, all_infostates_map)
