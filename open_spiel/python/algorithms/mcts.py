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

"""Monte-Carlo Tree Search algorithm for game play."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np

import pyspiel


class Evaluator(object):
  """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game for player 0.
  """

  def evaluate(self, state):
    """Returns evaluation on given state."""
    raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
  """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

  def __init__(self, n_rollouts):
    self.n_rollouts = n_rollouts

  def evaluate(self, state):
    """Returns evaluation on given state."""
    result = 0.0
    for _ in range(self.n_rollouts):
      working_state = state.clone()
      while not working_state.is_terminal():
        if working_state.is_chance_node():
          outcomes = working_state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = np.random.choice(action_list, p=prob_list)
          working_state.apply_action(action)
        else:
          action = random.choice(working_state.legal_actions())
          working_state.apply_action(action)
      result += working_state.player_return(0)

    return result / self.n_rollouts


class SearchNode(object):
  """A node in the search tree."""
  __slots__ = [
      "explore_count", "player_sign", "total_reward", "actions", "children"
  ]

  def __init__(self):
    self.explore_count = 0
    self.player_sign = 0
    self.total_reward = 0.0

    self.actions = []
    self.children = []

  def child_value(self, child_index, uct_c):
    """Returns the UCT value of child."""
    child = self.children[child_index]
    if child.explore_count == 0:
      return float("inf")

    return (
        self.player_sign * child.total_reward / child.explore_count +
        uct_c * math.sqrt(math.log(self.explore_count) / child.explore_count))

  def most_visited_action(self):
    """Returns the most visited action from this node."""
    return max((child.explore_count, a)
               for child, a in zip(self.children, self.actions))[1]


def _apply_tree_policy(root, state, uct_c):
  """Applies the UCT policy to play the game until reaching a new node."""
  visit_path = [root]
  working_state = state.clone()
  current_node = root
  while not working_state.is_terminal():
    if current_node.explore_count == 0:
      # For a new node, initialize its state, then return
      for action in working_state.legal_actions():
        current_node.actions.append(action)
        current_node.children.append(SearchNode())
      current_node.player_sign = 1 if working_state.current_player(
      ) == 0 else -1
      return visit_path, working_state

    if working_state.is_chance_node():
      # For chance nodes, rollout according to chance node's probability
      # distribution
      rand = random.random()
      cumulative_sum = 0
      for max_index, (_, prob) in enumerate(working_state.chance_outcomes()):
        cumulative_sum += prob
        if cumulative_sum > rand:
          break
    else:
      # Otherwise choose node with largest UCT value
      max_index = max((current_node.child_value(i, uct_c), i)
                      for i, _ in enumerate(current_node.actions))[1]

    working_state.apply_action(current_node.actions[max_index])
    current_node = current_node.children[max_index]
    visit_path.append(current_node)

  return visit_path, working_state


def mcts_search(state, uct_c, max_search_nodes, evaluator):
  """A vanilla Monte-Carlo Tree Search algorithm.

  This algorithm searches the game tree from the given state.
  At the leaf, the evaluator is called if the game state is not terminal.
  A total of max_search_nodes states are explored.

  At every node, the algorithm chooses the action with the highest UCT value,
  defined as: Q/N + c * sqrt(log(N) / N), where Q is the total reward after the
  action, and N is the number of times the action was explored in this
  position.  The input parameter c controls the balance between exploration and
  exploitation; higher values of c encourage exploration of under-explored
  nodes. Unseen actions are always explored first.

  At the end of the search, the chosen action is the action that has been
  explored most often. This is the action that is returned.

  This implementation only supports sequential 1-player or 2-player zero-sum
  games, with or without chance nodes.

  Arguments:
    state: pyspiel.State object, state to search from
    uct_c: the c value in the UCT algorithm.
    max_search_nodes: the maximum number of nodes to search
    evaluator: Evaluator object

  Returns:
    The most visited move from the root node.
  """
  root = SearchNode()
  for _ in range(max_search_nodes):
    visit_path, working_state = _apply_tree_policy(root, state, uct_c)
    if working_state.is_terminal():
      node_value = working_state.player_return(0)
    else:
      node_value = evaluator.evaluate(working_state)

    for node in visit_path:
      node.total_reward += node_value
      node.explore_count += 1

  return root.most_visited_action()


class MCTSBot(pyspiel.Bot):
  """Bot that uses Monte-Carlo Tree Search algorithm."""

  def __init__(self, game, player, uct_c, max_search_nodes, evaluator):
    super(MCTSBot, self).__init__(game, player)
    self.uct_c = uct_c
    self.max_search_nodes = max_search_nodes
    self.evaluator = evaluator

  def step(self, state):
    """Returns bot's policy and action at given state."""
    policy = []
    legal_actions = state.legal_actions(self.player_id())
    mcts_action = mcts_search(state, self.uct_c, self.max_search_nodes,
                              self.evaluator)
    for action in legal_actions:
      p = 1.0 if action == mcts_action else 0.0
      policy.append((action, p))

    return policy, mcts_action
