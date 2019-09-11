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
  the game for the specified player.
  """

  def evaluate(self, state, player):
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

  def evaluate(self, state, player):
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
      result += working_state.player_return(player)

    return result / self.n_rollouts


class SearchNode(object):
  """A node in the search tree.

  A SearchNode represents a state and possible continuations from it. Each child
  represents a possible action, and the expected result from doing so.

  Attributes:
    action: The action from the parent node's perspective. Not important for
      the root node, as the actions that lead to it are in the past.
    player_sign: +1 for moves by the root player or chance nodes,
      -1 for the opponent. Needed to update the `total_reward` to reflect
      the value of this move from the parent node's perspective.
    explore_count: How many times this node was explored.
    total_reward: The sum of rewards of rollouts through this node, from the
      parent node's perspective. The average reward of this node is
      `total_reward / explore_count`
    children: A list of SearchNodes representing the possible actions from
      this node, along with their expected rewards.
  """
  __slots__ = [
      "action", "player_sign", "explore_count", "total_reward", "children"]

  def __init__(self, action, player_sign):
    self.action = action
    self.player_sign = player_sign
    self.explore_count = 0
    self.total_reward = 0.0
    self.children = []

  def uct_value(self, parent_explore_count, uct_c, child_default_value):
    """Returns the UCT value of child."""
    if self.explore_count == 0:
      return child_default_value

    return self.total_reward / self.explore_count + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.explore_count)

  def most_visited_child(self):
    """Returns the most visited action from this node."""
    return max(self.children, key=lambda c: c.explore_count)

  def children_str(self, state=None):
    """Returns the string repr. of all children, in decreasing explore count.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    return "".join([
        "  {}\n".format(c.to_str(state))
        for c in sorted(self.children, key=lambda c: -c.explore_count)])

  def to_str(self, state=None):
    """Returns the string repr. of this node children.

    Looks like: "d4h: sign: 1, value:  244.0 / 2017 =  0.121,  20 children"

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    action = (state.action_to_string(state.current_player(), self.action)
              if state else str(self.action))
    return "{:>3}: sign: {}, value: {:6.1f} / {:4d} = {:6.3f}, {:3d} children".format(
        action, self.player_sign, self.total_reward, self.explore_count,
        self.total_reward / self.explore_count, len(self.children))

  def __str__(self):
    return self.to_str(None)


class MCTSBot(pyspiel.Bot):
  """Bot that uses Monte-Carlo Tree Search algorithm."""

  def __init__(self, game, player, uct_c, max_simulations, evaluator,
               verbose=False):
    """Initializes a MCTS Search algorithm in the form of a bot.

    Args:
      game: A pyspiel.Game to play.
      player: 0 or 1, for first or second player.
      uct_c: The exploration constant for UCT.
      max_simulations: How many iterations of MCTS to perform. Each simulation
        will result in one call to the evaluator. Memory usage should grow
        linearly with simulations * branching factor.
      How many nodes in the search tree should be evaluated.
        This is correlated with memory size and tree depth.
      evaluator: A `Evaluator` object to use to evaluate a leaf node.
      verbose: Whether to print information about the search tree before
        returning the action. Useful for confirming the search is working
        sensibly.
    """
    super(MCTSBot, self).__init__(game, player)
    self.uct_c = uct_c
    self.max_simulations = max_simulations
    self.evaluator = evaluator
    self.child_default_value = float("inf")
    self.player = player
    self.verbose = verbose

  def step(self, state):
    """Returns bot's policy and action at given state."""
    mcts_action = self.mcts_search(state)
    policy = [(action, (1.0 if action == mcts_action else 0.0))
              for action in state.legal_actions(self.player_id())]

    return policy, mcts_action

  def _apply_tree_policy(self, root, state):
    """Applies the UCT policy to play the game until reaching a leaf node.

    A leaf node is defined as a node that is terminal or has not been evaluated
    yet. If it reaches a node that has been evaluated before but hasn't been
    expanded, then expand it's children and continue.

    Args:
      root: The root node in the search tree.
      state: The state of the game at the root node.

    Returns:
      visit_path: A list of nodes descending from the root node to a leaf node.
      working_state: The state of the game at the leaf node.
    """
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    while not working_state.is_terminal() and current_node.explore_count > 0:
      if not current_node.children:
        # For a new node, initialize its state, then choose a child as normal.
        legal_actions = working_state.legal_actions()
        random.shuffle(legal_actions)  # Reduce bias from move generation order.
        player_sign = -1 if working_state.current_player() != self.player else 1
        current_node.children = [SearchNode(action, player_sign)
                                 for action in legal_actions]

      if working_state.is_chance_node():
        # For chance nodes, rollout according to chance node's probability
        # distribution
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        chosen_child = next(c for c in current_node.children
                            if c.action == action)
      else:
        # Otherwise choose node with largest UCT value
        chosen_child = max(
            current_node.children,
            key=lambda c: c.uct_value(current_node.explore_count, self.uct_c,  # pylint: disable=g-long-lambda
                                      self.child_default_value))

      working_state.apply_action(chosen_child.action)
      current_node = chosen_child
      visit_path.append(current_node)

    return visit_path, working_state

  def mcts_search(self, state):
    """A vanilla Monte-Carlo Tree Search algorithm.

    This algorithm searches the game tree from the given state.
    At the leaf, the evaluator is called if the game state is not terminal.
    A total of max_simulations states are explored.

    At every node, the algorithm chooses the action with the highest UCT value,
    defined as: Q/N + c * sqrt(log(N) / N), where Q is the total reward after
    the action, and N is the number of times the action was explored in this
    position.  The input parameter c controls the balance between exploration
    and exploitation; higher values of c encourage exploration of under-explored
    nodes. Unseen actions are always explored first.

    At the end of the search, the chosen action is the action that has been
    explored most often. This is the action that is returned.

    This implementation only supports sequential 1-player or 2-player zero-sum
    games, with or without chance nodes.

    Arguments:
      state: pyspiel.State object, state to search from

    Returns:
      The most visited move from the root node.
    """
    assert state.current_player() == self.player
    root = SearchNode(None, 1)
    for _ in range(self.max_simulations):
      visit_path, working_state = self._apply_tree_policy(root, state)
      if working_state.is_terminal():
        node_value = working_state.player_return(self.player)
      else:
        node_value = self.evaluator.evaluate(working_state, self.player)

      for node in visit_path:
        node.total_reward += node_value * node.player_sign
        node.explore_count += 1

    most_visited = root.most_visited_child()

    if self.verbose:
      print("Root:", root.to_str())
      print("Children:")
      print(root.children_str(working_state))
      print("Children of chosen:")
      chosen_state = state.clone()
      chosen_state.apply_action(most_visited.action)
      print(most_visited.children_str(chosen_state))

    return most_visited.action
