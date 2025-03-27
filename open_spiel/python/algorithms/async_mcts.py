# Copyright 2024 DeepMind Technologies Limited
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

"""An asynchronous Monte Carlo Tree Search algorithm for game play.

This implements asynchronous MCTS which runs evaluations asynchronously in
parallel. It is a very simplified version of the algorithm described in:

Schultz et al. '24. Mastering Board Games by External and Internal Planning
with Language Models. https://arxiv.org/abs/2412.12119

Unlike the paper which describes a entirely model-based approach, this
implementation does use a game engine internally. Also, this version uses
standard virtual losses (not the dynamic virtual counts described in the paper).
However, it does implement the asynchronous calls to the evaluators in the same
way as the algorithm in the paper.
"""

from __future__ import annotations
import concurrent.futures
import math
import time
from typing import Callable, Optional

from absl import logging
import numpy as np

import pyspiel


def robust_child_with_total_reward_tiebreaker(
    root: SearchNode,
) -> tuple[int, SearchNode]:
  """Returns the best action and associated child node.

  The child node with the most visits is chosen.
  In case of a tie, the child with the highest total reward is chosen.


  Args:
    root: The root node of the search tree.

  Returns:
    A tuple containing the best action and the associated child node.
  """
  selection_criteria = lambda node: (node.explore_count, node.total_reward)
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def robust_child(root: SearchNode) -> tuple[int, SearchNode]:
  """Returns the best action and associated child node.

  A child node with the most visits is chosen.

  Args:
    root: The root node of the search tree.

  Returns:
    A tuple containing the best action and the associated child node.
  """
  selection_criteria = lambda node: node.explore_count
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_child(root: SearchNode) -> tuple[int, SearchNode]:
  """Returns the best action and associated child node.

  A child node with the highest expected reward is chosen.

  Args:
    root: The root node of the search tree.

  Returns:
    A tuple containing the best action and the associated child node.
  """
  selection_criteria = (
      lambda node: node.total_reward / node.explore_count
      if node.explore_count
      else float("-inf")
  )
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_robust_child(
    root: SearchNode, find_robust: bool = False
) -> tuple[Optional[int], Optional[SearchNode]]:
  """Returns the best action and associated child node.

  A child node with the highest expected reward and most visits is chosen.
  If no such child exists, increase the number of simulations.

  Args:
    root: The root node of the search tree.
    find_robust: Whether to find a robust child node. E.g., if max compute is
      reached and max robust is not found.

  Returns:
    A tuple containing the best action and the associated child node.
  """
  if find_robust:
    best_action, best_child = robust_child(root)
  else:
    _, max_child_node = max_child(root)
    _, robust_child_node = robust_child(root)
    best_action, best_child = None, None
    for child in root.children:
      if child == max_child_node and child == robust_child_node:
        best_action, best_child = child.action, child
        break
  return best_action, best_child


def secure_child(
    root: SearchNode, secure_c: float = 1.0
) -> tuple[int, SearchNode]:
  """Returns the best action and associated child node.

  A child node with the most visits is chosen.

  Args:
    root: The root node of the search tree.
    secure_c: The constant used to calculate lower uncertainty bound

  Returns:
    A tuple containing the best action and the associated child node.
  """
  selection_criteria = (
      lambda node: node.total_reward / node.explore_count  # pylint: disable=g-long-ternary
      - secure_c / math.sqrt(node.explore_count)
      if node.explore_count
      else float("-inf")
  )
  best_child = max(root.children, key=selection_criteria)
  return best_child.action, best_child


def max_robust_secure_child(
    root: SearchNode, secure_c: float = 1.0, find_secure: bool = False
) -> tuple[Optional[int], Optional[SearchNode]]:
  """Returns the best action and associated child node.

  A child node with the most visits is chosen.

  Args:
    root: The root node of the search tree.
    secure_c: The constant used to calculate lower uncertainty bound.
    find_secure: Whether to find a secure child node.

  Returns:
    A tuple containing the best action and the associated child node.
  """
  if find_secure:
    best_action, best_child = secure_child(root, secure_c)
  else:
    best_action, best_child = max_robust_child(root)
  return best_action, best_child


class Evaluator:
  """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game. It returns the evaluation from all player's perspectives.
  """

  def prior_and_value(
      self, state: pyspiel.State
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    """Returrns a prior (list of (action, prior)) and values (np.array)."""
    raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
  """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

  def __init__(
      self,
      random_state: np.random.RandomState | None = None,
  ):
    self._random_state = random_state or np.random.RandomState()

  def prior_and_value(
      self, state: pyspiel.State
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    """Returns evaluation on given state."""
    # prior
    if state.is_chance_node():
      prior = state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      prior = [(action, 1.0 / len(legal_actions)) for action in legal_actions]
    # value
    working_state = state.clone()
    while not working_state.is_terminal():
      if working_state.is_chance_node():
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = self._random_state.choice(action_list, p=prob_list)
      else:
        action = self._random_state.choice(working_state.legal_actions())
      working_state.apply_action(action)
    value = np.array(working_state.returns())
    return prior, value


class SearchNode(object):
  """A node in the search tree.

  A SearchNode represents a state and possible continuations from it. Each child
  represents a possible action, and the expected result from doing so.

  Attributes:
    action: The action from the parent node's perspective. Not important for the
      root node, as the actions that lead to it are in the past.
    player: Which player made this action.
    prior: A prior probability for how likely this action will be selected.
    explore_count: How many times this node was explored.
    total_reward: The sum of rewards of rollouts through this node, from the
      parent node's perspective. The average reward of this node is
      `total_reward / explore_count`
    outcome: The rewards for all players if this is a terminal node or the
      subtree has been proven, otherwise None.
    children: A list of SearchNodes representing the possible actions from this
      node, along with their expected rewards.
    expanded: Whether this node has been expanded.
  """

  __slots__ = [
      "action",
      "player",
      "prior",
      "explore_count",
      "total_reward",
      "outcome",
      "children",
      "expanded",
  ]

  def __init__(self, action: int | None, player: int, prior: float):
    self.action = action
    self.player = player
    self.prior = prior
    self.explore_count = 0
    self.total_reward = 0.0
    self.outcome = None
    self.children = []
    self.expanded = False

  def uct_value(self, parent_explore_count: int, uct_c: float) -> float:
    """Returns the UCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    if self.explore_count == 0:
      return float("inf")

    return self.total_reward / self.explore_count + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.explore_count
    )

  def puct_value(self, parent_explore_count: int, uct_c: float) -> float:
    """Returns the PUCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    return (
        self.explore_count and self.total_reward / self.explore_count
    ) + uct_c * self.prior * math.sqrt(parent_explore_count) / (
        self.explore_count + 1
    )

  def sort_key(self):
    """Returns the best action from this node, either proven or most visited.

    This ordering leads to choosing:
    - Highest proven score > 0 over anything else, including a promising but
      unproven action.
    - A proven draw only if it has higher exploration than others that are
      uncertain, or the others are losses.
    - Uncertain action with most exploration over loss of any difficulty
    - Hardest loss if everything is a loss
    - Highest expected reward if explore counts are equal (unlikely).
    - Longest win, if multiple are proven (unlikely due to early stopping).
    """
    return (
        0 if self.outcome is None else self.outcome[self.player],
        self.explore_count,
        self.total_reward,
    )

  def best_child(self):
    """Returns the best child in order of the sort key."""
    return max(self.children, key=SearchNode.sort_key)

  def children_str(self, state=None):
    """Returns the string representation of this node's children.

    They are ordered based on the sort key, so order of being chosen to play.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    return "\n".join([
        c.to_str(state)
        for c in reversed(sorted(self.children, key=SearchNode.sort_key))
    ])

  def to_str(self, state=None):
    """Returns the string representation of this node.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    action = (
        state.action_to_string(state.current_player(), self.action)
        if state and self.action is not None
        else str(self.action)
    )
    return (
        "{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
        "outcome: {}, {:3d} children"
    ).format(
        action,
        self.player,
        self.prior,
        self.explore_count and self.total_reward / self.explore_count,
        self.explore_count,
        (
            "{:4.1f}".format(self.outcome[self.player])
            if self.outcome
            else "none"
        ),
        len(self.children),
    )

  def __str__(self):
    return self.to_str(None)


class MCTSBot(pyspiel.Bot):
  """Bot that uses Monte-Carlo Tree Search algorithm."""

  def __init__(
      self,
      game,
      uct_c,
      max_simulations,
      evaluator,
      solve=True,
      random_state=None,
      child_selection_fn=SearchNode.uct_value,
      best_child_fn: Callable[
          ..., tuple[Optional[int], Optional[SearchNode]]
      ] = robust_child_with_total_reward_tiebreaker,
      dirichlet_noise=None,
      verbose=False,
      dont_return_chance_node=False,
      virtual_loss: int = 10,  # virtual loss for async MCTS
      batch_size: int = 16,  # batch size for asynchronous MCTS
      secure_c: float = 1.0,  # secure constant for secure child selection
      simulations_multiplier: float = 1.0,
      max_additional_simulation_rounds: int = 0,
      timeout: float = 5.0,  # timeout for asynchronous MCTS
  ):
    """Initializes a MCTS Search algorithm in the form of a bot.

    In multiplayer games, or non-zero-sum games, the players will play the
    greedy strategy.

    Args:
      game: A pyspiel.Game to play.
      uct_c: The exploration constant for UCT.
      max_simulations: How many iterations of MCTS to perform. Each simulation
        will result in one call to the evaluator. Memory usage should grow
        linearly with simulations * branching factor. How many nodes in the
        search tree should be evaluated. This is correlated with memory size and
        tree depth.
      evaluator: A `Evaluator` object to use to evaluate a leaf node.
      solve: Whether to back up solved states.
      random_state: An optional numpy RandomState to make it deterministic.
      child_selection_fn: A function to select the child in the descent phase.
        The default is UCT.
      best_child_fn: A function to select the best child in root node after tree
        is built. The default is the child with the most visits.
      dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
        the policy at the root. This is from the alpha-zero paper.
      verbose: Whether to print information about the search tree before
        returning the action. Useful for confirming the search is working
        sensibly.
      dont_return_chance_node: If true, do not stop expanding at chance nodes.
        Enabled for AlphaZero.
      virtual_loss: the value to use for virtual loss in async MCTS.
      batch_size: The batch size for asynchronous MCTS.
      secure_c: The constant used to calculate lower uncertainty bound
      simulations_multiplier: The multiplier for search budget.
      max_additional_simulation_rounds: The maximum number of additional
        simulation rounds.
      timeout: The timeout for asynchronous MCTS.

    Raises:
      ValueError: if the game type isn't supported.
    """
    pyspiel.Bot.__init__(self)
    # Check that the game satisfies the conditions for this MCTS implementation.
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self._game = game
    self.uct_c = uct_c
    self.max_simulations = max_simulations
    self._max_simulations = max_simulations
    self.evaluator = evaluator
    self.verbose = verbose
    self.solve = solve
    self.max_utility = game.max_utility()
    self.min_utility = game.min_utility()
    self._dirichlet_noise = dirichlet_noise
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_fn = child_selection_fn
    self._best_child_fn = best_child_fn
    self.dont_return_chance_node = dont_return_chance_node
    self._root = None
    self.total_num_searches = 0
    self.total_search_time = 0
    # Async MCTS parameters
    self.virtual_loss = virtual_loss
    self.batch_size = batch_size
    self.timeout = timeout
    self.total_timeouts = 0
    self.total_eval_errors = 0
    # Child selection at root parameters
    self._secure_c = secure_c
    self._alternative_criteria = False
    self._max_additional_simulation_rounds = max_additional_simulation_rounds
    self._simulations_multiplier = simulations_multiplier

  def restart_at(self, state):
    pass

  def get_root(self):
    return self._root

  def _get_selection_function_arguments(self, root):
    if self._best_child_fn is secure_child:
      arguments = (root, self._secure_c)
    elif self._best_child_fn is max_robust_child:
      arguments = (root, self._alternative_criteria)
    elif self._best_child_fn is max_robust_secure_child:
      arguments = (root, self._secure_c, self._alternative_criteria)
    else:
      arguments = (root,)

    return arguments

  def step_with_policy(self, state):
    """Returns bot's policy and action at given state."""
    t1 = time.time()
    simulation_round = 0
    best_action = None
    best_child = None
    root = None
    while (
        best_action is None
        and simulation_round <= self._max_additional_simulation_rounds
    ):
      simulation_round += 1
      if simulation_round == self._max_additional_simulation_rounds:
        self._alternative_criteria = True
      root = self.mcts_search(state)
      assert root is not None, "Root is None"
      self._root = root
      arguments = self._get_selection_function_arguments(root)
      best_action, best_child = self._best_child_fn(*arguments)
      # Determine the number of additional simulations.
      if best_action is None:
        self.max_simulations = (
            int(self._simulations_multiplier * self._max_simulations)
        )
    assert best_action is not None, "Best action is None"
    assert best_child is not None, "Best child is None"
    assert self._root is not None, "Root is None"
    seconds = time.time() - t1
    self.total_search_time += seconds
    if self.verbose:
      print(
          "Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
              root.explore_count, seconds, root.explore_count / seconds
          )
      )
      print("Root:")
      print(root.to_str(state))
      print("Children:")
      print(root.children_str(state))
      if best_child.children:
        chosen_state = state.clone()
        chosen_state.apply_action(best_action)
        print("Children of chosen:")
        print(best_child.children_str(chosen_state))
    policy = [
        (action, (1.0 if action == best_action else 0.0))
        for action in state.legal_actions(state.current_player())
    ]
    # Rest max simulations to original value.
    self.max_simulations = self._max_simulations
    return policy, best_action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _add_virtual_losses(self, node):
    # Add virtual losses. This is applied to the nodes touched during the tree
    # policy action selection of the simulation (downward pass). This
    # discourages multiple threads from exploring the same node.
    node.total_reward += (self.virtual_loss * self.min_utility)
    node.explore_count += self.virtual_loss

  def _remove_virtual_losses(self, node):
    # Remove virtual losses. This is applied to the nodes touched during the
    # backpropagation phase (upward pass) to ensure that the fake losses are
    # removed once this path down the tree is done being simulated.
    node.total_reward -= (self.virtual_loss * self.min_utility)
    node.explore_count -= self.virtual_loss

  def _choose_next_node(self, visit_path, working_state, current_node):
    if working_state.is_chance_node():
      # For chance nodes, rollout according to chance node's probability
      # distribution
      outcomes = working_state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = self._random_state.choice(action_list, p=prob_list)
      chosen_child = next(
          c for c in current_node.children if c.action == action
      )
    else:
      # Otherwise choose node with largest UCT value
      chosen_child = max(
          current_node.children,
          key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
              c, current_node.explore_count, self.uct_c
          ),
      )
    working_state.apply_action(chosen_child.action)
    current_node = chosen_child
    self._add_virtual_losses(current_node)
    visit_path.append(current_node)
    return current_node

  def _apply_tree_policy(self, root, state):
    """Applies the UCT policy to play the game until reaching a leaf node.

    A leaf node is defined as a node that is terminal or has not been evaluated
    yet. If it reaches a node that has been evaluated before but hasn't been
    expanded, then expand its children and continue.

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
    self._add_virtual_losses(root)
    unexplored_explore_count = self.virtual_loss
    while (
        not working_state.is_terminal()
        and current_node.explore_count > unexplored_explore_count
    ) or (working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        return visit_path, working_state, current_node
      current_node = self._choose_next_node(
          visit_path, working_state, current_node
      )
    return visit_path, working_state, current_node

  def backpropagate(self, visit_path, returns):
    while visit_path:
      # For chance nodes, walk up the tree to find the decision-maker.
      decision_node_idx = -1
      while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
        decision_node_idx -= 1
      # Chance node targets are for the respective decision-maker.
      target_return = returns[visit_path[decision_node_idx].player]
      node = visit_path.pop()
      node.total_reward += target_return
      node.explore_count += 1
      self._remove_virtual_losses(node)
      assert node.explore_count >= 1

  def backpropagate_timeout(self, visit_path):
    while visit_path:
      # For chance nodes, walk up the tree to find the decision-maker.
      node = visit_path.pop()
      self._remove_virtual_losses(node)

  def expand(self, root, working_state, current_node, prior, value):
    # For a new node, initialize its state, then choose a child as normal.
    # prior, value = self.evaluator.prior_and_value(working_state)
    if current_node is root and self._dirichlet_noise:
      epsilon, alpha = self._dirichlet_noise
      noise = self._random_state.dirichlet([alpha] * len(prior))
      prior = [
          (a, (1 - epsilon) * p + epsilon * n)
          for (a, p), n in zip(prior, noise)
      ]
    # Reduce bias from move generation order.
    self._random_state.shuffle(prior)
    player = working_state.current_player()
    current_node.children = [
        SearchNode(action, player, prob) for action, prob in prior
    ]
    current_node.expanded = True

  def evaluate(
      self, working_state
  ) -> tuple[list[tuple[int, float]], np.ndarray]:
    if working_state.is_terminal():
      prior = []
      values = working_state.returns()
    else:
      prior, values = self.evaluator.prior_and_value(working_state)
    return prior, values

  def handle_leaf(self, prior, value, arguments, timeout=False):
    visit_path, working_state, node, root = arguments
    if timeout:
      self.backpropagate_timeout(visit_path)
      return
    assert node is not None
    if not node.expanded:
      self.expand(root, working_state, node, prior, value)
    if working_state.is_terminal():
      visit_path[-1].outcome = working_state.returns()
    self.backpropagate(visit_path, value)

  def async_mcts_search(self, state):
    root = SearchNode(None, state.current_player(), 1)
    # do one call up front, to ensure we have some children for the threads
    # to spread over.
    self._add_virtual_losses(root)
    working_state = state.clone()
    prior, value = self.evaluate(working_state)
    self.handle_leaf(
        prior, value, ([root], working_state, root, root), timeout=False
    )
    total_simulations = 1
    search_timeouts = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.batch_size
    ) as executor:
      while total_simulations < self.max_simulations:
        batch_timeouts = 0
        remaining_simulations = self.max_simulations - total_simulations
        num_to_queue = min(self.batch_size, remaining_simulations)
        futures = []
        arguments = []
        for _ in range(num_to_queue):
          visit_path, working_state, node = self._apply_tree_policy(root, state)
          arguments.append((visit_path, working_state, node, root))
          future = executor.submit(self.evaluate, working_state)
          futures.append(future)
        concurrent.futures.wait(
            futures,
            timeout=self.timeout,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        for i, future in enumerate(futures):
          if future.done():
            prior, value = future.result()
            self.handle_leaf(prior, value, arguments[i], timeout=False)
          else:
            batch_timeouts += 1
            search_timeouts += 1
            self.handle_leaf(None, None, arguments[i], timeout=True)
        total_simulations += num_to_queue
    if self.verbose:
      logging.info("Timeouts for this search: %d", search_timeouts)
    self.total_timeouts += search_timeouts
    self.total_num_searches += 1
    if self.verbose:
      logging.info(
          "Average timeouts per search: %g\n" +
          "Average eval errors per search: %g",
          self.total_timeouts / self.total_num_searches,
          self.total_eval_errors / self.total_num_searches)
    return root

  def mcts_search(self, state):
    """A vanilla Monte-Carlo Tree Search algorithm.

    This algorithm searches the game tree from the given state.
    At the leaf, the evaluator is called if the game state is not terminal.
    A total of max_simulations states are explored.

    At every node, the algorithm chooses the action with the highest PUCT value,
    defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
    reward after the action, and N is the number of times the action was
    explored in this position. The input parameter c controls the balance
    between exploration and exploitation; higher values of c encourage
    exploration of under-explored nodes. Unseen actions are always explored
    first.

    At the end of the search, the chosen action is the action that has been
    explored most often. This is the action that is returned.

    This implementation supports sequential n-player games, with or without
    chance nodes. All players maximize their own reward and ignore the other
    players' rewards. This corresponds to max^n for n-player games. It is the
    norm for zero-sum games, but doesn't have any special handling for
    non-zero-sum games. It doesn't have any special handling for imperfect
    information games.

    The implementation also supports backing up solved states, i.e. MCTS-Solver.
    The implementation is general in that it is based on a max^n backup (each
    player greedily chooses their maximum among proven children values, or there
    exists one child whose proven value is game.max_utility()), so it will work
    for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
    draw games). Also chance nodes are considered proven only if all children
    have the same value.

    Some references:
    - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
      https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
    - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
      https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
    - Silver, AlphaGo Zero: Starting from scratch, 2017
      https://deepmind.com/blog/article/alphago-zero-starting-scratch
    - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
      https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

    Arguments:
      state: pyspiel.State object, state to search from

    Returns:
      The most visited move from the root node.
    """
    return self.async_mcts_search(state)
