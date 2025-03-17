# Copyright 2023 DeepMind Technologies Limited
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

"""Backward induction for perfect information games."""

import enum
import random
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Mapping

import pyspiel


class TieBreakingPolicy(enum.Enum):
  """Different strategies for tie-breaking among equally good actions."""
  FIRST_ACTION = 0  # Choose the first action with the best value (default)
  LAST_ACTION = 1   # Choose the last action with the best value
  RANDOM_ACTION = 2  # Choose a random action among the best ones
  ALL_ACTIONS = 3   # Return all actions with the best value


def _compute_backward_induction(
    state: pyspiel.State,
    cache: Dict[str, Tuple[List[float], int, List[int]]],
    tie_breaking_policy: TieBreakingPolicy = TieBreakingPolicy.FIRST_ACTION
) -> Tuple[List[float], int, List[int]]:
  """Implementation of backward induction algorithm with memoization.
  
  Args:
    state: The current state to analyze
    cache: A dictionary mapping state strings to their computed results
    tie_breaking_policy: How to break ties when multiple actions are equally good
    
  Returns:
    A tuple of (values, best_action, best_actions) where:
    - values is a list of values for each player
    - best_action is the optimal action for the current player (-1 for terminal/chance)
    - best_actions is a list of all optimal actions (if TieBreakingPolicy.ALL_ACTIONS)
  """
  # Check if this state has already been computed
  state_str = state.to_string()
  if state_str in cache:
    return cache[state_str]
  
  # Handle terminal states
  if state.is_terminal():
    values = state.returns()
    cache[state_str] = (values, pyspiel.INVALID_ACTION, [])
    return cache[state_str]
  
  # Handle chance nodes
  if state.is_chance_node():
    num_players = state.get_game().num_players()
    values = [0.0] * num_players
    chance_outcomes = state.chance_outcomes()
    for action, prob in chance_outcomes:
      child = state.child(action)
      child_values, _, _ = _compute_backward_induction(child, cache, tie_breaking_policy)
      for p in range(num_players):
        values[p] += prob * child_values[p]
    cache[state_str] = (values, pyspiel.INVALID_ACTION, [])
    return cache[state_str]
  
  # For player nodes, compute the value that maximizes that player's payoff
  current_player = state.current_player()
  legal_actions = state.legal_actions()
  
  # Start with the first action as best
  best_action = legal_actions[0]
  child = state.child(best_action)
  best_values, _, _ = _compute_backward_induction(child, cache, tie_breaking_policy)
  best_value = best_values[current_player]
  
  # Track all actions that yield the best value (for tie-breaking)
  best_actions = [best_action]
  
  # Try all other actions
  for action in legal_actions[1:]:
    child = state.child(action)
    values, _, _ = _compute_backward_induction(child, cache, tie_breaking_policy)
    value = values[current_player]
    
    # If we found a strictly better action
    if value > best_value:
      best_value = value
      best_values = values
      best_action = action
      best_actions = [action]
    elif value == best_value:
      # This action is tied for best
      best_actions.append(action)
  
  # Apply tie-breaking policy
  if tie_breaking_policy == TieBreakingPolicy.FIRST_ACTION:
    # We already chose the first action as best initially
    pass
  elif tie_breaking_policy == TieBreakingPolicy.LAST_ACTION:
    # Choose the last action that tied for best
    best_action = best_actions[-1]
  elif tie_breaking_policy == TieBreakingPolicy.RANDOM_ACTION:
    # Choose a random action among the best
    if len(best_actions) > 1:
      best_action = random.choice(best_actions)
  elif tie_breaking_policy == TieBreakingPolicy.ALL_ACTIONS:
    # Leave best_actions as is to return all optimal actions
    pass
  
  cache[state_str] = (best_values, best_action, best_actions)
  return cache[state_str]


def backward_induction(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    tie_breaking_policy: TieBreakingPolicy = TieBreakingPolicy.FIRST_ACTION,
    allow_imperfect_information: bool = False
) -> Tuple[List[float], Dict[str, int]]:
  """Computes optimal values and policies for a perfect information game.
  
  Args:
    game: The game to analyze.
    state: An optional initial state to analyze from. If None, use the game's
      initial state.
    tie_breaking_policy: Policy for breaking ties between equally-valued actions.
    allow_imperfect_information: If True, allows running the algorithm on
      imperfect information games, but the solution may not be a Nash equilibrium.
      
  Returns:
    A tuple of:
    - values: the expected returns for each player at the given state
    - policy: a mapping from state strings to optimal actions for that state
    
  Raises:
    ValueError: If the game is not sequential (turn-based).
    RuntimeError: If the game has imperfect information and 
      allow_imperfect_information is False.
  """
  if game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Backward induction requires sequential games")
  
  if not allow_imperfect_information:
    if game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise RuntimeError(
          "Backward induction requires perfect information games. "
          "Use allow_imperfect_information=True to override this check.")
  elif game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    print("WARNING: Running backward induction on an imperfect information "
          "game. The result is NOT guaranteed to be a Nash equilibrium.",
          file=sys.stderr)
  
  root = state.clone() if state else game.new_initial_state()
  
  # Set fixed random seed for reproducible random tie-breaking
  random.seed(42)
  
  cache = {}
  values, _, _ = _compute_backward_induction(root, cache, tie_breaking_policy)
  
  # Extract the policy
  policy = {}
  for state_str, (_, best_action, _) in cache.items():
    if best_action != pyspiel.INVALID_ACTION:
      policy[state_str] = best_action
  
  return values, policy


def backward_induction_values(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    allow_imperfect_information: bool = False
) -> List[float]:
  """Helper function that returns just the values at the root.
  
  Args:
    game: The game to analyze.
    state: An optional initial state to analyze from. If None, use the game's
      initial state.
    allow_imperfect_information: If True, allows running the algorithm on
      imperfect information games, but the solution may not be a Nash equilibrium.
      
  Returns:
    A list of expected returns for each player at the given state.
  """
  values, _ = backward_induction(
      game, state, allow_imperfect_information=allow_imperfect_information)
  return values


def backward_induction_all_optimal_actions(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    allow_imperfect_information: bool = False
) -> Tuple[List[float], Dict[str, List[int]]]:
  """Returns all optimal actions when there are ties.
  
  Args:
    game: The game to analyze.
    state: An optional initial state to analyze from. If None, use the game's
      initial state.
    allow_imperfect_information: If True, allows running the algorithm on
      imperfect information games, but the solution may not be a Nash equilibrium.
      
  Returns:
    A tuple of:
    - values: the expected returns for each player at the given state
    - all_optimal_actions: a mapping from state strings to lists of optimal
      actions for that state
  """
  if game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Backward induction requires sequential games")
  
  if not allow_imperfect_information:
    if game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise RuntimeError(
          "Backward induction requires perfect information games. "
          "Use allow_imperfect_information=True to override this check.")
  elif game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    print("WARNING: Running backward induction on an imperfect information "
          "game. The result is NOT guaranteed to be a Nash equilibrium.",
          file=sys.stderr)
  
  root = state.clone() if state else game.new_initial_state()
  
  cache = {}
  values, _, _ = _compute_backward_induction(
      root, cache, TieBreakingPolicy.ALL_ACTIONS)
  
  # Extract all optimal actions
  all_optimal_actions = {}
  for state_str, (_, _, best_actions) in cache.items():
    if best_actions:
      all_optimal_actions[state_str] = best_actions
  
  return values, all_optimal_actions 