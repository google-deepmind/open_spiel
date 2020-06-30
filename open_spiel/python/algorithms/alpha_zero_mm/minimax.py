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

"""Implements the min-max algorithm with alpha-beta pruning.

Solves perfect play for deterministic, 2-players, perfect-information 0-sum
games.

See for example https://en.wikipedia.org/wiki/Alpha-beta_pruning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel


class minimaxBot(pyspiel.Bot):

  def __init__(self,
               game,
               evaluator,
               max_simulations=10):
   
    pyspiel.Bot.__init__(self) 
    self._game = game
    self.max_simulations = max_simulations
    self.evaluator = evaluator

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _alpha_beta(self, 
                  state, 
                  depth, 
                  alpha, 
                  beta,   
                  value_function,
                  maximizing_player_id):

    if state.is_terminal():
      return state.player_return(maximizing_player_id), None

    if depth == 0 and value_function is None:
      raise NotImplementedError(
          "We assume we can walk the full depth of the tree. "
          "Try increasing the maximum_depth or provide a value_function.")
    if depth == 0:
      return value_function(state), None

    player = state.current_player()
    best_action = -1
    if player == maximizing_player_id:
      value = -float("inf")
      # print(str(state))
      for action in state.legal_actions():
        
        child_state = state.clone()
        child_state.apply_action(action)
        child_value, _ = self._alpha_beta(child_state, depth - 1, alpha, beta,
                                    value_function, maximizing_player_id)
        # print(action, child_value)                                    
        if child_value > value:
          value = child_value
          best_action = action
        alpha = max(alpha, value)
        if alpha >= beta:
          break  # beta cut-off
      
      return value, best_action
    else:
      value = float("inf")
      for action in state.legal_actions():
        child_state = state.clone()
        child_state.apply_action(action)
        child_value, _ = self._alpha_beta(child_state, depth - 1, alpha, beta,
                                    value_function, maximizing_player_id)
        if child_value < value:
          value = child_value
          best_action = action
        beta = min(beta, value)
        if alpha >= beta:
          break  # alpha cut-off
      return value, best_action

  def alpha_beta_search(self, state=None,
                        value_function=None,
                        maximum_depth=30,
                        maximizing_player_id=None):

    if state is None:
      state = self._game.new_initial_state()
    if maximizing_player_id is None:
      maximizing_player_id = state.current_player()
    return self._alpha_beta(
        state.clone(),
        maximum_depth,
        alpha=-float("inf"),
        beta=float("inf"),
        value_function=value_function,
        maximizing_player_id=maximizing_player_id)

class SearchNode(object):
  __slots__ = [
      "action",
      "player",
      "prior",
      "value",
      "outcome",
      "children",
  ]

  def __init__(self, action, player, prior):
    self.action = action
    self.player = player
    self.prior = prior
    self.value = 0
    self.outcome = None
    self.children = []

  
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
    return (0 if self.outcome is None else self.outcome[self.player], self.value)

  def best_child(self):
    """Returns the best child in order of the sort key."""
    return max(self.children, key=SearchNode.sort_key)