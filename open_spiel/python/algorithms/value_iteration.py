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

"""Value iteration algorithm for solving a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python.algorithms import get_all_states
import pyspiel


def _add_transition(transitions, key, state):
  """Adds action transitions from given state."""
  for action in state.legal_actions():
    next_state = state.child(action)
    possibilities = []
    if next_state.is_chance_node():
      for chance_action, prob in next_state.chance_outcomes():
        realized_next_state = next_state.child(chance_action)
        possibilities.append([str(realized_next_state), prob])
    else:
      possibilities = [(str(next_state), 1)]

    transitions[(key, action)] = possibilities


def _initialize_maps(states, values, transitions):
  """Initialize the value and transition maps."""
  for key, state in states.items():
    if state.is_terminal():
      values[key] = state.player_return(0)
    else:
      values[key] = 0
      _add_transition(transitions, key, state)


def value_iteration(game, depth_limit, threshold):
  """Solves for the optimal value function of a game.

  For small games only! Solves the game using value iteration,
  with the maximum error for the value function less than threshold.
  This algorithm works for sequential 1-player games or 2-player zero-sum
  games, with or without chance nodes.

  Arguments:
    game: The game to analyze, as returned by `load_game`.
    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0
      means root-only, etc.
    threshold: Maximum error for state values..

  Returns:
    A `dict` with string keys and float values, mapping string encoding of
    states to the values of those states.
  """
  if game.num_players() not in (1, 2):
    raise ValueError("Game must be a 1-player or 2-player game")
  if (game.num_players() == 2 and
      game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM):
    raise ValueError("2-player games must be zero sum games")
  # We expect Value Iteration to be used with perfect information games, in
  # which `str` is assumed to display the state of the game.
  states = get_all_states.get_all_states(
      game, depth_limit, True, False, to_string=str)
  values = {}
  transitions = {}

  _initialize_maps(states, values, transitions)
  error = threshold + 1  # A value larger than threshold
  min_utility = game.min_utility()
  while error > threshold:
    error = 0
    for key, state in states.items():
      if state.is_terminal():
        continue
      player = state.current_player()
      value = min_utility if player == 0 else -min_utility
      for action in state.legal_actions():
        next_states = transitions[(key, action)]
        q_value = sum(p * values[next_state] for next_state, p in next_states)
        if player == 0:
          value = max(value, q_value)
        else:
          value = min(value, q_value)
      error = max(abs(values[key] - value), error)
      values[key] = value

  return values
