# Copyright 2019 DeepMind Technologies Limited
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

"""Value iteration algorithm for solving a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import lp_solver
import pyspiel


def _get_future_states(possibilities, state, reach=1.0):
  """Does a lookahead over chance nodes to all next states after (s,a).

  Also works if there are no chance nodes (i.e. base case).

  Arguments:
    possibilities:  an empty list, that will be filled with (str(next_state),
      transition probability) pairs for all possible next states
    state: the state following some s.apply_action(a), can be a chance node
    reach: chance reach probability of getting to this point from (s,a)
  Returns: nothing.
  """
  if not state.is_chance_node() or state.is_terminal():
    # Base case
    possibilities.append((str(state), reach))
  else:
    assert state.is_chance_node()
    for outcome, prob in state.chance_outcomes():
      next_state = state.child(outcome)
      _get_future_states(possibilities, next_state, reach * prob)


def _add_transition(transitions, key, state):
  """Adds action transitions from given state."""

  if state.is_simultaneous_node():
    for p0action in state.legal_actions(0):
      for p1action in state.legal_actions(1):
        next_state = state.clone()
        next_state.apply_actions([p0action, p1action])
        possibilities = []
        _get_future_states(possibilities, next_state)
        transitions[(key, p0action, p1action)] = possibilities
  else:
    for action in state.legal_actions():
      next_state = state.child(action)
      possibilities = []
      _get_future_states(possibilities, next_state)
      transitions[(key, action)] = possibilities


def _initialize_maps(states, values, transitions):
  """Initialize the value and transition maps."""
  for key, state in states.items():
    if state.is_terminal():
      values[key] = state.player_return(0)
    else:
      values[key] = 0
      _add_transition(transitions, key, state)


def value_iteration(game, depth_limit, threshold, cyclic_game=False):
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
    cyclic_game: set to True if the game has cycles (from state A we can get to
      state B, and from state B we can get back to state A).

  Returns:
    A `dict` with string keys and float values, mapping string encoding of
    states to the values of those states.
  """
  assert game.num_players() in (1,
                                2), ("Game must be a 1-player or 2-player game")
  if game.num_players() == 2:
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM, (
        "2-player games must be zero sum games")

  # Must be perfect information or one-shot (not imperfect information).
  assert (game.get_type().information == pyspiel.GameType.Information.ONE_SHOT
          or game.get_type().information ==
          pyspiel.GameType.Information.PERFECT_INFORMATION)

  # We expect Value Iteration to be used with perfect information games, in
  # which `str` is assumed to display the state of the game.
  states = get_all_states.get_all_states(
      game,
      depth_limit,
      True,
      False,
      to_string=str,
      stop_if_encountered=cyclic_game)
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
      elif state.is_simultaneous_node():
        # Simultaneous node. Assemble a matrix game from the child utilities.
        # and solve it using a matrix game solver.
        p0_utils = []  # row player
        p1_utils = []  # col player
        row = 0
        for p0action in state.legal_actions(0):
          # new row
          p0_utils.append([])
          p1_utils.append([])
          for p1action in state.legal_actions(1):
            # loop from left-to-right of columns
            next_states = transitions[(key, p0action, p1action)]
            joint_q_value = sum(
                p * values[next_state] for next_state, p in next_states)
            p0_utils[row].append(joint_q_value)
            p1_utils[row].append(-joint_q_value)
          row += 1
        stage_game = pyspiel.create_matrix_game(p0_utils, p1_utils)
        solution = lp_solver.solve_zero_sum_matrix_game(stage_game)
        value = solution[2]
      else:
        # Regular decision node
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
