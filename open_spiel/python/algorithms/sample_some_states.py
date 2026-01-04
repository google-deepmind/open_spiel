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

"""Example algorithm to sample some states from a game."""

import random
import pyspiel


def sample_some_states(
    game,
    max_states=100,
    make_distribution_fn=lambda states: [1 / len(states)] * len(states)):
  """Samples some states in the game.

  This can be run for large games, in contrast to `get_all_states`. It is useful
  for tests that need to check a predicate only on a subset of the game, since
  generating the whole game is infeasible.

  Currently only works for sequential games. For simultaneous games and mean
  field games it returns only the initial state.

  The algorithm maintains a list of states and repeatedly picks a random state
  from the list to expand until enough states have been sampled.

  Arguments:
    game: The game to analyze, as returned by `load_game`.
    max_states: The maximum number of states to return. Negative means no limit.
    make_distribution_fn: Function that takes a list of states and returns a
      corresponding distribution (as a list of floats). Only used for mean field
      games.

  Returns:
    A `list` of `pyspiel.State`.
  """
  if game.get_type().dynamics in [
      pyspiel.GameType.Dynamics.SIMULTANEOUS,
      pyspiel.GameType.Dynamics.MEAN_FIELD
  ]:
    return [game.new_initial_state()]
  states = []
  unexplored_actions = []
  indexes_with_unexplored_actions = set()

  def add_state(state):
    states.append(state)
    if state.is_terminal():
      unexplored_actions.append(None)
    else:
      indexes_with_unexplored_actions.add(len(states) - 1)
      unexplored_actions.append(set(state.legal_actions()))

  def expand_random_state():
    index = random.choice(list(indexes_with_unexplored_actions))
    state = states[index]
    if state.is_mean_field_node():
      child = state.clone()
      child.update_distribution(
          make_distribution_fn(child.distribution_support()))
      indexes_with_unexplored_actions.remove(index)
      return child
    else:
      actions = unexplored_actions[index]
      assert actions, f"Empty actions for state {state}"
      action = random.choice(list(actions))
      actions.remove(action)
      if not actions:
        indexes_with_unexplored_actions.remove(index)
      return state.child(action)

  add_state(game.new_initial_state())
  while (len(states) < max_states) and indexes_with_unexplored_actions:
    add_state(expand_random_state())

  if not states:
    raise ValueError("get_some_states sampled 0 states!")

  return states
