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

"""Example algorithm to get all states from a game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _get_subgames_states(state, all_states, depth_limit, depth,
                         include_terminals, include_chance_states, to_string,
                         stop_if_encountered):
  """Extract non-chance states for a subgame into the all_states dict."""
  if state.is_terminal():
    if include_terminals:
      # Include if not already present and then terminate recursion.
      state_str = to_string(state)
      if state_str not in all_states:
        all_states[state_str] = state.clone()
    return

  if depth > depth_limit >= 0:
    return

  if not state.is_chance_node() or include_chance_states:
    # Add only if not already present
    state_str = to_string(state)
    if state_str not in all_states:
      all_states[state_str] = state.clone()
    else:
      # We already saw this one. Stop the recursion if the flag is set
      if stop_if_encountered:
        return

  for action in state.legal_actions():
    state_for_search = state.child(action)
    _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1,
                         include_terminals, include_chance_states, to_string,
                         stop_if_encountered)


def get_all_states(game,
                   depth_limit=-1,
                   include_terminals=True,
                   include_chance_states=False,
                   to_string=lambda s: s.history_str(),
                   stop_if_encountered=True):
  """Gets all states in the game, indexed by their string representation.

  For small games only! Useful for methods that solve the  games explicitly,
  i.e. value iteration. Use this default implementation with caution as it does
  a recursive tree walk of the game and could easily fill up memory for larger
  games or games with long horizons.

  Currently only works for sequential games.

  Arguments:
    game: The game to analyze, as returned by `load_game`.
    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0
      means root-only, etc.
    include_terminals: If True, include terminal states.
    include_chance_states: If True, include chance node states.
    to_string: The serialization function. We expect this to be
      `lambda s: s.history_str()` as this enforces perfect recall, but for
        historical reasons, using `str` is also supported, but the goal is to
        remove this argument.
    stop_if_encountered: if this is set, do not keep recursively adding states
      if this state is already in the list. This allows support for games that
      have cycles.

  Returns:
    A `dict` with `to_string(state)` keys and `pyspiel.State` values containing
    all states encountered traversing the game tree up to the specified depth.
  """
  # Get the root state.
  state = game.new_initial_state()
  all_states = dict()

  # Then, do a recursive tree walk to fill up the map.
  _get_subgames_states(
      state=state,
      all_states=all_states,
      depth_limit=depth_limit,
      depth=0,
      include_terminals=include_terminals,
      include_chance_states=include_chance_states,
      to_string=to_string,
      stop_if_encountered=stop_if_encountered)

  if not all_states:
    raise ValueError("GetSubgameStates returned 0 states!")

  return all_states
