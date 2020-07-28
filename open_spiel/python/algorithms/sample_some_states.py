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

"""Example algorithm to sample some states from a game."""

import time
import random


def sample_some_states(game,
    max_rollouts=100,
    time_limit=1,
    depth_limit=-1,
    include_terminals=True,
    include_chance_states=True,
    to_string=lambda s: s.history_str()):
  """Samples some states in the game, indexed by their string representation.

  This can be run for large games, in contrast to `get_all_states`. It is useful
  for tests that need to check a predicate only on a subset of the game, since
  generating the whole game is infeasible. This function either makes specified
  number of rollouts or terminates after specified time limit.

  Currently only works for sequential games.

  Arguments:
    game: The game to analyze, as returned by `load_game`.
    max_rollouts: The maximum number of rollouts to make. Negative means no
      limit.
    time_limit: Time limit in seconds. Negative means no limit.
    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0
      means root-only, etc.
    include_terminals: If True, include terminal states.
    include_chance_states: If True, include chance node states.
    to_string: The serialization function. We expect this to be
      `lambda s: s.history_str()` as this enforces perfect recall, but for
        historical reasons, using `str` is also supported, but the goal is to
        remove this argument.

  Returns:
    A `dict` with `to_string(state)` keys and `pyspiel.State` values containing
    all states encountered traversing the game tree up to the specified depth.
  """
  if max_rollouts <= 0 and time_limit <= 0:
    raise ValueError("Both max_rollouts={} and time_limit={} are smaller "
                     "(or equal) than zero. The function would not return "
                     "a useful result.")
  some_states = dict()

  def is_time_out(t):
    return False if time_limit < 0 else time.time() - t > time_limit
  def rollouts_exceeded(n):
    return False if max_rollouts < 0 else n >= max_rollouts
  def add_state_if_not_present(state):
    nonlocal some_states
    state_str = to_string(state)
    if state_str not in some_states:
      some_states[state_str] = state.clone()

  start = time.time()
  num_rollouts = 0
  state = game.new_initial_state()
  while not is_time_out(start) and not rollouts_exceeded(num_rollouts):
    if not state.is_chance_node() or include_chance_states:
      add_state_if_not_present(state)

    # Do not continue beyond depth limit.
    if state.move_number() >= depth_limit >= 0:
      num_rollouts += 1
      state = game.new_initial_state()
      continue

    action = random.choice(state.legal_actions(state.current_player()))
    state.apply_action(action)

    if state.is_terminal():
      if include_terminals:
        add_state_if_not_present(state)
      num_rollouts += 1
      state = game.new_initial_state()

  if not some_states:
    raise ValueError("get_some_states sampled 0 states!")

  return some_states
