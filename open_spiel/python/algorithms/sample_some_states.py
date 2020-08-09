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

import random
import time


def sample_some_states(game,
                       max_states=100,
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
    max_states: The maximum number of states to return. Negative means no
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
    sampled states encountered traversing the game tree.
  """
  if (max_states < 0 and time_limit <= 0) or max_states == 0:
    raise ValueError("max_states={} and time_limit={} are smaller "
                     "(or equal) than zero. The function would not return "
                     "a useful result.".format(max_states, time_limit))

  def is_time_remaining(t):
    return time_limit < 0 or time.time() - t <= time_limit

  def collect_more_states(num_states):
    return max_states < 0 or num_states < max_states

  start = time.time()
  some_states = dict()
  state = game.new_initial_state()
  while is_time_remaining(start) and collect_more_states(len(some_states)):
    if (include_chance_states or not state.is_chance_node()) and \
       (include_terminals or not state.is_terminal()):
      # Add state if not present.
      state_str = to_string(state)
      if state_str not in some_states:
        some_states[state_str] = state.clone()

    # Restart when terminal is hit / do not continue beyond depth limit.
    if state.is_terminal() or state.move_number() >= depth_limit >= 0:
      state = game.new_initial_state()
      continue

    action = random.choice(state.legal_actions(state.current_player()))
    state.apply_action(action)

  if not some_states:
    raise ValueError("get_some_states sampled 0 states!")

  return some_states
