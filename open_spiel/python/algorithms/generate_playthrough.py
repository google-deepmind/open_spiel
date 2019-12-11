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

"""Functions to manipulate game playthoughs.

Used by examples/playthrough.py and tests/playthrough_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np

import pyspiel


def _escape(x):
  """Returns a newline-free backslash-escaped version of the given string."""
  x = x.replace("\\", R"\\")
  x = x.replace("\n", R"\n")
  return x


def playthrough(game_string, seed, alsologtostdout=False):
  """Returns a playthrough of the specified game as a single text.

  Actions are selected uniformly at random, including chance actions.

  Args:
    game_string: string, e.g. 'markov_soccer', with possible optional params,
      e.g. 'go(komi=4.5,board_size=19)'.
    seed: an integer to seed the random number generator governing action
      choices.
    alsologtostdout: Whether to also print the trace to stdout. This can be
      useful when an error occurs, to still be able to get context information.
  """
  lines = playthrough_lines(
      game_string=game_string, seed=seed, alsologtostdout=alsologtostdout)
  return "\n".join(lines) + "\n"


def playthrough_lines(game_string, seed, alsologtostdout=False):
  """Returns a playthrough of the specified game as a list of lines.

  Actions are selected uniformly at random, including chance actions.

  Args:
    game_string: string, e.g. 'markov_soccer' or 'kuhn_poker(players=4)'.
    seed: an integer to seed the random number generator governing action
      choices.
    alsologtostdout: Whether to also print the trace to stdout. This can be
      useful when an error occurs, to still be able to get context information.
  """
  lines = []
  if alsologtostdout:

    def add_line(v):
      print(v)
      lines.append(v)
  else:
    add_line = lines.append

  game = pyspiel.load_game(game_string)
  add_line("game: {}".format(game_string))
  if seed is None:
    seed = np.random.randint(2**32 - 1)
  add_line("seed: {}".format(seed))

  add_line("")
  game_type = game.get_type()
  add_line("GameType.chance_mode = {}".format(game_type.chance_mode))
  add_line("GameType.dynamics = {}".format(game_type.dynamics))
  add_line("GameType.information = {}".format(game_type.information))
  add_line("GameType.long_name = {}".format('"{}"'.format(game_type.long_name)))
  add_line("GameType.max_num_players = {}".format(game_type.max_num_players))
  add_line("GameType.min_num_players = {}".format(game_type.min_num_players))
  add_line("GameType.parameter_specification = {}".format("[{}]".format(
      ", ".join('"{}"'.format(param)
                for param in sorted(game_type.parameter_specification)))))
  add_line("GameType.provides_information_state_string = {}".format(
      game_type.provides_information_state_string))
  add_line("GameType.provides_information_state_tensor = {}".format(
      game_type.provides_information_state_tensor))
  add_line("GameType.provides_observation_string = {}".format(
      game_type.provides_observation_string))
  add_line("GameType.provides_observation_tensor = {}".format(
      game_type.provides_observation_tensor))
  add_line("GameType.reward_model = {}".format(game_type.reward_model))
  add_line("GameType.short_name = {}".format('"{}"'.format(
      game_type.short_name)))
  add_line("GameType.utility = {}".format(game_type.utility))

  add_line("")
  add_line("NumDistinctActions() = {}".format(game.num_distinct_actions()))
  add_line("MaxChanceOutcomes() = {}".format(game.max_chance_outcomes()))
  add_line("GetParameters() = {{{}}}".format(",".join(
      "{}={}".format(key, _escape(str(value)))
      for key, value in sorted(game.get_parameters().items()))))
  add_line("NumPlayers() = {}".format(game.num_players()))
  add_line("MinUtility() = {:.5}".format(game.min_utility()))
  add_line("MaxUtility() = {:.5}".format(game.max_utility()))
  try:
    utility_sum = game.utility_sum()
  except RuntimeError:
    utility_sum = None
  add_line("UtilitySum() = {}".format(utility_sum))
  if game_type.provides_information_state_tensor:
    add_line("InformationStateTensorShape() = {}".format(
        [int(x) for x in game.information_state_tensor_shape()]))
    add_line("InformationStateTensorSize() = {}".format(
        game.information_state_tensor_size()))
  if game_type.provides_observation_tensor:
    add_line("ObservationTensorShape() = {}".format(
        [int(x) for x in game.observation_tensor_shape()]))
    add_line("ObservationTensorSize() = {}".format(
        game.observation_tensor_size()))
  add_line("MaxGameLength() = {}".format(game.max_game_length()))
  add_line('ToString() = "{}"'.format(str(game)))

  players = list(range(game.num_players()))
  state = game.new_initial_state()
  state_idx = 0
  rng = np.random.RandomState(seed)

  while True:
    add_line("")
    add_line("# State {}".format(state_idx))
    for line in str(state).splitlines():
      add_line("# {}".format(line))
    add_line("IsTerminal() = {}".format(state.is_terminal()))
    add_line("History() = {}".format([int(a) for a in state.history()]))
    add_line('HistoryString() = "{}"'.format(state.history_str()))
    add_line("IsChanceNode() = {}".format(state.is_chance_node()))
    add_line("IsSimultaneousNode() = {}".format(state.is_simultaneous_node()))
    add_line("CurrentPlayer() = {}".format(state.current_player()))
    if game.get_type().provides_information_state_string:
      for player in players:
        add_line('InformationStateString({}) = "{}"'.format(
            player, _escape(state.information_state_string(player))))
    if game.get_type().provides_information_state_tensor:
      for player in players:
        vec = ", ".join(
            str(round(x, 5)) for x in state.information_state_tensor(player))
        add_line("InformationStateTensor({}) = [{}]".format(player, vec))
    if game.get_type().provides_observation_string:
      for player in players:
        add_line('ObservationString({}) = "{}"'.format(
            player, _escape(state.observation_string(player))))
    if game.get_type().provides_observation_tensor:
      for player in players:
        vec = ", ".join(
            str(round(x, 5)) for x in state.observation_tensor(player))
        add_line("ObservationTensor({}) = [{}]".format(player, vec))
    if game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC:
      add_line('SerializeState() = "{}"'.format(_escape(state.serialize())))
    if not state.is_chance_node():
      add_line("Rewards() = {}".format(state.rewards()))
      add_line("Returns() = {}".format(state.returns()))
    if state.is_terminal():
      break
    if state.is_chance_node():
      # In Python 2 and Python 3, the default number of decimal places displayed
      # is different. Thus, we hardcode a ".12" which is Python 2 behaviour.
      add_line("ChanceOutcomes() = [{}]".format(", ".join(
          "{{{}, {:.12f}}}".format(outcome, prob)
          for outcome, prob in state.chance_outcomes())))
    if state.is_simultaneous_node():
      for player in players:
        add_line("LegalActions({}) = [{}]".format(
            player, ", ".join(str(x) for x in state.legal_actions(player))))
      for player in players:
        add_line("StringLegalActions({}) = [{}]".format(
            player, ", ".join('"{}"'.format(state.action_to_string(player, x))
                              for x in state.legal_actions(player))))
      actions = [rng.choice(state.legal_actions(player)) for player in players]
      add_line("")
      add_line("# Apply joint action [{}]".format(
          format(", ".join(
              '"{}"'.format(state.action_to_string(player, action))
              for player, action in enumerate(actions)))))
      add_line("actions: [{}]".format(", ".join(
          str(action) for action in actions)))
      state.apply_actions(actions)
    else:
      add_line("LegalActions() = [{}]".format(", ".join(
          str(x) for x in state.legal_actions())))
      add_line("StringLegalActions() = [{}]".format(", ".join(
          '"{}"'.format(state.action_to_string(state.current_player(), x))
          for x in state.legal_actions())))
      action = rng.choice(state.legal_actions())
      add_line("")
      add_line('# Apply action "{}"'.format(
          state.action_to_string(state.current_player(), action)))
      add_line("action: {}".format(action))
      state.apply_action(action)
    state_idx += 1
  return lines


def content_lines(lines):
  """Return lines with content."""
  return [line for line in lines if line and line[0] == "#"]


def _playthrough_params(lines):
  """Returns the playthrough parameters from a playthrough record.

  Args:
    lines: The playthrough as a list of lines.

  Returns:
    A `dict` with entries:
      game_string: string, e.g. 'markov_soccer'.
      seed: an optional integerString to seed the random number generator
        governing action choices.
    Suitable for passing to playthrough to re-generate the playthrough.

  Raises:
    ValueError if the playthrough is not valid.
  """
  params = dict()
  for line in lines:
    match_game = re.match(r"^game: (.*)$", line)
    match_seed = re.match(r"^seed: (.*)$", line)
    if match_game:
      params["game_string"] = match_game.group(1)
    if match_seed:
      params["seed"] = int(match_seed.group(1))
    if "game_string" in params and "seed" in params:
      return params
  raise ValueError("Could not find params")


def replay(filename):
  """Re-runs the playthrough in the specified file. Returns (original, new)."""
  with open(filename, "r") as f:
    original = f.read()
  kwargs = _playthrough_params(original.splitlines())
  return (original, playthrough(**kwargs))


def update_path(path):
  """Regenerates all playthroughs in the path."""
  for filename in os.listdir(path):
    try:
      original, new = replay(os.path.join(path, filename))
      if original == new:
        print("        {}".format(filename))
      else:
        with open(os.path.join(path, filename), "w") as f:
          f.write(new)
        print("Updated {}".format(filename))
    except Exception as e:  # pylint: disable=broad-except
      print("{} failed: {}".format(filename, e))
      raise
