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

# Lint as python3
"""Functions to manipulate game playthoughs.

Used by examples/playthrough.py and tests/playthrough_test.py.

Note that not all states are fully represented in the playthrough.
See the logic in ShouldDisplayStateTracker for details.
"""

import collections
import os
import re
from typing import Optional

import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.mfg import games as mfgs  # pylint: disable=unused-import
from open_spiel.python.observation import make_observation
import pyspiel


def _escape(x):
  """Returns a newline-free backslash-escaped version of the given string."""
  x = x.replace("\\", R"\\")
  x = x.replace("\n", R"\n")
  return x


def _format_value(v):
  """Format a single value."""
  if v == 0:
    return "◯"
  elif v == 1:
    return "◉"
  else:
    return ValueError("Values must all be 0 or 1")


def _format_vec(vec):
  """Returns a readable format for a vector."""
  full_fmt = "".join(_format_value(v) for v in vec)
  short_fmt = None
  max_len = 250
  vec2int = lambda vec: int("".join("1" if b else "0" for b in vec), 2)
  if len(vec) > max_len:
    if all(v == 0 for v in vec):
      short_fmt = f"zeros({len(vec)})"
    elif all(v in (0, 1) for v in vec):
      sz = (len(vec) + 15) // 16
      # To reconstruct the original vector:
      # binvec = lambda n, x: [int(x) for x in f"{x:0>{n}b}"]
      short_fmt = f"binvec({len(vec)}, 0x{vec2int(vec):0>{sz}x})"
  if short_fmt and len(short_fmt) < len(full_fmt):
    return short_fmt
  else:
    return full_fmt


def _format_matrix(mat):
  return np.char.array([_format_vec(row) for row in mat])


def _format_tensor(tensor, tensor_name, max_cols=120):
  """Formats a tensor in an easy-to-view format as a list of lines."""
  if ((not tensor.shape) or (tensor.shape == (0,)) or (len(tensor.shape) > 3) or
      not np.logical_or(tensor == 0, tensor == 1).all()):
    vec = ", ".join(str(round(v, 5)) for v in tensor.ravel())
    return ["{} = [{}]".format(tensor_name, vec)]
  elif len(tensor.shape) == 1:
    return ["{}: {}".format(tensor_name, _format_vec(tensor))]
  elif len(tensor.shape) == 2:
    if len(tensor_name) + tensor.shape[0] + 2 < max_cols:
      lines = ["{}: {}".format(tensor_name, _format_vec(tensor[0]))]
      prefix = " " * (len(tensor_name) + 2)
    else:
      lines = ["{}:".format(tensor_name), _format_vec(tensor[0])]
      prefix = ""
    for row in tensor[1:]:
      lines.append(prefix + _format_vec(row))
    return lines
  elif len(tensor.shape) == 3:
    lines = ["{}:".format(tensor_name)]
    rows = []
    for m in tensor:
      formatted_matrix = _format_matrix(m)
      if (not rows) or (len(rows[-1][0] + formatted_matrix[0]) + 2 > max_cols):
        rows.append(formatted_matrix)
      else:
        rows[-1] = rows[-1] + "  " + formatted_matrix
    for i, big_row in enumerate(rows):
      if i > 0:
        lines.append("")
      for row in big_row:
        lines.append("".join(row))
    return lines


def playthrough(game_string,
                action_sequence,
                alsologtostdout=False,
                observation_params_string=None,
                seed: Optional[int] = None):
  """Returns a playthrough of the specified game as a single text.

  Actions are selected uniformly at random, including chance actions.

  Args:
    game_string: string, e.g. 'markov_soccer', with possible optional params,
      e.g. 'go(komi=4.5,board_size=19)'.
    action_sequence: A (possibly partial) list of action choices to make.
    alsologtostdout: Whether to also print the trace to stdout. This can be
      useful when an error occurs, to still be able to get context information.
    observation_params_string: Optional observation parameters for constructing
      an observer.
    seed: A(n optional) seed to initialize the random number generator from.
  """
  lines = playthrough_lines(game_string, alsologtostdout, action_sequence,
                            observation_params_string, seed)
  return "\n".join(lines) + "\n"


def format_shapes(d):
  """Returns a string representing the shapes of a dict of tensors."""
  if len(d) == 1:
    return str(list(d[min(d)].shape))
  else:
    return ", ".join(f"{key}: {list(value.shape)}" for key, value in d.items())


def _format_params(d, as_game=False):
  """Format a collection of params."""

  def fmt(val):
    if isinstance(val, dict):
      return _format_params(val, as_game=True)
    else:
      return _escape(str(val))

  if as_game:
    return d["name"] + "(" + ",".join(
        "{}={}".format(key, fmt(value))
        for key, value in sorted(d.items())
        if key != "name") + ")"
  else:
    return "{" + ",".join(
        "{}={}".format(key, fmt(value))
        for key, value in sorted(d.items())) + "}"


class ShouldDisplayStateTracker:
  """Determines whether a state is interesting enough to display."""

  def __init__(self):
    self.states_by_player = collections.defaultdict(int)

  def __call__(self, state) -> bool:
    """Returns True if a state is sufficiently interesting to display."""
    player = state.current_player()
    count = self.states_by_player[player]
    self.states_by_player[player] += 1
    if count == 0:
      # Always display the first state for a player
      return True
    elif player == -1:
      # For chance moves, display the first two only
      return count < 2
    else:
      # For regular player moves, display the first three and selected others
      return (count < 3) or (count % 10 == 0)


def playthrough_lines(game_string, alsologtostdout=False, action_sequence=None,
                      observation_params_string=None,
                      seed: Optional[int] = None):
  """Returns a playthrough of the specified game as a list of lines.

  Actions are selected uniformly at random, including chance actions.

  Args:
    game_string: string, e.g. 'markov_soccer' or 'kuhn_poker(players=4)'.
    alsologtostdout: Whether to also print the trace to stdout. This can be
      useful when an error occurs, to still be able to get context information.
    action_sequence: A (possibly partial) list of action choices to make.
    observation_params_string: Optional observation parameters for constructing
      an observer.
    seed: A(n optional) seed to initialize the random number generator from.
  """
  should_display_state_fn = ShouldDisplayStateTracker()
  lines = []
  action_sequence = action_sequence or []
  should_display = True

  def add_line(v, force=False):
    if force or should_display:
      if alsologtostdout:
        print(v)
      lines.append(v)

  game = pyspiel.load_game(game_string)
  add_line("game: {}".format(game_string))
  if observation_params_string:
    add_line("observation_params: {}".format(observation_params_string))
  if seed is None:
    seed = np.random.randint(2**32 - 1)
  game_type = game.get_type()

  default_observation = None
  try:
    observation_params = pyspiel.game_parameters_from_string(
        observation_params_string) if observation_params_string else None
    default_observation = make_observation(
        game,
        imperfect_information_observation_type=None,
        params=observation_params)
  except (RuntimeError, ValueError) as e:
    print("Warning: unable to build an observation: ", e)

  infostate_observation = None
  # TODO(author11) reinstate this restriction
  # if game_type.information in (pyspiel.IMPERFECT_INFORMATION,
  #                              pyspiel.ONE_SHOT):
  try:
    infostate_observation = make_observation(
        game, pyspiel.IIGObservationType(perfect_recall=True))
  except (RuntimeError, ValueError):
    pass

  public_observation = None
  private_observation = None

  # Instantiate factored observations only for imperfect information games,
  # as it would yield unncessarily redundant information for perfect info games.
  # The default observation is the same as the public observation, while private
  # observations are always empty.
  if game_type.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION:
    try:
      public_observation = make_observation(
          game,
          pyspiel.IIGObservationType(
              public_info=True,
              perfect_recall=False,
              private_info=pyspiel.PrivateInfoType.NONE))
    except (RuntimeError, ValueError):
      pass
    try:
      private_observation = make_observation(
          game,
          pyspiel.IIGObservationType(
              public_info=False,
              perfect_recall=False,
              private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
    except (RuntimeError, ValueError):
      pass

  add_line("")
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
  add_line("GameType.provides_factored_observation_string = {}".format(
      game_type.provides_factored_observation_string))
  add_line("GameType.reward_model = {}".format(game_type.reward_model))
  add_line("GameType.short_name = {}".format('"{}"'.format(
      game_type.short_name)))
  add_line("GameType.utility = {}".format(game_type.utility))

  add_line("")
  add_line("NumDistinctActions() = {}".format(game.num_distinct_actions()))
  add_line("PolicyTensorShape() = {}".format(game.policy_tensor_shape()))
  add_line("MaxChanceOutcomes() = {}".format(game.max_chance_outcomes()))
  add_line("GetParameters() = {}".format(_format_params(game.get_parameters())))
  add_line("NumPlayers() = {}".format(game.num_players()))
  add_line("MinUtility() = {:.5}".format(game.min_utility()))
  add_line("MaxUtility() = {:.5}".format(game.max_utility()))
  try:
    utility_sum = game.utility_sum()
  except RuntimeError:
    utility_sum = None
  add_line("UtilitySum() = {}".format(utility_sum))
  if infostate_observation and infostate_observation.tensor is not None:
    add_line("InformationStateTensorShape() = {}".format(
        format_shapes(infostate_observation.dict)))
    add_line("InformationStateTensorLayout() = {}".format(
        game.information_state_tensor_layout()))
    add_line("InformationStateTensorSize() = {}".format(
        len(infostate_observation.tensor)))
  if default_observation and default_observation.tensor is not None:
    add_line("ObservationTensorShape() = {}".format(
        format_shapes(default_observation.dict)))
    add_line("ObservationTensorLayout() = {}".format(
        game.observation_tensor_layout()))
    add_line("ObservationTensorSize() = {}".format(
        len(default_observation.tensor)))
  add_line("MaxGameLength() = {}".format(game.max_game_length()))
  add_line('ToString() = "{}"'.format(str(game)))

  players = list(range(game.num_players()))
  # Arbitrarily pick the last possible initial states (for all games
  # but multi-population MFGs, there will be a single initial state).
  state = game.new_initial_states()[-1]
  state_idx = 0
  rng = np.random.RandomState(seed)

  while True:
    should_display = should_display_state_fn(state)
    add_line("", force=True)
    add_line("# State {}".format(state_idx), force=True)
    for line in str(state).splitlines():
      add_line("# {}".format(line).rstrip())
    add_line("IsTerminal() = {}".format(state.is_terminal()))
    add_line("History() = {}".format([int(a) for a in state.history()]))
    add_line('HistoryString() = "{}"'.format(state.history_str()))
    add_line("IsChanceNode() = {}".format(state.is_chance_node()))
    add_line("IsSimultaneousNode() = {}".format(state.is_simultaneous_node()))
    add_line("CurrentPlayer() = {}".format(state.current_player()))
    if infostate_observation:
      for player in players:
        s = infostate_observation.string_from(state, player)
        if s is not None:
          add_line(f'InformationStateString({player}) = "{_escape(s)}"')
    if infostate_observation and infostate_observation.tensor is not None:
      for player in players:
        infostate_observation.set_from(state, player)
        for name, tensor in infostate_observation.dict.items():
          label = f"InformationStateTensor({player})"
          label += f".{name}" if name != "info_state" else ""
          for line in _format_tensor(tensor, label):
            add_line(line)
    if default_observation:
      for player in players:
        s = default_observation.string_from(state, player)
        if s is not None:
          add_line(f'ObservationString({player}) = "{_escape(s)}"')
    if public_observation:
      s = public_observation.string_from(state, 0)
      if s is not None:
        add_line('PublicObservationString() = "{}"'.format(_escape(s)))
      for player in players:
        s = private_observation.string_from(state, player)
        if s is not None:
          add_line(f'PrivateObservationString({player}) = "{_escape(s)}"')
    if default_observation and default_observation.tensor is not None:
      for player in players:
        default_observation.set_from(state, player)
        for name, tensor in default_observation.dict.items():
          label = f"ObservationTensor({player})"
          label += f".{name}" if name != "observation" else ""
          for line in _format_tensor(tensor, label):
            add_line(line)
    if game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC:
      add_line('SerializeState() = "{}"'.format(_escape(state.serialize())))
    if not state.is_chance_node():
      add_line("Rewards() = {}".format(state.rewards()))
      add_line("Returns() = {}".format(state.returns()))
    if state.is_terminal():
      break
    if state.is_chance_node():
      add_line("ChanceOutcomes() = {}".format(state.chance_outcomes()))
    if state.is_mean_field_node():
      add_line("DistributionSupport() = {}".format(
          state.distribution_support()))
      num_states = len(state.distribution_support())
      state.update_distribution(
          [1. / num_states] * num_states if num_states else [])
      if state_idx < len(action_sequence):
        assert action_sequence[state_idx] == "update_distribution", (
            f"Unexpected action at MFG node: {action_sequence[state_idx]}, "
            f"state: {state}, action_sequence: {action_sequence}")
      add_line("")
      add_line("# Set mean field distribution to be uniform", force=True)
      add_line("action: update_distribution", force=True)
    elif state.is_simultaneous_node():
      for player in players:
        add_line("LegalActions({}) = [{}]".format(
            player, ", ".join(str(x) for x in state.legal_actions(player))))
      for player in players:
        add_line("StringLegalActions({}) = [{}]".format(
            player, ", ".join('"{}"'.format(state.action_to_string(player, x))
                              for x in state.legal_actions(player))))
      if state_idx < len(action_sequence):
        actions = action_sequence[state_idx]
      else:
        actions = []
        for pl in players:
          legal_actions = state.legal_actions(pl)
          actions.append(0 if not legal_actions else rng.choice(legal_actions))
      add_line("")
      add_line("# Apply joint action [{}]".format(
          format(", ".join(
              '"{}"'.format(state.action_to_string(player, action))
              for player, action in enumerate(actions)))), force=True)
      add_line("actions: [{}]".format(", ".join(
          str(action) for action in actions)), force=True)
      state.apply_actions(actions)
    else:
      add_line("LegalActions() = [{}]".format(", ".join(
          str(x) for x in state.legal_actions())))
      add_line("StringLegalActions() = [{}]".format(", ".join(
          '"{}"'.format(state.action_to_string(state.current_player(), x))
          for x in state.legal_actions())))
      if state_idx < len(action_sequence):
        action = action_sequence[state_idx]
      else:
        action = rng.choice(state.legal_actions())
      add_line("")
      add_line('# Apply action "{}"'.format(
          state.action_to_string(state.current_player(), action)), force=True)
      add_line("action: {}".format(action), force=True)
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
      action_sequence: a list of action choices made in the playthrough.
    Suitable for passing to playthrough to re-generate the playthrough.

  Raises:
    ValueError if the playthrough is not valid.
  """
  params = {"action_sequence": []}
  for line in lines:
    match_game = re.match(r"^game: (.*)$", line)
    match_observation_params = re.match(r"^observation_params: (.*)$", line)
    match_action = re.match(r"^action: (.*)$", line)
    match_actions = re.match(r"^actions: \[(.*)\]$", line)
    if match_game:
      params["game_string"] = match_game.group(1)
    if match_observation_params:
      params["observation_params_string"] = match_observation_params.group(1)
    if match_action:
      matched = match_action.group(1)
      params["action_sequence"].append(matched if matched ==
                                       "update_distribution" else int(matched))
    if match_actions:
      params["action_sequence"].append(
          [int(x) for x in match_actions.group(1).split(", ")])
  if "game_string" in params:
    return params
  raise ValueError("Could not find params")


def _read_playthrough(filename):
  """Returns the content and the parsed arguments of a playthrough file."""
  with open(filename, "r", encoding="utf-8") as f:
    original = f.read()
  kwargs = _playthrough_params(original.splitlines())
  return original, kwargs


def replay(filename):
  """Re-runs the playthrough in the specified file. Returns (original, new)."""
  original, kwargs = _read_playthrough(filename)
  return (original, playthrough(**kwargs))


def update_path(path, shard_index=0, num_shards=1):
  """Regenerates all playthroughs in the path."""
  for filename in sorted(os.listdir(path))[shard_index::num_shards]:
    try:
      original, kwargs = _read_playthrough(os.path.join(path, filename))
      try:
        pyspiel.load_game(kwargs["game_string"])
      except pyspiel.SpielError as e:
        if "Unknown game" in str(e):
          print("[Skipped] Skipping game ", filename, " as ",
                kwargs["game_string"], " is not available.")
          continue
        else:
          raise
      new = playthrough(**kwargs)
      if original == new:
        print("        {}".format(filename))
      else:
        with open(os.path.join(path, filename), "w") as f:
          f.write(new)
        print("Updated {}".format(filename))
    except Exception as e:  # pylint: disable=broad-except
      print("{} failed: {}".format(filename, e))
      raise
