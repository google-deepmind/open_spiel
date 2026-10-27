# Copyright 2026 DeepMind Technologies Limited
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


"""What a loaded game reports: its session parameters and its structs.

`loader` loads games; this module holds what it asks of a loaded game: the
parameters a session reports, and whether the game's states and observations
serialise to structured JSON, probed once per game and parameters.
"""

from collections.abc import Mapping
import enum
import threading

from open_spiel.python.gui import contract
import pyspiel


def session_parameters(
    game_type: pyspiel.GameType,
    requested: Mapping[str, contract.ParameterValue],
    echoed: Mapping[str, contract.ParameterValue],
) -> dict[str, contract.ParameterValue]:
  """Every declared parameter's value in effect, as the session reports it.

  A game's `get_parameters()` can leave out parameters its type declares (a
  game played as a match declares the match's parameters and does not echo
  them), and the client needs a value for each one it offers. So a value
  echoed by the game wins, then one requested, then the declared default.

  Args:
    game_type: The game's type.
    requested: The parameters the session was opened with.
    echoed: What the loaded game's `get_parameters()` returned.

  Returns:
    The parameters, every scalar declared one included.
  """
  merged: dict[str, contract.ParameterValue] = {
      name: default
      for name, default in game_type.parameter_specification.items()
      if isinstance(default, (bool, int, float, str))
  }
  merged.update(requested)
  merged.update(echoed)
  return merged


class StructKind(enum.Enum):
  """The structs a game may provide.

  Attributes:
    STATE: The full state, from `State::ToStruct`.
    OBSERVATION: A player's observation, from `State::ToObservationStruct`.
  """

  STATE = "state"
  OBSERVATION = "observation"


# Whether each game, under given parameters, provides each kind of struct;
# keyed by `_struct_key`. pyspiel offers no non-throwing check (a game without
# the struct API throws from `State::ToStruct` or
# `State::ToObservationStruct`, and pyspiel prints every C++ SpielError it
# raises: "OpenSpiel exception: ToStruct is not implemented."), so each game is
# probed once per kind, on the first frame that asks, and the answer is kept
# here. The catalogue, which loads every game, never asks, so building it
# probes nothing.
_STRUCT_SUPPORT: dict[tuple[object, ...], bool] = {}
_STRUCT_SUPPORT_LOCK = threading.Lock()


def _struct_key(
    kind: StructKind,
    name: str,
    turn_based: bool,
    parameters: Mapping[str, contract.ParameterValue],
) -> tuple[object, ...]:
  """The cache key: the kind, and the game as served with its parameters."""
  return (
      kind,
      name,
      turn_based,
      tuple(sorted(parameters.items())),
  )


def struct_support(
    kind: StructKind,
    *,
    name: str,
    turn_based: bool,
    parameters: Mapping[str, contract.ParameterValue],
    game: pyspiel.Game,
) -> bool:
  """Whether the game provides the struct `kind`; probed once, then kept.

  Args:
    kind: The struct asked about.
    name: The game's short name.
    turn_based: Whether the game is served through the turn-based wrapper.
    parameters: The game's parameters, defaults included.
    game: The game as served.

  Returns:
    Whether the game's states provide the struct.
  """
  key = _struct_key(kind, name, turn_based, parameters)
  with _STRUCT_SUPPORT_LOCK:
    known = _STRUCT_SUPPORT.get(key)
    if known is None:
      state = game.new_initial_state()
      known = (
          _probe_struct(state)
          if kind == StructKind.STATE
          else _probe_observation_struct(state)
      )
      _STRUCT_SUPPORT[key] = known
  return known


def _probe_struct(state: pyspiel.State) -> bool:
  """Asks a state for its structured JSON; the one probe per game."""
  try:
    state.to_json()
  except pyspiel.SpielError:
    return False
  return True


def _probe_observation_struct(state: pyspiel.State) -> bool:
  """Asks a state for player 0's observation struct; one probe per game."""
  try:
    state.to_observation_struct(0).to_json()
  except pyspiel.SpielError:
    return False
  return True
