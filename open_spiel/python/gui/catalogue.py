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

"""The catalogue: every game the server knows, and whether it can serve it."""

import dataclasses
import enum

from open_spiel.python.gui import contract


class Dynamics(enum.StrEnum):
  """How players move, as the game declares it.

  Attributes:
    SEQUENTIAL: One player acts at a time.
    SIMULTANEOUS: Players act together; served through the turn-based wrapper.
    MEAN_FIELD: A mean-field game.
  """

  SEQUENTIAL = "sequential"
  SIMULTANEOUS = "simultaneous"
  MEAN_FIELD = "mean_field"


class ChanceMode(enum.StrEnum):
  """How the game uses chance, as the game declares it.

  Attributes:
    DETERMINISTIC: The game has no chance nodes.
    EXPLICIT_STOCHASTIC: Chance nodes list their outcomes and probabilities.
    SAMPLED_STOCHASTIC: Chance outcomes are sampled inside the state.
  """

  DETERMINISTIC = "deterministic"
  EXPLICIT_STOCHASTIC = "explicit_stochastic"
  SAMPLED_STOCHASTIC = "sampled_stochastic"


class Information(enum.StrEnum):
  """What the players observe, as the game declares it.

  Attributes:
    ONE_SHOT: A one-shot game.
    PERFECT_INFORMATION: Every player observes the full state.
    IMPERFECT_INFORMATION: Some state is hidden from some players.
  """

  ONE_SHOT = "one_shot"
  PERFECT_INFORMATION = "perfect_information"
  IMPERFECT_INFORMATION = "imperfect_information"


class ParameterType(enum.StrEnum):
  """The type of a game parameter's value.

  Attributes:
    BOOL: A boolean.
    INT: An integer.
    FLOAT: A real number.
    STRING: A string.
  """

  BOOL = "bool"
  INT = "int"
  FLOAT = "float"
  STRING = "string"


class ParameterKind(enum.StrEnum):
  """Which values a game parameter admits.

  Attributes:
    FREE: Any value of its type.
    RANGE: A number between `minimum` and `maximum`, inclusive.
    CHOICE: One of `choices`.
  """

  FREE = "free"
  RANGE = "range"
  CHOICE = "choice"


_PYTHON_TYPES = {
    ParameterType.BOOL: (bool,),
    ParameterType.INT: (int,),
    ParameterType.FLOAT: (int, float),
    ParameterType.STRING: (str,),
}


def _has_type(value: contract.ParameterValue, kind: ParameterType) -> bool:
  """Whether `value` has type `kind`; a boolean is not a number."""
  if isinstance(value, bool) and kind != ParameterType.BOOL:
    return False
  return isinstance(value, _PYTHON_TYPES[kind])


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
  """A game parameter, as a client needs it to offer the parameter.

  Attributes:
    name: The parameter's name, a key of a session's `parameters`.
    type: The type of the parameter's value.
    kind: Which values the parameter admits.
    default: The value the game uses when none is given.
    minimum: The least admitted value when `kind` is `range`, otherwise null.
    maximum: The greatest admitted value when `kind` is `range`, otherwise
      null.
    choices: The admitted values when `kind` is `choice`, otherwise null.
    description: What the parameter controls, or null when the game does not
      say.
  """

  name: str
  type: ParameterType
  kind: ParameterKind
  default: contract.ParameterValue
  minimum: float | None
  maximum: float | None
  choices: tuple[contract.ParameterValue, ...] | None
  description: str | None

  def __post_init__(self) -> None:
    contract.require(
        _has_type(self.default, self.type),
        f"Parameter {self.name} defaults to {self.default!r}, not a "
        f"{self.type}.",
    )
    is_range = self.kind == ParameterKind.RANGE
    contract.require(
        is_range == (self.minimum is not None) == (self.maximum is not None),
        f"Parameter {self.name} has bounds exactly when it is a range.",
    )
    if self.minimum is not None and self.maximum is not None:
      contract.require(
          self.type in (ParameterType.INT, ParameterType.FLOAT)
          and not isinstance(self.default, (bool, str))
          and self.minimum <= self.default <= self.maximum,
          f"Parameter {self.name} is a numeric range around its default.",
      )
    contract.require(
        (self.kind == ParameterKind.CHOICE) == (self.choices is not None),
        f"Parameter {self.name} has choices exactly when it is a choice.",
    )
    if self.choices is not None:
      contract.require(
          self.default in self.choices
          and all(_has_type(choice, self.type) for choice in self.choices),
          f"Parameter {self.name} chooses its default among {self.type}s.",
      )


@dataclasses.dataclass(frozen=True)
class CatalogueEntry:
  """A game the server knows, and whether it can serve it.

  Attributes:
    name: The game's short name, which a client sends to create a session.
    long_name: The game's display name.
    dynamics: How players move.
    chance_mode: How the game uses chance.
    information: What the players observe.
    min_seats: The fewest seats the game supports.
    max_seats: The most seats the game supports.
    num_players: The number of seats the game has with its default
      parameters, between `min_seats` and `max_seats`; null when not
      loadable.
    parameters: The game's parameters, ordered by name.
    loadable: Whether the server can serve the game.
    unavailable_reason: Why the server cannot serve the game; null when
      loadable.
    controllers: The `Controller.name`s that can play the game with its
      default parameters; empty when not loadable.
    renderer: The renderer its frames name in `Presentation.renderer`, from
      the game's registered adapter: the board a client draws it with, if
      the client has that renderer. Null for a game with no adapter, and for
      one that is not loadable; such a game is shown as its text.
  """

  name: str
  long_name: str
  dynamics: Dynamics
  chance_mode: ChanceMode
  information: Information
  min_seats: int
  max_seats: int
  num_players: int | None
  parameters: tuple[ParameterSpec, ...]
  loadable: bool
  unavailable_reason: str | None
  controllers: tuple[str, ...]
  renderer: str | None

  def __post_init__(self) -> None:
    contract.require(
        1 <= self.min_seats <= self.max_seats,
        f"{self.name} cannot support {self.min_seats} to {self.max_seats} "
        "seats.",
    )
    contract.require(
        self.loadable == (self.num_players is not None),
        f"{self.name} gives its default seat count exactly when loadable.",
    )
    contract.require(
        self.num_players is None
        or self.min_seats <= self.num_players <= self.max_seats,
        f"{self.name} has {self.num_players} default seats, outside "
        f"{self.min_seats} to {self.max_seats}.",
    )
    names = [parameter.name for parameter in self.parameters]
    contract.require(
        names == sorted(set(names)),
        f"{self.name} lists its parameters once each, ordered by name.",
    )
    contract.require(
        self.loadable == (self.unavailable_reason is None),
        f"{self.name} gives a reason exactly when it is not loadable.",
    )
    contract.require(
        self.loadable == (contract.HUMAN in self.controllers),
        f"{self.name} offers the human controller exactly when loadable.",
    )
    contract.require(
        self.loadable or not self.controllers,
        f"{self.name} is not loadable, so it offers no controllers.",
    )
    contract.require(
        self.renderer is None or (self.loadable and bool(self.renderer)),
        f"{self.name} names a renderer only when loadable, and never ''.",
    )


@dataclasses.dataclass(frozen=True)
class Catalogue:
  """Every game the server knows.

  Attributes:
    contract_version: The contract version the catalogue was built against.
    games: One entry per game, ordered by name.
  """

  contract_version: contract.ContractVersion = dataclasses.field(
      default=contract.CONTRACT_VERSION, kw_only=True
  )
  games: tuple[CatalogueEntry, ...]

  def __post_init__(self) -> None:
    names = [game.name for game in self.games]
    contract.require(
        names == sorted(set(names)), "Games are unique, ordered by name."
    )
