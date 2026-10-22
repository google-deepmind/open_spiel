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

"""The wire contract between the game server and its clients.

This module holds what every part of the contract shares, and the parts of a
session that a `frame.Frame` is built from. Every response body is a
`frame.Frame`, a `catalogue.Catalogue`, a `health.Health` or an
`errors.Error`, serialised with `to_json`; request bodies are defined in
`request.py`. `contract.schema.json` is generated from these definitions by
`schema.py`, and clients generate their types from it.

Constructors enforce the invariants stated in CONTRACT.md, so a value that
breaks them cannot be built.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Annotated, Literal

# The contract's version, incremented on every change that breaks clients.
ContractVersion = Literal[2]
CONTRACT_VERSION: ContractVersion = 2

# The value of a game parameter or a controller setting.
ParameterValue = bool | int | float | str

# A value as it appears on the wire.
JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["JsonValue"]
    | Mapping[str, "JsonValue"]
)

# A field that holds any JSON value; `to_json` checks it and the schema
# leaves it open.
ANY_JSON = "any JSON value"
Json = Annotated[object, ANY_JSON]

# The name of the controller that leaves a seat's moves to the client.
HUMAN = "human"


def require(condition: bool, message: str) -> None:
  """Raises `ValueError` with `message` unless the invariant holds."""
  if not condition:
    raise ValueError(message)


class ViewerKind(enum.StrEnum):
  """Who a frame is built for.

  Attributes:
    SEAT: A player, who sees what that player may observe.
    SPECTATOR: A watcher in no seat, who sees only public state.
    ORACLE: An omniscient observer, who sees the full state.
  """

  SEAT = "seat"
  SPECTATOR = "spectator"
  ORACLE = "oracle"


class MoverKind(enum.StrEnum):
  """Who acts at a position.

  Attributes:
    SEAT: A seat chooses the next action.
    CHANCE: Chance chooses the next action.
    TERMINAL: The game is over and nobody acts.
  """

  SEAT = "seat"
  CHANCE = "chance"
  TERMINAL = "terminal"


class SessionStatus(enum.StrEnum):
  """The state of a session's live game.

  Attributes:
    PLAYING: The game is in progress.
    PAUSED: A seat's controller failed; play resumes only after an undo, a
      reset or a change of that seat's controller.
    TERMINAL: The game is over.
  """

  PLAYING = "playing"
  PAUSED = "paused"
  TERMINAL = "terminal"


class ChancePolicy(enum.StrEnum):
  """How chance nodes are resolved.

  Attributes:
    AUTO: The server samples each outcome from a seeded generator.
    EXPOSE: The client chooses each outcome; in a game that hides
      information, only from the oracle view.
  """

  AUTO = "auto"
  EXPOSE = "expose"


@dataclasses.dataclass(frozen=True)
class Viewer:
  """The audience a frame is built for.

  Attributes:
    kind: Whether the viewer holds a seat, watches, or sees everything.
    seat: The viewer's seat when `kind` is `seat`, otherwise null.
  """

  kind: ViewerKind
  seat: int | None = None

  def __post_init__(self) -> None:
    require(
        (self.kind == ViewerKind.SEAT) == (self.seat is not None),
        f"A viewer of kind {self.kind} cannot have seat {self.seat}.",
    )


@dataclasses.dataclass(frozen=True)
class ToMove:
  """Who acts at a position.

  Attributes:
    kind: Whether a seat or chance acts, or the game is over.
    seat: The acting seat when `kind` is `seat`, otherwise null.
  """

  kind: MoverKind
  seat: int | None = None

  def __post_init__(self) -> None:
    require(
        (self.kind == MoverKind.SEAT) == (self.seat is not None),
        f"A mover of kind {self.kind} cannot have seat {self.seat}.",
    )


@dataclasses.dataclass(frozen=True)
class Controller:
  """A kind of agent that can play a seat.

  Attributes:
    name: The stable identifier used in requests and in `SeatView.controller`.
    label: The human-readable name.
    config_schema: A JSON Schema object that a seat's `config` for this
      controller must satisfy; `{}` when it takes no settings.
    unavailable_reason: Why the controller cannot play the session's game;
      null when it can. A seat is never played by a controller with a
      reason.
  """

  name: str
  label: str
  config_schema: dict[str, Json]
  unavailable_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class ControllerFailure:
  """Why a seat's controller stopped playing.

  Attributes:
    error: The name of the exception the controller raised.
    message: The exception's message; null for every viewer that is not
      omniscient, the failing seat included, since the message may describe
      any seat's private state.
  """

  error: str
  message: str | None


@dataclasses.dataclass(frozen=True)
class SeatView:
  """One seat of a session.

  Attributes:
    seat: The seat index, which is the OpenSpiel player id.
    controller: The `Controller.name` playing the seat.
    config: The controller's settings, satisfying its `config_schema`.
      Every viewer receives them, so they hold no secret.
    autoplay: Whether the server plays the seat's moves as soon as it is to
      move, rather than on an advance request. Never true for a human seat.
    failure: Why the controller failed, or null while it plays.
  """

  seat: int
  controller: str
  config: dict[str, ParameterValue]
  autoplay: bool
  failure: ControllerFailure | None

  def __post_init__(self) -> None:
    require(
        not (self.autoplay and self.controller == HUMAN),
        f"Seat {self.seat} is human, so it cannot autoplay.",
    )


@dataclasses.dataclass(frozen=True)
class Action:
  """A legal action at a position.

  Attributes:
    id: The OpenSpiel action id, which a client sends to play it.
    label: The game's text for the action.
  """

  id: int
  label: str


@dataclasses.dataclass(frozen=True)
class ChanceOutcome:
  """A chance outcome at a chance node.

  Attributes:
    id: The OpenSpiel action id, which a client sends to choose it.
    label: The game's text for the outcome.
    probability: The outcome's probability, in (0, 1].
  """

  id: int
  label: str
  probability: float

  def __post_init__(self) -> None:
    require(
        0.0 < self.probability <= 1.0,
        f"Outcome {self.id} cannot have probability {self.probability}.",
    )


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
  """One action of the live game, as the viewer may see it.

  Attributes:
    seat: The seat that acted, or null for a chance outcome.
    action: The OpenSpiel action id, or null when withheld from the viewer.
    label: The game's text for the action, or null when withheld.
    rewards: Each seat's reward for the action, in seat order, or null when
      withheld.
  """

  seat: int | None
  action: int | None
  label: str | None
  rewards: tuple[float, ...] | None

  def __post_init__(self) -> None:
    require(
        (self.action is None) == (self.label is None) == (self.rewards is None),
        "A history entry withholds its action, label and rewards together.",
    )

  @property
  def withheld(self) -> bool:
    """Whether the entry is withheld from the viewer."""
    return self.action is None


@dataclasses.dataclass(frozen=True)
class SessionInfo:
  """The fixed properties of a session.

  Attributes:
    id: The session's identifier.
    game: The game's short name.
    parameters: The game's parameters, defaults included.
    num_seats: The number of seats.
    chance_policy: How chance nodes are resolved.
    turn_based: Whether a simultaneous game is served through the turn-based
      wrapper, which has its seats act one after another.
    hides_information: Whether some seat may not see the full state. When
      false, every viewer is omniscient.
    seed: The chance generator's seed; null for every viewer but the oracle,
      since it determines every chance outcome to come.
    controllers: The controllers that can play a seat of this game.
  """

  id: str
  game: str
  parameters: dict[str, ParameterValue]
  num_seats: int
  chance_policy: ChancePolicy
  turn_based: bool
  hides_information: bool
  seed: int | None
  controllers: tuple[Controller, ...]

  def __post_init__(self) -> None:
    require(
        self.num_seats >= 1, f"A session cannot have {self.num_seats} seats."
    )
    require(
        any(
            controller.name == HUMAN and controller.unavailable_reason is None
            for controller in self.controllers
        ),
        "A session offers the human controller.",
    )


def to_json(value: object) -> JsonValue:
  """Converts a contract value to its wire form.

  Dataclasses become objects that carry every field, nulls included; enums
  become their values; tuples become lists.

  Args:
    value: A contract dataclass, or any value one holds.

  Returns:
    The value as dicts, lists, strings, numbers, booleans and None.

  Raises:
    TypeError: If `value` holds anything else.
  """
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
        field.name: to_json(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }
  if isinstance(value, enum.Enum):
    return to_json(value.value)
  if isinstance(value, (tuple, list)):
    return [to_json(item) for item in value]
  if isinstance(value, dict):
    return {str(key): to_json(item) for key, item in value.items()}
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  raise TypeError(f"The contract cannot carry {value!r}.")
