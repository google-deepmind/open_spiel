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
ContractVersion = Literal[7]
CONTRACT_VERSION: ContractVersion = 7

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
      reset or a change of that seat's controller. A seat whose controller
      could not be built (`SeatStatus.UNAVAILABLE`) resumes only after a
      change of its controller.
    TERMINAL: The game is over.
  """

  PLAYING = "playing"
  PAUSED = "paused"
  TERMINAL = "terminal"


class SeatStatus(enum.StrEnum):
  """Whether a seat's controller is ready to play.

  Attributes:
    READY: The controller is built and plays the seat; a human seat is always
      ready.
    PREPARING: The server is building the controller, which can take a while
      (a trained model loads and warms up). The seat does not move until it
      is ready; then autoplay goes on.
    UNAVAILABLE: The controller could not be built, or was not built within
      the build budget. The seat has a `failure`, so the session is paused
      until the seat is given a controller again.
  """

  READY = "ready"
  PREPARING = "preparing"
  UNAVAILABLE = "unavailable"


class ChancePolicy(enum.StrEnum):
  """How chance nodes are resolved.

  Attributes:
    AUTO: The server samples each outcome from a seeded generator.
    EXPOSE: The client chooses each outcome; in a game that hides
      information, only from the oracle view.
  """

  AUTO = "auto"
  EXPOSE = "expose"


class MctsLimits(enum.StrEnum):
  """How the served pyspiel's C++ MCTS (`pyspiel.MCTSBot`) bounds a search.

  The server finds out once, at startup, and picks MCTS's limits by it.

  Attributes:
    SIMULATIONS_OR_CLOCK: A search stops at its simulation count or its
      wall-clock time, whichever comes first. MCTS runs up to 10000
      simulations, 1000 by default, with a wall-clock backstop a little
      inside the move budget.
    SIMULATIONS_ONLY: A wall-clock time replaces the simulation count rather
      than bounding it. MCTS sets no wall-clock time, and its simulations are
      capped at 250, which fits the move budget on the slowest catalogued
      game.
  """

  SIMULATIONS_OR_CLOCK = "simulations_or_clock"
  SIMULATIONS_ONLY = "simulations_only"


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
      controller must satisfy; `{}` when it takes no settings. Each of its
      `properties` gives a setting's `type`, `default` and constraints; a
      setting listed in its `required` has no `default`, and a seat must be
      given it.
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
    status: Whether the controller is ready, being built, or could not be
      built.
    message: Why the seat is not ready, for every viewer: what the server is
      doing while it is `preparing`, or why it is `unavailable`; null when it
      is `ready`, and optional while it is `preparing`.
  """

  seat: int
  controller: str
  config: dict[str, ParameterValue]
  autoplay: bool
  failure: ControllerFailure | None
  status: SeatStatus = SeatStatus.READY
  message: str | None = None

  def __post_init__(self) -> None:
    require(
        not (self.autoplay and self.controller == HUMAN),
        f"Seat {self.seat} is human, so it cannot autoplay.",
    )
    require(
        self.status == SeatStatus.READY or self.controller != HUMAN,
        f"Seat {self.seat} is human, so it is always ready.",
    )
    require(
        self.status != SeatStatus.READY or self.message is None,
        f"Seat {self.seat} is ready, so it has no message.",
    )
    require(
        self.status != SeatStatus.PREPARING or self.failure is None,
        f"Seat {self.seat} is preparing, so it has not failed.",
    )
    require(
        self.status != SeatStatus.UNAVAILABLE
        or (self.failure is not None and self.message is not None),
        f"Seat {self.seat} is unavailable, so it has a failure and a message.",
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
class ActionHint:
  """What a game's adapter says about one legal action, for its renderer.

  Attributes:
    id: The legal action's OpenSpiel id.
    hint: The adapter's description of the action, such as the card it plays
      or the cells it joins.
  """

  id: int
  hint: Json


@dataclasses.dataclass(frozen=True)
class Presentation:
  """A position as the game's adapter presents it to one viewer.

  Attributes:
    renderer: The name of the client renderer that draws the view.
    view: The adapter's view of the position, holding only what the viewer
      may see.
    actions: The adapter's hints for the legal actions, each naming an action
      of the frame's `legal_actions` at most once; null exactly when the
      frame's `legal_actions` is null.
  """

  renderer: str
  view: Json
  actions: tuple[ActionHint, ...] | None


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
    public_label: What every viewer may read of a withheld action, as the
      game's adapter says, such as a card discarded face up; null when the
      action is shown in full or wholly hidden.
  """

  seat: int | None
  action: int | None
  label: str | None
  rewards: tuple[float, ...] | None
  public_label: str | None = None

  def __post_init__(self) -> None:
    require(
        (self.action is None) == (self.label is None) == (self.rewards is None),
        "A history entry withholds its action, label and rewards together.",
    )
    require(
        self.public_label is None or self.action is None,
        "Only a withheld history entry has a public label.",
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
