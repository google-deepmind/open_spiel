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

"""Request bodies.

Every session route also takes the viewer in its `viewer` query parameter;
see CONTRACT.md for the routes. Constructors enforce each body's invariants;
the request decoder answers a body that breaks them with `bad_request`.
"""

import dataclasses

from open_spiel.python.gui import contract

# The most seat actions one advance applies.
MAX_ADVANCE_PLIES = 1000


@dataclasses.dataclass(frozen=True)
class SeatConfig:
  """Who plays a seat.

  Attributes:
    controller: The `Controller.name` to play the seat.
    config: The controller's settings, satisfying its `config_schema`.
    autoplay: Whether the server plays the seat's moves as soon as it is to
      move. Must be false for a human seat.
  """

  controller: str
  config: dict[str, contract.ParameterValue]
  autoplay: bool

  def __post_init__(self) -> None:
    contract.require(
        not (self.autoplay and self.controller == contract.HUMAN),
        "A human seat cannot autoplay.",
    )


@dataclasses.dataclass(frozen=True)
class CreateSession:
  """Starts a session.

  Attributes:
    game: The game's short name.
    parameters: Game parameters; omitted ones take their defaults.
    seats: Who plays each seat, in seat order, one per seat of the game.
    chance_policy: How chance nodes are resolved.
    seed: The chance generator's seed, a non-negative integer, or null for
      the server to choose one.
  """

  game: str
  parameters: dict[str, contract.ParameterValue]
  seats: tuple[SeatConfig, ...]
  chance_policy: contract.ChancePolicy
  seed: int | None

  def __post_init__(self) -> None:
    contract.require(
        self.seed is None or self.seed >= 0,
        f"A session cannot have seed {self.seed}.",
    )


@dataclasses.dataclass(frozen=True)
class ApplyAction:
  """Plays an action for the seat to move.

  Attributes:
    action: The OpenSpiel action id, one of the frame's `legal_actions`.
    ply: The position to play at, the `ply` of the frame the action was
      chosen from: null, or `num_plies`, for the live position. An earlier
      ply plays from that position and takes back every later ply, as only
      an omniscient viewer may.
  """

  action: int
  ply: int | None


@dataclasses.dataclass(frozen=True)
class ApplyChanceOutcome:
  """Chooses the outcome at a chance node under the `expose` policy.

  Attributes:
    outcome: The OpenSpiel action id, one of the frame's `chance_outcomes`.
    ply: The position to choose at, as for `ApplyAction`.
  """

  outcome: int
  ply: int | None


@dataclasses.dataclass(frozen=True)
class Advance:
  """Lets controllers play their seats.

  Play stops when a human seat or an exposed chance node must act, the game
  ends, a controller fails, or `max_plies` seat actions have been applied.
  Under the `auto` policy, chance outcomes are sampled as they come and do
  not count.

  One request holds its session until play stops, so play is bounded.

  Attributes:
    max_plies: The most seat actions to apply, from 1 to 1000, or null for
      1000.
  """

  max_plies: int | None

  def __post_init__(self) -> None:
    contract.require(
        self.max_plies is None or 1 <= self.max_plies <= MAX_ADVANCE_PLIES,
        f"An advance cannot apply at most {self.max_plies} seat actions; "
        f"it applies from 1 to {MAX_ADVANCE_PLIES}.",
    )

  @property
  def limit(self) -> int:
    """The most seat actions to apply."""
    return MAX_ADVANCE_PLIES if self.max_plies is None else self.max_plies


@dataclasses.dataclass(frozen=True)
class Reset:
  """Restarts the live game, keeping the session's seats.

  Attributes:
    history: Actions to replay from the initial state, chance outcomes
      included, as in a frame's `history`; null or empty for a fresh game.
  """

  history: tuple[int, ...] | None
