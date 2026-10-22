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


"""Seats: who plays each seat of a session, and how a controller fails.

A controller that raises, or returns anything but a legal action, fails (see
`controllers.choose`): its seat records why, and nothing plays in its place
until the session is undone or reset, or the seat is given another controller.
"""

import dataclasses
import random

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
import pyspiel

_SEED_BOUND = 2**31


def random_seed() -> int:
  """Returns a fresh session seed."""
  return random.SystemRandom().randrange(_SEED_BOUND)


def derive_seed(seed: int, *path: object) -> int:
  """Derives an independent, reproducible seed from a session's seed."""
  key = "/".join(str(part) for part in (seed, *path))
  return random.Random(key).randrange(_SEED_BOUND)


@dataclasses.dataclass
class Seat:
  """Who plays a seat, and why they stopped if they failed.

  Attributes:
    spec: The controller and its settings.
    controller: The controller; a `controllers.Human` for a seat the client
      plays.
    autoplay: Whether the controller plays as soon as the seat is to move.
    failure: Why the controller failed, or None while it plays.
  """

  spec: controllers.Spec
  controller: controllers.Controller
  autoplay: bool
  failure: contract.ControllerFailure | None = None

  @classmethod
  def human(cls) -> "Seat":
    """Returns a seat the client plays."""
    return cls(
        spec=controllers.HUMAN_SPEC,
        controller=controllers.Human(),
        autoplay=False,
    )

  @classmethod
  def make(
      cls,
      spec: controllers.Spec,
      game: pyspiel.Game,
      *,
      autoplay: bool,
      seed: int,
  ) -> "Seat":
    """Returns a seat played as `spec` says.

    Args:
      spec: The controller and its settings.
      game: The game the seat plays.
      autoplay: Whether the controller plays as soon as the seat is to move.
      seed: Seeds the controller.

    Raises:
      controllers.UnavailableError: If the controller cannot play `game` with
        these settings, or a human seat is asked to autoplay.
    """
    controller = controllers.create(spec, game, seed=seed)
    if autoplay and isinstance(controller, controllers.Human):
      raise controllers.UnavailableError("A human seat cannot autoplay.")
    return cls(spec=spec, controller=controller, autoplay=autoplay)

  @property
  def is_human(self) -> bool:
    """Whether the client plays this seat."""
    return isinstance(self.controller, controllers.Human)

  def view(self, index: int) -> contract.SeatView:
    """Returns the seat as the contract shows it, failure message included."""
    return contract.SeatView(
        seat=index,
        controller=self.spec.kind,
        config=dict(self.spec.config),
        autoplay=self.autoplay,
        failure=self.failure,
    )

  def choose_action(self, state: pyspiel.State, index: int) -> int | None:
    """Asks the controller for this seat's action at `state`.

    Args:
      state: The live state, where this seat is to move. It is not changed.
      index: This seat's index.

    Returns:
      A legal action, or None if the controller failed; `failure` then says
      why.

    Raises:
      ValueError: If the seat is a human's (see `controllers.choose`).
    """
    try:
      return controllers.choose(self.controller, state, index)
    except controllers.ControllerError as error:
      self.failure = error.failure(visible=True)
      return None


class Seats:
  """Every seat of a session, in seat order."""

  def __init__(self, count: int) -> None:
    """Starts with `count` human seats."""
    self._seats = [Seat.human() for _ in range(count)]

  def __len__(self) -> int:
    return len(self._seats)

  def __getitem__(self, index: int) -> Seat:
    return self._seats[index]

  def __setitem__(self, index: int, seat: Seat) -> None:
    self._seats[index] = seat

  @property
  def failed(self) -> int | None:
    """The first seat whose controller failed, or None."""
    return next(
        (
            index
            for index, seat in enumerate(self._seats)
            if seat.failure is not None
        ),
        None,
    )

  def clear_failures(self) -> None:
    """Resumes every failed seat."""
    for seat in self._seats:
      seat.failure = None

  def views(self) -> tuple[contract.SeatView, ...]:
    """Returns every seat as the contract shows it."""
    return tuple(seat.view(index) for index, seat in enumerate(self._seats))
