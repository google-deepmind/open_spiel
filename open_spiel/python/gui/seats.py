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

A controller that raises, returns anything but a legal action, or overruns
the move budget fails (see `controllers.choose`): its seat records why, and
nothing plays in its place until the session is undone or reset, or the seat
is given another controller.

A seat's controller is built on a thread of its own (`builds`): until its
build finishes the seat is `preparing`, and does not move. A build that fails
leaves the seat `unavailable`, failed, until it is given a controller again.
"""

from collections.abc import Callable, Mapping
import dataclasses
import functools
import random
import time

from open_spiel.python.gui import builds
from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import errors
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
  """Who plays a seat, whether they are ready, and why they stopped.

  Attributes:
    spec: The controller and its settings.
    controller: The controller; a `controllers.Human` for a seat the client
      plays; None until it is built.
    autoplay: Whether the controller plays as soon as the seat is to move.
    failure: Why the controller failed, or None while it plays.
    status: Whether the controller is ready, being built, or not buildable.
    message: Why the seat is not ready, for every viewer; None when it is.
    build: The build of the controller while the seat is preparing.
  """

  spec: controllers.Spec
  controller: controllers.Controller | None
  autoplay: bool
  failure: contract.ControllerFailure | None = None
  status: contract.SeatStatus = contract.SeatStatus.READY
  message: str | None = None
  build: builds.Build[controllers.Controller] | None = None

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
    """Returns a seat played as `spec` says, its controller built here.

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
    _check_autoplay(controller, autoplay)
    return cls(spec=spec, controller=controller, autoplay=autoplay)

  @classmethod
  def preparing(
      cls,
      spec: controllers.Spec,
      game: pyspiel.Game,
      *,
      autoplay: bool,
      seed: int,
      on_built: "OnBuilt",
  ) -> "Seat":
    """Returns a seat played as `spec` says, its controller to be built.

    `spec` is checked here; the build is not started (`build.start`), and
    `on_built` hears when it finishes, on the thread that finished it. A
    human seat has nothing to build, and is returned ready.

    Args:
      spec: The controller and its settings.
      game: The game the seat plays.
      autoplay: Whether the controller plays as soon as the seat is to move.
      seed: Seeds the controller.
      on_built: Called with the build once it has finished.

    Raises:
      controllers.UnavailableError: If the controller is unknown, cannot play
        `game` or is given invalid settings, or a human seat is asked to
        autoplay.
    """
    build_with = controllers.builder(spec, game)
    if spec.kind == controllers.HUMAN:
      if autoplay:
        raise controllers.UnavailableError("A human seat cannot autoplay.")
      # Nothing to wait for: a person's seat is ready at once.
      return cls(spec=spec, controller=build_with(seed), autoplay=False)
    label = controllers.label(spec.kind)
    return cls(
        spec=spec,
        controller=None,
        autoplay=autoplay,
        status=contract.SeatStatus.PREPARING,
        message=(
            f"Building {label}, for at most "
            f"{builds.budget_seconds():g} seconds."
        ),
        build=builds.Build(
            lambda: build_with(seed),
            name=f"build-{spec.kind}",
            on_done=on_built,
        ),
    )

  @property
  def is_human(self) -> bool:
    """Whether the client plays this seat."""
    return isinstance(self.controller, controllers.Human)

  @property
  def ready(self) -> bool:
    """Whether the seat's controller is built and plays it."""
    return self.status == contract.SeatStatus.READY

  def moves(self, *, autoplay_only: bool) -> bool:
    """Whether the server may move the seat: a built controller plays it.

    Args:
      autoplay_only: Whether only a seat that autoplays may move.
    """
    return (
        self.ready
        and not self.is_human
        and (self.autoplay or not autoplay_only)
    )

  def finish(self, index: int, *, refuse: bool) -> None:
    """Takes the controller from the seat's finished build.

    A build that failed leaves the seat unavailable, unless `refuse`.

    Args:
      index: This seat's index.
      refuse: Whether a build that failed raises, as a request that built
        the controller itself would, rather than leaving the seat failed.

    Raises:
      ControllerUnavailableError: If `refuse`, and the controller could not
        play the game after all, or overran the build budget.
      Exception: If `refuse`, whatever else the build raised.
    """
    assert self.build is not None and self.status == (
        contract.SeatStatus.PREPARING
    )
    controller, error = self.build.outcome()
    if error is None:
      assert controller is not None
      try:
        _check_autoplay(controller, self.autoplay)
      except controllers.UnavailableError as e:
        error = e
    if error is not None and refuse:
      if isinstance(
          error, (controllers.UnavailableError, builds.OverBudgetError)
      ):
        raise errors.ControllerUnavailableError(
            _unavailable_message(self.spec.kind, error), seat=index
        ) from error
      raise error
    self.build = None
    if error is None:
      self.controller, self.status, self.message = (
          controller,
          contract.SeatStatus.READY,
          None,
      )
      return
    self.status = contract.SeatStatus.UNAVAILABLE
    self.message = _unavailable_message(self.spec.kind, error)
    self.failure = contract.ControllerFailure(
        error=(
            controllers.TIMEOUT
            if isinstance(error, builds.OverBudgetError)
            else type(error).__name__
        ),
        message=str(error),
    )

  def view(self, index: int) -> contract.SeatView:
    """Returns the seat as the contract shows it, failure message included."""
    return contract.SeatView(
        seat=index,
        controller=self.spec.kind,
        config=dict(self.spec.config),
        autoplay=self.autoplay,
        failure=self.failure,
        status=self.status,
        message=self.message,
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
    assert self.controller is not None and self.ready
    try:
      return controllers.choose(self.controller, state, index)
    except controllers.ControllerError as error:
      self.failure = error.failure(visible=True)
      return None


# Called with a seat's build once it has finished.
OnBuilt = Callable[[builds.Build[controllers.Controller]], None]

# Called with a seat's index and its build once the build has finished.
OnSeatBuilt = Callable[[int, builds.Build[controllers.Controller]], None]


def _check_autoplay(controller: controllers.Controller, autoplay: bool) -> None:
  """Refuses to autoplay a human seat."""
  if autoplay and isinstance(controller, controllers.Human):
    raise controllers.UnavailableError("A human seat cannot autoplay.")


def _unavailable_message(kind: str, error: Exception) -> str:
  """Says, for every viewer, why the controller `kind` was not built."""
  label = controllers.label(kind)
  if isinstance(error, controllers.UnavailableError):
    return str(error)
  if isinstance(error, builds.OverBudgetError):
    return (
        f"{label} was not ready within the {error.seconds:g} second build "
        "budget."
    )
  # What it raised may say anything, the seed included: the oracle alone
  # reads it, in the failure.
  return f"{label} could not be built: it raised {type(error).__name__}."


class Seats:
  """Every seat of a session, in seat order."""

  def __init__(
      self,
      count: int,
      game: pyspiel.Game,
      *,
      seed: int,
      on_built: OnSeatBuilt,
  ) -> None:
    """Starts with `count` human seats.

    Args:
      count: The number of seats.
      game: The game the seats play.
      seed: The session's seed, from which each seat's is derived.
      on_built: Called with a seat's index and its finished build, on the
        thread that finished it (see `assign`).
    """
    self._seats = [Seat.human() for _ in range(count)]
    self._game = game
    self._seed = seed
    self._on_built = on_built

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
    """Resumes every failed seat that has a controller to resume."""
    for seat in self._seats:
      if seat.status != contract.SeatStatus.UNAVAILABLE:
        seat.failure = None

  def assign(self, specs: Mapping[int, tuple[controllers.Spec, bool]]) -> None:
    """Gives seats controllers, each `(spec, autoplay)`, building them.

    Every spec is checked first. The builds then run at once, each on a
    thread of its own, and those that finish within `builds.wait_seconds`
    are used here. Any other seat is left preparing, and `on_built` hears
    when its build finishes; the caller then passes it to `Seat.finish`, if
    the seat still holds that build. The caller holds the lock `on_built`
    takes, so that no build finishes its seat meanwhile. A refusal leaves
    every seat as it was.

    Args:
      specs: The controller and whether it autoplays, by seat index.

    Raises:
      ControllerUnavailableError: If a controller cannot play the game, a
        human seat is asked to autoplay, or a build that finished here
        failed so.
    """
    made = {}
    for index, (spec, autoplay) in specs.items():
      try:
        made[index] = Seat.preparing(
            spec,
            self._game,
            autoplay=autoplay,
            seed=derive_seed(self._seed, "seat", index),
            on_built=functools.partial(self._on_built, index),
        )
      except controllers.UnavailableError as e:
        raise errors.ControllerUnavailableError(str(e), seat=index) from e
    before = {index: self._seats[index] for index in made}
    deadline = time.monotonic() + builds.wait_seconds()
    for index, seat in made.items():
      self._seats[index] = seat
      if seat.build is not None:
        seat.build.start()
    try:
      for index, seat in made.items():
        if seat.build is not None and seat.build.wait(
            deadline - time.monotonic()
        ):
          seat.finish(index, refuse=True)
    except BaseException:
      # The other builds, superseded, are ignored when they finish.
      for index, seat in before.items():
        self._seats[index] = seat
      raise

  def views(self) -> tuple[contract.SeatView, ...]:
    """Returns every seat as the contract shows it."""
    return tuple(seat.view(index) for index, seat in enumerate(self._seats))
