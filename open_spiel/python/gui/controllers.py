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


"""Controllers: what plays a seat.

A seat is played by exactly one controller. `human` leaves the seat's moves to
the client; every other controller chooses them itself. `descriptors` says
which controllers can play a game and why the others cannot (`available`
lists only those that can), `create` builds one for a seat, and `choose` asks
it for a move.

Controllers are registered by name with `register` (`controller_registry`);
`human`, `random` and `mcts` are registered here, and a program that serves
the GUI may register its own before it starts serving, since the catalogue
and the health report read the registry once.

No controller stands in for another. When one raises, returns anything but
a legal action, or takes longer than the move budget (`set_move_budget`,
10 seconds unless the server's `--move-budget-seconds` says otherwise),
`choose` raises `ControllerError` naming the seat, and the caller pauses that
seat. A controller chooses on a thread of its own, so a slow one holds up
only the request that asked it, and only for the budget; a move it returns
after that is discarded. A controller that may take long checks
`move_budget.check` as it works, as MCTS does between simulations, so that
one over its budget stops, and its thread exits, soon after. A call that
cannot check, as OpenSpiel's C++ MCTS cannot, must bound its own work to fit
the budget.

Programs that serve the GUI and third-party packages alike register their
controllers with `register`; `plugins` loads the packages' on request.

Every random choice a controller makes is seeded from its seed and the ply it
moves at, so a seat moves the same way at the same position however often it
is asked: after an undo, and when a game's history is replayed.
"""

from collections.abc import Callable, Mapping
import dataclasses
import random

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_registry
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import errors
from open_spiel.python.gui import mcts_controller
from open_spiel.python.gui import move_budget as move_budget_lib
import pyspiel

HUMAN = contract.HUMAN
RANDOM = "random"
MCTS = mcts_controller.NAME

# A controller's settings, as a client sends them.
Config = controller_settings.Config

# The failure name of a controller that did not choose within the budget.
TIMEOUT = "Timeout"

# The registry's names, which callers reach through this module.
UnavailableError = controller_registry.UnavailableError
Controller = controller_registry.Controller
Builder = controller_registry.Builder
Kind = controller_registry.Kind
plays_every_game = controller_registry.plays_every_game
register = controller_registry.register
unregister = controller_registry.unregister
kinds = controller_registry.kinds

# The settings' names, which a plugin declares its controller's settings with.
Setting = controller_settings.Setting
REQUIRED = controller_settings.REQUIRED

# MCTS's limits (`mcts_controller`), which a server sets once at startup.
configure_mcts = mcts_controller.configure
mcts_limits = mcts_controller.configured_limits

# The move budget's names, which callers reach through this module.
DEFAULT_MOVE_BUDGET_SECONDS = move_budget_lib.DEFAULT_SECONDS
set_move_budget = move_budget_lib.set_seconds
move_budget = move_budget_lib.seconds


class ControllerError(errors.ControllerFailedError):
  """A seat's controller failed to choose a legal action.

  As a refusal it names only the seat. What went wrong may describe private
  state, so `detail` reaches only an omniscient viewer, through `failure`.

  Attributes:
    seat: The seat whose controller failed.
    error: The name of the failure: the exception the controller raised, or
      `IllegalAction` when it returned something that is not a legal action.
    detail: What went wrong, which may describe private state.
  """

  def __init__(self, seat: int, error: str, detail: str) -> None:
    super().__init__(seat=seat)
    self.error = error
    self.detail = detail

  def failure(self, visible: bool) -> contract.ControllerFailure:
    """Returns the failure as a viewer sees it.

    Args:
      visible: Whether the viewer may see the failing seat's private state.
    """
    return contract.ControllerFailure(
        error=self.error, message=self.detail if visible else None
    )


@dataclasses.dataclass(frozen=True)
class Spec:
  """A request for a controller.

  Attributes:
    kind: The name the controller is registered under, such as `HUMAN`,
      `RANDOM` or `MCTS`.
    config: The controller's settings, satisfying its `config_schema`. A
      setting left out takes its default.
  """

  kind: str
  config: Config = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Descriptor:
  """A controller, and whether it can play a particular game.

  Attributes:
    controller: The controller as the contract describes it.
    unavailable_reason: Why the controller cannot play the game; None when it
      can.
  """

  controller: contract.Controller
  unavailable_reason: str | None

  @property
  def available(self) -> bool:
    """Whether the controller can play the game."""
    return self.unavailable_reason is None


# The request for a seat the client plays.
HUMAN_SPEC = Spec(kind=HUMAN)


class Human:
  """Leaves a seat's moves to the client; `choose` refuses to ask it."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Never called: `choose` refuses a human seat before asking."""
    del state
    raise AssertionError(f"Seat {seat} is played by a person.")


class Random:
  """Plays a uniformly random legal action, reproducibly from its seed."""

  def __init__(self, seed: int) -> None:
    self._seed = seed

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns one of `seat`'s legal actions, drawn uniformly."""
    rng = random.Random(controller_registry.choice_seed(self._seed, state))
    return int(rng.choice(state.legal_actions(seat)))


def _build_human(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> Controller:
  """Builds the human controller, which takes no settings and no seed."""
  del game, config, seed
  return Human()


def _build_random(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> Controller:
  """Builds the random controller, which takes no settings."""
  del game, config
  return Random(seed)


register(
    HUMAN,
    Kind("Human", (), reads_full_state=False, build=_build_human),
)
# Reads only the seat's legal actions.
register(
    RANDOM,
    Kind("Random", (), reads_full_state=False, build=_build_random),
)
# With the limits that fit the move budget with either behaviour of the C++
# search, until `configure_mcts` says which the served pyspiel has.
register(MCTS, mcts_controller.kind(contract.MctsLimits.SIMULATIONS_ONLY))


def _describe(name: str, kind: Kind) -> contract.Controller:
  """Returns the contract's description of the controller `name`."""
  return contract.Controller(
      name=name, label=kind.label, config_schema=kind.config_schema()
  )


def registered() -> tuple[contract.Controller, ...]:
  """Returns every controller, `human` first."""
  return tuple(_describe(name, kind) for name, kind in kinds().items())


def descriptors(game: pyspiel.Game) -> tuple[Descriptor, ...]:
  """Describes every controller, and whether it can play `game`.

  Args:
    game: The game as it is served, after any turn-based wrapping.

  Returns:
    One descriptor per controller, `human` first.
  """
  return tuple(
      Descriptor(_describe(name, kind), kind.unavailable_reason(game))
      for name, kind in kinds().items()
  )


def available(game: pyspiel.Game) -> tuple[contract.Controller, ...]:
  """Returns the controllers that can play `game`, `human` first."""
  return tuple(
      descriptor.controller
      for descriptor in descriptors(game)
      if descriptor.available
  )


def builder(spec: Spec, game: pyspiel.Game) -> Callable[[int], Controller]:
  """Checks `spec` against `game`, and returns what builds its controller.

  Everything a client can get wrong is checked here, at once; the build
  itself, which may take a while (a trained model loading and warming up),
  is left to the returned call, which a session makes on a thread of its own
  (`builds`).

  Args:
    spec: The controller and its settings.
    game: The game as it is served, after any turn-based wrapping.

  Returns:
    A call that builds the controller from the seed of its random choices.
    It raises `UnavailableError` if the controller finds it cannot play after
    all (a checkpoint that does not load), and whatever else its builder
    raises.

  Raises:
    UnavailableError: If the controller is unknown, cannot play `game`, or is
      given invalid settings.
  """
  registry = kinds()
  kind = registry.get(spec.kind)
  if kind is None:
    raise UnavailableError(
        f"There is no controller {controller_settings.shown(spec.kind)}; the "
        "controllers are "
        f"{', '.join(registry)}."
    )
  reason = kind.unavailable_reason(game)
  if reason is not None:
    raise UnavailableError(reason)
  config = kind.parse(spec.config)
  return lambda seed: kind.build(game, config, seed)


def label(name: str) -> str:
  """Returns the label of the controller registered as `name`, or `name`."""
  kind = kinds().get(name)
  return name if kind is None else kind.label


def create(spec: Spec, game: pyspiel.Game, *, seed: int) -> Controller:
  """Builds the controller `spec` asks for, to play a seat of `game`.

  Args:
    spec: The controller and its settings.
    game: The game as it is served, after any turn-based wrapping.
    seed: Seeds every random choice the controller makes.

  Returns:
    The controller.

  Raises:
    UnavailableError: If the controller is unknown, cannot play
      `game`, or is given invalid settings.
  """
  return builder(spec, game)(seed)


def choose(controller: Controller, state: pyspiel.State, seat: int) -> int:
  """Asks `controller` for `seat`'s action at `state`, within the budget.

  Args:
    controller: The controller playing `seat`.
    state: The position, where `seat` is to move. The controller is given a
      copy, on a thread of its own, so `state` does not change.
    seat: The seat to move.

  Returns:
    One of `seat`'s legal actions.

  Raises:
    ValueError: If `seat` is not to move at `state`, or `controller` is a
      human's, whose moves come from the client and never from here.
    ControllerError: If the controller raises, returns anything but one of
      `seat`'s legal actions, or does not return within the move budget.
  """
  if isinstance(controller, Human):
    raise ValueError(
        f"Seat {seat} is played by a person, whose moves come from the client."
    )
  if state.current_player() != seat:
    raise ValueError(
        f"Seat {seat} is not to move; player {state.current_player()} is."
    )
  legal = state.legal_actions(seat)
  # A copy: the controller may play on it, and the position stays as it is.
  copy = state.clone()
  try:
    action: object = move_budget_lib.run(
        lambda: controller.choose_action(copy, seat),
        name=f"controller-seat-{seat}",
    )
  except move_budget_lib.OverBudgetError as e:
    raise ControllerError(
        seat,
        TIMEOUT,
        f"Seat {seat}'s controller did not choose a move within the "
        f"{e.seconds:g} second move budget.",
    ) from e
  except Exception as e:
    # Whatever the failure, the seat pauses; nothing retries or covers for it.
    raise ControllerError(seat, type(e).__name__, str(e)) from e
  if (
      isinstance(action, bool)
      or not isinstance(action, int)
      or action not in legal
  ):
    raise ControllerError(
        seat,
        "IllegalAction",
        f"Seat {seat}'s controller chose {action!r}, which is not one of its "
        "legal actions.",
    )
  return action
