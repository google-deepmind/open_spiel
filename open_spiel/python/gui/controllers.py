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

No controller stands in for another. When one raises, or returns anything
but a legal action, `choose` raises `ControllerError` naming the seat, and the
caller pauses that seat.

Every random choice a controller makes is seeded from its seed and the ply it
moves at, so a seat moves the same way at the same position however often it
is asked: after an undo, and when a game's history is replayed.
"""

from collections.abc import Callable, Mapping
import dataclasses
import random
from typing import Protocol

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import errors
from open_spiel.python.gui import gil_safety
from open_spiel.python.gui import information
import pyspiel

HUMAN = contract.HUMAN
RANDOM = "random"
MCTS = "mcts"

# A controller's settings, as a client sends them.
Config = controller_settings.Config

# The cap on the MCTS search tree's memory, in megabytes.
_MCTS_MAX_MEMORY_MB = 1000

# The most simulations an MCTS move may run. A move runs while its session is
# locked, so this bounds how long the session's other requests wait.
_MCTS_MAX_SIMULATIONS = 10_000

_SEED_BOUND = 2**31


class UnavailableError(errors.ControllerUnavailableError):
  """A controller cannot be built for a seat.

  The controller is unknown, cannot play the game, or was given invalid
  settings. The message says which, and is safe to show to anyone.
  """


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
    kind: The controller's name: `HUMAN`, `RANDOM` or `MCTS`.
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


class Controller(Protocol):
  """Plays a seat."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the action `seat` plays at `state`, where it is to move."""


# Builds a controller for a game from its parsed settings and a seed.
_Builder = Callable[
    [pyspiel.Game, Mapping[str, contract.ParameterValue], int], Controller
]


@dataclasses.dataclass(frozen=True)
class _Kind:
  """A registered controller.

  Attributes:
    label: The human-readable name.
    settings: The settings the controller takes.
    reads_full_state: Whether the controller reads more of the state than the
      seat may see. Such a controller cannot play a game that hides
      information from its players.
    refusal: Returns why the controller cannot play a game, or None, beyond
      what `reads_full_state` already refuses.
    build: Builds the controller.
  """

  label: str
  settings: tuple[controller_settings.Setting, ...]
  reads_full_state: bool
  refusal: Callable[[pyspiel.Game], str | None]
  build: _Builder

  def unavailable_reason(self, game: pyspiel.Game) -> str | None:
    """Returns why the controller cannot play `game`, or None if it can."""
    game_type = game.get_type()
    if self.reads_full_state and information.hides_information(game_type):
      return (
          f"{self.label} reads the full state, which "
          f"{game_type.short_name} hides from its players."
      )
    return self.refusal(game)

  def config_schema(self) -> dict[str, contract.Json]:
    """Returns the JSON Schema of the controller's settings."""
    return controller_settings.schema(self.settings)

  def parse(self, config: Config) -> dict[str, contract.ParameterValue]:
    """Returns `config` with its defaults filled in.

    Args:
      config: The settings given for the controller, by name.

    Raises:
      UnavailableError: If `config` names an unknown setting or holds an
        invalid value.
    """
    try:
      return controller_settings.parse(self.settings, config, self.label)
    except controller_settings.InvalidSettingError as e:
      raise UnavailableError(str(e)) from e


def _choice_seed(seed: int, state: pyspiel.State) -> int:
  """Returns the seed of the choice made at `state`: one per ply."""
  key = f"{seed}/{len(state.history())}"
  return random.Random(key).randrange(_SEED_BOUND)


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
    rng = random.Random(_choice_seed(self._seed, state))
    return int(rng.choice(state.legal_actions(seat)))


class Mcts:
  """Plays the move preferred by OpenSpiel's Monte Carlo tree search.

  Each search runs a fixed number of simulations, each ending in a uniformly
  random rollout, and is seeded afresh for the ply it moves at, so the moves
  are reproducible from the seed. The search has no time limit: OpenSpiel's
  wall-clock budget replaces the simulation count, which would make the move
  depend on the machine's speed.
  """

  def __init__(
      self,
      game: pyspiel.Game,
      seed: int,
      *,
      max_simulations: int,
      uct_c: float,
      solve: bool,
  ) -> None:
    self._game = game
    self._seed = seed
    self._max_simulations = max_simulations
    self._uct_c = uct_c
    self._solve = solve

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the action the search prefers for `seat`."""
    del seat  # The search plays for the player to move, which is `seat`.
    seed = _choice_seed(self._seed, state)
    bot = pyspiel.MCTSBot(
        game=self._game,
        evaluator=pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=seed),
        uct_c=self._uct_c,
        max_simulations=self._max_simulations,
        max_memory_mb=_MCTS_MAX_MEMORY_MB,
        solve=self._solve,
        seed=seed,
        verbose=False,
    )
    return int(bot.step(state))


# The explicit None reads as "no refusal", which is what a refusal returns.
# pylint: disable-next=useless-return
def _plays_every_game(game: pyspiel.Game) -> str | None:
  """Refuses no game: returns None, the refusal callback's "can play"."""
  del game
  return None


def _mcts_refusal(game: pyspiel.Game) -> str | None:
  """Returns why MCTS cannot play `game`, or None if it can.

  The search plays forward from a clone of the full state, so it reads the
  full state (see `_Kind.reads_full_state`), and in a game whose state samples
  its own chance outcomes it would foresee them.

  Args:
    game: The game MCTS would play.
  """
  game_type = game.get_type()
  name = game_type.short_name
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    return f"MCTS needs a sequential game, and {name} is not one."
  if game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC:
    return (
        f"MCTS would foresee the chance outcomes of {name}, whose state "
        "samples them itself."
    )
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    return (
        f"MCTS needs rewards at the end of the game only, and {name} does not "
        "give them so."
    )
  if gil_safety.uses_python(game) and not gil_safety.python_games_are_safe():
    return (
        f"{name} is, or wraps, a game implemented in Python, and the "
        "installed pyspiel cannot run MCTS on games implemented in Python."
    )
  return None


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


def _build_mcts(
    game: pyspiel.Game, config: Mapping[str, contract.ParameterValue], seed: int
) -> Controller:
  """Builds the MCTS controller from its parsed settings."""
  return Mcts(
      game,
      seed,
      max_simulations=int(config["max_simulations"]),
      uct_c=float(config["uct_c"]),
      solve=bool(config["solve"]),
  )


_MCTS_SETTINGS = (
    controller_settings.Setting(
        "max_simulations",
        int,
        1000,
        "Simulations per move, each ending in a random rollout.",
        minimum=1,
        maximum=_MCTS_MAX_SIMULATIONS,
    ),
    controller_settings.Setting(
        "uct_c",
        float,
        2.0,
        "The exploration constant of the UCT selection rule.",
        minimum=0.0,
        maximum=100.0,
    ),
    controller_settings.Setting(
        "solve",
        bool,
        True,
        "Whether proven wins and losses are backed up the search tree.",
    ),
)

# Every controller, in the order clients list them.
_KINDS: Mapping[str, _Kind] = {
    HUMAN: _Kind(
        "Human",
        (),
        reads_full_state=False,
        refusal=_plays_every_game,
        build=_build_human,
    ),
    # Reads only the seat's legal actions.
    RANDOM: _Kind(
        "Random",
        (),
        reads_full_state=False,
        refusal=_plays_every_game,
        build=_build_random,
    ),
    MCTS: _Kind(
        "MCTS",
        _MCTS_SETTINGS,
        reads_full_state=True,
        refusal=_mcts_refusal,
        build=_build_mcts,
    ),
}


def _describe(name: str, kind: _Kind) -> contract.Controller:
  """Returns the contract's description of the controller `name`."""
  return contract.Controller(
      name=name, label=kind.label, config_schema=kind.config_schema()
  )


def registered() -> tuple[contract.Controller, ...]:
  """Returns every controller, `human` first."""
  return tuple(_describe(name, kind) for name, kind in _KINDS.items())


def descriptors(game: pyspiel.Game) -> tuple[Descriptor, ...]:
  """Describes every controller, and whether it can play `game`.

  Args:
    game: The game as it is served, after any turn-based wrapping.

  Returns:
    One descriptor per controller, `human` first.
  """
  return tuple(
      Descriptor(_describe(name, kind), kind.unavailable_reason(game))
      for name, kind in _KINDS.items()
  )


def available(game: pyspiel.Game) -> tuple[contract.Controller, ...]:
  """Returns the controllers that can play `game`, `human` first."""
  return tuple(
      descriptor.controller
      for descriptor in descriptors(game)
      if descriptor.available
  )


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
  kind = _KINDS.get(spec.kind)
  if kind is None:
    raise UnavailableError(
        f"There is no controller {controller_settings.shown(spec.kind)}; the "
        "controllers are "
        f"{', '.join(_KINDS)}."
    )
  reason = kind.unavailable_reason(game)
  if reason is not None:
    raise UnavailableError(reason)
  return kind.build(game, kind.parse(spec.config), seed)


def choose(controller: Controller, state: pyspiel.State, seat: int) -> int:
  """Asks `controller` for `seat`'s action at `state`.

  Args:
    controller: The controller playing `seat`.
    state: The position, where `seat` is to move. The controller is given a
      copy, so `state` does not change.
    seat: The seat to move.

  Returns:
    One of `seat`'s legal actions.

  Raises:
    ValueError: If `seat` is not to move at `state`, or `controller` is a
      human's, whose moves come from the client and never from here.
    ControllerError: If the controller raises, or returns anything but one of
      `seat`'s legal actions.
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
  try:
    # A copy: the controller may play on it, and the position stays as it is.
    action: object = controller.choose_action(state.clone(), seat)
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
