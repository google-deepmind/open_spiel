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


"""The registry of controllers: every kind of controller a seat may have.

A controller is registered by name with `register`, and `kinds` lists every
registered controller in the order registered, `human` first. `controllers`
registers `human`, `random` and `mcts`, and builds and asks controllers for
moves; a program that serves the GUI may register its own before it starts
serving, since the catalogue and the health report read the registry once.
"""

from collections.abc import Callable, Mapping
import dataclasses
import random
import re
import threading
import types
from typing import Protocol

import pyspiel

from open_spiel.python.gui import contract
from open_spiel.python.gui import controller_settings
from open_spiel.python.gui import errors
from open_spiel.python.gui import information

HUMAN = contract.HUMAN

# A controller's settings, as a client sends them.
Config = controller_settings.Config

_SEED_BOUND = 2**31


class UnavailableError(errors.ControllerUnavailableError):
  """A controller cannot be built for a seat.

  The controller is unknown, cannot play the game, or was given invalid
  settings. The message says which, and is safe to show to anyone.
  """


class Controller(Protocol):
  """Plays a seat."""

  def choose_action(self, state: pyspiel.State, seat: int) -> int:
    """Returns the action `seat` plays at `state`, where it is to move."""


# Builds a controller for a game from its parsed settings and a seed.
Builder = Callable[
    [pyspiel.Game, Mapping[str, contract.ParameterValue], int], Controller
]


# The explicit None reads as "no refusal", which is what a refusal returns.
# pylint: disable-next=useless-return
def plays_every_game(game: pyspiel.Game) -> str | None:
  """Refuses no game: returns None, the refusal callback's "can play"."""
  del game
  return None


@dataclasses.dataclass(frozen=True)
class Kind:
  """A kind of controller, as it is registered.

  Attributes:
    label: The human-readable name.
    settings: The settings the controller takes.
    reads_full_state: Whether the controller reads more of the state than the
      seat may see. Such a controller cannot play a game that hides
      information from its players.
    build: Builds the controller for a game from its parsed settings and a
      seed, which seeds every random choice it makes.
    refusal: Returns why the controller cannot play a game, or None, beyond
      what `reads_full_state` already refuses.
  """

  label: str
  settings: tuple[controller_settings.Setting, ...]
  reads_full_state: bool
  build: Builder
  refusal: Callable[[pyspiel.Game], str | None] = plays_every_game

  def __post_init__(self) -> None:
    if not self.label.strip():
      raise ValueError("A controller's label is blank.")

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

    Raises:
      UnavailableError: If `config` names an unknown setting, leaves out a
        required one or holds an invalid value; raised from the
        `controller_settings.InvalidSettingError` that says which.
    """
    try:
      return controller_settings.parse(self.settings, config, self.label)
    except controller_settings.InvalidSettingError as e:
      raise UnavailableError(str(e)) from e


# A controller's name: lower case letters, digits and underscores, starting
# with a letter, so that it reads the same in a URL, a log and a config file.
_NAME = re.compile(r"[a-z][a-z0-9_]*")

# Every controller, in the order clients list them: in the order registered,
# `human` first.
_REGISTRY: dict[str, Kind] = {}
_REGISTRY_LOCK = threading.Lock()


def register(name: str, kind: Kind) -> None:
  """Registers a controller under `name`.

  Register before serving: the catalogue and the health report list the
  controllers registered when they are first built.

  Args:
    name: The controller's stable identifier, as `Spec.kind` and the
      contract's `Controller.name` give it.
    kind: The controller.

  Raises:
    ValueError: If `name` is not lower case letters, digits and underscores
      starting with a letter, or a controller is already registered under it,
      or the first controller registered is not `human`.
  """
  if not _NAME.fullmatch(name):
    raise ValueError(
        f"Controller name {name!r} is not lower case letters, digits and "
        "underscores starting with a letter."
    )
  with _REGISTRY_LOCK:
    if name in _REGISTRY:
      raise ValueError(f"A controller is already registered as {name!r}.")
    if not _REGISTRY and name != HUMAN:
      raise ValueError(f"The first controller registered is {HUMAN!r}.")
    _REGISTRY[name] = kind


def replace(name: str, kind: Kind) -> None:
  """Registers `kind` in place of the controller registered as `name`.

  The controller keeps its place in the order. Replace before serving, as
  `register` says.

  Raises:
    ValueError: If no controller is registered as `name`, or `name` is
      `human`.
  """
  if name == HUMAN:
    raise ValueError(f"{HUMAN!r} cannot be replaced.")
  with _REGISTRY_LOCK:
    if name not in _REGISTRY:
      raise ValueError(f"No controller is registered as {name!r}.")
    _REGISTRY[name] = kind


def unregister(name: str) -> None:
  """Removes the controller registered under `name`.

  Raises:
    ValueError: If `name` is `human`, which every session offers, or no
      controller is registered under it.
  """
  if name == HUMAN:
    raise ValueError(f"{HUMAN!r} cannot be unregistered.")
  with _REGISTRY_LOCK:
    if _REGISTRY.pop(name, None) is None:
      raise ValueError(f"No controller is registered as {name!r}.")


def kinds() -> Mapping[str, Kind]:
  """Returns every registered controller by name, `human` first; read-only."""
  with _REGISTRY_LOCK:
    return types.MappingProxyType(dict(_REGISTRY))


def choice_seed(seed: int, state: pyspiel.State) -> int:
  """Returns the seed of a controller's choice at `state`: one per ply.

  Args:
    seed: The controller's seed.
    state: The position it chooses at.
  """
  key = f"{seed}/{len(state.history())}"
  return random.Random(key).randrange(_SEED_BOUND)
