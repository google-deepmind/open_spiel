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


"""Presenters: the registry of game adapters, and the presentations they make.

Each game's adapter (`adapter.Adapter`) draws a position for one viewer; the
frame carries the result as its `presentation`, and a game with no adapter
has none, so the client draws its text. The adapters in the `adapters`
package, one module per game, are registered when this module loads; a
program may `register` its own before it starts serving.

An adapter runs on the server, so it filters by the viewer itself: a seat sees
its own private information and what the rules make public, a spectator of a
game that hides information sees only what is public, and an omniscient
viewer sees everything.
"""

from collections.abc import Mapping, Sequence
import json
import threading
import types

import pyspiel

from open_spiel.python.gui import adapter as adapter_lib
from open_spiel.python.gui import adapters as adapters_package
from open_spiel.python.gui import contract

Adapter = adapter_lib.Adapter
Viewpoint = adapter_lib.Viewpoint

_REGISTRY: dict[str, Adapter] = dict(adapters_package.ADAPTERS)
_REGISTRY_LOCK = threading.Lock()


def register(game: str, adapter: Adapter) -> None:
  """Registers `adapter` for the game with the short name `game`.

  Raises:
    ValueError: If an adapter is already registered for `game`, or the
      adapter names no renderer.
  """
  if not getattr(adapter, "renderer", "").strip():
    raise ValueError(f"The adapter for {game!r} names no renderer.")
  with _REGISTRY_LOCK:
    if game in _REGISTRY:
      raise ValueError(f"An adapter is already registered for {game!r}.")
    _REGISTRY[game] = adapter


def unregister(game: str) -> None:
  """Removes the adapter registered for `game`.

  Raises:
    ValueError: If no adapter is registered for `game`.
  """
  with _REGISTRY_LOCK:
    if _REGISTRY.pop(game, None) is None:
      raise ValueError(f"No adapter is registered for {game!r}.")


def adapters() -> Mapping[str, Adapter]:
  """Returns every registered adapter by game short name; read-only."""
  with _REGISTRY_LOCK:
    return types.MappingProxyType(dict(_REGISTRY))


def adapter_for(game: str) -> Adapter | None:
  """Returns the adapter registered for `game`, or None."""
  return adapters().get(game)


def public_action_label(state: pyspiel.State, action: int) -> str | None:
  """Returns what a viewer who may not see `action` may read of it.

  Asks the adapter of `state`'s game (`Adapter.public_action_label`), with
  the state before the action; None when the action is hidden from such a
  viewer, and for a game with no adapter.

  Raises:
    TypeError: If the adapter's label is not a str.
  """
  adapter = adapter_for(str(state.get_game().get_type().short_name))
  if adapter is None:
    return None
  label = adapter.public_action_label(state, action)
  if label is not None and not isinstance(label, str):
    raise TypeError(f"A public action label is a str, not {label!r}.")
  return label


def _checked_json(value: object, what: str) -> contract.JsonValue:
  """Returns `value` after checking that it encodes as strict JSON."""
  try:
    json.dumps(value, allow_nan=False)
  except (TypeError, ValueError) as e:
    raise TypeError(f"{what} is not JSON: {e}") from e
  return value  # type: ignore[return-value]


def present(
    game: str,
    state: pyspiel.State,
    viewpoint: Viewpoint,
    legal_actions: Sequence[int] | None,
) -> contract.Presentation | None:
  """Returns the adapter's presentation of `state`; None with no adapter.

  Args:
    game: The game's short name.
    state: The position to present.
    viewpoint: Who the presentation is for.
    legal_actions: The legal actions the viewer is given, or None when it is
      given none; only these are described.

  Raises:
    TypeError: If the adapter returns anything that is not JSON.
  """
  adapter = adapter_for(game)
  if adapter is None:
    return None
  view = adapter.present(state, viewpoint)
  if not isinstance(view, dict):
    raise TypeError(f"The {game} adapter's view is not a JSON object.")
  hints: tuple[contract.ActionHint, ...] | None = None
  if legal_actions is not None:
    described = adapter.describe_actions(state, legal_actions, viewpoint)
    hints = tuple(
        contract.ActionHint(
            id=action,
            hint=_checked_json(described[action], f"The {game} hint"),
        )
        for action in legal_actions
        if action in described
    )
  return contract.Presentation(
      renderer=adapter.renderer,
      view=_checked_json(view, f"The {game} view"),
      actions=hints,
  )
