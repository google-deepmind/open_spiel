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


"""Controller plugins: bots that other packages register, loaded on request.

A plugin is a module with a function `register_into(controllers)`, which is
given the `controllers` module and registers its bots with
`controllers.register`. Nothing loads a plugin unless asked to: a bare
`python -m open_spiel.python.gui` imports none. `--controllers` names the
plugins to load, comma separated:

  python -m open_spiel.python.gui --controllers my_bots.gui_plugin

The name `auto` loads every entry point an installed distribution declares
in the group `open_spiel.gui.controllers`, which may name a module or its
`register_into` itself:

  [project.entry-points."open_spiel.gui.controllers"]
  my_bots = "my_bots.gui_plugin"

Plugins load before the server starts, so the catalogue and the health
report list their bots. Each one fails loud: a module that cannot be
imported, one with no `register_into`, and a `register_into` that raises,
among them a name already registered, each stop the server from starting
with a `PluginError` that names the plugin.

This module is the one place the GUI imports a module by name: the names
come from the command line or the installed entry points, never from the
GUI's own source.
"""

from collections.abc import Callable, Sequence
import importlib
import importlib.metadata
import types

from absl import logging

from open_spiel.python.gui import controllers

# The entry-point group that installed distributions declare plugins in.
ENTRY_POINT_GROUP = "open_spiel.gui.controllers"

# The `--controllers` name that loads every plugin in ENTRY_POINT_GROUP.
AUTO = "auto"

# What a plugin calls with the `controllers` module.
RegisterInto = Callable[[types.ModuleType], object]


class PluginError(Exception):
  """A controller plugin could not be loaded or could not register."""


def parse(value: str) -> tuple[str, ...]:
  """Returns the plugin names in a `--controllers` value.

  Args:
    value: Comma-separated module names, `auto` among them if wanted.

  Raises:
    PluginError: If a name is empty.
  """
  names = tuple(name.strip() for name in value.split(","))
  if not all(names):
    raise PluginError(
        f"--controllers {value!r} has an empty name; give module names "
        "separated by commas."
    )
  return names


def load(names: Sequence[str]) -> tuple[str, ...]:
  """Loads the plugins `names` gives, in order, each registering its bots.

  Args:
    names: Module names, or `AUTO` for every plugin in ENTRY_POINT_GROUP.

  Returns:
    The names of the controllers the plugins registered, in order.

  Raises:
    PluginError: If a plugin cannot be imported, has no `register_into`, or
      its `register_into` raises.
  """
  before = set(controllers.kinds())
  for name in names:
    if name == AUTO:
      _load_entry_points()
    else:
      _register(f"module {name!r}", _module_hook(name))
  added = tuple(kind for kind in controllers.kinds() if kind not in before)
  logging.info("Controller plugins registered: %s", ", ".join(added) or "none")
  return added


def _module_hook(name: str) -> RegisterInto:
  """Imports the plugin module `name` and returns its `register_into`."""
  try:
    module = importlib.import_module(name)
  except Exception as e:
    raise PluginError(
        f"Controller plugin module {name!r} cannot be imported: "
        f"{type(e).__name__}: {e}"
    ) from e
  return _hook_of(module, f"module {name!r}")


def _hook_of(loaded: object, plugin: str) -> RegisterInto:
  """Returns the `register_into` of a loaded module, or the callable itself."""
  if isinstance(loaded, types.ModuleType):
    hook = getattr(loaded, "register_into", None)
    if not callable(hook):
      raise PluginError(
          f"Controller plugin {plugin} has no function register_into"
          "(controllers)."
      )
    return hook  # type: ignore[no-any-return]
  if not callable(loaded):
    raise PluginError(
        f"Controller plugin {plugin} is neither a module nor a callable "
        f"register_into, but {type(loaded).__name__}."
    )
  return loaded


def _load_entry_points() -> None:
  """Loads every plugin in ENTRY_POINT_GROUP, in name order."""
  points = sorted(
      importlib.metadata.entry_points(group=ENTRY_POINT_GROUP),
      key=lambda point: (point.name, point.value),
  )
  if not points:
    logging.warning(
        "--controllers %s: no installed distribution declares an entry "
        "point in %s.",
        AUTO,
        ENTRY_POINT_GROUP,
    )
  for point in points:
    plugin = f"entry point {point.name!r} ({point.value})"
    try:
      loaded = point.load()
    except Exception as e:
      raise PluginError(
          f"Controller plugin {plugin} cannot be loaded: "
          f"{type(e).__name__}: {e}"
      ) from e
    _register(plugin, _hook_of(loaded, plugin))


def _register(plugin: str, hook: RegisterInto) -> None:
  """Calls a plugin's `register_into` with the `controllers` module."""
  try:
    hook(controllers)
  except Exception as e:
    raise PluginError(
        f"Controller plugin {plugin} failed to register: "
        f"{type(e).__name__}: {e}"
    ) from e
