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

"""Scoped registrations for tests of code that registers into this package.

The open registries (metrics sinks, params publishers, series emitters, PG
loss variants) are process-global: a test that registers into them, or needs
them empty, uses these context managers, which restore the registry on exit
(also on an exception), instead of patching the private globals.

scoped_metrics       replaces the metrics auto sinks, params publishers and
                     series emitters for the block
registered_metrics   what the metrics registries hold now
scoped_loss          a PG LossSpec registered under its name for the block,
                     in place of any spec registered under it
open_view            the package's own game rows, set_* defaults and a
                     plugin-free child environment for the block, in a
                     process where a plugin also changed them (the
                     package conftest applies it to every test module)
setting_drift        the set_* settings not at the package's default
"""

import contextlib
import os
import sys
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from open_spiel.python.scale import metrics
from open_spiel.python.scale import plugins
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.learners import pg

__all__ = [
    "open_view",
    "registered_metrics",
    "scoped_loss",
    "scoped_metrics",
    "setting_drift",
]

_Publisher = Callable[[Mapping[str, Any]], bool]
_Emitter = Callable[[Mapping[str, Any]], None]


def registered_metrics() -> dict[str, tuple[Callable[..., Any], ...]]:
  """The metrics registries' current entries, in registration order.

  Returns:
    {"auto_sinks", "params_publishers", "series_emitters"} -> a tuple each.
  """
  # pylint: disable=protected-access
  return {
      "auto_sinks": tuple(metrics._AUTO),
      "params_publishers": tuple(metrics._PARAMS_PUBLISHERS),
      "series_emitters": tuple(metrics._SERIES_EMITTERS),
  }


def _callables(
    name: str, fns: Sequence[Callable[..., Any]]
) -> list[Callable[..., Any]]:
  """Checks a sequence of callables.

  Args:
    name: what it is (for errors).
    fns: the candidate sequence.

  Returns:
    fns as a list.

  Raises:
    TypeError: fns is not a sequence (or is a string), or holds a
      non-callable.
  """
  if isinstance(fns, (str, bytes)) or not isinstance(fns, Sequence):
    raise TypeError(f"{name}: a sequence of callables, got {fns!r}")
  for fn in fns:
    if not callable(fn):
      raise TypeError(f"{name}: {fn!r} is not callable")
  return list(fns)


@contextlib.contextmanager
def scoped_metrics(
    auto_sinks: Optional[Sequence[Callable[..., Any]]] = None,
    params_publishers: Optional[Sequence[_Publisher]] = None,
    series_emitters: Optional[Sequence[_Emitter]] = None,
) -> Iterator[None]:
  """Within the block, the given metrics registries hold exactly these entries.

  A registry passed as None is left as it is (registrations inside the block
  still roll back on exit). Every registry is restored on exit.

  Args:
    auto_sinks: the auto sink factories (metrics.register_auto_sink).
    params_publishers: the params publishers (register_params_publisher).
    series_emitters: the series emitters (register_series_emitter).

  Yields:
    None.

  Raises:
    TypeError: a registry is not a sequence of callables.
  """
  # pylint: disable=protected-access
  regs = (
      (metrics._AUTO, "auto_sinks", auto_sinks),
      (metrics._PARAMS_PUBLISHERS, "params_publishers", params_publishers),
      (metrics._SERIES_EMITTERS, "series_emitters", series_emitters),
  )
  # pylint: enable=protected-access
  new = [None if fns is None else _callables(n, fns) for _, n, fns in regs]
  saved = [list(reg) for reg, _, _ in regs]
  try:
    for (reg, _, _), fns in zip(regs, new):
      if fns is not None:
        reg[:] = fns  # in place: modules holding the list object see it
    yield
  finally:
    for (reg, _, _), old in zip(regs, saved):
      reg[:] = old


@contextlib.contextmanager
def scoped_loss(spec: pg.LossSpec) -> Iterator[pg.LossSpec]:
  """Within the block, spec is the PG loss registered under spec.name.

  Replaces any spec already registered under the name (pg.register_loss
  refuses that) and restores the registry on exit.

  Args:
    spec: the LossSpec.

  Yields:
    spec.

  Raises:
    TypeError: spec is not a LossSpec.
    ValueError: spec names a built-in loss.
  """
  if not isinstance(spec, pg.LossSpec):
    raise TypeError(f"scoped_loss takes a pg.LossSpec, got {type(spec)}")
  if spec.name in pg.PG_LOSSES:
    raise ValueError(f"{spec.name!r} is a built-in loss")
  reg = pg._EXTENSION_LOSSES  # pylint: disable=protected-access
  saved = dict(reg)
  reg[spec.name] = spec
  try:
    yield spec
  finally:
    reg.clear()
    reg.update(saved)


#: The rules_sha scheme the package itself uses (a plugin may set another).
OPEN_RULES_SHA_TAG = {
    "tag": f"rules/{env_registry.RULES_SHA_VERSION}",
    "source": env_registry.CANONICAL_SOURCE,
}

_PKG = __name__.rsplit(".", 2)[0]

#: What each plugin setter (plugins.py: the set_* extension points) holds in
#: the package's own process: by module (relative to the package), the
#: global it sets and its default given the module. A dict, list or set is
#: set in place, anything else rebound. setters_test checks every default
#: against a fresh process with no plugin (setting_drift).
_OPEN_SETTINGS = {
    "envs.registry": (
        ("_RULES_SHA_TAG", lambda m: OPEN_RULES_SHA_TAG),
        ("_SOURCE_ALIASES", lambda m: {}),
    ),
    "envs.gin": (("_DEFAULT_ENGINE", lambda m: {"name": "reference"}),),
    "core.decide": (("_SCOPE", lambda m: {"name": m.DEFAULT_SCOPE}),),
    "ckpt.codecs": (("TREE_SHA_TAG", lambda m: m.DEFAULT_TREE_SHA_TAG),),
    "ckpt.formats": (("_WRITER", lambda m: {"name": m.NATIVE}),),
    "ckpt.provenance": (
        (
            "_STATE",
            lambda m: {
                "attempt_id_env": None,
                "parse": m._as_string,  # pylint: disable=protected-access
                "code_repo": None,
            },
        ),
    ),
    "sources.deals": (
        ("HASH_TAG", lambda m: b"deal_hash\0"),
        ("INDEX_FORMAT", lambda m: "deal_index"),
    ),
    "sources.shards": (("MANIFEST_FORMAT", lambda m: "trajectory_shards"),),
    "league.runner": (("EVAL_SERIES", lambda m: "eval"),),
    "launch.preemption": (("_DEFAULT", lambda m: ["sigterm"]),),
    # Not set_* but the same class: a plugin's registrations here replace
    # or hide the package's configs (overlays, hidden names) or change every
    # config hash (hash transforms, recorded forms).
    "league.config_common": (
        ("_PACKAGES", lambda m: [f"{_PKG}.configs"]),
        ("_RESOLVERS", lambda m: {}),
        ("_OVERLAYS", lambda m: {}),
        ("_HIDDEN", lambda m: set()),
        ("_HASH_TRANSFORMS", lambda m: []),
    ),
    "core.hash_compat": (
        ("_ADDED", lambda m: {}),
        ("_CLASS_TRANSFORMS", lambda m: {}),
        ("_KEYSETS", lambda m: {}),
        ("_STRING_TRANSFORMS", lambda m: []),
    ),
}
SETTING_MODULES = tuple(f"{_PKG}.{rel}" for rel in _OPEN_SETTINGS)
#: The package's top-level set_* functions, by module, whose settings
#: _OPEN_SETTINGS holds (envs.registry's register_rules_source_alias is held
#: too) ...
HELD_SETTERS = {
    "envs.registry": ("set_rules_sha_tag",),
    "envs.gin": ("set_default_engine",),
    "core.decide": ("set_scope",),
    "ckpt.codecs": ("set_tree_sha_tag",),
    "ckpt.formats": ("set_default_writer",),
    "ckpt.provenance": ("set_attempt_id_env", "set_code_repo"),
    "sources.deals": ("set_hash_tag", "set_index_format"),
    "sources.shards": ("set_manifest_format",),
    "league.runner": ("set_eval_series",),
    "launch.preemption": ("set_default_sources",),
}
#: ... and those it leaves, each with why. setters_test fails on a set_* in
#: neither (a new extension point must say what open_view does with it).
UNHELD_SETTERS = {
    "ckpt.formats.set_native": (
        "not an extension point: ckpt.artifact sets "
        "the native format once, at its import"
    ),
}


def _loaded(rel):
  """The package's module `rel` when it is imported, else None.

  A module not imported yet holds its defaults when it is: no setter of it
  can have run.

  Args:
    rel: the module, relative to the package.

  Returns:
    the module, or None.
  """
  return sys.modules.get(f"{_PKG}.{rel}")


def _settings():
  """The settings of the imported modules of _OPEN_SETTINGS.

  Returns:
    [(module, global, default)].
  """
  out = []
  for rel, specs in _OPEN_SETTINGS.items():
    m = _loaded(rel)
    if m is not None:
      out.extend((m, name, default(m)) for name, default in specs)
  return out


def setting_drift() -> list[tuple[str, Any, Any]]:
  """The settings whose value is not the package's default.

  Empty in a process where no plugin set one (modules not imported are
  not read).

  Returns:
    [(module.global, value, default)].
  """
  return [
      (f"{m.__name__}.{name}", getattr(m, name), default)
      for m, name, default in _settings()
      if getattr(m, name) != default
  ]


def _set(module, name, value) -> None:
  """Sets a setting: a dict, list or set in place, anything else rebound.

  Args:
    module: the module.
    name: the global.
    value: its new value.
  """
  current = getattr(module, name)
  if isinstance(current, (dict, set)):
    current.clear()
    current.update(value)
  elif isinstance(current, list):
    current[:] = list(value)
  else:
    setattr(module, name, value)


def _copy(value):
  """A dict's, a list's or a set's copy, else the value itself.

  Args:
    value: a value.

  Returns:
    the copy.
  """
  return type(value)(value) if isinstance(value, (dict, list, set)) else value


@contextlib.contextmanager
def open_view() -> Iterator[None]:
  """Within the block, the package's process-global state is its own.

  A plugin (plugins.py) changes that state for its whole process, at import:
  it replaces built-in game rows (env_registry.register(..., replace=True)),
  sets defaults through the set_* extension points (the rules_sha scheme and
  its source aliases, the default artifact writer, the gin engine, the
  recorded tags and formats, provenance, preemption sources), adds config
  packages, resolvers, overlays, hidden names and hash transforms (which
  replace or hide the package's configs and change their hashes), and names
  itself in the environment (plugins.ENV) so every child process loads it.
  One pytest session that collects the package's tests and the tests of a
  package with a plugin (whose conftest loads it) runs the package's tests
  under the plugin, and they fail although they pass alone. Within the
  block:

    - every built-in game name maps to its built-in row (env_registry.BUILTIN);
    - every setting of an imported module in _OPEN_SETTINGS holds the
      package's default (the rules_sha tag is OPEN_RULES_SHA_TAG, no rules
      source alias, config overlay or hash transform is registered, the
      package's configs are the only config package, ...);
    - plugins.ENV is not in os.environ, so a child Python loads no plugin.

  What a plugin registered under new names (games, formats, sinks, ...)
  stays registered: the package's tests do not name it. On exit (also on an
  exception) everything is put back as it was, containers in place. Without
  a plugin loaded the block changes nothing. State read before the block
  (at collection: module-level code, parametrize arguments) still sees the
  plugin's; a test module that reads such state at collection reads it
  inside its own `with open_view():`.

  Yields:
    None.
  """
  games = env_registry._GAMES  # pylint: disable=protected-access
  replaced = {
      g.name: games.get(g.name)
      for g in env_registry.BUILTIN
      if games.get(g.name) is not g
  }
  settings = _settings()
  saved = [_copy(getattr(m, name)) for m, name, _ in settings]
  saved_env = os.environ.get(plugins.ENV)
  try:
    for g in env_registry.BUILTIN:
      if g.name in replaced:
        games[g.name] = g
    for m, name, default in settings:
      _set(m, name, default)
    os.environ.pop(plugins.ENV, None)
    yield
  finally:
    for name, row in replaced.items():
      if row is None:
        games.pop(name, None)
      else:
        games[name] = row
    for (m, name, _), old in zip(settings, saved):
      _set(m, name, old)
    if saved_env is None:
      os.environ.pop(plugins.ENV, None)
    else:
      os.environ[plugins.ENV] = saved_env
