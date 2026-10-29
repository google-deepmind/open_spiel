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

"""Resolution of the dotted paths that configs and artifacts record.

Configs and artifacts name callables and classes by string: a routed policy's
route function, a belief sampler class, a network factory, each as
"pkg.module:attr". A path resolver maps a recorded string to the path the
code has now, so a module can move without invalidating the strings that name
it. Every registry that turns a recorded string into an object goes through
`resolve_path` first. With `registry` this module:

  registry.register_path_resolver(
      lambda p: p.replace("acme.old_routes", "acme.routes", 1)
      if p.startswith("acme.old_routes") else None)
  fn = registry.load_object("acme.old_routes:endgame")   # acme.routes.endgame

Resolvers run in registration order and the first non-None answer wins. When
none answers, a path into a module this package moved (MOVED_MODULES) resolves
to the module's new home; any other path resolves to itself.

A path that starts with "." names a module of this package by its place in
it (".envs.gin.endgame:in_domain"), whatever name the package is imported
under: an artifact that records such paths loads in any install of the
package. It is made absolute before the resolvers see it.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Callable, Optional, Sequence

PathResolver = Callable[[str], Optional[str]]

_RESOLVERS: list[PathResolver] = []

_PACKAGE = __name__.rpartition(".")[0]

#: Modules that moved inside this package, old to new (package-relative).
#: Strings recorded before a move (configs, manifests) still resolve.
MOVED_MODULES: dict[str, str] = {
    "configs.common": "league.config_common",
    "learners.advantages": "core.advantages",
    "learners.schedules": "core.schedules",
    "league.layout": "ckpt.layout",
    "eval.arena_engine": "core.arena_engine",
    "eval.deadline": "core.deadline",
    "eval.game_tree": "core.game_tree",
    "belief.particle": "policies.particle_belief",
}


def moved_module_path(dotted: str) -> Optional[str]:
  """The current form of a path into a module listed in MOVED_MODULES.

  Args:
    dotted: A recorded path ("pkg.module" or "pkg.module:attr").

  Returns:
    The path with the moved module's new name, or None when it names no
    moved module.
  """
  mod, sep, attr = dotted.partition(":")
  for old, new in MOVED_MODULES.items():
    old_mod = f"{_PACKAGE}.{old}"
    if mod == old_mod or mod.startswith(old_mod + "."):
      return f"{_PACKAGE}.{new}{mod[len(old_mod):]}{sep}{attr}"
  return None


def absolute_path(dotted: str) -> str:
  """`dotted` with a leading "." replaced by this package's name.

  Args:
    dotted: A recorded path; relative when it starts with ".".

  Returns:
    The absolute path ("<package>.module[:attr]"); an absolute path as given.

  Raises:
    ValueError: a relative path whose module part is empty or starts with
      another "." (only one level, this package, is meant).
  """
  if not dotted.startswith("."):
    return dotted
  module = dotted[1:].partition(":")[0]
  if not module or module.startswith("."):
    raise ValueError(
        f"a relative path is '.module' or '.module:attr', got {dotted!r}"
    )
  return _PACKAGE + dotted


def register_path_resolver(fn: PathResolver) -> None:
  """Adds `fn` to the resolvers `resolve_path` consults.

  Args:
    fn: maps a dotted path ("pkg.module" or "pkg.module:attr") to its current
      form, or returns None when it does not apply. Registering the same
      function twice has no effect.

  Raises:
    TypeError: fn is not callable.
  """
  if not callable(fn):
    raise TypeError(f"path resolver {fn!r} is not callable")
  if fn not in _RESOLVERS:
    _RESOLVERS.append(fn)


def absent_package(
    missing: Optional[str], packages: Sequence[str]
) -> Optional[str]:
  """The entry of `packages` that a ModuleNotFoundError shows is absent.

  An entry is a package's full import name (it may be nested, "a.b.games").
  It is returned when `missing` is the entry, a module inside it or a package
  holding it, and the entry does not import; otherwise None (the error is
  then not about an optional package, and the caller re-raises it).

  Args:
    missing: The ModuleNotFoundError's module name (its `name`); may be None.
    packages: The optional packages' full import names.

  Returns:
    The absent entry, or None.
  """
  missing = str(missing or "")
  if not missing:
    return None
  for pkg in packages:
    if (
        missing == pkg
        or missing.startswith(pkg + ".")
        or pkg.startswith(missing + ".")
    ):
      try:
        found = importlib.util.find_spec(pkg) is not None
      except ModuleNotFoundError:  # a package holding it does not import
        found = False
      return None if found else pkg
  return None


def resolve_path(dotted: str) -> str:
  """The current form of a recorded dotted path.

  Args:
    dotted: The recorded path ("pkg.module" or "pkg.module:attr"), or one
      relative to this package (".module" or ".module:attr", module doc).

  Returns:
    The first non-None resolver answer; else the moved module's new path
    (`moved_module_path`); else `dotted` itself (made absolute).

  Raises:
    TypeError: `dotted` is not a string, or a resolver returned a non-string.
    ValueError: a relative path that names no module (".", ".:attr", "..x").
  """
  if not isinstance(dotted, str):
    raise TypeError(f"a dotted path is a string, got {dotted!r}")
  dotted = absolute_path(dotted)
  for fn in _RESOLVERS:
    got = fn(dotted)
    if got is not None:
      if not isinstance(got, str):
        raise TypeError(f"path resolver {fn!r} returned {got!r} for {dotted!r}")
      return got
  moved = moved_module_path(dotted)
  return dotted if moved is None else moved


def load_object(path: str) -> Any:
  """The object a "pkg.module:attr" path names, after `resolve_path`.

  Args:
    path: The recorded path.

  Returns:
    The named object.

  Raises:
    ValueError: `path` has no ":attr" part.
    ImportError, AttributeError: the module or attribute does not exist.
  """
  resolved = resolve_path(path)
  if ":" not in resolved:
    raise ValueError(f"want 'module:attr', got {path!r}")
  mod, attr = resolved.split(":", 1)
  obj = importlib.import_module(mod)
  for part in attr.split("."):
    obj = getattr(obj, part)
  return obj
