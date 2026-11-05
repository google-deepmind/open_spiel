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

The same package installed under another name (a release imports it as
open_spiel.python.<name>) still reads what an install under its earlier name
recorded: `register_package_alias(old)` makes every path under `old` resolve
into this package, and `register_object_alias` names one moved class or
function by its recorded path without setting anything on a module. Pickles
go through the same resolution with `loads` (or `Unpickler`), whose
`find_class` resolves each global a pickle names:

  registry.register_package_alias("acme_rl")    # recorded acme_rl.* paths
  state = registry.loads(path.read_bytes())     # acme_rl.core.types:Foo ok
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import pickle
import re
from typing import Any, Callable, Optional, Sequence

PathResolver = Callable[[str], Optional[str]]

_RESOLVERS: list[PathResolver] = []

_PACKAGE = __name__.rpartition(".")[0]

#: This package's import name: its source name in a checkout, its release
#: name (open_spiel.python.<name>) in an install of the release.
PACKAGE: str = _PACKAGE

_MODULE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

# Package names recorded paths may use for this package (package aliases).
_PACKAGE_ALIASES: list[str] = []

# Recorded "pkg.module:attr" paths that name one object (object aliases).
_OBJECTS: dict[str, Any] = {}

_MISSING = object()

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


def register_package_alias(recorded: str) -> None:
  """Paths recorded under package name `recorded` resolve into this package.

  An install of this package under another import name (a release imports it
  as open_spiel.python.<name>) then reads configs, artifacts and pickles that
  an install under `recorded` wrote: "<recorded>.envs.kuhn:KuhnEnv" resolves
  to "<PACKAGE>.envs.kuhn:KuhnEnv" before any path resolver sees it.
  Registering this package's own name, or the same alias twice, has no
  effect.

  Args:
    recorded: The package name the paths were recorded under.

  Raises:
    ValueError: `recorded` is not a dotted module name, or it holds this
      package or lies inside it (a path under it would be ambiguous).
  """
  if not isinstance(recorded, str) or not _MODULE.match(recorded):
    raise ValueError(
        f"a package alias is a dotted module name, got {recorded!r}"
    )
  if recorded == _PACKAGE or recorded in _PACKAGE_ALIASES:
    return
  if recorded.startswith(_PACKAGE + ".") or _PACKAGE.startswith(recorded + "."):
    raise ValueError(
        f"package alias {recorded!r} holds or lies inside this package"
        f" ({_PACKAGE!r}): a path under it would be ambiguous"
    )
  for other in _PACKAGE_ALIASES:
    if recorded.startswith(other + ".") or other.startswith(recorded + "."):
      raise ValueError(
          f"package alias {recorded!r} overlaps the alias {other!r}"
      )
  _PACKAGE_ALIASES.append(recorded)


def package_aliases() -> tuple[str, ...]:
  """The package names registered with `register_package_alias`, in order.

  Returns:
    The names, oldest registration first.
  """
  return tuple(_PACKAGE_ALIASES)


def package_path(dotted: str) -> str:
  """`dotted` made absolute, with a package alias replaced by PACKAGE.

  Args:
    dotted: A recorded path ("pkg.module[:attr]" or ".module[:attr]").

  Returns:
    The path as this install names it; a path outside this package and its
    aliases as given.

  Raises:
    TypeError: `dotted` is not a string.
    ValueError: a relative path that names no module (`absolute_path`).
  """
  if not isinstance(dotted, str):
    raise TypeError(f"a dotted path is a string, got {dotted!r}")
  dotted = absolute_path(dotted)
  mod = dotted.partition(":")[0]
  for alias in _PACKAGE_ALIASES:
    if mod == alias or mod.startswith(alias + "."):
      return _PACKAGE + dotted[len(alias) :]
  return dotted


def register_object_alias(path: str, obj: Any) -> None:
  """The recorded path `path` ("pkg.module:attr") names `obj`.

  For a class or function that moved while the module it was recorded in
  still exists: `load_object(path)` and unpickling (`loads`, `Unpickler`)
  return `obj`, and nothing is set on the module. An object alias is looked
  up before the path is resolved, under any package alias of the path.

  Args:
    path: The recorded path.
    obj: The object it names now.

  Raises:
    TypeError: `path` is not a string.
    ValueError: `path` is not "module:attr", or it already names another
      object.
  """
  if not isinstance(path, str):
    raise TypeError(f"a dotted path is a string, got {path!r}")
  mod, sep, attr = absolute_path(path).partition(":")
  if not sep or not _MODULE.match(mod) or not _MODULE.match(attr):
    raise ValueError(f"want 'module:attr', got {path!r}")
  got = _OBJECTS.get(path, _MISSING)
  if got is not _MISSING and got is not obj:
    raise ValueError(f"{path!r} already names {got!r}")
  _OBJECTS[path] = obj


def _object_alias(path: str) -> Any:
  """The registered object at `path` or at another name of its package.

  Args:
    path: a dotted object path.

  Returns:
    The object, or _MISSING when none is registered under any alias.
  """
  got = _OBJECTS.get(path, _MISSING)
  if got is not _MISSING or not _OBJECTS:
    return got
  want = package_path(path)
  for key, obj in _OBJECTS.items():
    if package_path(key) == want:
      return obj
  return _MISSING


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
    (`moved_module_path`); else `dotted` itself. The resolvers and the moved
    modules see the path as `package_path` gives it: absolute, and under
    PACKAGE where it was recorded under a package alias.

  Raises:
    TypeError: `dotted` is not a string, or a resolver returned a non-string.
    ValueError: a relative path that names no module (".", ".:attr", "..x").
  """
  dotted = package_path(dotted)
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
    The object an object alias gives the path (before or after resolution),
    else the named attribute.

  Raises:
    ValueError: `path` has no ":attr" part.
    ImportError, AttributeError: the module or attribute does not exist.
  """
  obj = _object_alias(path) if isinstance(path, str) else _MISSING
  if obj is not _MISSING:
    return obj
  resolved = resolve_path(path)
  obj = _object_alias(resolved)
  if obj is not _MISSING:
    return obj
  if ":" not in resolved:
    raise ValueError(f"want 'module:attr', got {path!r}")
  mod, attr = resolved.split(":", 1)
  obj = importlib.import_module(mod)
  for part in attr.split("."):
    obj = getattr(obj, part)
  return obj


class Unpickler(pickle.Unpickler):
  """A pickle.Unpickler whose globals go through this module's resolution.

  `find_class(module, name)` looks the recorded "module:name" up as
  `load_object` does: an object alias first, then `resolve_path` (relative
  and package-aliased paths, path resolvers, MOVED_MODULES), then the
  attribute of the resolved module. A pickle written by an install of this
  package under another name, or before a module moved, loads unchanged and
  no module is altered to make it load. Globals outside this package resolve
  to themselves unless a resolver maps them.
  """

  def find_class(self, module: str, name: str) -> Any:
    """The object a pickled global names, resolved as the class doc says.

    Args:
      module: the recorded module path.
      name: the recorded attribute name.

    Returns:
      The object `module:name` resolves to.

    Raises:
      TypeError: path resolution turned the global into a path with no
        attribute.
    """
    path = f"{module}:{name}"
    obj = _object_alias(path)
    if obj is not _MISSING:
      return obj
    resolved = resolve_path(path)
    obj = _object_alias(resolved)
    if obj is not _MISSING:
      return obj
    mod, sep, attr = resolved.partition(":")
    if not sep:
      raise TypeError(
          f"path resolution turned the pickled global {path!r} into"
          f" {resolved!r}, which names no attribute"
      )
    return super().find_class(mod, attr)


def loads(data: bytes) -> Any:
  """`pickle.loads(data)` through `Unpickler` (recorded paths resolve).

  Only for pickles this package's own processes wrote (a run's state): like
  pickle.loads, it runs what the pickle names.

  Args:
    data: The pickle's bytes.

  Returns:
    The unpickled object.
  """
  return Unpickler(io.BytesIO(data)).load()
