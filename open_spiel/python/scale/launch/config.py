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

"""The backend config file (YAML), how it is read, and the backend registry.

One checked-in file describes where runs execute:

  backend: local                  backend: gcp
  root: /data/runs                project: ...   (every field; see gcp.py)
  python: python3

  backend = config.make_backend(config.load("backend.yaml"))

Every field a backend reads is required: a missing or unknown field raises
BackendFileError naming it, and nothing defaults to any project, bucket, network
or account. `${NAME}` anywhere in a value is replaced by the environment
variable NAME (an unset one raises), so one file can name a deployment's
resources through its own environment. Three names are built in (the
environment may override them), so a file can ship this package and the
games it trains on wherever they are installed and whatever their import
names:

  PACKAGE_ROOT        the directory the package is imported from (a sys.path
                      entry)
  PACKAGE_PATH        the package folder relative to it (its import name,
                      dotted parts as folders)
  GAMES_PACKAGE_PATH  the JAX games package folder relative to PACKAGE_ROOT
                      (deps.GAMES_PACKAGE: every game but kuhn_poker_jax
                      needs it); a file that uses it raises
                      BackendFileError when the games package is not
                      installed under PACKAGE_ROOT

The `backend` field picks a factory registered with register_backend(kind,
factory); "local" and "gcp" are built in.
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import Any, Callable, Mapping

from open_spiel.python.scale import deps
from open_spiel.python.scale.launch import local

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def builtins() -> dict[str, Any]:
  """The built-in names (module doc), from this module's location.

  GAMES_PACKAGE_PATH is a BackendFileError (raised where a file uses it) when
  the games package is not importable from PACKAGE_ROOT under its import name.

  Returns:
    {name: value}, a BackendFileError value for an unavailable name.

  Raises:
    BackendFileError: the package is not at its folder under its import name.
  """
  parts = __name__.split(".")[:-2]  # <package>.launch.config
  pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  root = pkg_dir
  for _ in parts:
    root = os.path.dirname(root)
  if os.path.realpath(os.path.join(root, *parts)) != os.path.realpath(pkg_dir):
    raise BackendFileError(
        f"package {'.'.join(parts)!r} is not at {pkg_dir} under its import name"
    )
  return {
      "PACKAGE_ROOT": root,
      "PACKAGE_PATH": "/".join(parts),
      "GAMES_PACKAGE_PATH": _games_path(root),
  }


def _games_path(root: str) -> Any:
  """deps.GAMES_PACKAGE's folder relative to root, or a BackendFileError.

  deps.GAMES_PACKAGE's folder relative to root, or the BackendFileError saying
  why it is not there.

  Args:
    root: the code root the package is shipped from.

  Returns:
    the folder relative to root, or the BackendFileError saying why not.
  """
  name = deps.GAMES_PACKAGE
  rel = name.replace(".", "/")
  try:
    spec = importlib.util.find_spec(name)
  except ModuleNotFoundError:  # a package holding it does not import
    spec = None
  if spec is None or not spec.submodule_search_locations:
    return BackendFileError(
        f"the JAX games package ({name}) is not installed here, so it cannot "
        "be "
        "shipped; install it (it comes with OpenSpiel) or drop "
        "${GAMES_PACKAGE_PATH} (then only kuhn_poker_jax runs)"
    )
  found = os.path.realpath(list(spec.submodule_search_locations)[0])
  if found != os.path.realpath(os.path.join(root, rel)):
    return BackendFileError(
        f"the JAX games package ({name}) is at {found}, not under {root} where "
        "this package is; set GAMES_PACKAGE_PATH and the code root to match"
    )
  return rel


class BackendFileError(ValueError):
  """The backend config file is malformed."""


#: factory(body, base_dir, dry_run) -> Backend. body: the config without its
#: `backend` field; base_dir: the config file's directory (relative paths
#: resolve against it); dry_run: create nothing.
BackendFactory = Callable[[Mapping[str, Any], str, bool], Any]
_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(kind: str, factory: BackendFactory) -> None:
  """Makes `backend: <kind>` in a config file build its Backend with factory.

  Registering a kind twice raises ValueError.

  Args:
    kind: the config's `backend` value.
    factory: factory(body, dry_run=...) -> Backend.

  Raises:
    ValueError: the kind is empty or already registered.
  """
  if not isinstance(kind, str) or not kind:
    raise ValueError(f"backend kind {kind!r}: want a non-empty string")
  if kind in _BACKENDS:
    raise ValueError(f"backend kind {kind!r} is already registered")
  _BACKENDS[kind] = factory


def backends() -> tuple[str, ...]:
  """The registered backend kinds, sorted.

  Returns:
    the kinds.
  """
  return tuple(sorted(_BACKENDS))


def interpolate(value: Any, environ: Mapping[str, str], where: str = "") -> Any:
  """`value` with every ${NAME} replaced from environ, recursively.

  Args:
    value: a config value (str, mapping, list or scalar).
    environ: the names and their values.
    where: the value's path, for messages.

  Returns:
    the interpolated value.
  """
  if isinstance(value, str):

    def sub(m):
      """The value of the ${NAME} that m matched.

      Args:
        m: the match.

      Returns:
        the value.

      Raises:
        BackendFileError: the name is not set, or is an unavailable built-in.
      """
      name = m.group(1)
      if name not in environ:
        raise BackendFileError(
            f"{where or 'config'}: ${{{name}}} is not set in the environment"
        )
      got = environ[name]
      # a built-in name that is not available (builtins)
      if isinstance(got, BackendFileError):
        raise BackendFileError(f"{where or 'config'}: ${{{name}}}: {got}")
      return got

    return _VAR.sub(sub, value)
  if isinstance(value, Mapping):
    return {
        k: interpolate(v, environ, f"{where}.{k}" if where else str(k))
        for k, v in value.items()
    }
  if isinstance(value, (list, tuple)):
    return [
        interpolate(v, environ, f"{where}[{i}]") for i, v in enumerate(value)
    ]
  return value


_NoneType = type(None)


def require(d: Any, where: str, keys: Mapping[str, Any]) -> dict[str, Any]:
  """`d` as a dict, checked against `keys`.

  `d` as a dict, which must be a mapping with exactly `keys`
  ({name: type or tuple of types}). None is accepted only where the types
  include it; a bool is accepted only where bool is listed.

  Args:
    d: the value.
    where: its path, for messages.
    keys: {name: type or tuple of types}.

  Returns:
    d as a dict.

  Raises:
    BackendFileError: d is not a mapping, misses or adds a field, or a field
        has the wrong type.
  """
  if not isinstance(d, Mapping):
    raise BackendFileError(f"{where}: want a mapping, got {type(d).__name__}")
  missing = sorted(set(keys) - set(d))
  unknown = sorted(set(d) - set(keys))
  if missing:
    raise BackendFileError(
        f"{where}: missing field(s) {missing} (every field is required; there "
        "are no defaults)"
    )
  if unknown:
    raise BackendFileError(
        f"{where}: unknown field(s) {unknown}; fields: {sorted(keys)}"
    )
  out = dict(d)
  for k, t in keys.items():
    ts = t if isinstance(t, tuple) else (t,)
    v = out[k]
    if v is None and _NoneType not in ts:
      raise BackendFileError(f"{where}.{k}: may not be null")
    if v is not None and not isinstance(
        v, tuple(x for x in ts if x is not _NoneType)
    ):
      raise BackendFileError(
          f"{where}.{k}: want {' | '.join(x.__name__ for x in ts)}, got "
          f"{type(v).__name__} ({v!r})"
      )
    if isinstance(v, bool) and bool not in ts:
      raise BackendFileError(f"{where}.{k}: want {ts}, got a bool")
  return out


def load(path: str, environ: Mapping[str, str] = None) -> dict[str, Any]:
  """The parsed, interpolated config file (a mapping with a backend field).

  Args:
    path: the config file.
    environ: the environment (default os.environ, with the built-in names).

  Returns:
    the config.

  Raises:
    BackendFileError: the file is not a mapping with a backend field.
  """
  # Lazy: an optional dependency (PyYAML reads the file; the launcher host
  # needs it only for a config file).
  import yaml  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  with open(path, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  if not isinstance(raw, Mapping) or "backend" not in raw:
    raise BackendFileError(f"{path}: want a mapping with a 'backend' field")
  cfg = interpolate(
      raw, {**builtins(), **(os.environ if environ is None else environ)}
  )
  cfg["_dir"] = os.path.dirname(os.path.abspath(path))
  return cfg


def make_backend(cfg: Mapping[str, Any], *, dry_run: bool = False):
  """The Backend a loaded config names (its `backend` field).

  Args:
    cfg: a loaded config.
    dry_run: build the Backend for a dry run.

  Returns:
    the Backend.

  Raises:
    BackendFileError: an unregistered kind, or the factory's own field errors.
  """
  kind = cfg.get("backend")
  if kind not in _BACKENDS:
    raise BackendFileError(f"backend {kind!r}: want one of {list(backends())}")
  body = {k: v for k, v in cfg.items() if k not in ("backend", "_dir")}
  return _BACKENDS[kind](body, cfg.get("_dir", os.getcwd()), dry_run)


def _local(body: Mapping[str, Any], base_dir: str, dry_run: bool):
  """The local backend of a config body.

  Args:
    body: The config body.
    base_dir: The config's folder.
    dry_run: Unused.

  Returns:
    The LocalBackend.
  """
  del dry_run
  d = require(body, "local", {"root": str, "python": str})
  root = (
      d["root"]
      if os.path.isabs(d["root"])
      else os.path.join(base_dir, d["root"])
  )
  return local.LocalBackend(root, python=d["python"])


def _gcp(body: Mapping[str, Any], base_dir: str, dry_run: bool):
  """The GCP backend of a config body.

  Args:
    body: The config body.
    base_dir: The config's folder.
    dry_run: Whether to only log commands.

  Returns:
    The GcpBackend.
  """
  # Lazy: a cycle (gcp imports this module).
  from open_spiel.python.scale.launch import gcp  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return gcp.GcpBackend(gcp.GcpConfig.parse(body, base_dir), dry_run=dry_run)


register_backend("local", _local)
register_backend("gcp", _gcp)
