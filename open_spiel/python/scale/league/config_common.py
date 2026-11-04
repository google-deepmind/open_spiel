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

"""Config loading, command-line overrides and config hashing for training runs.

Every run config is a `league.runner.LeagueRunConfig`: an env, a learner kind
(`league.kinds`), a sampler rule and a controller. A config module defines
`get_config()`. `load_config` takes a config name (a module of this
package, `configs`, or of a package registered with
`register_config_package`), a "<prefix><rest>" name that a resolver
registered with `register_config_resolver` builds, or a path to a .py file.

  cfg = common.load_config("mmd_kuhn_poker_jax")
  cfg = common.apply_overrides(cfg, ["seed=3", "league.num_envs=64"])
  cfg = common.with_overrides(cfg, num_updates=500)
  common.config_hash(cfg), common.resume_hash(cfg)

Overrides. `apply_overrides` sets fields by dotted path
("policy_kwargs.network.torso.widths=(8,)", or "learner.learner.lr=1e-4": a
LeagueRunConfig's `learner` is the learner kind's spec, and that spec's
`learner` field is the optimiser config), refusing unknown paths and type
changes. `with_overrides` sets the run length, the batch geometry and the seed
in every place the run's parts hold them; the learner kind says where
(`LearnerKind.with_run_updates`, `with_batch`, `with_seed`).

Hashing. `config_hash` is the SHA-256 of the canonical JSON of the config:
`plain(cfg)` (every dataclass field, nothing else) passed through each
registered hash transform in registration order. A transform is a
canonicalisation step: it decides which form of a config identifies it, for
example by dropping a field that only labels a run.
`resume_hash` is the same hash with the run-length fields cleared when the
learner's step sizes do not depend on the run length, so such a run may be
extended on resume; a resumed run must match it.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from typing import Any, Callable, Mapping, Optional

from open_spiel.python.scale import plugins
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry


def _runner():
  """The runner module, imported on first use.

  Returns:
    The runner module.
  """
  # Lazy: a cycle (runner imports this module).
  from open_spiel.python.scale.league import runner  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return runner


def env_names() -> tuple[str, ...]:
  """Every trainable game: the env registry's names, after the plugins load.

  Returns:
    The names.
  """
  plugins.load_env()
  return env_registry.names()


def _check_env(name: str) -> None:
  known = env_names()
  if name not in known:
    raise KeyError(f"unknown env {name!r}; registered: {known}")


def learner_kind(cfg: Any) -> str:
  """The name of the learner kind of a run config.

  Args:
    cfg: a run config.

  Returns:
    The kind name.

  Raises:
    TypeError: cfg is not a league.runner.LeagueRunConfig.
  """
  runner = _runner()
  if not isinstance(cfg, runner.LeagueRunConfig):
    raise TypeError(
        f"{type(cfg).__name__} is not a run config "
        "(league.runner.LeagueRunConfig)"
    )
  return cfg.learner.kind


def _kind(cfg: Any):
  """The learner kind of `cfg`.

  Args:
    cfg: The run config.

  Returns:
    The kind.
  """
  # Lazy: cycle (kinds.bc imports this module); configs import without jax.
  from open_spiel.python.scale.league import kinds  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  learner_kind(cfg)
  return kinds.of(cfg.learner)


def run_length(cfg: Any) -> int:
  """The run's length in league iterations (a one-slot run updates once each).

  Args:
    cfg: a run config.

  Returns:
    The length.
  """
  learner_kind(cfg)
  return int(cfg.num_iterations)


def with_overrides(
    cfg: Any,
    *,
    num_updates: Optional[int] = None,
    num_envs: Optional[int] = None,
    num_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Any:
  """Cfg with the run size and seed set everywhere the run's parts hold them.

  Args:
    cfg: a run config.
    num_updates: the run length in league iterations. Under a single-learner
      controller the learner's schedule horizon moves with it
      (`LearnerKind.with_run_updates`).
    num_envs: the Arena's rows per update (`LearnerKind.with_batch` applies it
      to the learner's own copy, if it keeps one).
    num_steps: plies per update (refused by a kind that sets them itself).
    seed: the run's seed, also given to the League and the learner kind.

  Returns:
    The updated run config.

  Raises:
    ValueError: a value is invalid or the kind refuses it.
  """
  kind = _kind(cfg)
  runner = _runner()
  kw: dict[str, Any] = {}
  league, learner = cfg.league, cfg.learner
  if num_updates is not None:
    n = int(num_updates)
    kw["num_iterations"] = n
    if runner.single_learner_key(cfg.controller) is not None:
      learner = kind.with_run_updates(
          learner, n * int(league.updates_per_iteration)
      )
  if num_envs is not None:
    league = dataclasses.replace(league, num_envs=int(num_envs))
    learner = kind.with_batch(learner, int(num_envs))
  if num_steps is not None:
    if getattr(kind, "fixed_num_steps", False):
      raise ValueError(
          f"num_steps does not apply to {kind.name} configs (one whole game "
          "per row)"
      )
    league = dataclasses.replace(league, num_steps=int(num_steps))
  if seed is not None:
    kw["seed"] = int(seed)
    league = dataclasses.replace(league, seed=int(seed))
    learner = kind.with_seed(learner, int(seed))
  if league is not cfg.league:
    kw["league"] = league
  if learner is not cfg.learner:
    kw["learner"] = learner
  return dataclasses.replace(cfg, **kw) if kw else cfg


def _parse_value(text: str) -> Any:
  """A Python literal, or the text itself.

  Args:
    text: The value text.

  Returns:
    The parsed value.
  """
  try:
    return ast.literal_eval(text)
  except (ValueError, SyntaxError):
    return text  # a bare string (e.g. an encoder name)


def _coerce(old: Any, new: Any, path: str) -> Any:
  """The override `new` for a field holding `old`, coerced to its type.

  An int becomes a float for a float field, and a list or tuple becomes the
  field's sequence type.

  Args:
    old: the field's current value.
    new: the override.
    path: the override's dotted path, for the error message.

  Returns:
    the value to set.

  Raises:
    TypeError: new's type does not match old's.
  """
  if isinstance(old, bool) or isinstance(new, bool):
    if (
        not (isinstance(old, bool) and isinstance(new, bool))
        and old is not None
    ):
      raise TypeError(f"{path}: {new!r} for a {type(old).__name__} field")
    return new
  if isinstance(old, float) and isinstance(new, int):
    return float(new)
  if isinstance(old, tuple) and isinstance(new, list):
    return tuple(new)
  if isinstance(old, list) and isinstance(new, tuple):
    return list(new)
  if old is not None and new is not None and not isinstance(new, type(old)):
    raise TypeError(
        f"{path}: {new!r} ({type(new).__name__}) for a {type(old).__name__} "
        "field"
    )
  return new


def _set_path(obj: Any, keys: list[Any], value: Any, path: str) -> Any:
  """A copy of `obj` with the field at `keys` set to `value`.

  Descends through dataclasses, mappings and tuples (by index).

  Args:
    obj: the config, or a part of it.
    keys: the remaining path components.
    value: the new value.
    path: the full dotted path, for error messages.

  Returns:
    the updated copy of obj.

  Raises:
    KeyError: a component names no field, key or index of its parent.
  """
  head, rest = keys[0], keys[1:]
  if dataclasses.is_dataclass(obj):
    names = {f.name for f in dataclasses.fields(obj)}
    if head not in names:
      raise KeyError(f"{path}: {type(obj).__name__} has no field {head!r}")
    cur = getattr(obj, head)
    new = (
        _set_path(cur, rest, value, path) if rest else _coerce(cur, value, path)
    )
    return dataclasses.replace(obj, **{head: new})
  if isinstance(obj, Mapping):
    if head not in obj:
      raise KeyError(f"{path}: no key {head!r} (keys {sorted(obj)})")
    cur = obj[head]
    new = (
        _set_path(cur, rest, value, path) if rest else _coerce(cur, value, path)
    )
    return {**dict(obj), head: new}
  if isinstance(obj, tuple) and rest:
    # In a tuple of config parts (the in-loop evals, for example) an index
    # descends into one element; a path that ends at the tuple sets all of it.
    if not head.isdigit() or int(head) >= len(obj):
      raise KeyError(
          f"{path}: {head!r} is not an index 0..{len(obj) - 1} of a "
          f"{len(obj)}-tuple (index {head})"
      )
    i = int(head)
    return obj[:i] + (_set_path(obj[i], rest, value, path),) + obj[i + 1 :]
  raise KeyError(
      f"{path}: cannot descend into a {type(obj).__name__} at {head!r}"
  )


#: Override names that `with_overrides` applies to every part holding them.
WITH_OVERRIDES = ("seed", "num_envs", "num_steps")


def run_length_paths(cfg: Any) -> tuple[str, ...]:
  """Dotted paths of the fields that set the run length (refused as overrides).

  Args:
    cfg: a run config.

  Returns:
    The paths.
  """
  return ("num_iterations",) + tuple(
      f"learner.{p}" for p in _kind(cfg).run_length_paths()
  )


def apply_overrides(cfg: Any, overrides) -> Any:
  """Cfg with each "dotted.path=value" applied.

  The value is a Python literal, else a bare string. "seed", "num_envs" and
  "num_steps" go through `with_overrides`.

  Args:
    cfg: a run config.
    overrides: the "dotted.path=value" items.

  Returns:
    The updated run config.

  Raises:
    ValueError: an item is not path=value, or sets a run-length field (set the
      run length with `with_overrides(num_updates=...)`, which moves the
      schedule horizon with it).
    KeyError: an unknown path.
    TypeError: a value of another type than the field's.
  """
  for item in overrides or ():
    if "=" not in item:
      raise ValueError(f"override {item!r} is not path=value")
    path, text = item.split("=", 1)
    path = path.strip()
    if path in run_length_paths(cfg):
      raise ValueError(
          f"override {path!r}: set the run length with "
          "with_overrides(cfg, num_updates=...)"
      )
    value = _parse_value(text.strip())
    # fields several config parts share (with_overrides)
    if path in WITH_OVERRIDES:
      if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}: {value!r} is not an int")
      cfg = with_overrides(cfg, **{path: value})
      continue
    cfg = _set_path(cfg, path.split("."), value, path)
  learner_kind(cfg)
  return cfg


# -- hashing -------------------------------------------------------------------

HashTransform = Callable[[dict[str, Any]], dict[str, Any]]
_HASH_TRANSFORMS: list[HashTransform] = []


def register_hash_transform(fn: HashTransform) -> None:
  """Adds a transform that `config_dict`, `config_hash` and `resume_hash` apply.

  Args:
    fn: maps the JSON-ready config dict to the dict that is hashed. Transforms
      run in registration order; registering one twice has no effect. A
      transform must not mutate its input.

  Raises:
    TypeError: fn is not callable.
  """
  if not callable(fn):
    raise TypeError(f"hash transform {fn!r} is not callable")
  if fn not in _HASH_TRANSFORMS:
    _HASH_TRANSFORMS.append(fn)


def plain(x: Any) -> Any:
  """X as nested dicts, lists and scalars: every dataclass field, nothing else.

  Args:
    x: a config value.

  Returns:
    The plain value.
  """
  if dataclasses.is_dataclass(x) and not isinstance(x, type):
    return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
  if isinstance(x, Mapping):
    return {str(k): plain(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [plain(v) for v in x]
  return x


def _transformed(d: dict[str, Any]) -> dict[str, Any]:
  """`d` after every registered hash transform.

  Args:
    d: The config dict.

  Returns:
    The transformed dict.

  Raises:
    TypeError: A transform returned something else.
  """
  for fn in _HASH_TRANSFORMS:
    d = fn(d)
    if not isinstance(d, dict):
      raise TypeError(
          f"hash transform {fn!r} returned {type(d).__name__}, not a dict"
      )
  return d


def config_dict(cfg: Any) -> dict[str, Any]:
  """The dict a config hashes as: `plain(cfg)` through every hash transform.

  Args:
    cfg: a run config.

  Returns:
    The dict.
  """
  return _transformed(json.loads(json.dumps(plain(cfg))))


def _sha(d: dict[str, Any]) -> str:
  """The digest of a dict's canonical JSON.

  Args:
    d: The dict.

  Returns:
    The hex SHA-256.
  """
  return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def config_hash(cfg: Any) -> str:
  """The hash of a run config's `config_dict`.

  Args:
    cfg: a run config.

  Returns:
    The hex digest.
  """
  return _sha(config_dict(cfg))


def _clear(d: dict[str, Any], path: str) -> None:
  *head, last = path.split(".")
  for k in head:
    d = d[k]
  if last not in d:
    raise KeyError(f"run-length path {path!r} is not in the config")
  d[last] = None


def resume_hash(cfg: Any) -> str:
  """The hash a resumed run must match.

  It is `config_hash` with the run-length fields (`LearnerKind.
  extendable_hash_fields`, plus num_iterations) cleared when the learner's
  step sizes do not depend on the run length (`constant_lr()` of the learner
  spec), so such a run may be extended; a run whose schedule horizon is its
  length may not.

  Args:
    cfg: a run config.

  Returns:
    The hex digest.
  """
  kind = _kind(cfg)
  d = json.loads(json.dumps(plain(cfg)))
  if cfg.learner.constant_lr():
    _clear(d, "num_iterations")
    for p in kind.extendable_hash_fields():
      _clear(d, f"learner.{p}")
  d = _transformed(d)
  # Domain separation: the resume hash of a config never equals its
  # config_hash, even when no run-length field is cleared.
  d["kind"] = "league"
  return _sha(d)


# -- builders ------------------------------------------------------------------


def build_env_by_name(name: str, **kwargs: Any) -> types.Env:
  """The env `name` built with kwargs (KeyError naming the registered envs).

  Args:
    name: a registered env name.
    **kwargs: env options.

  Returns:
    The env.
  """
  _check_env(name)
  return env_registry.make(name, **kwargs)


def env_spec(cfg: Any) -> "env_registry.EnvSpec":
  """The EnvSpec of a run config: its game and env_kwargs, defaults filled.

  Args:
    cfg: a run config.

  Returns:
    The EnvSpec.
  """
  _check_env(cfg.env)
  return env_registry.spec(cfg.env, **dict(cfg.env_kwargs))


def build_env(cfg: Any) -> types.Env:
  """The env of a run config.

  Args:
    cfg: a run config.

  Returns:
    The env.
  """
  return build_env_by_name(cfg.env, **dict(cfg.env_kwargs))


def rules_sha(env: types.Env) -> str:
  """The env's rules identity (envs.registry.rules_sha).

  Args:
    env: an env.

  Returns:
    The rules hash.
  """
  return env_registry.rules_sha(env)


def network_policy_kwargs(
    game: str,
    encoder: str,
    obs_shape,
    num_actions: int,
    num_players: int,
    network: Mapping[str, Any],
) -> dict[str, Any]:
  """arch_kwargs of a "network" policy (networks.policy.NetworkPolicy).

  `encoder` names the function that builds the features and `obs_format` the
  feature format the policy declares; for a registry encoder the two are the
  same name, which NetworkPolicy checks, so the encoder name is passed as both.

  Args:
    game: the game name.
    encoder: the encoder name, also the declared obs_format.
    obs_shape: the encoder's output shape.
    num_actions: the game's action count.
    num_players: the game's seat count.
    network: the network config.

  Returns:
    The arch_kwargs.
  """
  return dict(
      game=game,
      obs_format=encoder,
      num_actions=int(num_actions),
      num_players=int(num_players),
      encoder=encoder,
      obs_shape=[int(d) for d in obs_shape],
      network=dict(network),
      value_layout="seat",
  )


# -- loading -------------------------------------------------------------------


#: Modules of this package that are helpers, not configs.
HELPER_MODULES = ("__init__.py", "common.py", "recipes.py")
# The configs package (this module is league.config_common), then
# register_config_package.
_PACKAGES: list[Any] = [__name__.rsplit(".", 2)[0] + ".configs"]
# name prefix -> fn(rest) -> config; register_config_resolver
_RESOLVERS: dict[str, Any] = {}


def register_config_package(package: str) -> None:
  """Makes every config module of `package` a config name.

  A config module is a module of `package` whose source defines
  `get_config` at top level (read, not imported; see defines_get_config), so
  a file a test runner generates beside the configs is not a config.
  Modules starting with "_" and the helpers the package lists in its
  HELPER_MODULES are skipped too (only those: a module of `package` named
  like one of this package's helpers is still a config when it defines
  get_config). A config name defined in two registered packages raises
  ValueError when config names are resolved (config_modules).

  Args:
    package: the package's dotted name.
  """
  assert isinstance(package, str) and package, package
  if package not in _PACKAGES:
    _PACKAGES.append(package)


def register_config_resolver(prefix: str, fn) -> None:
  """Makes load_config("<prefix><rest>") return fn(rest).

  For example for a table of generated configs.

  Args:
    prefix: the config name prefix.
    fn: rest -> run config.

  Raises:
    ValueError: the prefix is taken.
  """
  assert isinstance(prefix, str) and prefix and callable(fn), (prefix, fn)
  if prefix in _RESOLVERS and _RESOLVERS[prefix] is not fn:
    raise ValueError(f"config prefix {prefix!r} is already registered")
  _RESOLVERS[prefix] = fn


def defines_get_config(path: str) -> bool:
  """Whether the Python source at `path` binds `get_config` at top level.

  Read with ast, not imported, so listing config names runs no config code
  and needs none of the configs' dependencies. A def, an assignment or an
  import that binds the name counts.

  Args:
    path: a .py file.

  Returns:
    True when the module would have a get_config attribute.

  Raises:
    SyntaxError: the file is not valid Python (the message names it).
  """
  with open(path, encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=path)
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      names = [node.name]
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      names = [t.id for t in targets if isinstance(t, ast.Name)]
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      names = [a.asname or a.name.split(".")[0] for a in node.names]
    else:
      continue
    if "get_config" in names:
      return True
  return False


def _package_configs(package: str) -> dict[str, Any]:
  """{config name: module path} of the config modules of `package`.

  Args:
    package: a registered config package's dotted name.

  Returns:
    the mapping, over the modules register_config_package describes.
  """
  mod = importlib.import_module(package)
  here = os.path.dirname(mod.__file__)
  own = package == _PACKAGES[0]
  helpers = set(HELPER_MODULES if own else getattr(mod, "HELPER_MODULES", ()))
  return {
      f[:-3]: f"{package}.{f[:-3]}"
      for f in sorted(os.listdir(here))
      if f.endswith(".py")
      and not f.startswith("_")
      and f not in helpers
      and defines_get_config(os.path.join(here, f))
  }


def config_modules() -> dict[str, Any]:
  """{config name: module path} over every registered config package.

  Returns:
    The mapping.

  Raises:
    ValueError: a config name is defined in two registered packages.
  """
  plugins.load_env()
  out: dict[str, Any] = {}
  for pkg in _PACKAGES:
    for name, mod in _package_configs(pkg).items():
      if name in out:
        raise ValueError(f"config {name!r} is in {out[name]} and {mod}")
      out[name] = mod
  return out


def config_names() -> tuple[str, ...]:
  """Every config name, sorted.

  Returns:
    The names.
  """
  return tuple(sorted(config_modules()))


def config_module(name: str):
  """The module of config `name` (registered packages), or None.

  Args:
    name: the config name.

  Returns:
    The module, or None.
  """
  mods = config_modules()
  return importlib.import_module(mods[name]) if name in mods else None


def load_config(name_or_path: str) -> Any:
  """The run config a config name, a resolver name or a .py path builds.

  Args:
    name_or_path: a config name, a resolver name or a .py path.

  Returns:
    The run config.

  Raises:
    KeyError: an unknown config name.
    FileNotFoundError: a path that does not exist.
    ModuleNotFoundError: the config's game needs a package that is not
      installed; the message names it and a game that needs none.
    TypeError: get_config() returned something other than a run config.
  """
  try:
    return _load_config(name_or_path)
  except ModuleNotFoundError as e:
    root = env_registry.absent_game_package(e.name)
    if root is None or isinstance(e, env_registry.MissingGamePackageError):
      raise
    raise ModuleNotFoundError(
        f"config {name_or_path!r} needs {env_registry.GAME_PACKAGES[root]}, "
        "which is not "
        "installed; without it the trainer runs "
        f"{', '.join(env_registry.BUILTIN_STANDALONE)} "
        "(e.g. --config=mmd_kuhn_poker_jax)",
        name=root,
    ) from e


def _load_config(name_or_path: str) -> Any:
  """Loads a run config by resolver prefix, file path or config name.

  Args:
    name_or_path: a resolver-prefixed name, a .py path or a config name.

  Returns:
    The run config.

  Raises:
    FileNotFoundError: a path that does not exist.
    KeyError: an unknown config name.
  """
  plugins.load_env()
  for prefix, fn in _RESOLVERS.items():
    if name_or_path.startswith(prefix):
      cfg = fn(name_or_path[len(prefix) :])
      learner_kind(cfg)
      return cfg
  if name_or_path.endswith(".py") or os.sep in name_or_path:
    path = os.path.abspath(name_or_path)
    if not os.path.exists(path):
      raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "_config_file_" + hashlib.sha256(path.encode()).hexdigest()[:12], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
  else:
    mod = config_module(name_or_path)
    if mod is None:
      raise KeyError(
          f"unknown config {name_or_path!r}; known: {config_names()}"
      )
  cfg = mod.get_config()
  learner_kind(cfg)
  return cfg
