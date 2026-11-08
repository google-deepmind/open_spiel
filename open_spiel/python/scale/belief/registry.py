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

"""The table of belief samplers, by name.

Search records the sampler it uses by registered name and constructor
kwargs, so an artifact rebuilds its sampler without naming arbitrary code,
and a learner chooses a sampler by name in its config.

  sampler = registry.build("perfect_information", env)
  sampler = registry.build("leduc", env, opponent=model)  # takes an opponent
  name = registry.name_of(sampler)                        # "leduc"
  same = registry.build(name, env, **sampler.kwargs)

A sampler class is called as `cls(env, **kwargs)`. A class whose constructor
has an `opponent` parameter takes the opponent model, a Policy (or its
`{"arch_name", "arch_kwargs"}` record), through `build(..., opponent=...)`.
Every sampler exposes `kwargs`, the JSON constructor arguments besides
`env`, so `build(name_of(s), env, **s.kwargs)` rebuilds it.

The generic samplers (perfect_information, enumeration, particle) are
registered when the table is first read. A game's own samplers (leduc, gin)
live with the game and are declared in its GameSpec.components; their module
is imported only when the sampler is asked for by name, so naming none needs
no game dependency.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional, Union

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry

#: Where a sampler's params come from: "none" (it has none, or its own
#: init_params), "own" (the searching policy's params; the sampler models the
#: opponent with the network being searched) or "opponent" (an opponent
#: policy's params or table, set by the caller).
PARAMS_SOURCES = ("none", "own", "opponent")

_PKG = __name__.rpartition(".")[0]  # this package, under its import name
_BUILTIN_MODULES = tuple(
    f"{_PKG}.{m}" for m in ("perfect_information", "enumeration")
) + (
    # The particle sampler simulates an opponent policy, so it lives above
    # the policies (policies.particle_belief) and is loaded by name.
    f"{_PKG.rpartition('.')[0]}.policies.particle_belief",
)
#: The env-registry component kind of belief samplers (GameSpec.components).
COMPONENT_KIND = "belief"
# name -> the module whose import registers it (register_lazy).
_LAZY: dict[str, str] = {}

_CLASSES: dict[str, type[Any]] = {}
_loading = False


def _load_builtins() -> None:
  global _loading
  if _loading:
    return
  _loading = True
  try:
    for mod in _BUILTIN_MODULES:
      importlib.import_module(mod)
  finally:
    _loading = False


def register(name: str, cls: type[Any]) -> type[Any]:
  """Registers sampler class `cls` under `name`.

  A class may be registered under several names (aliases); `name_of` reports
  the first. Re-registering a name with the same class is a no-op.

  Args:
    name: the sampler name.
    cls: the sampler class.

  Returns:
    `cls`, so this works as a decorator.

  Raises:
    ValueError: `name` is empty or registered to a different class.
    TypeError: `cls` is not a class.
  """
  if not isinstance(name, str) or not name:
    raise ValueError(f"sampler name must be a non-empty string, got {name!r}")
  if not isinstance(cls, type):
    raise TypeError(f"register a sampler class, got {type(cls).__name__}")
  old = _CLASSES.get(name)
  if old is not None and old is not cls:
    raise ValueError(
        f"belief sampler {name!r} is already registered as {old.__qualname__}"
    )
  _CLASSES[name] = cls
  return cls


def register_lazy(name: str, module: str) -> None:
  """Registers `name` as provided by `module`, imported when first asked for.

  Args:
    name: the sampler name.
    module: the module whose import registers it.

  Raises:
    ValueError: `name` is already registered to another module.
  """
  if _LAZY.get(name, module) != module:
    raise ValueError(
        f"belief sampler {name!r} is already lazily registered by {_LAZY[name]}"
    )
  _LAZY[name] = module


def names() -> tuple[str, ...]:
  """Every sampler name: registered and lazy.

  Returns:
    The sorted names.
  """
  _load_builtins()
  return tuple(
      sorted(
          set(_CLASSES)
          | set(_LAZY)
          | set(env_registry.component_names(COMPONENT_KIND))
      )
  )


def _lazy_module(name: str) -> Optional[str]:
  """The module that registers the sampler `name` lazily, if any.

  Args:
    name: The sampler name.

  Returns:
    The module name, or None.
  """
  return _LAZY.get(name) or env_registry.component_module(COMPONENT_KIND, name)


def get(name: str) -> type[Any]:
  """The sampler class registered as `name`.

  Args:
    name: the sampler name.

  Returns:
    The registered class.

  Raises:
    KeyError: no sampler of that name; the message lists the known names.
  """
  _load_builtins()
  if name not in _CLASSES:
    module = _lazy_module(name)
    if module is not None:
      importlib.import_module(module)
  if name not in _CLASSES:
    raise KeyError(f"no belief sampler {name!r}; known: {names()}")
  return _CLASSES[name]


def takes_opponent(name: str) -> bool:
  """Whether sampler `name` is constructed with an opponent model.

  Args:
    name: the sampler name.

  Returns:
    True when its constructor has an `opponent` parameter.
  """
  return "opponent" in inspect.signature(get(name)).parameters


def build(
    name: str,
    env: types.Env,
    *,
    opponent: Optional[Union[types.Policy, dict[str, Any]]] = None,
    **kwargs: Any,
):
  """The sampler `name` for `env`.

  Args:
    name: a registered sampler name.
    env: the Env whose worlds are sampled.
    opponent: the opponent model, required by samplers whose constructor takes
      one (`takes_opponent`), refused by the others.
    **kwargs: further constructor arguments.

  Returns:
    The constructed sampler.

  Raises:
    KeyError: unknown name.
    ValueError: `opponent` is missing for a sampler that needs it, or given to
      one that does not.
  """
  cls = get(name)
  if takes_opponent(name):
    if opponent is None and "opponent" not in kwargs:
      raise ValueError(
          f"belief sampler {name!r} is built from an opponent model; "
          "pass opponent"
      )
    if opponent is not None:
      kwargs["opponent"] = opponent
  elif opponent is not None:
    raise ValueError(f"belief sampler {name!r} takes no opponent model")
  return cls(env, **kwargs)


def name_of(sampler: Any) -> str:
  """The registered name of a sampler object's class (its first name).

  Args:
    sampler: a sampler object.

  Returns:
    The first name its class was registered under.

  Raises:
    KeyError: the class is not registered (an unregistered sampler cannot be
      recorded in an artifact).
  """
  _load_builtins()
  cls = type(sampler)
  for n, c in _CLASSES.items():
    if c is cls:
      return n
  raise KeyError(
      f"{cls.__module__}:{cls.__qualname__} is not a registered belief sampler "
      f"({names()}); register it with belief.registry.register"
  )


def check(sampler: Any) -> None:
  """Checks that `sampler` declares the BeliefSampler attributes.

  Args:
    sampler: a sampler object.

  Raises:
    ValueError: an attribute is missing or has the wrong value.
  """
  for attr, allowed in (
      ("is_perfect_information", (True, False)),
      ("params_source", PARAMS_SOURCES),
  ):
    v = getattr(sampler, attr, None)
    if v not in allowed:
      raise ValueError(
          f"{type(sampler).__name__}.{attr} = {v!r}; want one of {allowed}"
      )
  for attr in (
      "init_params",
      "init_carry",
      "observe",
      "sample_worlds",
      "kwargs",
      "stateful",
      "capability",
  ):
    if not hasattr(sampler, attr):
      raise ValueError(
          f"{type(sampler).__name__} has no {attr!r} (BeliefSampler protocol)"
      )
