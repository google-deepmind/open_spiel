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

"""The architecture registry: arch_name -> Policy constructor.

A policy is chosen by name and JSON kwargs, so a config, a learner and the
artifact loader all build it the same way, and an artifact records exactly
what rebuilds it.

  policy = registry.build("uniform", dict(game="kuhn_poker", num_actions=2,
                                          num_players=2))
  registry.register("my_net", MyNetPolicy)         # MyNetPolicy(**arch_kwargs)
  with registry.temporary("probe", ProbePolicy):  # registered for the block
    ...

`build(arch_name, arch_kwargs)` calls the constructor and checks that the
policy's spec echoes both, so `build(spec.arch_name, spec.arch_kwargs)`
reproduces a policy. Registering a name twice raises.

The game-agnostic architectures register when the registry is first used.
Game-specific architectures (scripted policies, game searches) live in
modules listed by a game's `GameSpec.extras` or `register_lazy`; they are
imported only when a name is not found, so naming none needs no game
dependency.

Kwargs translators (`register_kwargs_translator`) rewrite arch_kwargs before
every build, so a plugin can accept another spelling of an architecture's
kwargs (for example a schema its own artifacts record) and map it onto the
one the constructor takes.
"""

import contextlib
import importlib
from typing import Any, Callable, Mapping, Optional

from open_spiel.python.scale import registry as path_registry
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry

_CTORS: dict[str, Callable[..., types.Policy]] = {}
# arch_name -> (ctor, nesting depth)
_TEMPORARY: dict[str, tuple[Callable[..., types.Policy], int]] = {}

# Modules whose import registers the built-in architectures.
_ROOT = __name__.rsplit(".", 2)[0]
_BUILTIN_MODULES = tuple(
    f"{_ROOT}.{m}"
    for m in (
        "policies.uniform",
        "policies.tabular",
        "policies.routed",
        "policies.opening",
        "networks.policy",
        "policies.search",
        "policies.rollout",
    )
)
#: arch_name -> the module whose import registers it (register_lazy).
LAZY: dict[str, str] = {}

KwargsTranslator = Callable[
    [str, Mapping[str, Any]], Optional[Mapping[str, Any]]
]
_TRANSLATORS: list[KwargsTranslator] = []


def register_lazy(arch_name: str, module: str) -> None:
  """Registers `arch_name` as provided by `module`, imported when asked for.

  Args:
    arch_name: the architecture name.
    module: the module whose import registers it.

  Raises:
    ValueError: a name or module that is not a non-empty string, or a name
      lazily registered by another module.
  """
  if not (isinstance(arch_name, str) and arch_name and isinstance(module, str)):
    raise ValueError(
        f"register_lazy({arch_name!r}, {module!r}): want two non-empty strings"
    )
  if LAZY.get(arch_name, module) != module:
    raise ValueError(
        f"architecture {arch_name!r} is already lazily registered by "
        f"{LAZY[arch_name]}"
    )
  LAZY[arch_name] = module


def register(arch_name: str, ctor: Callable[..., types.Policy]) -> None:
  """Registers `ctor(**arch_kwargs) -> Policy` as `arch_name`.

  Args:
    arch_name: the architecture name.
    ctor: the policy constructor.

  Raises:
    ValueError: the name is empty or already registered.
  """
  if not isinstance(arch_name, str) or not arch_name:
    raise ValueError(f"arch_name must be a non-empty string, got {arch_name!r}")
  if arch_name in _CTORS:
    raise ValueError(f"architecture {arch_name!r} is already registered")
  _CTORS[arch_name] = ctor


@contextlib.contextmanager
def temporary(arch_name: str, ctor: Callable[..., types.Policy]):
  """Registers arch_name for the duration of the block (test architectures).

  Nested blocks for the same (arch_name, ctor) are allowed; the name is
  removed when the outermost block exits. A name registered permanently, or
  to another ctor, raises ValueError.

  Args:
    arch_name: the architecture name.
    ctor: the policy constructor.

  Yields:
    Nothing; the name is registered inside the block.
  """
  if _TEMPORARY.get(arch_name, (None, 0))[0] is ctor:
    _TEMPORARY[arch_name] = (ctor, _TEMPORARY[arch_name][1] + 1)
  else:
    register(arch_name, ctor)
    _TEMPORARY[arch_name] = (ctor, 1)
  try:
    yield
  finally:
    c, n = _TEMPORARY[arch_name]
    if n == 1:
      del _TEMPORARY[arch_name]
      _CTORS.pop(arch_name, None)
    else:
      _TEMPORARY[arch_name] = (c, n - 1)


def register_kwargs_translator(fn: KwargsTranslator) -> None:
  """Adds a translator of arch_kwargs, run before every build.

  Translators run in registration order.

  Args:
    fn: fn(arch_name, arch_kwargs) -> translated arch_kwargs, or None to leave
      them.

  Raises:
    TypeError: `fn` is not callable.
  """
  if not callable(fn):
    raise TypeError(
        f"a kwargs translator must be callable, got {type(fn).__name__}"
    )
  if fn not in _TRANSLATORS:
    _TRANSLATORS.append(fn)


def translate(arch_name: str, arch_kwargs: Mapping[str, Any]) -> dict[str, Any]:
  """arch_kwargs after every registered translator.

  Args:
    arch_name: the architecture name.
    arch_kwargs: its kwargs.

  Returns:
    The translated kwargs.
  """
  kw = dict(arch_kwargs)
  for fn in _TRANSLATORS:
    out = fn(arch_name, kw)
    if out is not None:
      kw = dict(out)
  return kw


def registered() -> tuple[str, ...]:
  """Every architecture name: the registered ones and the lazy ones.

  Returns:
    The sorted names.
  """
  _load_builtins()
  return tuple(sorted(set(_CTORS) | set(LAZY)))


def _load_builtins() -> None:
  for mod in _BUILTIN_MODULES:
    importlib.import_module(mod)


def _load_named(arch_name: str) -> None:
  """Imports the module that registers arch_name.

  That is its register_lazy module, else the extras of the game its "<game>."
  prefix names.

  Args:
    arch_name: the architecture name.
  """
  if arch_name in _CTORS:
    return
  if arch_name in LAZY:
    importlib.import_module(LAZY[arch_name])
  else:
    env_registry.load_extras_for(arch_name)


def _ctor(arch_name: str) -> Callable[..., types.Policy]:
  """The constructor of `arch_name`.

  Args:
    arch_name: The architecture.

  Returns:
    The constructor.

  Raises:
    KeyError: The architecture is unknown.
  """
  _load_builtins()
  _load_named(arch_name)
  if arch_name not in _CTORS:
    raise KeyError(
        f"unknown architecture {arch_name!r}; registered: {registered()}"
    )
  return _CTORS[arch_name]


def build(arch_name: str, arch_kwargs: Mapping[str, Any]) -> types.Policy:
  """The Policy for (arch_name, arch_kwargs).

  Args:
    arch_name: the architecture name.
    arch_kwargs: its constructor kwargs (translated first).

  Returns:
    The built Policy.

  Raises:
    KeyError: unknown architecture.
    ValueError: the built policy's spec does not echo arch_name and the
      (translated) arch_kwargs.
  """
  ctor = _ctor(arch_name)
  kwargs = translate(arch_name, arch_kwargs)
  pol = ctor(**kwargs)
  if pol.spec.arch_name != arch_name or dict(pol.spec.arch_kwargs) != kwargs:
    raise ValueError(
        f"architecture {arch_name!r} built a policy whose spec is "
        f"({pol.spec.arch_name!r}, {dict(pol.spec.arch_kwargs)}), not "
        f"({arch_name!r}, {kwargs})"
    )
  return pol


def build_part(part: Any) -> types.Policy:
  """A policy object as is, or the policy of a policy record.

  Args:
    part: a Policy or an `{"arch_name", "arch_kwargs"}` record.

  Returns:
    The Policy.

  Raises:
    TypeError: `part` is neither.
    ValueError: a record with other keys.
  """
  if isinstance(part, Mapping):
    if set(part) != {"arch_name", "arch_kwargs"}:
      raise ValueError(
          "a policy record has keys arch_name and arch_kwargs, got "
          f"{sorted(part)}"
      )
    return build(part["arch_name"], part["arch_kwargs"])
  if not hasattr(part, "spec"):
    raise TypeError(
        f"want a Policy or a policy record, got {type(part).__name__}"
    )
  return part


def policy_class(arch_name: str):
  """The Policy class an architecture builds.

  That is the constructor when it is a class, else its `policy_class`
  attribute, else None. Class-level facts (decider, prior_dependence,
  emits_target_policy) are read from it without building.

  Args:
    arch_name: the architecture name.

  Returns:
    The class, or None.
  """
  ctor = _ctor(arch_name)
  if isinstance(ctor, type):
    return ctor
  return getattr(ctor, "policy_class", None)


def build_from_spec(spec: types.PolicySpec) -> types.Policy:
  """The Policy a PolicySpec records.

  Args:
    spec: the PolicySpec.

  Returns:
    build(spec.arch_name, spec.arch_kwargs).
  """
  return build(spec.arch_name, spec.arch_kwargs)


def resolve_callable(path: str) -> Callable[..., Any]:
  """The attribute a 'pkg.module:attr' path names, after registry.resolve_path.

  Args:
    path: the 'pkg.module:attr' path.

  Returns:
    The named object.
  """
  return path_registry.load_object(path)
