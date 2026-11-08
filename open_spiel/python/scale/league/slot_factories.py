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

"""Slot factories: the operators an expert-iteration run builds.

Slot factories: the operators an expert-iteration run builds over its
base network.

An AZ / ExIt run trains one base network (a BCLearner) and acts with an
operator built over it once, when the driver is constructed; each generation
swaps params only. A factory, named in the spec (`kinds.ExIt.factory`, with
`factory_kwargs`), is that recipe:

  build(base, env, spec)        the operator Policy (built once)
  params(operator, base_params) the operator's params for base params
                                (the per-generation swap)
  unwrap(operator, op_params)   the base params inside operator params
  base_carry(init_carry)        the base's carry inside the operator's (the
                                AZ path needs a stateless base: PlyReplay
                                stores single plies)
  plain_member(base)            the plain network as a pool member, gate
                                entrant and plain sibling artifact
                                (AZDriver.plain_policy): the base itself
                                (PlainBase), or e.g. the base routed to a
                                scripted part it cannot play
  plain_params(plain, base_params)  plain_member's params for base params
  unplain_params(plain, plain_params)  the base params inside them

  Every hook is declared on SlotFactory and checked at get(); PlainBase
  gives the identity plain_member / plain_params / unplain_params.

  "az_search" (the default, spec.factory "")  az.build's SearchPolicy over
      the base with the AZConfig's MCTS, wrap {"base": params, "sampler":
      {}}: the AZ kind's operator.

Other factories are registered by the modules that define them (a game's
search, for example) and documented there.

A name is registered once.

A factory whose module must not be imported unless named is registered by
name with `register_lazy` (name -> module), or declared by its game in
GameSpec.components (kind "slot_factory"), and imported by `get` on first
use.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Protocol

from open_spiel.python.scale.belief import registry as belief_registry
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry


class SlotFactory(Protocol):
  """An operator recipe (module doc).

  Attributes:
    name: the factory's registered name.
  """

  name: str

  # A protocol stub: Returns documents what implementations return.
  # pylint: disable-next=redundant-returns-doc
  def build(
      self, base: types.Policy, env: types.Env, spec: Any
  ) -> types.Policy:
    """The operator Policy over `base` (built once).

    Args:
      base: the base network's policy.
      env: the league's env.
      spec: the kind's spec.

    Returns:
      The operator policy.
    """
    ...

  def params(self, operator: types.Policy, base_params: Any) -> Any:
    """The operator's params for base params (the per-generation swap).

    Args:
      operator: the operator policy.
      base_params: the base network's params.

    Returns:
      The operator params.
    """
    ...

  def unwrap(self, operator: types.Policy, op_params: Any) -> Any:
    """The base params inside operator params.

    Args:
      operator: the operator policy.
      op_params: the operator params.

    Returns:
      The base params.
    """
    ...

  def base_carry(self, init_carry: Any) -> Any:
    """The base's carry inside the operator's.

    Args:
      init_carry: the operator's initial carry.

    Returns:
      The base's carry.
    """
    ...

  def plain_member(self, base: types.Policy) -> types.Policy:
    """The plain network as a pool member, gate entrant and plain sibling.

    Args:
      base: the base network's policy.

    Returns:
      The plain policy.
    """
    ...

  def plain_params(self, plain: types.Policy, base_params: Any) -> Any:
    """plain_member's params for base params.

    Args:
      plain: the plain policy.
      base_params: the base network's params.

    Returns:
      The plain params.
    """
    ...

  def unplain_params(self, plain: types.Policy, plain_params: Any) -> Any:
    """The base params inside plain params.

    Args:
      plain: the plain policy.
      plain_params: the plain params.

    Returns:
      The base params.
    """
    ...


#: The hooks every factory has (SlotFactory; get() checks them).
HOOKS = (
    "build",
    "params",
    "unwrap",
    "base_carry",
    "plain_member",
    "plain_params",
    "unplain_params",
)


class PlainBase:
  """The plain member is the base network itself.

  The plain member is the base network itself (SlotFactory's plain hooks
  as identities): mix in for a factory whose base plays every ply.
  """

  def plain_member(self, base):
    """The base itself.

    Args:
      base: the base network's policy.

    Returns:
      base.
    """
    return base

  def plain_params(self, plain, base_params):
    """The base params themselves.

    Args:
      plain: the plain policy (unused).
      base_params: the base network's params.

    Returns:
      base_params.
    """
    del plain
    return base_params

  def unplain_params(self, plain, plain_params):
    """The plain params themselves.

    Args:
      plain: the plain policy (unused).
      plain_params: the plain params.

    Returns:
      plain_params.
    """
    del plain
    return plain_params


_FACTORIES: dict[str, Callable[[Mapping[str, Any]], SlotFactory]] = {}
DEFAULT = "az_search"
#: name -> the module whose import registers it (get() imports it on first use).
#: A game's own factories are declared in its GameSpec.components instead.
LAZY: dict[str, str] = {}
#: The env-registry component kind of slot factories (GameSpec.components).
COMPONENT_KIND = "slot_factory"


def register_lazy(name: str, module: str) -> None:
  """`name` is registered by importing `module`, when get() first asks for it.

  Args:
    name: the factory name.
    module: the module that registers it.

  Raises:
    ValueError: name is already lazily registered to another module.
  """
  assert isinstance(name, str) and name and isinstance(module, str), (
      name,
      module,
  )
  if LAZY.get(name, module) != module:
    raise ValueError(
        f"slot factory {name!r} is already lazily registered by {LAZY[name]}"
    )
  LAZY[name] = module


def register(
    name: str, ctor: Callable[[Mapping[str, Any]], SlotFactory]
) -> None:
  """Registers ctor(factory_kwargs) -> SlotFactory under name.

  Args:
    name: the factory name.
    ctor: builds the factory from its kwargs.

  Raises:
    ValueError: name is already registered.
  """
  assert isinstance(name, str) and name, name
  if name in _FACTORIES:
    raise ValueError(f"slot factory {name!r} is already registered")
  _FACTORIES[name] = ctor


def names() -> tuple[str, ...]:
  """The registered names (a LAZY one once its module is imported).

  Returns:
    The names, sorted.
  """
  return tuple(sorted(_FACTORIES))


def get(name: str, kwargs: Mapping[str, Any]) -> SlotFactory:
  """The factory `name` built with kwargs, importing a lazy one first.

  Args:
    name: the factory name; DEFAULT when empty.
    kwargs: the factory kwargs.

  Returns:
    The SlotFactory.

  Raises:
    KeyError: no factory is registered under `name`.
    TypeError: the factory lacks a SlotFactory hook.
  """
  name = name or DEFAULT
  if name not in _FACTORIES:
    module = LAZY.get(name) or env_registry.component_module(
        COMPONENT_KIND, name
    )
    if module is not None:
      importlib.import_module(module)
  if name not in _FACTORIES:
    lazy = sorted(set(LAZY) | set(env_registry.component_names(COMPONENT_KIND)))
    raise KeyError(
        f"no slot factory {name!r}; registered: {names()}, lazy: {lazy}"
    )
  f = _FACTORIES[name](dict(kwargs))
  assert f.name == name, (f.name, name)
  missing = [h for h in HOOKS if not callable(getattr(f, h, None))]
  if missing:
    raise TypeError(
        f"slot factory {name!r} lacks the SlotFactory hooks {missing} "
        "(PlainBase "
        "gives the plain ones)"
    )
  return f


class AZSearch(PlainBase):
  """The default: the AZConfig's SearchPolicy (module doc)."""

  name = DEFAULT

  def __init__(self, kwargs: Mapping[str, Any]):
    if kwargs:
      raise ValueError(
          f"the {DEFAULT} factory takes no factory_kwargs (its MCTS is the "
          f"AZConfig's): {sorted(kwargs)}"
      )

  def build(self, base, env, spec):
    """az.build's SearchPolicy over the base with the AZConfig's MCTS.

    Args:
      base: the base network's policy.
      env: the league's env.
      spec: the kind's spec.

    Returns:
      The search policy.
    """
    # Lazy: this module imports without jax and mctx.
    from open_spiel.python.scale.policies import search  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

    cfg = spec.az
    opponent = base if belief_registry.takes_opponent(cfg.sampler) else None
    return search.SearchPolicy(
        base,
        env,
        belief_registry.build(cfg.sampler, env, opponent=opponent),
        cfg.mcts(),
    )

  def params(self, operator, base_params):
    """{"base": base_params, "sampler": {}}.

    Args:
      operator: the operator policy (unused).
      base_params: the base network's params.

    Returns:
      The operator params.
    """
    del operator
    return {"base": base_params, "sampler": {}}

  def unwrap(self, operator, op_params):
    """The "base" entry of the operator params.

    Args:
      operator: the operator policy (unused).
      op_params: the operator params.

    Returns:
      The base params.
    """
    del operator
    return op_params["base"]

  def base_carry(self, init_carry):
    """The "base" entry of the operator's carry.

    Args:
      init_carry: the operator's initial carry.

    Returns:
      The base's carry.
    """
    return init_carry["base"]


register(DEFAULT, AZSearch)
