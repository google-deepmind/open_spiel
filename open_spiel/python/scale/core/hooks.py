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

"""Declared hooks: the optional methods and attributes an object may define.

A hook is an extension point the framework reads from an object a deployment
hands it: a league controller, a promotion gate, a run hook, an env, an
evaluator, a match engine, a canonical deal form. Each kind of object has one
`Protocol` that lists every hook it may define, with its kind (a method or a
value) and its meaning. The framework reads a hook only through its Protocol,
never with getattr(obj, name, None), so a hook used wrongly raises instead of
being silently skipped:

  CONTROLLER = hooks.Protocol("league controller", [
      hooks.method("supports_sides", "supports_sides(sides) -> bool ..."),
      hooks.value("pairwise", "reads the payoff as a two-player game",
                  check=hooks.is_bool),
  ])
  fn = CONTROLLER.get(ctl, "supports_sides")    # the hook, or None
  CONTROLLER.require(engine, "table_keys")      # the hook, or raises
  CONTROLLER.check(ctl)                         # every hook, validated

* Reading a name the Protocol does not declare raises UndeclaredHookError
  (a KeyError): a misspelled read is a bug at the reading site.
* A hook of the wrong kind raises MalformedHookError (a TypeError): a method
  hook that is not callable, a value hook that fails its check.
* `require` raises MissingHookError (a TypeError) when the object does not
  define the hook.
* An AttributeError raised inside a hook defined on the object's class (a
  property that fails) propagates; it never reads as "not defined".
* `check(obj)` validates every declared hook (call it where the object is
  registered or handed in); a Protocol built with at_least_one=True also
  refuses an object that defines none of its hooks (a run hook that hooks
  nothing).

An object defines a hook by having the attribute: a method, a class
attribute, a property, or (for a function such as a canonical deal form) a
function attribute. A hook whose value is None counts as not defined, so a
base class may declare `name = None` as the documented default.

Every Protocol is registered in PROTOCOLS by its name when built (a second
Protocol of the same name raises); `describe()` prints them all.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence


class UndeclaredHookError(KeyError):
  """A read of a name the Protocol does not declare."""


class MalformedHookError(TypeError):
  """A hook of the wrong kind or value."""


class MissingHookError(TypeError):
  """A required hook the object does not define."""


@dataclasses.dataclass(frozen=True)
class Hook:
  """One declared hook.

  Attributes:
    name: the attribute name.
    doc: its signature and meaning, as the Protocol documents it.
    is_method: a method (must be callable) or a value.
    check: None, or a function of a defined value that raises (or returns
      False) when the value is malformed.
  """

  name: str
  doc: str
  is_method: bool
  check: Optional[Callable[[Any], Any]] = None


def method(name: str, doc: str) -> Hook:
  """A method hook: defined means callable.

  Args:
    name: the attribute name.
    doc: its signature and meaning.

  Returns:
    The Hook.
  """
  return Hook(name, doc, True)


def value(
    name: str, doc: str, check: Optional[Callable[[Any], Any]] = None
) -> Hook:
  """A value hook, validated by `check` when defined.

  Args:
    name: the attribute name.
    doc: its meaning.
    check: None, or a function of the value that raises or returns False when
      the value is malformed.

  Returns:
    The Hook.
  """
  return Hook(name, doc, False, check)


def is_bool(v: Any) -> bool:
  """A check: v is a bool (numpy bools included).

  Args:
    v: the value.

  Returns:
    Whether v is a bool.
  """
  return isinstance(v, bool) or type(v).__name__ == "bool_"


_MISSING = object()

#: Every Protocol built, by name (Protocol.__init__).
PROTOCOLS: dict[str, "Protocol"] = {}


def _defined_on_class(obj: Any, name: str) -> bool:
  """Whether `name` is defined on obj's class (its MRO), not the instance.

  Args:
    obj: the object.
    name: the attribute name.

  Returns:
    Whether a class of obj's MRO defines `name`.
  """
  cls = obj if isinstance(obj, type) else type(obj)
  return any(name in vars(k) for k in cls.__mro__)


class Protocol:
  """The declared hooks of one kind of object (module doc).

  Attributes:
    name: the kind of object, for errors ("league controller").
    hooks: {name: Hook}, in declaration order.
    at_least_one: check() refuses an object that defines none of the hooks.
  """

  def __init__(
      self, name: str, hooks: Sequence[Hook], *, at_least_one: bool = False
  ):
    if not hooks:
      raise ValueError(f"hook protocol {name!r} declares no hooks")
    by_name = {}
    for h in hooks:
      if not isinstance(h, Hook):
        raise TypeError(f"hook protocol {name!r}: {h!r} is not a Hook")
      if h.name in by_name:
        raise ValueError(f"hook protocol {name!r} declares {h.name!r} twice")
      by_name[h.name] = h
    if name in PROTOCOLS:
      raise ValueError(f"a hook protocol named {name!r} is already declared")
    self.name = name
    self.hooks = by_name
    self.at_least_one = bool(at_least_one)
    PROTOCOLS[name] = self

  def __repr__(self) -> str:
    return f"hooks.Protocol({self.name!r}, {list(self.hooks)})"

  def _hook(self, name: str) -> Hook:
    """The declared Hook `name`.

    Args:
      name: the hook name.

    Returns:
      The Hook.

    Raises:
      UndeclaredHookError: the Protocol does not declare `name`.
    """
    h = self.hooks.get(name)
    if h is None:
      raise UndeclaredHookError(
          f"{name!r} is not a {self.name} hook (declared: {list(self.hooks)})"
      )
    return h

  def get(self, obj: Any, name: str) -> Any:
    """The hook `name` of obj, or None when obj does not define it.

    Args:
      obj: the object.
      name: a declared hook name.

    Returns:
      The hook (a bound method, a function or a value), or None.

    Raises:
      UndeclaredHookError: the Protocol does not declare `name`.
      MalformedHookError: a method hook that is not callable, or a value hook
        that fails its check.
      AttributeError: a hook defined on obj's class raised it (a failing
        property).
    """
    h = self._hook(name)
    try:
      v = getattr(obj, name)
    except AttributeError:
      if _defined_on_class(obj, name):
        raise
      v = _MISSING
    if v is _MISSING or v is None:
      return None
    what = getattr(obj, "__name__", None) or type(obj).__name__
    if h.is_method and not callable(v):
      raise MalformedHookError(
          f"{self.name} {what}: hook {name!r} is a method ({h.doc}), got a "
          f"{type(v).__name__}"
      )
    if h.check is not None:
      try:
        ok = h.check(v)
      except (TypeError, ValueError) as e:
        raise MalformedHookError(
            f"{self.name} {what}: hook {name!r} = {v!r} is malformed ({h.doc})"
        ) from e
      if isinstance(ok, bool) and not ok:
        raise MalformedHookError(
            f"{self.name} {what}: hook {name!r} = {v!r} is malformed ({h.doc})"
        )
    return v

  def require(self, obj: Any, name: str) -> Any:
    """The hook `name` of obj; raises when obj does not define it.

    Args:
      obj: the object.
      name: a declared hook name.

    Returns:
      The hook.

    Raises:
      MissingHookError: obj does not define it.
      UndeclaredHookError, MalformedHookError: as get().
    """
    v = self.get(obj, name)
    if v is None:
      what = getattr(obj, "__name__", None) or type(obj).__name__
      raise MissingHookError(
          f"{self.name} {what} defines no {name} hook ({self.hooks[name].doc})"
      )
    return v

  def defined(self, obj: Any) -> tuple[str, ...]:
    """The declared hooks obj defines, validated (get()), in order.

    Args:
      obj: the object.

    Returns:
      The hook names.
    """
    return tuple(n for n in self.hooks if self.get(obj, n) is not None)

  def check(self, obj: Any) -> Any:
    """Validates every declared hook of obj (module doc).

    Args:
      obj: the object.

    Returns:
      obj.

    Raises:
      MissingHookError: at_least_one and obj defines none of the hooks.
      MalformedHookError, AttributeError: as get().
    """
    if not self.defined(obj) and self.at_least_one:
      what = getattr(obj, "__name__", None) or type(obj).__name__
      raise MissingHookError(
          f"{self.name} {what} defines none of its hooks {list(self.hooks)}"
      )
    return obj


def describe() -> str:
  """Every declared Protocol and its hooks, one per line.

  Returns:
    The text.
  """
  lines = []
  for name in sorted(PROTOCOLS):
    p = PROTOCOLS[name]
    lines.append(f"{name}:")
    for h in p.hooks.values():
      kind = "method" if h.is_method else "value"
      lines.append(f"  {h.name} ({kind}): {h.doc}")
  return "\n".join(lines)
