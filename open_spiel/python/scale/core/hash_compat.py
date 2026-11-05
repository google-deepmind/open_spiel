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

"""Config fields added after configs were hashed.

They are left out of the hash at their default.

`league.config_common.config_hash` (also as `configs.common`) and
`resume_hash` hash every dataclass field, and a resumed run must reproduce
its recorded hash. A field added to a config class therefore changes the
hash of every config of that class unless it is left out while it holds the
default it was added with. Declare such a field
with `added`:

  @dataclasses.dataclass(frozen=True)
  class LeagueConfig:
    ...
    payoff_view: str = hash_compat.added("auto")

`config_common` then hashes the field only when it holds something other
than its default; at its default the key is absent from `config_dict`, so
every config built before the field existed keeps its config_hash and
resume_hash. A field is compared with its default in plain form
(`common.plain`), so an equal tuple, dataclass or string all count as the
default.

How it is applied. Registered hash transforms
(`common.register_hash_transform`) may recognise a dataclass's dict by its
exact key set, so deleting a key before they run would hide the dict from
them. The hashed dict therefore
carries OMITTED in place of the value of every added field at its default
while the transforms run, and `drop_omitted`, the last step of
`common.config_dict` and `resume_hash`, deletes those keys. A transform sees
the full key set and never needs to know these fields.

Recorded forms. A deployment whose runs were hashed by earlier code, before
a field was declared with `added` or before a field or module was renamed,
registers how to put the hashed dict back to the form those runs recorded:

  * `register_added_fields(cls, group, also)`: a group of fields of `cls`
    added after runs were launched on it is left out while every field of
    the group holds its default (and the optional predicate `also` on cls's
    field dict holds); a group with any field set otherwise is hashed whole;
  * `register_class_transform(cls, fn)`: fn rewrites the hashed field dict of
    every `cls`, for example a renamed field under its old name;
  * `register_string_transform(fn)`: fn maps every string in the hashed dict,
    for example a moved module's path back to its recorded one.

A dataclass's dict is recognised by its exact key set (the added fields'
keys are still present then, holding OMITTED), so two registered classes
with the same field names are refused. `common.config_dict` and
`resume_hash` apply `recorded_form` before the registered hash transforms;
with nothing registered it changes nothing.
"""

import dataclasses
from typing import Any, Callable, Mapping, Optional

#: The dataclass field metadata key that marks a field as added (`added`).
METADATA_KEY = "hash_compat.added"

#: Stands in for the value of an added field at its default while the
#: registered hash transforms run; `drop_omitted` deletes every key holding it.
OMITTED = "__hash_compat_omitted_at_default__"


def added(default: Any) -> Any:
  """A dataclass field with `default`, left out of the config hash at it.

  Left out while it holds `default`. Use only for an immutable default (None,
  a number, a string, a tuple, a frozen dataclass).

  Args:
    default: the field's default.

  Returns:
    The dataclasses.field.

  Raises:
    TypeError: a mutable default.
  """
  if isinstance(default, (list, dict, set)):
    raise TypeError(
        f"an added field's default must be immutable, got {default!r}"
    )
  # A field factory: the caller calls it in its dataclass body.
  # pylint: disable-next=invalid-field-call
  return dataclasses.field(default=default, metadata={METADATA_KEY: True})


def added_kwarg(key: str, value: Any, default: Any) -> dict[str, Any]:
  """{key: value}, or {} when value is `default`.

  The keyword-argument form of `added`: a key added after configs were
  hashed to a hashed kwargs mapping (a config's controller_kwargs) is left
  out of that mapping at the default it was added with, so every config that
  does not set it keeps its config_hash and resume_hash. The consumer reads
  a missing key as the default.

  Args:
    key: the keyword.
    value: its value.
    default: the default it was added with (immutable).

  Returns:
    The kwargs to merge into the mapping.

  Raises:
    TypeError: a mutable default.
  """
  if isinstance(default, (list, dict, set)):
    raise TypeError(
        f"an added kwarg's default must be immutable, got {default!r}"
    )
  return {} if value == default else {str(key): value}


def is_added(f: dataclasses.Field) -> bool:
  """Whether dataclass field `f` was declared with `added`.

  Args:
    f: the field.

  Returns:
    bool.
  """
  return bool(f.metadata.get(METADATA_KEY, False))


def added_fields(cls: type[Any]) -> dict[str, Any]:
  """{name: default} of the added fields of dataclass `cls`.

  Args:
    cls: the dataclass.

  Returns:
    The dict.

  Raises:
    TypeError: cls is not a dataclass.
  """
  if not dataclasses.is_dataclass(cls):
    raise TypeError(f"{cls!r} is not a dataclass")
  return {f.name: f.default for f in dataclasses.fields(cls) if is_added(f)}


def at_default(
    obj: Any, f: dataclasses.Field, plain: Callable[[Any], Any]
) -> bool:
  """Whether added field `f` of dataclass instance `obj` holds its default.

  Compared in plain form.

  Args:
    obj: the dataclass instance.
    f: the added field.
    plain: the plain-form function (common.plain).

  Returns:
    bool.

  Raises:
    ValueError: f is not an added field, or has no default.
  """
  if not is_added(f):
    raise ValueError(f"{type(obj).__name__}.{f.name} is not an added field")
  if f.default is dataclasses.MISSING:
    raise ValueError(
        f"added field {type(obj).__name__}.{f.name} has no default"
    )
  return plain(getattr(obj, f.name)) == plain(f.default)


def drop_omitted(d: Any) -> Any:
  """`d` with every dict key whose value is OMITTED removed (recursively).

  Args:
    d: the hashed dict (or any value inside it).

  Returns:
    The value without the omitted keys.

  Raises:
    ValueError: OMITTED appears anywhere other than as a dict value (a
      transform moved it into a list or a key), so the hash would be wrong.
  """
  if isinstance(d, Mapping):
    out = {}
    for k, v in d.items():
      if k == OMITTED:
        raise ValueError("hash_compat: the omitted marker is a dict key")
      if isinstance(v, str) and v == OMITTED:
        continue
      out[k] = drop_omitted(v)
    return out
  if isinstance(d, (list, tuple)):
    for v in d:
      if isinstance(v, str) and v == OMITTED:
        raise ValueError(
            "hash_compat: the omitted marker is a list element (a hash "
            "transform moved a field value into a list)"
        )
    return [drop_omitted(v) for v in d]
  return d


# -- recorded forms (module doc "Recorded forms") ----------------------------

#: (fields -> the default each was added with, predicate on the class's
#: original field dict or None).
Group = tuple[Mapping[str, Any], Optional[Callable[[Mapping[str, Any]], bool]]]
_ADDED: dict[str, tuple[Group, ...]] = {}  # class name -> groups
# class name -> [fn(field dict) -> field dict]
_CLASS_TRANSFORMS: dict[
    str, list[Callable[[dict[str, Any]], dict[str, Any]]]
] = {}
_KEYSETS: dict[frozenset[str], str] = {}  # field-name set -> class name
_STRING_TRANSFORMS: list[Callable[[str], str]] = []


def _register_class(cls: type[Any]) -> str:
  """Registers dataclass cls's key set.

  Args:
    cls: the dataclass.

  Returns:
    Its name.

  Raises:
    TypeError: cls is not a dataclass.
    ValueError: another registered class has the same field names.
  """
  if not dataclasses.is_dataclass(cls):
    raise TypeError(f"{cls!r} is not a dataclass")
  keys = frozenset(f.name for f in dataclasses.fields(cls))
  name = cls.__name__
  got = _KEYSETS.get(keys)
  if got is not None and got != name:
    raise ValueError(
        f"{name} and {got} have the same fields; their dicts are ambiguous"
    )
  _KEYSETS[keys] = name
  return name


def register_added_fields(
    cls: type[Any],
    group: Mapping[str, Any],
    also: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> None:
  """Leaves `group` out of cls's hashed dict while each field is at its default.

  Args:
    cls: the config dataclass.
    group: field -> the default it was added with.
    also: optional predicate on cls's original field dict that must also hold.

  Raises:
    TypeError: cls is not a dataclass, or `also` is not callable.
    ValueError: a field of `group` is not a field of cls.
  """
  name = _register_class(cls)
  if also is not None and not callable(also):
    raise TypeError(f"{also!r} is not callable")
  unknown = set(group) - {f.name for f in dataclasses.fields(cls)}
  if unknown:
    raise ValueError(f"{sorted(unknown)} are not fields of {name}")
  _ADDED[name] = _ADDED.get(name, ()) + ((dict(group), also),)


def register_class_transform(
    cls: type[Any], fn: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
  """Makes fn rewrite the hashed field dict of every cls (after added groups).

  Registering the same fn twice has no effect.

  Args:
    cls: the config dataclass.
    fn: field dict -> field dict; must not mutate its input.

  Raises:
    TypeError: cls is not a dataclass, or fn is not callable.
  """
  if not callable(fn):
    raise TypeError(f"class transform {fn!r} is not callable")
  name = _register_class(cls)
  fns = _CLASS_TRANSFORMS.setdefault(name, [])
  if fn not in fns:
    fns.append(fn)


def register_string_transform(fn: Callable[[str], str]) -> None:
  """Makes fn map every string value in the hashed dict (registration order).

  Registering the same fn twice has no effect.

  Args:
    fn: str -> str.

  Raises:
    TypeError: fn is not callable.
  """
  if not callable(fn):
    raise TypeError(f"string transform {fn!r} is not callable")
  if fn not in _STRING_TRANSFORMS:
    _STRING_TRANSFORMS.append(fn)


def _at_defaults(
    d: Mapping[str, Any],
    group: Mapping[str, Any],
    plain: Callable[[Any], Any],
) -> bool:
  """Whether every field of `group` is in d at its (plain) default.

  Args:
    d: a hashed field dict.
    group: {field: default}.
    plain: maps a default to its hashed (plain) form.

  Returns:
    True when each field is present and equal to its plain default.
  """
  return all(k in d and d[k] == plain(v) for k, v in group.items())


def recorded_form(x: Any, plain: Callable[[Any], Any]) -> Any:
  """Plain value x (a hashed dict or a value in it) in its recorded form.

  Every dict whose key set is a registered class's loses the added groups at
  their defaults, then goes through that class's transforms; every string
  goes through the string transforms. With nothing registered x is returned
  unchanged (as a copy).

  Args:
    x: a plain value (common.plain).
    plain: the plain-form function (common.plain), for the groups' defaults.

  Returns:
    The recorded form.
  """
  if isinstance(x, dict):
    name = _KEYSETS.get(frozenset(x))
    skip: set[str] = set()
    if name is not None:
      for group, also in _ADDED.get(name, ()):
        if _at_defaults(x, group, plain) and (also is None or also(x)):
          skip |= set(group)
    out = {k: recorded_form(v, plain) for k, v in x.items() if k not in skip}
    for fn in _CLASS_TRANSFORMS.get(name, ()) if name is not None else ():
      out = fn(out)
    return out
  if isinstance(x, list):
    return [recorded_form(v, plain) for v in x]
  if isinstance(x, str):
    for fn in _STRING_TRANSFORMS:
      x = fn(x)
    return x
  return x
