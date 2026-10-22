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


"""Decoding of request bodies into the contract's request dataclasses.

Decoding is strict, as the contract specifies: an object carries every field
of its type and no other, numbers keep their JSON types (a boolean is not a
number, and an integer field refuses `1.5`), and enums take their wire values.
"""

import dataclasses
import enum
import json
import math
import types
import typing
from typing import Any, TypeVar

from open_spiel.python.gui import errors

_T = TypeVar("_T")

_SCALAR_NAMES: dict[object, str] = {
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    str: "a string",
    types.NoneType: "null",
}


class DecodeError(errors.BadRequestError):
  """A request body that does not decode to the expected type."""


def parse(body: bytes) -> object:
  """Parses a request body as JSON.

  Args:
    body: The raw body, which must be UTF-8.

  Returns:
    The parsed value.

  Raises:
    DecodeError: If the body is not UTF-8 JSON, holds a non-finite number,
      holds an integer longer than Python converts, or nests too deeply.
  """
  try:
    return json.loads(body.decode("utf-8"), parse_constant=_refuse_constant)
  except RecursionError as error:
    raise DecodeError("The body nests too deeply.") from error
  except ValueError as error:
    # Covers UnicodeDecodeError, json.JSONDecodeError and the integer digit
    # limit, which json reports as a plain ValueError.
    raise DecodeError(f"The body is not JSON: {error}") from error


def _refuse_constant(name: str) -> float:
  """Refuses the `NaN` and `Infinity` literals that `json` accepts."""
  raise DecodeError(f"The body holds {name}, which JSON does not allow.")


def decode(cls: type[_T], value: object) -> _T:
  """Decodes a parsed JSON value as an instance of the dataclass `cls`.

  Args:
    cls: A dataclass whose fields are booleans, numbers, strings, None,
      enums, dataclasses, `tuple[X, ...]`, `dict[str, X]` or unions of these.
    value: A value returned by `parse`.

  Returns:
    The decoded instance.

  Raises:
    DecodeError: If `value` does not match `cls`. The message names the
      offending field.
  """
  return typing.cast(_T, _decode(cls, value, "body"))


def _decode(hint: Any, value: object, path: str) -> object:
  """Decodes `value` as the type `hint`; `path` names it in errors."""
  origin = typing.get_origin(hint)
  if origin in (typing.Union, types.UnionType):
    return _decode_union(typing.get_args(hint), value, path)
  if origin is tuple:
    return _decode_tuple(typing.get_args(hint), value, path)
  if origin is dict:
    return _decode_dict(typing.get_args(hint), value, path)
  if dataclasses.is_dataclass(hint) and isinstance(hint, type):
    return _decode_dataclass(hint, value, path)
  if isinstance(hint, type) and issubclass(hint, enum.Enum):
    return _decode_enum(hint, value, path)
  if hint in (bool, int, float, str, types.NoneType):
    if not _is_scalar(hint, value):
      raise DecodeError(f"{path} is not {_describe(hint)}.")
    return float(typing.cast(float, value)) if hint is float else value
  raise TypeError(f"Cannot decode {hint!r}.")


def _is_scalar(hint: type[Any], value: object) -> bool:
  """Whether `value` is a JSON scalar of the type `hint`."""
  if hint is bool or isinstance(value, bool):
    return hint is bool and isinstance(value, bool)
  if hint is float:
    return isinstance(value, (int, float)) and math.isfinite(value)
  return isinstance(value, hint)


def _describe(hint: Any) -> str:
  """Names the JSON values of the scalar type `hint`."""
  return _SCALAR_NAMES.get(hint, repr(hint))


def _decode_union(members: tuple[Any, ...], value: object, path: str) -> object:
  """Decodes `value` as the first member of a union that accepts it."""
  if value is None and types.NoneType in members:
    return None
  candidates = [member for member in members if member is not types.NoneType]
  if len(candidates) == 1:
    return _decode(candidates[0], value, path)
  for member in candidates:
    try:
      return _decode(member, value, path)
    except DecodeError:
      continue
  descriptions = [_describe(member) for member in candidates]
  raise DecodeError(
      f"{path} is not {', '.join(descriptions[:-1])} or {descriptions[-1]}."
  )


def _decode_tuple(args: tuple[Any, ...], value: object, path: str) -> object:
  """Decodes a JSON list as a `tuple[X, ...]`."""
  if len(args) != 2 or args[1] is not Ellipsis:
    raise TypeError(f"Cannot decode tuple{list(args)}.")
  if not isinstance(value, list):
    raise DecodeError(f"{path} is not a list.")
  return tuple(
      _decode(args[0], item, f"{path}[{index}]")
      for index, item in enumerate(value)
  )


def _decode_dict(args: tuple[Any, ...], value: object, path: str) -> object:
  """Decodes a JSON object as a `dict[str, X]`."""
  if args[0] is not str:
    raise TypeError(f"Cannot decode a dict keyed by {args[0]!r}.")
  if not isinstance(value, dict):
    raise DecodeError(f"{path} is not an object.")
  return {
      key: _decode(args[1], item, f"{path}.{key}")
      for key, item in value.items()
  }


def _decode_enum(cls: type[enum.Enum], value: object, path: str) -> object:
  """Decodes an enum from its wire value."""
  try:
    return cls(value)
  except ValueError:
    choices = ", ".join(repr(member.value) for member in cls)
    raise DecodeError(f"{path} is not one of {choices}.") from None


def _decode_dataclass(cls: type[Any], value: object, path: str) -> object:
  """Decodes a JSON object that carries exactly the fields of `cls`."""
  if not isinstance(value, dict):
    raise DecodeError(f"{path} is not an object.")
  hints = typing.get_type_hints(cls)
  fields = [field for field in dataclasses.fields(cls) if field.init]
  names = {field.name for field in fields}
  missing = sorted(names - value.keys())
  extra = sorted(value.keys() - names)
  if missing:
    raise DecodeError(f"{path} lacks the fields {', '.join(missing)}.")
  if extra:
    raise DecodeError(f"{path} has unknown fields {', '.join(extra)}.")
  arguments = {
      field.name: _decode(
          hints[field.name], value[field.name], f"{path}.{field.name}"
      )
      for field in fields
  }
  try:
    return cls(**arguments)
  except ValueError as error:
    raise DecodeError(f"{path} is invalid: {error}") from error
