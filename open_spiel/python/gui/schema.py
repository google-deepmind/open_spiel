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

"""Generates the JSON Schema of the wire contract.

The schema is derived from the dataclasses and enums of the contract modules;
the descriptions of types, fields and values come from their docstrings. To
regenerate the checked-in schema after changing the contract:

  python -m open_spiel.python.gui.schema
"""

from collections.abc import Sequence
import dataclasses
import enum
import inspect
import json
import os
import re
import types
import typing

from absl import app

from open_spiel.python.gui import catalogue
from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
from open_spiel.python.gui import frame
from open_spiel.python.gui import health
from open_spiel.python.gui import request

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "contract.schema.json")

# Every response body is one of these.
RESPONSE_TYPES: tuple[type[object], ...] = (
    frame.Frame,
    catalogue.Catalogue,
    health.Health,
    contract.SessionClosed,
    errors.Error,
)

# Every request body is one of these.
REQUEST_TYPES: tuple[type[object], ...] = (
    request.CreateSession,
    request.ApplyAction,
    request.ApplyChanceOutcome,
    request.Advance,
    request.Reset,
    request.SeatConfig,
)

Schema = dict[str, contract.JsonValue]

_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_PRIMITIVES: dict[type[object], str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    type(None): "null",
}
_ATTRIBUTE = re.compile(r"^  (\w+): (.*)$")


def _docstring(cls: type[object]) -> str:
  """Returns the docstring of `cls`, which every contract type has."""
  if not cls.__doc__:
    raise ValueError(f"{cls.__name__} has no docstring.")
  return inspect.cleandoc(cls.__doc__)


def _description(cls: type[object]) -> str:
  """Returns the prose of a docstring, without its `Attributes:` section."""
  prose = _docstring(cls).split("\nAttributes:", maxsplit=1)[0].strip()
  return "\n\n".join(" ".join(part.split()) for part in prose.split("\n\n"))


def _attributes(cls: type[object], names: Sequence[str]) -> dict[str, str]:
  """Returns the description of each name in a docstring's `Attributes:`.

  Args:
    cls: The documented class.
    names: The names `cls` must document, and no others.

  Returns:
    Each name's description, on one line.

  Raises:
    ValueError: If the docstring does not document exactly `names`.
  """
  lines = _docstring(cls).splitlines()
  if "Attributes:" not in lines:
    raise ValueError(f"{cls.__name__} documents no attributes.")
  descriptions: dict[str, str] = {}
  name = ""
  for line in lines[lines.index("Attributes:") + 1 :]:
    if match := _ATTRIBUTE.match(line):
      name, descriptions[name] = match.groups()
    elif line.startswith("    ") and name:
      descriptions[name] += " " + line.strip()
    else:
      break
  if sorted(descriptions) != sorted(names):
    raise ValueError(
        f"{cls.__name__} documents {sorted(descriptions)}, not {sorted(names)}."
    )
  return descriptions


class _Builder:
  """Accumulates the definitions reachable from the contract's bodies."""

  def __init__(self) -> None:
    self.definitions: dict[str, Schema] = {}

  def reference(self, cls: type[object]) -> Schema:
    """Defines `cls` once and returns a reference to its definition."""
    if cls.__name__ not in self.definitions:
      self.definitions[cls.__name__] = {}  # Reserved against recursion.
      if issubclass(cls, enum.Enum):
        self.definitions[cls.__name__] = self._enum(cls)
      else:
        self.definitions[cls.__name__] = self._object(cls)
    return {"$ref": f"#/$defs/{cls.__name__}"}

  def _enum(self, cls: type[enum.Enum]) -> Schema:
    """Returns the definition of a string enum."""
    descriptions = _attributes(cls, [member.name for member in cls])
    values = "\n".join(
        f"- `{member.value}`: {descriptions[member.name]}" for member in cls
    )
    return {
        "type": "string",
        "enum": [str(member.value) for member in cls],
        "description": f"{_description(cls)}\n\n{values}",
    }

  def _object(self, cls: type[object]) -> Schema:
    """Returns the definition of a dataclass."""
    if not dataclasses.is_dataclass(cls):
      raise TypeError(f"{cls.__name__} is neither a dataclass nor an enum.")
    hints = typing.get_type_hints(cls, include_extras=True)
    names = [field.name for field in dataclasses.fields(cls)]
    descriptions = _attributes(cls, names)
    properties: Schema = {
        name: self.schema_of(hints[name]) | {"description": descriptions[name]}
        for name in names
    }
    return {
        "type": "object",
        "description": _description(cls),
        "properties": properties,
        "required": list(names),
        "additionalProperties": False,
    }

  def schema_of(self, hint: object) -> Schema:
    """Returns the schema of a type annotation."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Annotated:
      return {} if contract.ANY_JSON in args[1:] else self.schema_of(args[0])
    if isinstance(hint, type) and hint in _PRIMITIVES:
      return {"type": _PRIMITIVES[hint]}
    if origin is typing.Literal and len(args) == 1:
      return {"const": args[0]}
    if origin in (types.UnionType, typing.Union):
      return self._union(args)
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
      return {"type": "array", "items": self.schema_of(args[0])}
    if origin is dict and args[0] is str:
      return {"type": "object", "additionalProperties": self.schema_of(args[1])}
    if isinstance(hint, type):
      return self.reference(hint)
    raise TypeError(f"The contract cannot carry {hint!r}.")

  def _union(self, members: Sequence[object]) -> Schema:
    """Returns the schema of a union of types."""
    if all(isinstance(m, type) and m in _PRIMITIVES for m in members):
      names = [_PRIMITIVES[m] for m in members if isinstance(m, type)]
      if "number" in names and "integer" in names:
        names.remove("integer")  # Every integer is a number.
      return {"type": list(names)}
    return {"anyOf": [self.schema_of(member) for member in members]}


def build() -> Schema:
  """Returns the JSON Schema of every request and response body."""
  builder = _Builder()
  responses = [builder.reference(cls) for cls in RESPONSE_TYPES]
  for cls in REQUEST_TYPES:
    builder.reference(cls)
  return {
      "$schema": _DIALECT,
      "title": "OpenSpiel GUI contract",
      "description": (
          f"Version {contract.CONTRACT_VERSION} of the contract. Every "
          "response body is one of these; request bodies are among the "
          "definitions. Generated by schema.py; do not edit."
      ),
      "oneOf": list(responses),
      "$defs": dict(sorted(builder.definitions.items())),
  }


def render() -> str:
  """Returns the schema as the text of `contract.schema.json`."""
  return json.dumps(build(), indent=2) + "\n"


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
    f.write(render())


if __name__ == "__main__":
  app.run(main)
