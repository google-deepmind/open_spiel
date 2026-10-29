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


"""A controller's settings: their JSON Schema, and the values they accept.

A client sends a controller's settings as JSON. `schema` describes them, and
`parse` checks what a client sent against them and fills in the defaults.

A setting is a boolean, an integer, a number or a string. One with a default
may be left out; one declared with `default=REQUIRED` must be sent. A number
may be bounded; a string may be bounded in length and held to a pattern; a
string or a number may be held to a closed set of `choices`. For example, a
controller that plays a checkpoint takes:

  Setting("checkpoint", str, REQUIRED, "The checkpoint to play.",
          min_length=1)
  Setting("deterministic", bool, True, "Whether it plays its best move.")
  Setting("temperature", float, 1.0, "The sampling temperature.",
          minimum=0.0)
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import math
import re
from typing import Final

from open_spiel.python.gui import contract

# A controller's settings, as a client sends them.
Config = Mapping[str, contract.ParameterValue]

# The Python type a setting takes.
SettingType = type[bool] | type[int] | type[float] | type[str]

# The longest client-sent text an error message repeats.
_MAX_SHOWN = 40

# The JSON Schema type of each setting type, and the values it accepts.
_JSON_TYPES: Mapping[SettingType, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}
_ACCEPTED: Mapping[SettingType, tuple[type[object], ...]] = {
    bool: (bool,),
    int: (int,),
    float: (int, float),
    str: (str,),
}


class _Required(enum.Enum):
  """The marker of a setting that has no default: `REQUIRED`."""

  REQUIRED = "REQUIRED"

  def __repr__(self) -> str:
    return "REQUIRED"


# The default of a setting a client must send.
REQUIRED: Final = _Required.REQUIRED


class InvalidSettingError(ValueError):
  """A client sent a setting the controller does not take, or a bad value.

  Also the error of a client that leaves out a required setting. The message
  is safe to show to anyone.
  """


def shown(value: object) -> str:
  """Returns `value`'s repr, cut short if a client sent something long."""
  text = repr(value)
  if len(text) <= _MAX_SHOWN:
    return text
  return f"{text[:_MAX_SHOWN]}..."


def _characters(count: int) -> str:
  """Returns "1 character" or "`count` characters"."""
  return "1 character" if count == 1 else f"{count} characters"


@dataclasses.dataclass(frozen=True)
class Setting:
  """One setting of a controller, and the values that it admits.

  The constructor checks the declaration, so a controller cannot declare a
  setting whose default its own constraints refuse.

  Attributes:
    name: The setting's name, as a client sends it.
    kind: The setting's type: bool, int, float or str.
    default: The value a client that leaves the setting out gets, or
      `REQUIRED` for a setting a client must send.
    description: What the setting does, for a client to show.
    minimum: The least value a number setting takes, or None for no bound.
    maximum: The greatest value a number setting takes, or None for no bound.
    min_length: The fewest characters a string setting takes, or None.
    max_length: The most characters a string setting takes, or None.
    pattern: A regular expression a string setting must contain a match of,
      as JSON Schema's `pattern` does; anchor it with ^ and $ to match the
      whole string. Keep to the syntax Python and JavaScript share, since a
      client checks it too. None for no pattern.
    choices: The only values the setting takes, for a string or number
      setting, in the order a client offers them; None for no closed set.
  """

  name: str
  kind: SettingType
  default: contract.ParameterValue | _Required
  description: str
  minimum: float | None = None
  maximum: float | None = None
  min_length: int | None = None
  max_length: int | None = None
  pattern: str | None = None
  choices: tuple[contract.ParameterValue, ...] | None = None

  def __post_init__(self) -> None:
    if self.kind not in _JSON_TYPES:
      raise ValueError(
          f"Setting {self.name!r} has kind {self.kind!r}, not bool, int, "
          "float or str."
      )
    is_number = self.kind in (int, float)
    if not is_number and (self.minimum is not None or self.maximum is not None):
      raise ValueError(f"Setting {self.name!r} is not a number to bound.")
    is_string = self.kind is str
    has_text_constraint = (
        self.min_length is not None
        or self.max_length is not None
        or self.pattern is not None
    )
    if not is_string and has_text_constraint:
      raise ValueError(
          f"Setting {self.name!r} is not a string to bound in length or "
          "hold to a pattern."
      )
    for length in (self.min_length, self.max_length):
      if length is not None and length < 0:
        raise ValueError(
            f"Setting {self.name!r} has length bound {length!r}, not a "
            "whole number of characters."
        )
    if (
        self.min_length is not None
        and self.max_length is not None
        and self.min_length > self.max_length
    ):
      raise ValueError(f"Setting {self.name!r} admits no length.")
    if self.pattern is not None:
      try:
        re.compile(self.pattern)
      except re.error as e:
        raise ValueError(
            f"Setting {self.name!r} has an invalid pattern: {e}."
        ) from e
    if self.choices is not None:
      if self.kind is bool:
        raise ValueError(
            f"Setting {self.name!r} is a boolean, which has its own choices."
        )
      if not self.choices or len(set(self.choices)) != len(self.choices):
        raise ValueError(
            f"Setting {self.name!r} has no choices, or repeats one."
        )
      for choice in self.choices:
        self._check_declared(choice, "choice")
    if self.default is not REQUIRED:
      self._check_declared(self.default, "default")

  def _check_declared(self, value: contract.ParameterValue, what: str) -> None:
    """Raises ValueError if the setting refuses its own `value`."""
    try:
      self.parse(value)
    except InvalidSettingError as e:
      raise ValueError(f"The {what} is refused: {e}") from e

  @property
  def required(self) -> bool:
    """Whether a client must send the setting, having no default."""
    return self.default is REQUIRED

  def schema(self) -> dict[str, contract.Json]:
    """Returns the setting's JSON Schema."""
    result: dict[str, contract.Json] = {"type": _JSON_TYPES[self.kind]}
    if self.default is not REQUIRED:
      result["default"] = self.default
    result["description"] = self.description
    optional: dict[str, contract.Json | None] = {
        "minimum": self.minimum,
        "maximum": self.maximum,
        "minLength": self.min_length,
        "maxLength": self.max_length,
        "pattern": self.pattern,
        "enum": None if self.choices is None else list(self.choices),
    }
    result.update(
        (keyword, value)
        for keyword, value in optional.items()
        if value is not None
    )
    return result

  def _bounds(self) -> str:
    """Describes the values the setting's bounds admit."""
    if self.minimum is not None and self.maximum is not None:
      return f"lies in [{self.minimum}, {self.maximum}]"
    if self.minimum is not None:
      return f"is at least {self.minimum}"
    if self.maximum is not None:
      return f"is at most {self.maximum}"
    return "is a finite number"

  def _lengths(self) -> str:
    """Describes the lengths the setting's length bounds admit."""
    if self.min_length is not None and self.max_length is not None:
      return f"{self.min_length} to {self.max_length} characters long"
    if self.min_length is not None:
      return f"at least {_characters(self.min_length)} long"
    assert self.max_length is not None, self.name
    return f"at most {_characters(self.max_length)} long"

  def parse(self, value: contract.ParameterValue) -> contract.ParameterValue:
    """Returns `value` as the setting's type.

    Args:
      value: The value a client sent.

    Raises:
      InvalidSettingError: If `value` has the wrong type, is not finite, lies
        out of bounds, has a length out of bounds, does not match the
        pattern or is not one of the choices. A boolean is never taken for a
        number.
    """
    # bool subclasses int, so a boolean must be refused for a number, and a
    # number for a boolean, before the type itself is checked.
    is_boolean_as_wanted = isinstance(value, bool) == (self.kind is bool)
    has_accepted_type = isinstance(value, _ACCEPTED[self.kind])
    if not is_boolean_as_wanted or not has_accepted_type:
      raise InvalidSettingError(
          f"Setting {self.name!r} takes {_JSON_TYPES[self.kind]} values, not "
          f"{shown(value)}."
      )
    parsed: contract.ParameterValue
    if isinstance(value, str):
      parsed = self._parse_string(value)
    elif isinstance(value, bool):
      return value  # A boolean, for a boolean setting.
    else:
      parsed = self._parse_number(value)
    if self.choices is not None and parsed not in self.choices:
      raise InvalidSettingError(
          f"Setting {self.name!r} is one of "
          f"{', '.join(map(shown, self.choices))}, so it cannot be "
          f"{shown(value)}."
      )
    return parsed

  def _parse_string(self, value: str) -> str:
    """Returns `value`, a string, if its length and pattern are admitted."""
    if (self.min_length is not None and len(value) < self.min_length) or (
        self.max_length is not None and len(value) > self.max_length
    ):
      raise InvalidSettingError(
          f"Setting {self.name!r} is {self._lengths()}, so it cannot be "
          f"{shown(value)}."
      )
    if self.pattern is not None and re.search(self.pattern, value) is None:
      raise InvalidSettingError(
          f"Setting {self.name!r} matches {shown(self.pattern)}, so it cannot "
          f"be {shown(value)}."
      )
    return value

  def _parse_number(self, value: int | float) -> int | float:
    """Returns `value`, a number, as the setting's type, if within bounds."""
    try:
      number = float(value)
    except OverflowError:  # An integer too large for a float.
      number = math.inf
    if (
        not math.isfinite(number)
        or (self.minimum is not None and number < self.minimum)
        or (self.maximum is not None and number > self.maximum)
    ):
      raise InvalidSettingError(
          f"Setting {self.name!r} {self._bounds()}, so it cannot be "
          f"{shown(value)}."
      )
    return int(value) if self.kind is int else float(value)


def schema(settings: Sequence[Setting]) -> dict[str, contract.Json]:
  """Returns the JSON Schema of a controller that takes `settings`.

  The schema lists the settings a client must send in `required`, which it
  leaves out when there are none.
  """
  if not settings:
    return {}
  result: dict[str, contract.Json] = {
      "type": "object",
      "properties": {setting.name: setting.schema() for setting in settings},
      "additionalProperties": False,
  }
  required = [setting.name for setting in settings if setting.required]
  if required:
    result["required"] = required
  return result


def parse(
    settings: Sequence[Setting], config: Config, label: str
) -> dict[str, contract.ParameterValue]:
  """Returns `config` with its defaults filled in.

  Args:
    settings: The settings the controller takes.
    config: The settings a client sent.
    label: The controller's human-readable name, for the error message.

  Raises:
    InvalidSettingError: If `config` names an unknown setting, leaves out a
      required one or holds an invalid value.
  """
  known = {setting.name: setting for setting in settings}
  unknown = sorted(set(config) - set(known))
  if unknown:
    raise InvalidSettingError(
        f"{label} takes no setting {', '.join(map(shown, unknown))}."
    )
  missing = [
      name
      for name, setting in known.items()
      if setting.required and name not in config
  ]
  if missing:
    raise InvalidSettingError(
        f"{label} needs the setting {', '.join(map(shown, missing))}."
    )
  parsed: dict[str, contract.ParameterValue] = {}
  for name, setting in known.items():
    value = config.get(name, setting.default)
    assert not isinstance(value, _Required), name  # Refused as missing above.
    parsed[name] = setting.parse(value)
  return parsed
