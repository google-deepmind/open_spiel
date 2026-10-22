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
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math

from open_spiel.python.gui import contract

# A controller's settings, as a client sends them.
Config = Mapping[str, contract.ParameterValue]

# The Python type a setting takes.
SettingType = type[bool] | type[int] | type[float]

# The longest client-sent text an error message repeats.
_MAX_SHOWN = 40

# The JSON Schema type of each setting type, and the values it accepts.
_JSON_TYPES: Mapping[SettingType, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
}
_ACCEPTED: Mapping[SettingType, tuple[type[object], ...]] = {
    bool: (bool,),
    int: (int,),
    float: (int, float),
}


class InvalidSettingError(ValueError):
  """A client sent a setting the controller does not take, or a bad value.

  The message is safe to show to anyone.
  """


def shown(value: object) -> str:
  """Returns `value`'s repr, cut short if a client sent something long."""
  text = repr(value)
  if len(text) <= _MAX_SHOWN:
    return text
  return f"{text[:_MAX_SHOWN]}..."


@dataclasses.dataclass(frozen=True)
class Setting:
  """One setting of a controller, and the bounds that it must lie within.

  Attributes:
    name: The setting's name, as a client sends it.
    kind: The setting's type.
    default: The value a client that leaves the setting out gets.
    description: What the setting does, for a client to show.
    minimum: The least value the setting takes, or None for no bound.
    maximum: The greatest value the setting takes, or None for no bound.
  """

  name: str
  kind: SettingType
  default: contract.ParameterValue
  description: str
  minimum: float | None = None
  maximum: float | None = None

  def schema(self) -> dict[str, contract.Json]:
    """Returns the setting's JSON Schema."""
    result: dict[str, contract.Json] = {
        "type": _JSON_TYPES[self.kind],
        "default": self.default,
        "description": self.description,
    }
    if self.minimum is not None:
      result["minimum"] = self.minimum
    if self.maximum is not None:
      result["maximum"] = self.maximum
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

  def parse(self, value: contract.ParameterValue) -> contract.ParameterValue:
    """Returns `value` as the setting's type.

    Args:
      value: The value a client sent.

    Raises:
      InvalidSettingError: If `value` has the wrong type, is not finite or
        lies out of bounds. A boolean is never taken for a number.
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      return value  # A boolean, for a boolean setting.
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
    return self.kind(value)


def schema(settings: Sequence[Setting]) -> dict[str, contract.Json]:
  """Returns the JSON Schema of a controller that takes `settings`."""
  if not settings:
    return {}
  return {
      "type": "object",
      "properties": {setting.name: setting.schema() for setting in settings},
      "additionalProperties": False,
  }


def parse(
    settings: Sequence[Setting], config: Config, label: str
) -> dict[str, contract.ParameterValue]:
  """Returns `config` with its defaults filled in.

  Args:
    settings: The settings the controller takes.
    config: The settings a client sent.
    label: The controller's human-readable name, for the error message.

  Raises:
    InvalidSettingError: If `config` names an unknown setting or holds an
      invalid value.
  """
  known = {setting.name: setting for setting in settings}
  unknown = sorted(set(config) - set(known))
  if unknown:
    raise InvalidSettingError(
        f"{label} takes no setting {', '.join(map(shown, unknown))}."
    )
  return {
      name: setting.parse(config.get(name, setting.default))
      for name, setting in known.items()
  }
