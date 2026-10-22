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

"""Tests for the generated JSON Schema of the wire contract."""

import dataclasses
import json

from absl.testing import absltest

from open_spiel.python.gui import catalogue
from open_spiel.python.gui import contract
from open_spiel.python.gui import errors
from open_spiel.python.gui import frame
from open_spiel.python.gui import health
from open_spiel.python.gui import request
from open_spiel.python.gui import schema

_DATACLASSES = {}
for _module in (catalogue, contract, errors, frame, health, request):
  _DATACLASSES.update({
      name: value
      for name, value in vars(_module).items()
      if isinstance(value, type) and dataclasses.is_dataclass(value)
  })


def _definitions() -> dict[str, dict[str, object]]:
  """The schema's definitions, read back from its rendered text."""
  definitions: dict[str, dict[str, object]] = json.loads(schema.render())[
      "$defs"
  ]
  return definitions


def _properties(name: str) -> dict[str, dict[str, object]]:
  """The properties of the schema's definition of `name`."""
  properties = _definitions()[name]["properties"]
  assert isinstance(properties, dict)
  return properties


class SchemaTest(absltest.TestCase):

  def test_the_checked_in_schema_is_current(self) -> None:
    with open(schema.SCHEMA_PATH, encoding="utf-8") as f:
      checked_in = f.read()
    self.assertEqual(
        checked_in,
        schema.render(),
        "contract.schema.json is stale; regenerate it with "
        "`python -m open_spiel.python.gui.schema`.",
    )

  def test_every_contract_type_is_defined(self) -> None:
    self.assertLessEqual(set(_DATACLASSES), set(_definitions()))

  def test_objects_require_exactly_their_fields(self) -> None:
    for name, definition in _definitions().items():
      if name not in _DATACLASSES:
        continue
      with self.subTest(name):
        fields = [f.name for f in dataclasses.fields(_DATACLASSES[name])]
        self.assertEqual(list(_properties(name)), fields)
        self.assertEqual(definition["required"], fields)
        self.assertIs(definition["additionalProperties"], False)

  def test_every_type_field_and_value_is_described(self) -> None:
    for name, definition in _definitions().items():
      with self.subTest(name):
        self.assertTrue(definition["description"])
        for prop in _properties(name).values() if name in _DATACLASSES else ():
          self.assertTrue(prop["description"])

  def test_enums_list_their_wire_values(self) -> None:
    self.assertEqual(
        _definitions()["ViewerKind"]["enum"], ["seat", "spectator", "oracle"]
    )

  def test_nullable_fields_admit_null(self) -> None:
    frame_properties = _properties("Frame")
    self.assertEqual(frame_properties["state_text"]["type"], ["string", "null"])
    legal_actions = frame_properties["legal_actions"]["anyOf"]
    assert isinstance(legal_actions, list)
    self.assertIn({"type": "null"}, legal_actions)

  def test_json_fields_are_open(self) -> None:
    self.assertEqual(set(_properties("Frame")["struct"]), {"description"})

  def test_the_contract_version_is_a_constant(self) -> None:
    for name in ("Frame", "Catalogue", "Health", "Error"):
      with self.subTest(name):
        self.assertEqual(
            _properties(name)["contract_version"]["const"],
            contract.CONTRACT_VERSION,
        )

  def test_responses_are_the_roots(self) -> None:
    roots = json.loads(schema.render())["oneOf"]
    self.assertEqual(
        roots,
        [{"$ref": f"#/$defs/{cls.__name__}"} for cls in schema.RESPONSE_TYPES],
    )


if __name__ == "__main__":
  absltest.main()
