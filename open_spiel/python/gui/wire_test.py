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


"""Tests for decoding request bodies."""

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import request
from open_spiel.python.gui import wire


def _seat(**changes: object) -> dict[str, object]:
  """Returns a `SeatConfig` body with `changes`."""
  return {"controller": "human", "config": {}, "autoplay": False} | changes


def _create(**changes: object) -> dict[str, object]:
  """Returns a `CreateSession` body with `changes`."""
  body: dict[str, object] = {
      "game": "chess",
      "parameters": {},
      "seats": [_seat(), _seat()],
      "chance_policy": "auto",
      "seed": None,
  }
  return body | changes


class ParseTest(absltest.TestCase):

  def test_parses_json(self) -> None:
    self.assertEqual(wire.parse(b'{"action": 3}'), {"action": 3})

  def test_refuses_a_body_that_is_not_json(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "not JSON"):
      wire.parse(b"{action: 3}")

  def test_refuses_a_body_that_is_not_utf8(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "not JSON"):
      wire.parse(b'{"game": "\xff"}')

  def test_refuses_an_integer_longer_than_python_converts(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "not JSON"):
      wire.parse(b"1" * 5000)

  def test_refuses_a_body_that_nests_too_deeply(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "nests too deeply"):
      wire.parse(b"[" * 100000)

  def test_refuses_a_body_one_level_past_the_nesting_cap(self) -> None:
    depth = wire.MAX_NESTING + 1
    with self.assertRaisesRegex(wire.DecodeError, "nests too deeply"):
      wire.parse(b"[" * depth + b"]" * depth)

  def test_parses_a_body_at_the_nesting_cap(self) -> None:
    depth = wire.MAX_NESTING
    self.assertIsInstance(wire.parse(b"[" * depth + b"]" * depth), list)

  def test_does_not_count_brackets_inside_strings(self) -> None:
    text = '["' + "[" * (wire.MAX_NESTING + 1) + '\\"{"]'
    self.assertEqual(
        wire.parse(text.encode()), ["[" * (wire.MAX_NESTING + 1) + '"{']
    )

  def test_refuses_non_finite_numbers(self) -> None:
    for literal in (b"NaN", b"Infinity", b"-Infinity"):
      with self.subTest(literal=literal):
        with self.assertRaisesRegex(wire.DecodeError, "does not allow"):
          wire.parse(b'{"max_plies": ' + literal + b"}")


class DecodeTest(absltest.TestCase):

  def test_decodes_nested_dataclasses_enums_and_tuples(self) -> None:
    decoded = wire.decode(request.CreateSession, _create(seed=7))
    self.assertEqual(
        decoded,
        request.CreateSession(
            game="chess",
            parameters={},
            seats=(
                request.SeatConfig("human", {}, False),
                request.SeatConfig("human", {}, False),
            ),
            chance_policy=contract.ChancePolicy.AUTO,
            seed=7,
        ),
    )

  def test_parameter_values_keep_their_json_types(self) -> None:
    decoded = wire.decode(
        request.CreateSession,
        _create(parameters={"b": True, "i": 9, "f": 6.5, "s": "x"}),
    )
    self.assertEqual(
        decoded.parameters, {"b": True, "i": 9, "f": 6.5, "s": "x"}
    )
    self.assertIsInstance(decoded.parameters["b"], bool)
    self.assertIsInstance(decoded.parameters["i"], int)

  def test_a_nullable_field_takes_null(self) -> None:
    self.assertEqual(
        wire.decode(request.Advance, {"max_plies": None}),
        request.Advance(max_plies=None),
    )

  def test_a_body_that_breaks_its_invariants_is_a_bad_request(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, r"body\.seats\[0\] is invalid: .*human seat"
    ):
      wire.decode(request.CreateSession, _create(seats=[_seat(autoplay=True)]))
    with self.assertRaisesRegex(wire.DecodeError, r"body is invalid: .*seed"):
      wire.decode(request.CreateSession, _create(seed=-1))
    with self.assertRaisesRegex(wire.DecodeError, "body is invalid"):
      wire.decode(request.Advance, {"max_plies": 0})

  def test_refuses_a_missing_field(self) -> None:
    body = _create()
    del body["seed"]
    with self.assertRaisesRegex(wire.DecodeError, "body lacks the fields seed"):
      wire.decode(request.CreateSession, body)

  def test_refuses_an_unknown_field(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, "body has unknown fields player"
    ):
      wire.decode(request.ApplyAction, {"action": 1, "ply": None, "player": 0})

  def test_a_boolean_is_not_an_integer(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, r"body\.action is not an integer"
    ):
      wire.decode(request.ApplyAction, {"action": True, "ply": None})

  def test_an_integer_field_refuses_a_fraction(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, r"body\.action is not an integer"
    ):
      wire.decode(request.ApplyAction, {"action": 1.5, "ply": None})

  def test_an_enum_takes_only_its_wire_values(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, r"body\.chance_policy is not one of 'auto', 'expose'"
    ):
      wire.decode(request.CreateSession, _create(chance_policy="AUTO"))

  def test_errors_name_the_nested_field(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError, r"body\.seats\[1\]\.autoplay is not a boolean"
    ):
      wire.decode(
          request.CreateSession, _create(seats=[_seat(), _seat(autoplay=1)])
      )

  def test_a_parameter_must_be_a_scalar(self) -> None:
    with self.assertRaisesRegex(
        wire.DecodeError,
        r"body\.parameters\.size is not a boolean, an integer, a number or a "
        "string",
    ):
      wire.decode(request.CreateSession, _create(parameters={"size": [9]}))

  def test_refuses_a_body_that_is_not_an_object(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "body is not an object"):
      wire.decode(request.ApplyAction, [1])

  def test_reports_a_broken_invariant_as_a_decode_error(self) -> None:
    with self.assertRaisesRegex(wire.DecodeError, "body is invalid"):
      wire.decode(contract.Viewer, {"kind": "seat", "seat": None})


if __name__ == "__main__":
  absltest.main()
