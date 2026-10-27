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


"""Tests for the settings a controller takes."""

from collections.abc import Callable

from absl.testing import absltest

from open_spiel.python.gui import controller_settings

_Setting = controller_settings.Setting
_Invalid = controller_settings.InvalidSettingError
_REQUIRED = controller_settings.REQUIRED


class SettingTest(absltest.TestCase):

  def test_a_lower_bound_alone_reads_at_least(self) -> None:
    setting = _Setting("c", float, 1.0, "A constant.", minimum=0.0)
    with self.assertRaisesRegex(
        _Invalid, r"^Setting 'c' is at least 0.0, so it cannot be -1.0.$"
    ):
      setting.parse(-1.0)

  def test_an_upper_bound_alone_reads_at_most(self) -> None:
    setting = _Setting("n", int, 1, "A count.", maximum=5)
    with self.assertRaisesRegex(
        _Invalid, r"^Setting 'n' is at most 5, so it cannot be 6.$"
    ):
      setting.parse(6)

  def test_both_bounds_read_lies_in(self) -> None:
    setting = _Setting("n", int, 1, "A count.", minimum=1, maximum=5)
    with self.assertRaisesRegex(_Invalid, r"lies in \[1, 5\]"):
      setting.parse(0)

  def test_an_unbounded_number_must_be_finite(self) -> None:
    setting = _Setting("c", float, 1.0, "A constant.")
    for value in (float("inf"), float("-inf"), float("nan")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(_Invalid, "is a finite number"):
          setting.parse(value)

  def test_a_number_within_bounds_takes_the_settings_type(self) -> None:
    setting = _Setting("c", float, 1.0, "A constant.", minimum=0.0)
    parsed = setting.parse(3)
    self.assertIsInstance(parsed, float)
    self.assertEqual(parsed, 3.0)

  def test_a_boolean_is_not_a_number_nor_a_number_a_boolean(self) -> None:
    with self.assertRaisesRegex(_Invalid, "takes integer values"):
      _Setting("n", int, 1, "A count.").parse(True)
    with self.assertRaisesRegex(_Invalid, "takes boolean values"):
      _Setting("b", bool, True, "A switch.").parse(0)

  def test_a_long_value_is_cut_short_in_the_message(self) -> None:
    with self.assertRaises(_Invalid) as raised:
      _Setting("n", int, 1, "A count.").parse("x" * 10_000)
    self.assertLess(len(str(raised.exception)), 120)


class StringSettingTest(absltest.TestCase):

  def test_a_string_takes_text(self) -> None:
    self.assertEqual(_Setting("s", str, "", "Text.").parse("abc"), "abc")

  def test_a_string_takes_no_number_nor_a_number_text(self) -> None:
    with self.assertRaisesRegex(_Invalid, "takes string values, not 1.$"):
      _Setting("s", str, "", "Text.").parse(1)
    with self.assertRaisesRegex(_Invalid, "takes integer values, not '1'.$"):
      _Setting("n", int, 1, "A count.").parse("1")

  def test_length_bounds(self) -> None:
    setting = _Setting("s", str, "ab", "Text.", min_length=2, max_length=3)
    self.assertEqual(setting.parse("abc"), "abc")
    for value in ("a", "abcd"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            _Invalid,
            "^Setting 's' is 2 to 3 characters long, so it cannot be "
            f"'{value}'.$",
        ):
          setting.parse(value)

  def test_one_length_bound_alone(self) -> None:
    with self.assertRaisesRegex(_Invalid, "is at least 1 character long"):
      _Setting("s", str, "a", "Text.", min_length=1).parse("")
    with self.assertRaisesRegex(_Invalid, "is at most 1 character long"):
      _Setting("s", str, "", "Text.", max_length=1).parse("ab")

  def test_a_pattern_is_searched_for_as_json_schema_does(self) -> None:
    unanchored = _Setting("s", str, "x1", "Text.", pattern="[0-9]")
    self.assertEqual(unanchored.parse("a1b"), "a1b")
    anchored = _Setting("s", str, "1", "Text.", pattern="^[0-9]+$")
    with self.assertRaises(_Invalid) as raised:
      anchored.parse("a1")
    self.assertEqual(
        str(raised.exception),
        "Setting 's' matches '^[0-9]+$', so it cannot be 'a1'.",
    )

  def test_choices_close_the_set(self) -> None:
    setting = _Setting("s", str, "a", "Text.", choices=("a", "b"))
    self.assertEqual(setting.parse("b"), "b")
    with self.assertRaisesRegex(
        _Invalid, "^Setting 's' is one of 'a', 'b', so it cannot be 'c'.$"
    ):
      setting.parse("c")

  def test_number_choices(self) -> None:
    setting = _Setting("n", int, 1, "A count.", choices=(1, 2))
    self.assertEqual(setting.parse(2), 2)
    with self.assertRaisesRegex(_Invalid, "is one of 1, 2"):
      setting.parse(3)

  def test_the_schema_gives_the_string_constraints(self) -> None:
    setting = _Setting(
        "s", str, "ab", "Text.", min_length=1, max_length=8, pattern="^[a-z]+$"
    )
    self.assertEqual(
        setting.schema(),
        {
            "type": "string",
            "default": "ab",
            "description": "Text.",
            "minLength": 1,
            "maxLength": 8,
            "pattern": "^[a-z]+$",
        },
    )

  def test_the_schema_gives_choices_as_an_enum(self) -> None:
    setting = _Setting("s", str, "a", "Text.", choices=("a", "b"))
    self.assertEqual(setting.schema()["enum"], ["a", "b"])


class DeclarationTest(absltest.TestCase):

  def test_a_default_the_constraints_refuse_is_refused(self) -> None:
    cases: tuple[Callable[[], _Setting], ...] = (
        lambda: _Setting("s", str, "", "Text.", min_length=1),
        lambda: _Setting("s", str, "a", "Text.", choices=("b",)),
        lambda: _Setting("s", str, 1, "Text."),
        lambda: _Setting("n", int, 0, "A count.", minimum=1),
        lambda: _Setting("n", int, 1.5, "A count."),
    )
    for declare in cases:
      with self.assertRaisesRegex(ValueError, "The default is refused"):
        declare()

  def test_a_constraint_of_the_wrong_kind_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "not a number to bound"):
      _Setting("s", str, "", "Text.", minimum=0.0)
    with self.assertRaisesRegex(ValueError, "not a string to bound"):
      _Setting("n", int, 1, "A count.", pattern="1")
    with self.assertRaisesRegex(ValueError, "its own choices"):
      _Setting("b", bool, True, "A switch.", choices=(True,))

  def test_bad_lengths_patterns_and_choices_are_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "admits no length"):
      _Setting("s", str, _REQUIRED, "Text.", min_length=3, max_length=2)
    with self.assertRaisesRegex(ValueError, "not a whole number"):
      _Setting("s", str, _REQUIRED, "Text.", min_length=-1)
    with self.assertRaisesRegex(ValueError, "invalid pattern"):
      _Setting("s", str, _REQUIRED, "Text.", pattern="(")
    with self.assertRaisesRegex(ValueError, "no choices, or repeats one"):
      _Setting("s", str, _REQUIRED, "Text.", choices=())
    with self.assertRaisesRegex(ValueError, "no choices, or repeats one"):
      _Setting("s", str, _REQUIRED, "Text.", choices=("a", "a"))
    with self.assertRaisesRegex(ValueError, "The choice is refused"):
      _Setting("s", str, _REQUIRED, "Text.", choices=("a", 1))

  def test_an_unknown_kind_is_refused(self) -> None:
    with self.assertRaisesRegex(ValueError, "not bool, int, float or str"):
      _Setting("l", list, [], "A list.")  # type: ignore[arg-type]


class RequiredTest(absltest.TestCase):

  # The checkpoint bot's settings, the contract this was built for.
  _SETTINGS = (
      _Setting("checkpoint", str, _REQUIRED, "The checkpoint.", min_length=1),
      _Setting("deterministic", bool, True, "Plays its best move."),
      _Setting("temperature", float, 1.0, "Sampling.", minimum=0.0),
  )

  def test_a_setting_with_no_default_is_required(self) -> None:
    self.assertTrue(self._SETTINGS[0].required)
    self.assertFalse(self._SETTINGS[1].required)
    self.assertEqual(repr(_REQUIRED), "REQUIRED")

  def test_a_missing_required_setting_is_refused_by_name(self) -> None:
    with self.assertRaisesRegex(
        _Invalid, "^Checkpoint needs the setting 'checkpoint'.$"
    ):
      controller_settings.parse(self._SETTINGS, {}, "Checkpoint")
    with self.assertRaisesRegex(_Invalid, "needs the setting 'checkpoint'"):
      controller_settings.parse(
          self._SETTINGS, {"temperature": 0.5}, "Checkpoint"
      )

  def test_every_missing_required_setting_is_named(self) -> None:
    settings = (
        _Setting("a", str, _REQUIRED, "A."),
        _Setting("b", int, _REQUIRED, "B."),
    )
    with self.assertRaisesRegex(_Invalid, "^Bot needs the setting 'a', 'b'.$"):
      controller_settings.parse(settings, {}, "Bot")

  def test_a_required_setting_sent_is_checked_and_defaults_fill_the_rest(
      self,
  ) -> None:
    self.assertEqual(
        controller_settings.parse(
            self._SETTINGS,
            {"checkpoint": "runs/run_1/checkpoint_100"},
            "Checkpoint",
        ),
        {
            "checkpoint": "runs/run_1/checkpoint_100",
            "deterministic": True,
            "temperature": 1.0,
        },
    )
    with self.assertRaisesRegex(_Invalid, "at least 1 character long"):
      controller_settings.parse(self._SETTINGS, {"checkpoint": ""}, "C")

  def test_a_required_setting_of_any_type(self) -> None:
    for kind, value in ((bool, False), (int, 3), (float, 0.5), (str, "x")):
      with self.subTest(kind=kind):
        settings = (_Setting("v", kind, _REQUIRED, "V."),)
        self.assertEqual(
            controller_settings.parse(settings, {"v": value}, "Bot"),
            {"v": value},
        )
        with self.assertRaises(_Invalid):
          controller_settings.parse(settings, {}, "Bot")

  def test_the_schema_lists_required_settings_without_a_default(self) -> None:
    self.assertEqual(
        controller_settings.schema(self._SETTINGS),
        {
            "type": "object",
            "properties": {
                "checkpoint": {
                    "type": "string",
                    "description": "The checkpoint.",
                    "minLength": 1,
                },
                "deterministic": {
                    "type": "boolean",
                    "default": True,
                    "description": "Plays its best move.",
                },
                "temperature": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Sampling.",
                    "minimum": 0.0,
                },
            },
            "additionalProperties": False,
            "required": ["checkpoint"],
        },
    )


class ParseTest(absltest.TestCase):

  _SETTINGS = (
      _Setting("n", int, 3, "A count.", minimum=1),
      _Setting("b", bool, False, "A switch."),
  )

  def test_defaults_fill_what_is_left_out(self) -> None:
    self.assertEqual(
        controller_settings.parse(self._SETTINGS, {"n": 5}, "Bot"),
        {"n": 5, "b": False},
    )

  def test_an_unknown_setting_is_refused_by_name(self) -> None:
    with self.assertRaisesRegex(_Invalid, "^Bot takes no setting 'depth'.$"):
      controller_settings.parse(self._SETTINGS, {"depth": 1}, "Bot")

  def test_no_settings_have_an_empty_schema(self) -> None:
    self.assertEqual(controller_settings.schema(()), {})

  def test_the_schema_admits_no_unknown_settings(self) -> None:
    schema = controller_settings.schema(self._SETTINGS)
    self.assertIs(schema["additionalProperties"], False)
    self.assertNotIn("required", schema)  # None is required.
    self.assertEqual(
        schema["properties"],
        {
            "n": {
                "type": "integer",
                "default": 3,
                "description": "A count.",
                "minimum": 1,
            },
            "b": {
                "type": "boolean",
                "default": False,
                "description": "A switch.",
            },
        },
    )


if __name__ == "__main__":
  absltest.main()
