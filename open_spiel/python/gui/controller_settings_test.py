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

from absl.testing import absltest

from open_spiel.python.gui import controller_settings

_Setting = controller_settings.Setting
_Invalid = controller_settings.InvalidSettingError


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
