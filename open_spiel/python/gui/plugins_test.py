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


"""Tests for the controller plugin loader."""

import importlib.metadata
from unittest import mock

from absl.testing import absltest

import pyspiel

from open_spiel.python.gui import cli
from open_spiel.python.gui import controllers
from open_spiel.python.gui import fake_plugin
from open_spiel.python.gui import loader
from open_spiel.python.gui import plugins

_PLUGIN = "open_spiel.python.gui.fake_plugin"


def _entry_point(name: str, value: str) -> importlib.metadata.EntryPoint:
  """Returns an entry point in the plugin group, as a distribution has."""
  return importlib.metadata.EntryPoint(
      name=name, value=value, group=plugins.ENTRY_POINT_GROUP
  )


class PluginsTestCase(absltest.TestCase):
  """Unregisters whatever a test's plugins registered."""

  def setUp(self) -> None:
    super().setUp()
    before = set(controllers.kinds())

    def unregister_added() -> None:
      for name in set(controllers.kinds()) - before:
        controllers.unregister(name)

    self.addCleanup(unregister_added)

  def entry_points(self, *points: importlib.metadata.EntryPoint) -> mock.Mock:
    """Makes `points` the installed entry points for the test."""
    patcher = mock.patch.object(
        importlib.metadata, "entry_points", return_value=list(points)
    )
    self.addCleanup(patcher.stop)
    return patcher.start()


class LoadTest(PluginsTestCase):

  def test_a_plugin_module_registers_a_controller(self) -> None:
    self.assertEqual(plugins.load([_PLUGIN]), (fake_plugin.NAME,))
    self.assertIn(fake_plugin.NAME, controllers.kinds())

  def test_a_plugins_controller_is_in_the_catalogue(self) -> None:
    plugins.load([_PLUGIN])
    game_type = pyspiel.load_game("tic_tac_toe").get_type()
    # pylint: disable-next=protected-access
    entry = loader._catalogue_entry(game_type)
    self.assertIn(fake_plugin.NAME, entry.controllers)

  def test_a_plugins_controller_plays(self) -> None:
    plugins.load([_PLUGIN])
    game = pyspiel.load_game("tic_tac_toe")
    controller = controllers.create(
        controllers.Spec(fake_plugin.NAME), game, seed=0
    )
    state = game.new_initial_state()
    self.assertEqual(controllers.choose(controller, state, 0), 0)

  def test_no_names_load_nothing(self) -> None:
    entry_points = self.entry_points(_entry_point("fake", _PLUGIN))
    self.assertEqual(plugins.load([]), ())
    entry_points.assert_not_called()

  def test_an_unknown_module_fails_loud(self) -> None:
    with self.assertRaisesRegex(
        plugins.PluginError,
        r"'open_spiel\.python\.gui\.no_such_plugin' cannot be imported: "
        "ModuleNotFoundError",
    ) as raised:
      plugins.load(["open_spiel.python.gui.no_such_plugin"])
    self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)

  def test_a_module_that_fails_to_import_fails_loud(self) -> None:
    with mock.patch(
        "importlib.import_module", side_effect=RuntimeError("no GPU")
    ):
      with self.assertRaisesRegex(
          plugins.PluginError,
          "'broken' cannot be imported: RuntimeError: no GPU",
      ):
        plugins.load(["broken"])

  def test_a_module_without_register_into_fails_loud(self) -> None:
    with self.assertRaisesRegex(
        plugins.PluginError,
        r"module 'open_spiel\.python\.gui\.errors' has no function "
        r"register_into\(controllers\)",
    ):
      plugins.load(["open_spiel.python.gui.errors"])

  def test_a_name_registered_twice_fails_loud(self) -> None:
    plugins.load([_PLUGIN])
    with self.assertRaisesRegex(
        plugins.PluginError,
        "failed to register: ValueError: A controller is already registered "
        "as 'first_legal'",
    ):
      plugins.load([_PLUGIN])

  def test_a_builtin_name_fails_loud(self) -> None:
    with mock.patch.object(fake_plugin, "NAME", controllers.RANDOM):
      with self.assertRaisesRegex(plugins.PluginError, "already registered"):
        plugins.load([_PLUGIN])


class EntryPointTest(PluginsTestCase):

  def test_auto_loads_an_entry_point_naming_a_module(self) -> None:
    entry_points = self.entry_points(_entry_point("fake", _PLUGIN))
    self.assertEqual(plugins.load([plugins.AUTO]), (fake_plugin.NAME,))
    entry_points.assert_called_once_with(group=plugins.ENTRY_POINT_GROUP)

  def test_auto_loads_an_entry_point_naming_register_into(self) -> None:
    self.entry_points(_entry_point("fake", f"{_PLUGIN}:register_into"))
    self.assertEqual(plugins.load([plugins.AUTO]), (fake_plugin.NAME,))

  def test_auto_with_no_entry_points_loads_nothing(self) -> None:
    self.entry_points()
    with self.assertLogs(level="WARNING"):
      self.assertEqual(plugins.load([plugins.AUTO]), ())

  def test_an_entry_point_that_fails_to_load_fails_loud(self) -> None:
    self.entry_points(_entry_point("gone", "no_such_package.plugin"))
    with self.assertRaisesRegex(
        plugins.PluginError,
        r"entry point 'gone' \(no_such_package\.plugin\) cannot be loaded: "
        "ModuleNotFoundError",
    ):
      plugins.load([plugins.AUTO])

  def test_an_entry_point_naming_neither_fails_loud(self) -> None:
    self.entry_points(_entry_point("name", f"{_PLUGIN}:NAME"))
    with self.assertRaisesRegex(
        plugins.PluginError, "neither a module nor a callable register_into"
    ):
      plugins.load([plugins.AUTO])


class CommandLineTest(PluginsTestCase):

  def test_no_flag_loads_no_plugin(self) -> None:
    self.assertEqual(cli.parse_args([]).controllers, [])

  def test_names_are_comma_separated_and_the_flag_repeats(self) -> None:
    args = cli.parse_args(["--controllers", "a, b", "--controllers=auto"])
    with mock.patch.object(plugins, "load", return_value=()) as load:
      cli.load_plugins(args.controllers)
    load.assert_called_once_with(["a", "b", "auto"])

  def test_an_empty_name_fails_loud(self) -> None:
    with self.assertRaisesRegex(plugins.PluginError, "empty name"):
      cli.load_plugins(["a,,b"])

  def test_the_flag_loads_the_plugin(self) -> None:
    self.assertEqual(cli.load_plugins([_PLUGIN]), (fake_plugin.NAME,))

  def test_the_move_budget_defaults_to_ten_seconds(self) -> None:
    self.assertEqual(cli.parse_args([]).move_budget_seconds, 10.0)
    self.assertEqual(
        cli.parse_args(["--move-budget-seconds=2.5"]).move_budget_seconds, 2.5
    )

  def test_a_move_budget_that_is_not_positive_is_a_usage_error(self) -> None:
    for value in ("0", "-1", "inf", "soon"):
      with self.subTest(value=value):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
          cli.parse_args([f"--move-budget-seconds={value}"])


if __name__ == "__main__":
  absltest.main()
