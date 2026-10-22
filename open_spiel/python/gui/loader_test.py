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


"""Tests for loading games and listing which ones the server serves."""

import re
from unittest import mock

from absl.testing import absltest

from open_spiel.python.gui import catalogue
from open_spiel.python.gui import contract
from open_spiel.python.gui import loader


class LoadTest(absltest.TestCase):

  def test_loads_a_game_with_its_default_parameters(self) -> None:
    loaded = loader.load("tic_tac_toe", {})
    self.assertEqual(loaded.num_seats, 2)
    self.assertFalse(loaded.turn_based)
    self.assertFalse(loaded.hides_information)

  def test_reports_parameters_with_their_defaults(self) -> None:
    loaded = loader.load("kuhn_poker", {"players": 3})
    self.assertEqual(loaded.parameters, {"players": 3})
    self.assertEqual(loaded.num_seats, 3)

  def test_widens_an_integer_given_for_a_real_parameter(self) -> None:
    komi = loader.load("go", {"komi": 7}).parameters["komi"]
    self.assertIsInstance(komi, float)
    self.assertEqual(komi, 7.0)

  def test_wraps_a_simultaneous_game_to_be_turn_based(self) -> None:
    loaded = loader.load("matrix_rps", {})
    self.assertTrue(loaded.turn_based)
    self.assertTrue(loaded.hides_information)
    self.assertEqual(loaded.parameters, {})

  def test_treats_imperfect_information_as_hidden(self) -> None:
    loaded = loader.load("kuhn_poker", {})
    self.assertTrue(loaded.hides_information)
    self.assertTrue(loaded.has_seat_views)

  def test_refuses_an_unknown_game(self) -> None:
    with self.assertRaises(loader.UnknownGameError):
      loader.load("no_such_game", {})

  def test_refuses_parameters_the_game_refuses(self) -> None:
    with self.assertRaises(loader.InvalidParametersError):
      loader.load("tic_tac_toe", {"no_such_parameter": 1})

  def test_a_refusal_logs_what_the_game_raised_and_sends_none_of_it(
      self,
  ) -> None:
    with self.assertLogs(level="INFO") as logs:
      with self.assertRaises(loader.InvalidParametersError) as raised:
        loader.load("tic_tac_toe", {"no_such_parameter": 1})
    self.assertEqual(
        str(raised.exception),
        "tic_tac_toe refuses these parameters; the server's log has the "
        "details.",
    )
    self.assertIn("no_such_parameter", "\n".join(logs.output))

  def test_a_refusal_at_the_start_sends_none_of_what_the_game_raised(
      self,
  ) -> None:
    with self.assertLogs(level="INFO"):
      with self.assertRaises(loader.InvalidParametersError) as raised:
        loader.load("go", {"board_size": 0})
    self.assertEqual(
        str(raised.exception),
        "go cannot start with these parameters; the server's log has the "
        "details.",
    )

  def test_refuses_parameters_the_game_rejects_only_when_it_starts(
      self,
  ) -> None:
    with self.assertRaises(loader.InvalidParametersError):
      loader.load("go", {"board_size": 0})

  def test_refuses_mean_field_games(self) -> None:
    with self.assertRaisesRegex(loader.UnavailableError, "population"):
      loader.load("mfg_crowd_modelling", {})

  def test_refuses_games_that_cannot_be_rebuilt_from_history(self) -> None:
    with self.assertRaisesRegex(loader.UnavailableError, "rebuilt"):
      loader.load("negotiation", {})


class ServerPathParametersTest(absltest.TestCase):

  def test_refuses_a_file_other_than_the_default(self) -> None:
    refused = set()
    for name, parameter in sorted(loader.SERVER_PATH_PARAMETERS):
      if name not in loader.registered_types():
        continue
      refused.add(name)
      with self.subTest(f"{name}.{parameter}"):
        with self.assertRaisesRegex(
            loader.InvalidParametersError, "names a file on the server"
        ) as raised:
          loader.load(name, {parameter: "/etc/hostname"})
        self.assertNotIn("/etc/hostname", str(raised.exception))
    # Games every build registers, so the loop is never empty.
    self.assertEqual({"efg_game", "nfg_game"} - refused, set())

  def test_admits_the_default_given_explicitly(self) -> None:
    loaded = loader.load("python_hangman", {"word_list_file": ""})
    self.assertEqual(loaded.parameters["word_list_file"], "")

  def test_lists_every_registered_parameter_that_looks_like_a_path(
      self,
  ) -> None:
    looks_like_a_path = re.compile(r"file|path|dir|root", re.IGNORECASE)
    found = set()
    for name, game_type in loader.registered_types().items():
      for parameter, default in game_type.parameter_specification.items():
        if isinstance(default, str) and looks_like_a_path.search(parameter):
          found.add((name, parameter))
    # The scan finds the parameters every build registers, so it is not empty.
    self.assertIn(("efg_game", "filename"), found)
    self.assertIn(("nfg_game", "filename"), found)
    self.assertEqual(found - loader.SERVER_PATH_PARAMETERS, set())

  def test_no_game_with_a_nested_game_parameter_is_served(self) -> None:
    # The table names top-level parameters only; a wrapper could pass a path
    # inside its nested game. Such games are unavailable, which keeps them out.
    for name, game_type in loader.registered_types().items():
      if any(
          not isinstance(default, (bool, int, float, str))
          for default in game_type.parameter_specification.values()
      ):
        with self.subTest(name):
          self.assertIsNotNone(loader.unavailable_reason(game_type))

  def test_every_listed_parameter_is_a_string_its_game_declares(self) -> None:
    for name, parameter in sorted(loader.SERVER_PATH_PARAMETERS):
      game_type = loader.registered_types().get(name)
      if game_type is None:
        continue
      with self.subTest(f"{name}.{parameter}"):
        self.assertIsInstance(
            game_type.parameter_specification.get(parameter), str
        )

  def test_the_catalogue_offers_only_the_default(self) -> None:
    entry = next(
        entry
        for entry in loader.build_catalogue().games
        if entry.name == "python_hangman"
    )
    spec = {spec.name: spec for spec in entry.parameters}["word_list_file"]
    self.assertEqual(spec.kind, catalogue.ParameterKind.CHOICE)
    self.assertEqual(spec.choices, ("",))


class StructSupportTest(absltest.TestCase):
  """Struct support is probed lazily, once per game and parameters."""

  def setUp(self) -> None:
    super().setUp()
    loader._STRUCT_SUPPORT.clear()  # pylint: disable=protected-access
    self.probe = self.enter_context(
        mock.patch.object(
            loader,
            "_probe_struct",
            wraps=loader._probe_struct,  # pylint: disable=protected-access
        )
    )

  def test_reports_whether_the_game_provides_a_struct(self) -> None:
    self.assertTrue(loader.load("tic_tac_toe", {}).provides_struct)
    self.assertFalse(loader.load("kuhn_poker", {}).provides_struct)

  def test_loading_does_not_probe(self) -> None:
    loader.load("kuhn_poker", {})
    loader.load("matrix_rps", {})
    self.probe.assert_not_called()

  def test_probes_once_per_game_across_loads_and_frames(self) -> None:
    first = loader.load("kuhn_poker", {})
    second = loader.load("kuhn_poker", {})
    for _ in range(5):
      self.assertFalse(first.provides_struct)
      self.assertFalse(second.provides_struct)
    self.probe.assert_called_once()

  def test_probes_each_set_of_parameters_separately(self) -> None:
    self.assertFalse(loader.load("kuhn_poker", {}).provides_struct)
    self.assertFalse(loader.load("kuhn_poker", {"players": 3}).provides_struct)
    self.assertEqual(self.probe.call_count, 2)


class CatalogueTest(absltest.TestCase):

  entries: dict[str, catalogue.CatalogueEntry]

  @classmethod
  def setUpClass(cls) -> None:
    super().setUpClass()
    cls.entries = {
        entry.name: entry for entry in loader.build_catalogue().games
    }

  def test_lists_every_registered_game(self) -> None:
    self.assertEqual(set(self.entries), set(loader.registered_types()))

  def test_a_loadable_game_offers_the_human_controller(self) -> None:
    entry = self.entries["tic_tac_toe"]
    self.assertTrue(entry.loadable)
    self.assertIn(contract.HUMAN, entry.controllers)

  def test_an_unservable_game_says_why(self) -> None:
    entry = self.entries["mfg_crowd_modelling"]
    self.assertFalse(entry.loadable)
    self.assertEqual(
        entry.unavailable_reason,
        "Mean-field games need a population distribution to play.",
    )
    self.assertEqual(entry.controllers, ())

  def test_a_game_needing_parameters_is_not_loadable(self) -> None:
    entry = self.entries["efg_game"]
    self.assertFalse(entry.loadable)
    self.assertEqual(
        entry.unavailable_reason, "The game needs parameters to load."
    )

  def test_gives_the_default_seat_count(self) -> None:
    self.assertEqual(self.entries["tic_tac_toe"].num_players, 2)
    routing = self.entries["python_dynamic_routing"]
    self.assertTrue(routing.loadable)
    self.assertEqual(routing.num_players, 5)
    self.assertEqual(routing.min_seats, 1)

  def test_an_unservable_game_has_no_default_seat_count(self) -> None:
    self.assertIsNone(self.entries["mfg_crowd_modelling"].num_players)

  def test_lists_a_game_served_as_hiding_information_as_imperfect(
      self,
  ) -> None:
    imperfect = catalogue.Information.IMPERFECT_INFORMATION
    # Each declares perfect information: hangman keeps its word, and
    # oshi_zumo's moves are simultaneous.
    for name in ("python_hangman", "oshi_zumo"):
      with self.subTest(name):
        self.assertEqual(self.entries[name].information, imperfect)
    self.assertEqual(
        self.entries["tic_tac_toe"].information,
        catalogue.Information.PERFECT_INFORMATION,
    )

  def test_describes_each_parameter(self) -> None:
    komi = {spec.name: spec for spec in self.entries["go"].parameters}["komi"]
    self.assertEqual(komi.type, catalogue.ParameterType.FLOAT)
    self.assertEqual(komi.default, 7.5)

  def test_includes_games_written_in_python(self) -> None:
    self.assertTrue(self.entries["python_tic_tac_toe"].loadable)


if __name__ == "__main__":
  absltest.main()
