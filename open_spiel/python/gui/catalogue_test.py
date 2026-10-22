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

"""Tests for the catalogue's invariants."""

import dataclasses

from absl.testing import absltest

from open_spiel.python.gui import catalogue

_BOARD_SIZE = catalogue.ParameterSpec(
    name="board_size",
    type=catalogue.ParameterType.INT,
    kind=catalogue.ParameterKind.RANGE,
    default=19,
    minimum=2,
    maximum=25,
    choices=None,
    description=None,
)
_KOMI = catalogue.ParameterSpec(
    name="komi",
    type=catalogue.ParameterType.FLOAT,
    kind=catalogue.ParameterKind.FREE,
    default=7.5,
    minimum=None,
    maximum=None,
    choices=None,
    description="Points given to white.",
)
_GO = catalogue.CatalogueEntry(
    name="go",
    long_name="Go",
    dynamics=catalogue.Dynamics.SEQUENTIAL,
    chance_mode=catalogue.ChanceMode.DETERMINISTIC,
    information=catalogue.Information.PERFECT_INFORMATION,
    min_seats=2,
    max_seats=2,
    num_players=2,
    parameters=(_BOARD_SIZE, _KOMI),
    loadable=True,
    unavailable_reason=None,
    controllers=("human", "random", "mcts"),
)


class ParameterSpecTest(absltest.TestCase):

  def test_the_default_has_the_parameter_type(self) -> None:
    with self.assertRaisesRegex(ValueError, "defaults to"):
      dataclasses.replace(_KOMI, default="7.5")
    with self.assertRaisesRegex(ValueError, "defaults to"):
      dataclasses.replace(_BOARD_SIZE, default=True)

  def test_a_range_has_bounds_around_its_default(self) -> None:
    with self.assertRaisesRegex(ValueError, "bounds"):
      dataclasses.replace(_BOARD_SIZE, maximum=None)
    with self.assertRaisesRegex(ValueError, "bounds"):
      dataclasses.replace(_KOMI, minimum=0.0)
    with self.assertRaisesRegex(ValueError, "around its default"):
      dataclasses.replace(_BOARD_SIZE, maximum=9)

  def test_a_choice_lists_its_default(self) -> None:
    rules = catalogue.ParameterSpec(
        name="rules",
        type=catalogue.ParameterType.STRING,
        kind=catalogue.ParameterKind.CHOICE,
        default="japanese",
        minimum=None,
        maximum=None,
        choices=("chinese", "japanese"),
        description=None,
    )
    with self.assertRaisesRegex(ValueError, "among"):
      dataclasses.replace(rules, default="korean")
    with self.assertRaisesRegex(ValueError, "choices exactly"):
      dataclasses.replace(rules, choices=None)


class CatalogueEntryTest(absltest.TestCase):

  def test_the_seat_range_is_ordered(self) -> None:
    with self.assertRaisesRegex(ValueError, "seats"):
      dataclasses.replace(_GO, min_seats=3)

  def test_parameters_are_unique_and_ordered_by_name(self) -> None:
    with self.assertRaisesRegex(ValueError, "parameters"):
      dataclasses.replace(_GO, parameters=(_KOMI, _BOARD_SIZE))

  def test_an_unloadable_game_gives_a_reason_and_no_controllers(self) -> None:
    dataclasses.replace(
        _GO,
        num_players=None,
        loadable=False,
        unavailable_reason="No replay.",
        controllers=(),
    )
    with self.assertRaisesRegex(ValueError, "reason"):
      dataclasses.replace(_GO, num_players=None, loadable=False, controllers=())
    with self.assertRaisesRegex(ValueError, "controllers"):
      dataclasses.replace(
          _GO,
          num_players=None,
          loadable=False,
          unavailable_reason="No replay.",
          controllers=("random",),
      )

  def test_the_default_seat_count_is_given_exactly_when_loadable(self) -> None:
    with self.assertRaisesRegex(ValueError, "default seat count"):
      dataclasses.replace(_GO, num_players=None)
    with self.assertRaisesRegex(ValueError, "default seat count"):
      dataclasses.replace(
          _GO,
          loadable=False,
          unavailable_reason="No replay.",
          controllers=(),
      )

  def test_the_default_seat_count_is_within_the_seat_range(self) -> None:
    with self.assertRaisesRegex(ValueError, "default seats"):
      dataclasses.replace(_GO, num_players=3)

  def test_a_loadable_game_offers_the_human_controller(self) -> None:
    with self.assertRaisesRegex(ValueError, "human"):
      dataclasses.replace(_GO, controllers=("random",))


class CatalogueTest(absltest.TestCase):

  def test_games_are_unique_and_ordered_by_name(self) -> None:
    chess = dataclasses.replace(_GO, name="chess")
    catalogue.Catalogue(games=(chess, _GO))
    for games in ((_GO, chess), (_GO, _GO)):
      with self.assertRaisesRegex(ValueError, "ordered by name"):
        catalogue.Catalogue(games=games)


if __name__ == "__main__":
  absltest.main()
