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


"""Tests for deciding whether a game hides information."""

from absl.testing import absltest

from open_spiel.python.games import hangman  # pylint: disable=unused-import
from open_spiel.python.gui import information
import pyspiel


def _hides(game: pyspiel.Game) -> bool:
  """Whether `game` hides information, from its type."""
  return information.hides_information(game.get_type())


class HidesInformationTest(absltest.TestCase):

  def test_a_game_of_perfect_information_hides_nothing(self) -> None:
    self.assertFalse(_hides(pyspiel.load_game("tic_tac_toe")))

  def test_a_game_of_imperfect_information_hides(self) -> None:
    self.assertTrue(_hides(pyspiel.load_game("kuhn_poker")))

  def test_a_game_that_hides_despite_its_declaration_hides(self) -> None:
    game = pyspiel.load_game("python_hangman")
    self.assertEqual(
        game.get_type().information,
        pyspiel.GameType.Information.PERFECT_INFORMATION,
    )
    self.assertTrue(_hides(game))

  def test_a_simultaneous_game_served_turn_based_hides(self) -> None:
    self.assertTrue(_hides(pyspiel.load_game_as_turn_based("oshi_zumo")))


if __name__ == "__main__":
  absltest.main()
