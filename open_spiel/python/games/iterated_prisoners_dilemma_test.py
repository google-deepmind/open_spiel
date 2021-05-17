# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for iterated_prisoners_dilemma.py."""

from absl.testing import absltest

from open_spiel.python.games import iterated_prisoners_dilemma  # pylint: disable=unused-import
import pyspiel


class IteratedPrisonersDilemmaTest(absltest.TestCase):

  def test_game_as_turn_based(self):
    """Check the game can be converted to a turn-based game."""
    game = pyspiel.load_game("python_iterated_prisoners_dilemma")
    turn_based = pyspiel.convert_to_turn_based(game)
    pyspiel.random_sim_test(
        turn_based, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based_via_string(self):
    """Check the game can be created as a turn-based game from a string."""
    game = pyspiel.load_game(
        "turn_based_simultaneous_game(game=python_iterated_prisoners_dilemma())"
    )
    pyspiel.random_sim_test(
        game, num_sims=10, serialize=False, verbose=True)

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_iterated_prisoners_dilemma")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)


if __name__ == "__main__":
  absltest.main()
