# Copyright 2019 DeepMind Technologies Limited
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

"""Tests the C++ nfg_game methods exposed to Python."""

from absl.testing import absltest

import pyspiel


class NFGGameTest(absltest.TestCase):

  def test_pd(self):
    pd_nfg_string = ("""NFG 1 R "OpenSpiel export of matrix_pd()"
{ "Player 0" "Player 1" } { 2 2 }

5 5
10 0
0 10
1 1
""")
    game = pyspiel.load_nfg_game(pd_nfg_string)
    # First (row) player utilities (player, row, col)
    self.assertEqual(game.player_utility(0, 0, 0), 5)
    self.assertEqual(game.player_utility(0, 1, 0), 10)
    self.assertEqual(game.player_utility(0, 0, 1), 0)
    self.assertEqual(game.player_utility(0, 1, 1), 1)
    # Now, second (column) player
    self.assertEqual(game.player_utility(1, 0, 0), 5)
    self.assertEqual(game.player_utility(1, 1, 0), 0)
    self.assertEqual(game.player_utility(1, 0, 1), 10)
    self.assertEqual(game.player_utility(1, 1, 1), 1)

  def test_native_export_import(self):
    """Check that we can import games that we've exported.

    We do not do any additional checking here, as these methods are already
    being extensively tested in nfg_test.cc. The purpose of this test is only
    to check that the python wrapping works.
    """
    game_strings = [
        "matrix_rps", "matrix_shapleys_game", "matrix_pd", "matrix_sh",
        "blotto(players=2,coins=5,fields=3)",
        "blotto(players=3,coins=5,fields=3)"
    ]
    for game_string in game_strings:
      game = pyspiel.load_game(game_string)
      nfg_text = pyspiel.game_to_nfg_string(game)
      nfg_game = pyspiel.load_nfg_game(nfg_text)
      self.assertIsNotNone(nfg_game)


if __name__ == "__main__":
  absltest.main()
