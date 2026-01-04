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

"""Tests the C++ nfg_writer methods exposed to Python."""

from absl.testing import absltest

import pyspiel


class NFGWriterTest(absltest.TestCase):

  def test_rps(self):
    expected_rps_nfg = ("""NFG 1 R "OpenSpiel export of matrix_rps()"
{ "Player 0" "Player 1" } { 3 3 }

0 0
1 -1
-1 1
-1 1
0 0
1 -1
1 -1
-1 1
0 0
""")
    game = pyspiel.load_game("matrix_rps")
    nfg_text = pyspiel.game_to_nfg_string(game)
    self.assertEqual(nfg_text, expected_rps_nfg)

  def test_pd(self):
    expected_pd_nfg = ("""NFG 1 R "OpenSpiel export of matrix_pd()"
{ "Player 0" "Player 1" } { 2 2 }

5 5
10 0
0 10
1 1
""")
    game = pyspiel.load_game("matrix_pd")
    nfg_text = pyspiel.game_to_nfg_string(game)
    self.assertEqual(nfg_text, expected_pd_nfg)

  def test_mp3p(self):
    expected_mp3p_nfg = ("""NFG 1 R "OpenSpiel export of matching_pennies_3p()"
{ "Player 0" "Player 1" "Player 2" } { 2 2 2 }

1 1 -1
-1 1 1
-1 -1 -1
1 -1 1
1 -1 1
-1 -1 -1
-1 1 1
1 1 -1
""")
    game = pyspiel.load_game("matching_pennies_3p")
    nfg_text = pyspiel.game_to_nfg_string(game)
    self.assertEqual(nfg_text, expected_mp3p_nfg)


if __name__ == "__main__":
  absltest.main()
