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

"""Tests for Game::MaxGameStringLength."""

import unittest
import pyspiel

class MaxGameStringLengthTest(unittest.TestCase):

  def test_kuhn_poker(self):
    # 2 players: "0 1 pbp" -> 7 chars
    game = pyspiel.load_game("kuhn_poker", {"players": 2})
    self.assertEqual(game.max_game_string_length(), 7)

    # 3 players: "0 1 2 pbppb" -> 3 + 2 (spaces) + 5 (bets) + 1 (space) = 11?
    # Logic in cc:
    # int card_len = std::to_string(num_players_).length(); // 1 for n=3
    # int cards_str_len = num_players_ * card_len + (num_players_ - 1); // 3*1 + 2 = 5 ("0 1 2")
    # int betting_str_len = MaxGameLength(); // 3*2 - 1 = 5
    # total = cards_str_len + 1 + betting_str_len = 5 + 1 + 5 = 11.
    game3 = pyspiel.load_game("kuhn_poker", {"players": 3})
    self.assertEqual(game3.max_game_string_length(), 11)

  def test_unimplemented_game(self):
    # Default implementation returns 0.
    game = pyspiel.load_game("tic_tac_toe")
    self.assertEqual(game.max_game_string_length(), 0)

if __name__ == "__main__":
  unittest.main()
