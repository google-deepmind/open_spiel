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

"""Tests for the game-specific functions for chess."""


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel

blackjack = pyspiel.blackjack

NUM_SIM_GAMES = 10
SEED = 87375711


class GamesBlackjackTest(parameterized.TestCase):

  def test_blackjack_game_sim(self):
    game = pyspiel.load_game("blackjack")
    for _ in range(NUM_SIM_GAMES):
      print("----------------")
      print("New game")
      print("----------------")
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = np.random.choice(action_list, p=prob_list)
          print("Chance samples", state.action_to_string(action))
        else:
          print("Player turn")
          print("My cards:", blackjack.cards_to_strings(state.cards(0)))
          print("My best total:", state.get_best_player_total(0))
          print(
              "Dealer's cards:",
              blackjack.cards_to_strings(state.cards(state.dealer_id())),
          )
          print(
              "Dealer's best total:",
              state.get_best_player_total(state.dealer_id()),
          )
          actions = state.legal_actions()
          action = np.random.choice(actions)
          print("Action chosen:", state.action_to_string(action))
        state.apply_action(action)
        print("")
      print("Terminal state: ")
      print(str(state))
      print("Returns:", state.returns())
      print("")

  def test_card_to_string_conversion(self):
    for i in range(52):
      self.assertEqual(i, blackjack.get_card_by_string(
          blackjack.card_to_string(i)))

  def test_blackjack_three_aces(self):
    game = pyspiel.load_game("blackjack")
    state = game.new_initial_state()
    self.assertTrue(state.is_chance_node())
    # Player's cards
    state.apply_action(blackjack.get_card_by_string("D6"))
    state.apply_action(blackjack.get_card_by_string("DA"))
    # Dealer's cards
    state.apply_action(blackjack.get_card_by_string("CQ"))
    state.apply_action(blackjack.get_card_by_string("C3"))
    # Play starts.
    self.assertFalse(state.is_chance_node())
    self.assertListEqual(blackjack.cards_to_strings(state.cards(0)),
                         ["D6", "DA"])
    self.assertEqual(state.get_best_player_total(0), 17)
    state.apply_action(blackjack.HIT)
    state.apply_action(blackjack.get_card_by_string("SA"))
    self.assertEqual(state.get_best_player_total(0), 18)
    self.assertListEqual(blackjack.cards_to_strings(state.cards(0)),
                         ["D6", "DA", "SA"])
    state.apply_action(blackjack.HIT)
    state.apply_action(blackjack.get_card_by_string("CA"))
    self.assertEqual(state.get_best_player_total(0), 19)
    self.assertListEqual(blackjack.cards_to_strings(state.cards(0)),
                         ["D6", "DA", "SA", "CA"])
    state.apply_action(blackjack.HIT)
    state.apply_action(blackjack.get_card_by_string("C2"))
    self.assertEqual(state.get_best_player_total(0), 21)
    state.apply_action(blackjack.STAND)
    self.assertListEqual(blackjack.cards_to_strings(state.cards(0)),
                         ["D6", "DA", "SA", "CA", "C2"])
    # Dealer's turn.
    # Dealer has a 13, must hit.
    self.assertTrue(state.is_chance_node())
    self.assertListEqual(
        blackjack.cards_to_strings(state.cards(state.dealer_id())),
        ["CQ", "C3"])
    state.apply_action(blackjack.get_card_by_string("HA"))
    self.assertEqual(state.get_best_player_total(1), 14)
    self.assertTrue(state.is_chance_node())
    self.assertListEqual(
        blackjack.cards_to_strings(state.cards(state.dealer_id())),
        ["CQ", "C3", "HA"])
    state.apply_action(blackjack.get_card_by_string("S3"))
    self.assertEqual(state.get_best_player_total(1), 17)
    self.assertListEqual(
        blackjack.cards_to_strings(state.cards(state.dealer_id())),
        ["CQ", "C3", "HA", "S3"])
    # Dealer must stop on 17. This should be a terminal state.
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0])


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
