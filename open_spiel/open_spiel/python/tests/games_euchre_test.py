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

"""Tests for the game-specific functions for euchre."""


from absl.testing import absltest

import pyspiel
euchre = pyspiel.euchre


class GamesEuchreTest(absltest.TestCase):

  def test_bindings(self):
    self.assertEqual(euchre.JACK_RANK, 2)
    self.assertEqual(euchre.NUM_SUITS, 4)
    self.assertEqual(euchre.NUM_CARDS_PER_SUIT, 6)
    self.assertEqual(euchre.NUM_CARDS, 24)
    self.assertEqual(euchre.PASS_ACTION, 24)
    self.assertEqual(euchre.CLUBS_TRUMP_ACTION, 25)
    self.assertEqual(euchre.DIAMONDS_TRUMP_ACTION, 26)
    self.assertEqual(euchre.HEARTS_TRUMP_ACTION, 27)
    self.assertEqual(euchre.SPADES_TRUMP_ACTION, 28)
    self.assertEqual(euchre.GO_ALONE_ACTION, 29)
    self.assertEqual(euchre.PLAY_WITH_PARTNER_ACTION, 30)
    self.assertEqual(euchre.MAX_BIDS, 8)
    self.assertEqual(euchre.NUM_TRICKS, 5)
    self.assertEqual(euchre.FULL_HAND_SIZE, 5)
    game = pyspiel.load_game('euchre')
    state = game.new_initial_state()
    self.assertEqual(state.num_cards_dealt(), 0)
    self.assertEqual(state.num_cards_played(), 0)
    self.assertEqual(state.num_passes(), 0)
    self.assertEqual(state.upcard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.discard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.trump_suit(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.left_bower(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.right_bower(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.declarer(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.declarer_partner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.first_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.second_defender(), pyspiel.PlayerId.INVALID)
    self.assertIsNone(state.declarer_go_alone())
    self.assertEqual(state.lone_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.active_players(), [True, True, True, True])
    self.assertEqual(state.dealer(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.current_phase(), euchre.Phase.DEALER_SELECTION)
    self.assertEqual(state.current_trick_index(), 0)
    self.assertEqual(state.card_holder(), [None] * 24)
    self.assertEqual(euchre.card_rank(8), euchre.JACK_RANK)
    self.assertEqual(euchre.card_rank(8, euchre.Suit.CLUBS), 100)
    self.assertEqual(euchre.card_suit(8), euchre.Suit.CLUBS)
    self.assertEqual(euchre.card_suit(8, euchre.Suit.SPADES),
                     euchre.Suit.SPADES)
    self.assertEqual(euchre.card_string(8), 'CJ')
    trick = state.tricks()[state.current_trick_index()]
    self.assertEqual(trick.winning_card(), pyspiel.INVALID_ACTION)
    self.assertEqual(trick.led_suit(), euchre.Suit.INVALID_SUIT)
    self.assertEqual(trick.trump_suit(), euchre.Suit.INVALID_SUIT)
    self.assertFalse(trick.trump_played())
    self.assertEqual(trick.leader(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.winner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.cards(), [pyspiel.INVALID_ACTION])
    trick = state.current_trick()
    self.assertEqual(trick.led_suit(), euchre.Suit.INVALID_SUIT)
    self.assertEqual(trick.trump_suit(), euchre.Suit.INVALID_SUIT)
    self.assertFalse(trick.trump_played())
    self.assertEqual(trick.leader(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.winner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.cards(), [pyspiel.INVALID_ACTION])


if __name__ == '__main__':
  absltest.main()
