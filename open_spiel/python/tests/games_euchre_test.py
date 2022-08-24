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


class GamesEuchreTest(absltest.TestCase):

  def test_bindings(self):
    game = pyspiel.load_game('euchre')
    self.assertEqual(game.max_bids(), 8)
    self.assertEqual(game.num_cards(), 24)
    state = game.new_initial_state()
    self.assertEqual(state.num_cards_dealt(), 0)
    self.assertEqual(state.num_cards_played(), 0)
    self.assertEqual(state.num_passes(), 0)
    self.assertEqual(state.upcard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.discard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.trump_suit(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.left_bower(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.declarer(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.first_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.declarer_partner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.second_defender(), pyspiel.PlayerId.INVALID)
    self.assertIsNone(state.declarer_go_alone())
    self.assertEqual(state.lone_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.active_players(), [True, True, True, True])
    self.assertEqual(state.dealer(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.current_phase(), state.Phase.DEALER_SELECTION)
    self.assertEqual(state.card_holder(), [None] * 24)
    self.assertEqual(state.card_rank(3), 0)
    self.assertEqual(state.card_rank(4), 1)
    self.assertEqual(state.card_string(0), 'C9')
    self.assertEqual(state.card_string(23), 'SA')
    self.assertEqual(state.card_suit(0), state.Suit.CLUBS)
    self.assertEqual(state.card_suit(23), state.Suit.SPADES)
    self.assertEqual(state.current_trick(), 0)

    trick = state.tricks()[0]
    self.assertEqual(trick.leader(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.winner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.led_suit(), state.Suit.INVALID_SUIT)
    self.assertEqual(trick.cards(), [-1])


if __name__ == '__main__':
  absltest.main()
