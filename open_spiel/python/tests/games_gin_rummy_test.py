# Copyright 2022 DeepMind Technologies Limited
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

"""Tests for the game-specific functions for gin rummy."""


from absl.testing import absltest

import pyspiel


class GamesGinRummyTest(absltest.TestCase):

  def test_bindings(self):
    game = pyspiel.load_game('gin_rummy')
    self.assertFalse(game.oklahoma())
    self.assertEqual(game.knock_card(), 10)
    self.assertEqual(game.draw_upcard_action(), 52)
    self.assertEqual(game.draw_stock_action(), 53)
    self.assertEqual(game.pass_action(), 54)
    self.assertEqual(game.knock_action(), 55)
    self.assertEqual(game.meld_action_base(), 56)
    state = game.new_initial_state()
    self.assertEqual(state.current_phase(), state.Phase.DEAL)
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertIsNone(state.upcard())
    self.assertEqual(state.stock_size(), 52)
    self.assertEqual(state.hands(), [[], []])
    self.assertEqual(state.discard_pile(), [])
    self.assertEqual(state.deadwood(), [0, 0])
    self.assertEqual(state.knocked(), [False, False])
    self.assertEqual(state.pass_on_first_upcard(), [False, False])
    self.assertEqual(state.layed_melds(), [[], []])
    self.assertEqual(state.layoffs(), [])
    utils = pyspiel.GinRummyUtils(13, 4, 10)  # 13 ranks, 4 suits, 10 hand size
    self.assertEqual(utils.card_string(0), 'As')
    self.assertEqual(utils.hand_to_string([0, 1, 2]),
                     '+--------------------------+\n'
                     '|As2s3s                    |\n'
                     '|                          |\n'
                     '|                          |\n'
                     '|                          |\n'
                     '+--------------------------+\n')
    self.assertEqual(utils.card_int('As'), 0)
    self.assertEqual(utils.card_ints_to_card_strings([0, 1, 2]),
                     ['As', '2s', '3s'])
    self.assertEqual(utils.card_strings_to_card_ints(['As', '2s', '3s']),
                     [0, 1, 2])
    self.assertEqual(utils.card_value(0), 1)
    self.assertEqual(utils.total_card_value([50, 51]), 20)
    self.assertEqual(utils.total_card_value([[0, 1], [50, 51]]), 23)
    self.assertEqual(utils.card_rank(51), 12)
    self.assertEqual(utils.card_suit(51), 3)
    self.assertTrue(utils.is_consecutive([0, 1, 2]))
    self.assertTrue(utils.is_rank_meld([0, 13, 26]))
    self.assertTrue(utils.is_suit_meld([0, 1, 2]))
    self.assertEqual(utils.rank_melds([0, 1, 13, 26]), [[0, 13, 26]])
    self.assertEqual(utils.suit_melds([0, 5, 6, 7]), [[5, 6, 7]])
    self.assertEqual(utils.all_melds([0, 5, 6, 7, 13, 26]),
                     [[0, 13, 26], [5, 6, 7]])
    self.assertEqual(utils.all_meld_groups([0, 5, 6, 7, 13, 26]),
                     [[[0, 13, 26], [5, 6, 7]], [[5, 6, 7], [0, 13, 26]]])
    self.assertEqual(utils.best_meld_group([0, 5, 6, 7, 13, 26]),
                     [[0, 13, 26], [5, 6, 7]])
    self.assertEqual(utils.min_deadwood([0, 1, 2], 3), 0)
    self.assertEqual(utils.min_deadwood([0, 1, 2]), 0)
    self.assertEqual(utils.rank_meld_layoff([0, 13, 26]), 39)
    self.assertEqual(utils.suit_meld_layoffs([0, 1, 2]), [3])
    self.assertEqual(utils.legal_melds([0, 1, 2, 3], 10), [65, 66, 109])
    self.assertEqual(utils.legal_discards([0, 1, 2], 10), [0, 1, 2])
    self.assertEqual(utils.all_layoffs([65], [3]), [4])
    self.assertEqual(utils.meld_to_int([0, 1, 2]), 65)
    self.assertEqual(utils.int_to_meld[65], [0, 1, 2])


if __name__ == '__main__':
  absltest.main()
