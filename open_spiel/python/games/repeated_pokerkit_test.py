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

# Lint as python3
"""Tests for OpenSpiel 'Repeated' PokerkitWrapper."""

import itertools
import json
import logging
import random

from absl.testing import absltest
from absl.testing import parameterized

# Hack to handle an inability to import pokerkit gracefully. Necessary since
# OpenSpiel may support earlier versions of python that pokerkit doesn't.
IMPORTED_ALL_LIBRARIES = False
try:
  # pylint: disable=g-import-not-at-top
  import pokerkit

  from open_spiel.python.games import pokerkit_wrapper
  from open_spiel.python.games import repeated_pokerkit
  import pyspiel
  # pylint: disable=g-import-not-at-top
  IMPORTED_ALL_LIBRARIES = True
except ImportError as e:
  logging.error("Failed to import needed pokerkit libraries: %s", e)
  pokerkit_wrapper = None
  repeated_pokerkit = None
  pokerkit = None
  pyspiel = None

if IMPORTED_ALL_LIBRARIES:
  assert pokerkit is not None
  assert pokerkit_wrapper is not None
  assert repeated_pokerkit is not None
  assert pyspiel is not None

  ACTION_CHECK_OR_CALL = pokerkit_wrapper.ACTION_CHECK_OR_CALL
  ACTION_FOLD = pokerkit_wrapper.ACTION_FOLD

  Card = pokerkit.Card
  ACE = pokerkit.Rank.ACE
  DEUCE = pokerkit.Rank.DEUCE
  KING = pokerkit.Rank.KING
  QUEEN = pokerkit.Rank.QUEEN
  JACK = pokerkit.Rank.JACK
  TEN = pokerkit.Rank.TEN
  NINE = pokerkit.Rank.NINE
  EIGHT = pokerkit.Rank.EIGHT
  SEVEN = pokerkit.Rank.SEVEN
  SIX = pokerkit.Rank.SIX
  FIVE = pokerkit.Rank.FIVE
  FOUR = pokerkit.Rank.FOUR
  TREY = pokerkit.Rank.TREY
  SPADE = pokerkit.Suit.SPADE
  CLUB = pokerkit.Suit.CLUB
  HEART = pokerkit.Suit.HEART
  DIAMOND = pokerkit.Suit.DIAMOND

  # E.g.
  # [
  #   {variant: "NoLimitTexasHoldem"}
  #   {variant: "FixedLimitTexasHoldem"}
  #   ...
  # ]
  _ALL_VARIANTS = [
      {"variant": variant}
      for variant in pokerkit_wrapper.SUPPORTED_VARIANT_MAP.keys()
  ]
  # E.g.
  # [
  #   {variant: "NoLimitTexasHoldem", reset_stacks: True},
  #   {variant: "NoLimitTexasHoldem", reset_stacks: False},
  #   {variant: "FixedLimitTexasHoldem", reset_stacks: True},
  #   {variant: "FixedLimitTexasHoldem", reset_stacks: False},
  #   ...
  # ]
  _ALL_VARIANTS_CASH_AND_TOURNAMENT_PARAMS = [
      params | {"reset_stacks": reset_stacks}
      for params, reset_stacks in itertools.product(
          _ALL_VARIANTS, [True, False]
      )
  ]

  class RepeatedPokerkitTest(parameterized.TestCase):
    """Test the OpenSpiel game 'Repeated Pokerkit'."""

    def _load_repeated_pokerkit_game_with_variant(self, variant, params):
      """Loads a repeated_pokerkit game with variant-specific param filtering."""
      use_bring_in = variant in pokerkit_wrapper.VARIANT_PARAM_USAGE["bring_in"]
      use_blinds = (
          variant
          in pokerkit_wrapper.VARIANT_PARAM_USAGE["raw_blinds_or_straddles"]
      )
      use_fixed_bet_sizes = (
          variant in pokerkit_wrapper.VARIANT_PARAM_USAGE["small_bet"]
      )
      if use_fixed_bet_sizes:
        self.assertIn(variant, pokerkit_wrapper.VARIANT_PARAM_USAGE["big_bet"])

      if not use_bring_in:
        params["pokerkit_game_params"].pop("bring_in")
      if not use_blinds:
        self.assertTrue(use_bring_in and use_fixed_bet_sizes)
        params["pokerkit_game_params"].pop("blinds")
        params["rotate_dealer"] = False
      if not use_fixed_bet_sizes:
        self.assertTrue(use_blinds and not use_bring_in)
        params["pokerkit_game_params"].pop("small_bet")
        params["pokerkit_game_params"].pop("big_bet")

      return pyspiel.load_game("python_repeated_pokerkit", params)

    def test_parse_blind_schedule(self):
      blind_schedule_str = "1:1/2;2:2/3"
      blind_schedule = repeated_pokerkit.parse_blind_schedule(
          blind_schedule_str
      )
      self.assertLen(blind_schedule, 2)
      self.assertEqual(blind_schedule[0].num_hands, 1)
      self.assertEqual(blind_schedule[0].small_blind, 1)
      self.assertEqual(blind_schedule[0].big_blind, 2)
      self.assertEqual(blind_schedule[1].num_hands, 2)
      self.assertEqual(blind_schedule[1].small_blind, 2)
      self.assertEqual(blind_schedule[1].big_blind, 3)

    def test_parse_bet_size_schedule(self):
      bet_size_schedule_str = "1:10/20;2:20/30"
      bet_size_schedule = repeated_pokerkit.parse_bet_size_schedule(
          bet_size_schedule_str
      )
      self.assertLen(bet_size_schedule, 2)
      self.assertEqual(bet_size_schedule[0].num_hands, 1)
      self.assertEqual(bet_size_schedule[0].small_bet_size, 10)
      self.assertEqual(bet_size_schedule[0].big_bet_size, 20)
      self.assertEqual(bet_size_schedule[1].num_hands, 2)
      self.assertEqual(bet_size_schedule[1].small_bet_size, 20)
      self.assertEqual(bet_size_schedule[1].big_bet_size, 30)

    def test_parse_bring_in_schedule(self):
      bring_in_schedule_str = "1:5;2:10"
      bring_in_schedule = repeated_pokerkit.parse_bring_in_schedule(
          bring_in_schedule_str
      )
      self.assertLen(bring_in_schedule, 2)
      self.assertEqual(bring_in_schedule[0].num_hands, 1)
      self.assertEqual(bring_in_schedule[0].bring_in, 5)
      self.assertEqual(bring_in_schedule[1].num_hands, 2)
      self.assertEqual(bring_in_schedule[1].bring_in, 10)

    def test_random_sim_cash_game_headsup(self):
      # Note the reset_stacks=True
      game = pyspiel.load_game(
          "python_repeated_pokerkit(max_num_hands=20,reset_stacks=True,"
          "rotate_dealer=True,pokerkit_game_params=python_pokerkit_wrapper())"
      )
      pyspiel.random_sim_test(game, num_sims=5, serialize=False, verbose=True)

    def test_random_sim_tournament_headsup(self):
      # Note the reset_stacks=False
      game = pyspiel.load_game(
          "python_repeated_pokerkit(max_num_hands=20,reset_stacks=False,"
          "rotate_dealer=True,pokerkit_game_params=python_pokerkit_wrapper())"
      )
      pyspiel.random_sim_test(game, num_sims=5, serialize=False, verbose=True)

    @parameterized.parameters(*_ALL_VARIANTS_CASH_AND_TOURNAMENT_PARAMS)
    def test_random_sim_all_variants_cash_and_tournament_headsup(
        self,
        variant,
        # True -> cash game, False -> tournament
        reset_stacks,
    ):
      params = {
          "max_num_hands": 10,
          "reset_stacks": reset_stacks,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "variant": variant,
              "num_players": 2,
              "blinds": "2 4",
              "bring_in": 4,
              "small_bet": 8,
              "big_bet": 16,
              "stack_sizes": "64 64",
          },
      }
      game = self._load_repeated_pokerkit_game_with_variant(variant, params)
      # NOTE: Consider carefully before increasing num_sims any further. As of
      # 2025, num_sims=3 is already pushing it (and e.g. num_sims=5 takes ~50%
      # longer than num_sims=3 does).
      pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=False)

    @parameterized.parameters(*_ALL_VARIANTS_CASH_AND_TOURNAMENT_PARAMS)
    def test_random_sim_all_variants_cash_and_tournament_6max(
        self,
        variant,
        # True -> cash game, False -> tournament
        reset_stacks,
    ):
      params = {
          "max_num_hands": 3,
          "reset_stacks": reset_stacks,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "variant": variant,
              "num_players": 6,
              "blinds": "2 4",
              "bring_in": 4,
              "small_bet": 8,
              "big_bet": 16,
              "stack_sizes": "30 30 30 30 30 30",
          },
      }
      game = self._load_repeated_pokerkit_game_with_variant(variant, params)
      # NOTE: Consider carefully before increasing num_sims any further. As of
      # 2025, num_sims=2 is already pushing it (and e.g. num_sims=3 takes ~50%
      # longer than num_sims=2 does).
      pyspiel.random_sim_test(game, num_sims=2, serialize=False, verbose=False)

    def test_blind_schedule_progression(self):
      blind_schedule_str = "1:10/20;2:20/40;1:50/100;1:1000/1000"
      params = {
          "max_num_hands": 100,
          "reset_stacks": False,
          "rotate_dealer": True,
          "blind_schedule": blind_schedule_str,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "1 2",  # Initial blinds should be overridden
              "stack_sizes": "1000 1000",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()

      # TODO: b/444333187 - consider refactoring to be "DAMP" instead of "DRY".
      expected_blinds = [
          (10, 20),  # Hand 0
          (20, 40),  # Hand 1
          (20, 40),  # Hand 2
          (50, 100),  # Hand 3
          (1000, 1000),  # Hands 4+
      ]
      seen_hands = set()
      current_hand = state._hand_number
      self.assertEqual(state._hand_number, 0)
      while not state.is_terminal():
        self.assertEqual(state._hand_number, current_hand)
        if current_hand not in seen_hands:
          seen_hands.add(current_hand)
          small_blind, big_blind = (
              expected_blinds[-1]
              if current_hand >= len(expected_blinds)
              else expected_blinds[current_hand]
          )
          self.assertEqual(
              state._small_blind,
              small_blind,
          )
          self.assertEqual(
              state._big_blind,
              big_blind,
          )

        if state.is_chance_node():
          action = random.choice([o for o, _ in state.chance_outcomes()])
        else:
          legal_actions = state.legal_actions()
          # Choose to check/call in all situations to ensure we reach the final
          # blind schedule level (at which point everyone will be forced
          # all-in).
          if ACTION_CHECK_OR_CALL in legal_actions:
            action = ACTION_CHECK_OR_CALL
          else:
            action = random.choice(state.legal_actions())

        state.apply_action(action)
        # apply_action handles the transition to the next hand when the
        # pokerkit_wrapper state is terminal.
        self.assertBetween(state._hand_number, current_hand, current_hand + 1)
        current_hand = state._hand_number

    def test_initially_scheduled_blinds_larger_than_stacks(self):
      blind_schedule_str = "1:200/400"
      params = {
          "max_num_hands": 1,
          "reset_stacks": False,
          "rotate_dealer": True,
          "blind_schedule": blind_schedule_str,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "1 2",  # Initial blinds should be overridden
              "stack_sizes": "100 100",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      self.assertEqual(state._hand_number, 0)
      self.assertEqual(state._small_blind, 200)
      self.assertEqual(state._big_blind, 400)
      self.assertEqual(state._stacks, [100, 100])
      # With 2 players, player 1 is dealer/SB and player 0 is BB in hand 0.
      # P1 posts SB=min(100, 200)=100, P0 posts BB=min(100, 400)=100.
      # Both players are all-in.
      # Pokerkit state stacks are seat-based. If P0=seat0=BB, P1=seat1=SB:
      # Seat 0 stack: 100-100=0. Seat 1 stack: 100-100=0.
      self.assertEqual(
          state.pokerkit_wrapper_state._wrapped_state.stacks, [0, 0]
      )
      self.assertEqual(
          state.pokerkit_wrapper_state._wrapped_state.bets, [100, 100]
      )

      # 4 chance nodes for hole cards
      for _ in range(4):
        self.assertTrue(state.is_chance_node())
        action = random.choice([o for o, _ in state.chance_outcomes()])
        state.apply_action(action)

      # After hole cards, betting round should be skipped as players are all in,
      # and board cards should be dealt.
      # 5 chance nodes for board cards
      for _ in range(5):
        self.assertTrue(state.is_chance_node())
        action = random.choice([o for o, _ in state.chance_outcomes()])
        state.apply_action(action)

      self.assertTrue(state.is_terminal())

    def test_later_stage_blinds_larger_than_stacks(self):
      blind_schedule_str = "1:1/2;1:200/400"
      params = {
          "max_num_hands": 5,
          "reset_stacks": False,
          "rotate_dealer": True,
          "blind_schedule": blind_schedule_str,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "1 2",
              "stack_sizes": "100 100",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      state_struct = state.to_struct()
      self.assertEqual(state_struct.hand_number, 0)
      self.assertEqual(state_struct.stacks, [100, 100])
      self.assertEqual(state_struct.dealer, 1)
      self.assertEqual(state_struct.small_blind, 1)
      self.assertEqual(state_struct.big_blind, 2)

      # 4 chance nodes for hole cards
      for _ in range(4):
        self.assertTrue(state.is_chance_node())
        action = random.choice([o for o, _ in state.chance_outcomes()])
        state.apply_action(action)

      # SB immediately folds preflop to start the second hand with higher
      # blinds.
      state.apply_action(ACTION_FOLD)

      self.assertFalse(state.is_terminal())
      self.assertTrue(state.is_chance_node())
      state_struct = state.to_struct()
      self.assertEqual(state_struct.hand_number, 1)
      self.assertEqual(state_struct.stacks, [101, 99])
      self.assertEqual(state_struct.dealer, 0)
      self.assertEqual(state_struct.small_blind, 200)
      self.assertEqual(state_struct.big_blind, 400)

      # 4 chance nodes for hole cards first, but then afterwards betting round
      # should be skipped as players are all in, and board cards should be
      # dealt, meaning +5 more chance nodes => 9 total chance nodes.
      #
      # Unlike before, we can't have a player fold so to have a deterministic
      # winner we need to choose the specific cards ourselves.
      # Result should be AA778 winning vs 77668 (with the 22 being counterfeit)
      dealt_cards = [
          Card(ACE, SPADE),
          Card(DEUCE, SPADE),
          Card(ACE, HEART),
          Card(DEUCE, DIAMOND),
          Card(SIX, CLUB),  # Flop 1
          Card(SIX, HEART),  # Flop 2
          Card(SEVEN, SPADE),  # Flop 3
          Card(SEVEN, HEART),  # Turn
          Card(EIGHT, DIAMOND),  # River
      ]
      for card in dealt_cards:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      self.assertFalse(state.is_terminal())
      state_struct = state.to_struct()
      self.assertEqual(state_struct.hand_number, 2)
      self.assertEqual(state_struct.stacks, [2, 198])
      self.assertEqual(state_struct.small_blind, 200)
      self.assertEqual(state_struct.big_blind, 400)

      # Same again as before, but to accomodate dealer button rotation we have
      # to reverse the hole cards.
      dealt_cards = [
          Card(DEUCE, SPADE),
          Card(ACE, SPADE),
          Card(DEUCE, DIAMOND),
          Card(ACE, HEART),
          Card(SIX, CLUB),  # Flop 1
          Card(SIX, HEART),  # Flop 2
          Card(SEVEN, SPADE),  # Flop 3
          Card(SEVEN, HEART),  # Turn
          Card(EIGHT, DIAMOND),  # River
      ]
      for card in dealt_cards:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      self.assertTrue(state.is_terminal())
      state_struct = state.to_struct()
      self.assertEqual(state_struct.hand_number, 3)
      self.assertEqual(state_struct.stacks, [0, 200])
      self.assertEqual(state_struct.small_blind, 200)
      self.assertEqual(state_struct.big_blind, 400)

    def test_stacks_carried_over(self):
      params = {
          "max_num_hands": 3,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 3,
              "blinds": "10 20",
              "stack_sizes": "100 100 100",  # Players 0 1 2: SB BB BTN
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()

      # Hand 0: P0 will win, P1 will bust, P2 will not be affected:

      self.assertEqual(state._hand_number, 0)
      self.assertEqual(state._stacks, [100, 100, 100])
      # Play a hand where P0 wins all of P1's chips and none of P2's chips.
      # P0 (SB) posts 10, P1 (BB) posts 20

      # First hole card (P0, then P1, then P2)
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(ACE, SPADE)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(DEUCE, SPADE)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(KING, HEART)]
      )

      # Second hole card (P0, then P1, then P2)
      # P0 has pocket Aces, P1 pockets Twos, P2 pocket Kings.
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(ACE, CLUB)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(DEUCE, CLUB)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(KING, SPADE)]
      )

      # P2 folds preflop
      state.apply_action(ACTION_FOLD)
      # P0 calls to 20 (10 additional), P1 checks
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)
      # Flop
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(SEVEN, SPADE)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(SEVEN, HEART)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(SEVEN, DIAMOND)]
      )
      # P0 bets 80 (all-in), P1 calls
      state.apply_action(80)
      state.apply_action(ACTION_CHECK_OR_CALL)
      # Turn/River
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(SIX, CLUB)]
      )
      state.apply_action(
          state.pokerkit_wrapper_game.card_to_int[Card(SIX, DIAMOND)]
      )
      assert not state.is_terminal()

      # P0 wins the pot. P0 stack: 100 + 100 = 200. P1 stack: 100 - 100 = 0.
      # P2 stack is unchanged. It's now the next hand and only P0 and P2 are
      # still playing (ie in the underlying pokerkit game).
      self.assertEqual(state._hand_number, 1)
      self.assertEqual(state._stacks, [200, 0, 100])
      # Button rotates from P2 to P0 since P0 is still active. But, this is
      # heads up now, meaning:
      # - P0 (former SB) now SB/BTN: 200 - 10 => 190
      # - P2 (former BTN) now BB: 100 - 20 => 80
      self.assertEqual(
          state.pokerkit_wrapper_state._wrapped_state.stacks,
          # NOTE: This is the pokerkit's stacks, meaning that the
          # last element is the BTN (or in this case, SB/BTN). So
          # this makes sense given the rotation from the prior hand.
          [80, 190],
      )

    def test_stack_rotation_multiway(self):
      params = {
          "max_num_hands": 10,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "variant": "NoLimitTexasHoldem",
              "num_players": 4,
              "blinds": "1 2",
              "stack_sizes": "200 200 200 200",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      # Each time Seat 0 / SB will win the hand. So in the next hand we should
      # be seeing the BTN / Seat 3 (which is their new seat post rotation)
      # gaining the chips, while everyone else has lost some.
      board = [
          Card(ACE, DIAMOND),
          Card(KING, DIAMOND),
          Card(QUEEN, DIAMOND),
          Card(JACK, HEART),
          Card(SEVEN, HEART),
      ]
      hole_cards = [
          Card(ACE, SPADE),  # SB card 0
          Card(KING, SPADE),
          Card(QUEEN, SPADE),
          Card(JACK, SPADE),
          Card(ACE, CLUB),  # SB Card 1
          Card(KING, CLUB),
          Card(QUEEN, CLUB),
          Card(JACK, CLUB),
      ]
      expected_dealers = [3, 0, 1, 2, 3, 0, 1, 2, 3, 0]

      # RepeatedPokerkit only updates stacks at the end of each hand and keeps
      # them all unrotated.
      expected_stacks_repeated_pokerkit = [
          [200, 200, 200, 200],
          [215, 195, 195, 195],  # P0 won last hand (as SB), + 15 chips
          [210, 210, 190, 190],  # P1 won last hand (as SB), + 15 chips
          [205, 205, 205, 185],  # P2 won last hand (as SB), + 15 chips
          # 'Symmetry' of win/loss naturally results in us reaching the same
          # game state we had at the start.
          [200, 200, 200, 200],  # P3 won last hand (as SB), + 15 chips
          [215, 195, 195, 195],  # P0 won last hand (as SB), + 15 chips
          [210, 210, 190, 190],  # P1 won last hand (as SB), + 15 chips
          [205, 205, 205, 185],  # P2 won last hand (as SB), + 15 chips
          # Again, same symmetry.
          [200, 200, 200, 200],  # P3 won last hand (as SB), + 15 chips
          [215, 195, 195, 195],  # P0 won last hand (as SB), + 15 chips
          [210, 210, 190, 190],  # P1 won last hand (as SB), + 15 chips
      ]
      # The PokerkitWrapper however will have rotated, plus by the time we're
      # checking, already removed additional chips to handle the blinds.
      expected_stacks_pokerkit_wrapper = [
          [199, 198, 200, 200],
          [194, 193, 195, 215],
          [189, 188, 210, 210],
          [184, 203, 205, 205],
          # 'Symmetry' of win/loss naturally results in us reaching the same
          # game state we had at the start.
          [199, 198, 200, 200],
          [194, 193, 195, 215],
          [189, 188, 210, 210],
          [184, 203, 205, 205],
          # Again, same symmetry.
          [199, 198, 200, 200],
          [194, 193, 195, 215],
          # After the tenth hand we reach max_num_hands and so won't ever have
          # started the next hand, meaning the rotation will still match the
          # last hand and blinds won't have been removed. Hence this differs
          # from above.
          # (Note however how the *rotation* still differs from the unrotated
          # RepeatedPokerkit's stacks above at this point!)
          [210, 190, 190, 210],
      ]

      for i in range(10):
        self.assertEqual(state._hand_number, i)
        self.assertEqual(state._dealer, expected_dealers[i])

        # RepeatedPokerkit's stacks should only care about the results following
        # the end of the prior hand + generally are not rotated.
        self.assertEqual(state._stacks, expected_stacks_repeated_pokerkit[i])
        self.assertEqual(
            state.to_struct().stacks,
            expected_stacks_repeated_pokerkit[i],
        )
        # Pokerkit's stacks should include the exact current situation in the
        # current hand, meaning they will have already removed chips coming from
        # the blinds AND will also be rotated.
        self.assertEqual(
            state.pokerkit_wrapper_state._wrapped_state.stacks,
            expected_stacks_pokerkit_wrapper[i],
        )
        self.assertEqual(
            state.to_struct().pokerkit_state_struct.stacks,
            expected_stacks_pokerkit_wrapper[i],
        )

        # Deal hole cards: 4 players * 2 cards = 8 chance nodes
        for i in range(8):
          self.assertTrue(state.is_chance_node())
          state.apply_action(
              state.pokerkit_wrapper_game.card_to_int[hole_cards[i]]
          )

        # Preflop betting: call around to BB
        self.assertFalse(state.is_chance_node())
        for _ in range(4):
          state.apply_action(ACTION_CHECK_OR_CALL)

        # Flop
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[board[0]])
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[board[1]])
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[board[2]])

        # Flop betting: first player bets 3, everyone calls. This means all
        # players have contributed 5 chips total.
        state.apply_action(3)
        state.apply_action(ACTION_CHECK_OR_CALL)
        state.apply_action(ACTION_CHECK_OR_CALL)
        state.apply_action(ACTION_CHECK_OR_CALL)

        # Turn
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[board[3]])
        # Turn betting: check around
        for _ in range(4):
          state.apply_action(ACTION_CHECK_OR_CALL)

        # River
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[board[4]])
        # River betting: check around
        for _ in range(4):
          state.apply_action(ACTION_CHECK_OR_CALL)

      self.assertEqual(state._hand_number, 9)
      self.assertTrue(state.is_terminal())

      final_stacks_index = 10
      assert final_stacks_index == len(expected_stacks_repeated_pokerkit) - 1
      self.assertEqual(
          state._stacks, expected_stacks_repeated_pokerkit[final_stacks_index]
      )
      self.assertEqual(
          state.to_struct().stacks,
          expected_stacks_repeated_pokerkit[final_stacks_index],
      )
      self.assertEqual(
          state.pokerkit_wrapper_state._wrapped_state.stacks,
          expected_stacks_pokerkit_wrapper[final_stacks_index],
      )
      self.assertEqual(
          state.to_struct().pokerkit_state_struct.stacks,
          expected_stacks_pokerkit_wrapper[final_stacks_index],
      )

    def test_eliminates_players_at_exactly_zero_chips(self):
      params = {
          "max_num_hands": 3,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 3,
              "blinds": "10 20",
              "stack_sizes": "100 100 100",  # Players 0 1 2: SB BB BTN
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()

      # Deal each player two random cards (choice doesn't matter)
      # 3 players * 2 hole cards => 6 chance nodes
      for _ in range(6):
        action = random.choice([o for o, _ in state.chance_outcomes()])
        state.apply_action(action)

      # P3 BTN raises to *almost* all-in (1 chip behind)
      state.apply_action(99)
      # P1 SB Shoves all-in
      state.apply_action(100)
      # P2 BB folds => 80 chips left
      state.apply_action(ACTION_FOLD)
      # BTN folds => 1 chip left (should not bust). BB wins the pot and the next
      # hand starts.
      self.assertEqual(state._hand_number, 0)
      state.apply_action(ACTION_FOLD)
      self.assertEqual(state._hand_number, 1)
      self.assertEqual(state._stacks, [219, 80, 1])

      # Ensure *three* players here are still active, not just two.
      self.assertEqual(state.num_players(), 3)
      self.assertEqual(state._num_active_players, 3)
      # Similarly: 3 players * 2 hole cards => 6 chance nodes
      for _ in range(6):
        self.assertTrue(state.is_chance_node())
        self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
        action = random.choice([o for o, _ in state.chance_outcomes()])
        state.apply_action(action)

        # Double check that nothing unexpectedly changes out from underneath
        # us here as we apply actions.
        self.assertEqual(state.num_players(), 3)
        self.assertEqual(state._num_active_players, 3)

      # P1 BTN bets 40
      state.apply_action(40)
      # P2 SB calls
      state.apply_action(ACTION_CHECK_OR_CALL)
      # P3 BB was already all-in due to having less than one Big Blind, so
      # gameplay should immediately procede to dealing the flop despite there
      # being 3 players.
      self.assertTrue(state.is_chance_node())

      # Definitely excessive, but triple checking this again just to be certain.
      self.assertEqual(state.num_players(), 3)
      self.assertEqual(state._num_active_players, 3)

    def test_multiway_all_but_one_players_bust_simultaneously(self):
      params = {
          "max_num_hands": 2,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 3,
              "blinds": "10 20",
              "stack_sizes": "100 100 100",  # Players 0 1 2: SB BB BTN
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()

      # Hand 0: P0 wins, P1 and P2 bust.
      # 3 players: P0=SB, P1=BB, P2=BTN
      self.assertEqual(state._hand_number, 0)
      self.assertEqual(state._stacks, [100, 100, 100])

      # Hole cards:
      # P0: Ac As
      # P1: Kc Ks
      # P2: Qc Qs
      hole_cards = [
          Card(ACE, CLUB),
          Card(KING, CLUB),
          Card(QUEEN, CLUB),
          Card(ACE, SPADE),
          Card(KING, SPADE),
          Card(QUEEN, SPADE),
      ]
      for card in hole_cards:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      self.assertEqual(state.current_player(), 2)  # P2 BTN acts first preflop

      # P2 BTN raises all-in to 100
      state.apply_action(100)
      # P0 SB calls all-in (has 90 behind, action is for 100)
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)
      # P1 BB calls all-in (has 80 behind, action is for 100)
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Flop, turn, river
      board = [
          Card(TREY, HEART),
          Card(FOUR, DIAMOND),
          Card(SIX, SPADE),
          Card(SEVEN, HEART),
          Card(EIGHT, HEART),
      ]
      for card in board:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      # Hand 0 finished, P0 won, P1 and P2 busted.
      # With only 1 player left, game should be terminal.
      self.assertTrue(state.is_terminal())
      self.assertEqual(state._hand_number, 1)
      self.assertEqual(state._stacks, [300, 0, 0])
      self.assertEqual(state._num_active_players, 1)

    def test_6max_3players_bust_then_split_pot(self):
      params = {
          "max_num_hands": 2,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 6,
              "blinds": "10 20",
              "stack_sizes": "100 100 100 100 100 100",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()

      # Hand 0: 4-way all-in preflop, P0 wins, P1,P2,P3 bust.
      # 6 players: P0=SB, P1=BB, P2=UTG, P3=MP, P4=CO, P5=BTN
      self.assertEqual(state._hand_number, 0)
      self.assertEqual(state._stacks, [100] * 6)

      # Hole cards:
      # P0: Ac As
      # P1: Kc Ks
      # P2: Qc Qs
      # P3: Jc Js
      # P4: 2c 2s
      # P5: 3c 3s
      hole_cards_h0 = [
          Card(ACE, CLUB),
          Card(KING, CLUB),
          Card(QUEEN, CLUB),
          Card(JACK, CLUB),
          Card(DEUCE, CLUB),
          Card(TREY, CLUB),
          Card(ACE, SPADE),
          Card(KING, SPADE),
          Card(QUEEN, SPADE),
          Card(JACK, SPADE),
          Card(DEUCE, SPADE),
          Card(TREY, SPADE),
      ]
      for card in hole_cards_h0:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      self.assertEqual(state.current_player(), 2)  # P2 UTG acts first preflop

      # P2 UTG raises all-in to 100
      state.apply_action(100)
      # P3 MP calls all-in
      self.assertEqual(state.current_player(), 3)
      state.apply_action(ACTION_CHECK_OR_CALL)
      # P4 CO folds
      self.assertEqual(state.current_player(), 4)
      state.apply_action(ACTION_FOLD)
      # P5 BTN folds
      self.assertEqual(state.current_player(), 5)
      state.apply_action(ACTION_FOLD)
      # P0 SB calls all-in
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)
      # P1 BB calls all-in
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Board cards
      board_h0 = [
          Card(SEVEN, DIAMOND),
          Card(SEVEN, SPADE),
          Card(SEVEN, CLUB),
          Card(EIGHT, DIAMOND),
          Card(EIGHT, HEART),
      ]
      for card in board_h0:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      # Hand 0 finished. P0 won 300 chips. P1,P2,P3 busted.
      # Players 0,4,5 remain active.
      self.assertEqual(state._hand_number, 1)
      self.assertEqual(state._stacks, [400, 0, 0, 0, 100, 100])
      self.assertEqual(state._num_active_players, 3)
      self.assertFalse(state.is_terminal())

      # Hand 1: Players 0,4,5 play. 3-way split pot.
      # P0=BTN, P4=SB, P5=BB. Stacks: P0=400, P4=100, P5=100. Blinds 10/20.
      # Active players sorted are 0,4,5 -> player_map {0:0, 1:4, 2:5}, so
      # seat0=P0, seat1=P4, seat2=P5.
      # P0 is dealer, so P4 is SB, P5 is BB.
      # Hole cards for hand 1:
      # P0: 2h 3h, P4: 4s 5s, P5: 6c 7c
      hole_cards_h1 = [
          Card(DEUCE, HEART),  # P0 card 1
          Card(FOUR, SPADE),  # P4 card 1
          Card(SIX, CLUB),  # P5 card 1
          Card(TREY, HEART),  # P0 card 2
          Card(FIVE, SPADE),  # P4 card 2
          Card(SEVEN, CLUB),  # P5 card 2
      ]
      for card in hole_cards_h1:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      # Players 0,4,5 are active. P0=BTN, P4=SB, P5=BB.
      # P0 P4 and P5 all raise as much as possible (or call) to force an all-in
      # situation.
      # In 3-handed, BTN acts first preflop.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(max(state.legal_actions()))
      self.assertEqual(state.current_player(), 4)
      state.apply_action(max(state.legal_actions()))
      self.assertEqual(state.current_player(), 5)
      state.apply_action(max(state.legal_actions()))

      # Board Ad Kd Qd Jd Td - royal flush on board.
      board_h1 = [
          Card(ACE, DIAMOND),
          Card(KING, DIAMOND),
          Card(QUEEN, DIAMOND),
          Card(JACK, DIAMOND),
          Card(TEN, DIAMOND),
      ]
      for card in board_h1:
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.pokerkit_wrapper_game.card_to_int[card])

      # Hand 1 finished, split pot. Stacks unchanged.
      # max_num_hands will have been reached, so game should be terminal and
      # the hand number should remain the same as before.
      self.assertEqual(state._hand_number, 1)
      self.assertTrue(state.is_terminal())
      self.assertEqual(state._stacks, [400, 0, 0, 0, 100, 100])
      self.assertEqual(state._num_active_players, 3)
      self.assertEqual(
          state.returns(), [300.0, -100.0, -100.0, -100.0, 0.0, 0.0]
      )

    def test_max_num_hands_limit(self):
      """Tests that the game terminates when max_num_hands is reached."""
      params = {
          "max_num_hands": 1,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "50 100",
              "stack_sizes": "1000 1000",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      self.assertEqual(state._hand_number, 0)
      self.assertFalse(state.is_terminal())

      # Play one hand: SB folds immediately preflop.
      # 4 chance nodes for hole cards
      for _ in range(4):
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.chance_outcomes()[0][0])

      self.assertFalse(state.is_chance_node())
      # P1 is SB and folds.
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_FOLD)

      # Hand is over, max_num_hands=1 is reached, game should be terminal.
      self.assertTrue(state.is_terminal())
      self.assertEqual(state._hand_number, 0)
      # P1 is SB in hand 0 and folded. P0 is BB.
      # P1 stack: 1000 - 50 = 950. P0 stack: 1000 + 50 = 1050.
      self.assertEqual(state._stacks, [1050, 950])

    def test_returns_in_terminal_state(self):
      """Tests that returns() are correct when state is terminal."""
      params = {
          "max_num_hands": 1,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "50 100",
              "stack_sizes": "1000 1000",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      self.assertEqual(state._hand_number, 0)
      self.assertFalse(state.is_terminal())

      # Play one hand: SB folds immediately preflop.
      # 4 chance nodes for hole cards
      for _ in range(4):
        self.assertTrue(state.is_chance_node())
        state.apply_action(state.chance_outcomes()[0][0])

      self.assertFalse(state.is_chance_node())
      # P1 is SB and folds.
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_FOLD)

      # Hand is over, max_num_hands=1 is reached, game should be terminal.
      self.assertTrue(state.is_terminal())
      # P1 is SB in hand 0 and folded. P0 is BB.
      # P1 stack: 1000 - 50 = 950. P0 stack: 1000 + 50 = 1050.
      self.assertEqual(state.returns(), [50.0, -50.0])

    def test_pokerkit_state_to_struct_to_json_two_hands(self):
      params = {
          "max_num_hands": 2,
          "reset_stacks": False,
          "rotate_dealer": True,
          "pokerkit_game_params": {
              "name": "python_pokerkit_wrapper",
              "num_players": 2,
              "blinds": "50 100",
              "stack_sizes": "1000 1000",
          },
      }
      game = pyspiel.load_game("python_repeated_pokerkit", params)
      state = game.new_initial_state()
      self.assertEqual(state._hand_number, 0)

      state_struct = state.to_struct()
      state_json = state.to_json()
      self.assertEqual(state_struct.to_json(), state_json)

      # --- First hand (hand 0) ---
      self.assertIsInstance(state_json, str)
      try:
        data_h0 = json.loads(state_json)
      except json.JSONDecodeError as e:
        self.fail(f"Failed to parse JSON {state_json}, error: {e}")

      self.assertEqual(state_struct.hand_number, 0)
      self.assertEqual(data_h0["hand_number"], 0)
      self.assertEqual(state_struct.is_terminal, False)
      self.assertEqual(data_h0["is_terminal"], False)
      self.assertEqual(state_struct.stacks, [1000, 1000])
      self.assertEqual(data_h0["stacks"], [1000, 1000])
      self.assertEqual(state_struct.dealer, 1)
      self.assertEqual(data_h0["dealer"], 1)
      self.assertEqual(state_struct.seat_to_player, {0: 0, 1: 1})
      self.assertEqual(data_h0["seat_to_player"], [[0, 0], [1, 1]])
      self.assertEqual(state_struct.player_to_seat, {0: 0, 1: 1})
      self.assertEqual(data_h0["player_to_seat"], [[0, 0], [1, 1]])
      self.assertEqual(state_struct.small_blind, 50)
      self.assertEqual(data_h0["small_blind"], 50)
      self.assertEqual(state_struct.big_blind, 100)
      self.assertEqual(data_h0["big_blind"], 100)
      self.assertEqual(state_struct.small_bet_size, -1)
      self.assertEqual(data_h0["small_bet_size"], -1)
      self.assertEqual(state_struct.big_bet_size, -1)
      self.assertEqual(data_h0["big_bet_size"], -1)
      self.assertEqual(state_struct.bring_in, -1)
      self.assertEqual(data_h0["bring_in"], -1)
      self.assertEqual(state_struct.hand_returns, [[0.0, 0.0]])
      self.assertEqual(data_h0["hand_returns"], [[0.0, 0.0]])

      pk_state_struct_h0 = state_struct.pokerkit_state_struct
      pk_data_h0 = data_h0["pokerkit_state_struct"]

      self.assertEqual(pk_data_h0["is_terminal"], False)
      self.assertEqual(
          state.pokerkit_wrapper_state.is_terminal(), pk_data_h0["is_terminal"]
      )
      self.assertEqual(pk_data_h0["current_player"], pyspiel.PlayerId.CHANCE)
      self.assertEqual(
          state.pokerkit_wrapper_state.current_player(),
          pk_data_h0["current_player"],
      )
      # Stacks in pokerkit state are after blinds: 1000-50=950, 1000-100=900
      self.assertEqual(pk_data_h0["stacks"], [900, 950])
      self.assertEqual(pk_state_struct_h0.stacks, pk_data_h0["stacks"])
      self.assertEqual(pk_data_h0["bets"], [100, 50])
      self.assertEqual(pk_state_struct_h0.bets, pk_data_h0["bets"])

      # Play until hand is over, having P1 fold immediately preflop.
      while state._hand_number == 0:
        if state.is_chance_node():
          state.apply_action(state.chance_outcomes()[0][0])
        else:
          state.apply_action(ACTION_FOLD)

      # --- Second hand (hand 1) ---
      self.assertEqual(state._hand_number, 1)
      state_struct_h1 = state.to_struct()
      state_json_h1 = state.to_json()
      self.assertEqual(state_struct_h1.to_json(), state_json_h1)
      self.assertIsInstance(state_json_h1, str)
      try:
        data_h1 = json.loads(state_json_h1)
      except json.JSONDecodeError as e:
        self.fail(f"Failed to parse JSON {state_json_h1}, error: {e}")

      self.assertEqual(state_struct_h1.hand_number, 1)
      self.assertEqual(data_h1["hand_number"], 1)
      self.assertEqual(state_struct_h1.is_terminal, False)
      self.assertEqual(data_h1["is_terminal"], False)
      # P1 folded in hand 0, so P0 wins 50 from P1.
      # Stacks before hand 1: P0=1000+50=1050, P1=1000-50=950.
      self.assertEqual(state_struct_h1.stacks, [1050, 950])
      self.assertEqual(data_h1["stacks"], [1050, 950])
      self.assertEqual(state_struct_h1.dealer, 0)
      self.assertEqual(data_h1["dealer"], 0)
      self.assertEqual(state_struct_h1.seat_to_player, {0: 1, 1: 0})
      self.assertEqual(data_h1["seat_to_player"], [[0, 1], [1, 0]])
      self.assertEqual(state_struct_h1.player_to_seat, {0: 1, 1: 0})
      self.assertEqual(data_h1["player_to_seat"], [[0, 1], [1, 0]])
      self.assertEqual(state_struct_h1.small_blind, 50)
      self.assertEqual(data_h1["small_blind"], 50)
      self.assertEqual(state_struct_h1.big_blind, 100)
      self.assertEqual(data_h1["big_blind"], 100)
      self.assertEqual(state_struct_h1.small_bet_size, -1)
      self.assertEqual(data_h1["small_bet_size"], -1)
      self.assertEqual(state_struct_h1.big_bet_size, -1)
      self.assertEqual(data_h1["big_bet_size"], -1)
      self.assertEqual(state_struct_h1.bring_in, -1)
      self.assertEqual(data_h1["bring_in"], -1)
      self.assertEqual(
          state_struct_h1.hand_returns, [[50.0, -50.0], [0.0, 0.0]]
      )
      self.assertEqual(data_h1["hand_returns"], [[50.0, -50.0], [0.0, 0.0]])

      pk_state_struct_h1 = state_struct_h1.pokerkit_state_struct
      pk_data_h1 = data_h1["pokerkit_state_struct"]

      self.assertEqual(pk_data_h1["is_terminal"], False)
      self.assertEqual(
          pk_state_struct_h1.is_terminal, pk_data_h1["is_terminal"]
      )
      self.assertEqual(pk_data_h1["current_player"], pyspiel.PlayerId.CHANCE)
      self.assertEqual(
          pk_state_struct_h1.current_player, pk_data_h1["current_player"]
      )
      # Dealer rotates, P1 was dealer in hand 0, so P0 is dealer in hand 1.
      # Heads-up: dealer is SB. P0=SB=50, P1=BB=100.
      # Pokerkit state stacks based on seats:
      # P0 is seat 1 (SB), stack 1050-50=1000
      # P1 is seat 0 (BB), stack 950-100=850
      self.assertEqual(pk_data_h1["stacks"], [850, 1000])
      self.assertEqual(pk_state_struct_h1.stacks, pk_data_h1["stacks"])
      self.assertEqual(pk_data_h1["bets"], [100, 50])
      self.assertEqual(pk_state_struct_h1.bets, pk_data_h1["bets"])

      # Finally double check we can reach terminal state as expected
      while not state.is_terminal():
        if state.is_chance_node():
          state.apply_action(state.chance_outcomes()[0][0])
        else:
          # Now this time the other player folds - meaning we're inverting what
          # just happened last hand.
          state.apply_action(ACTION_FOLD)

      state_struct_terminal = state.to_struct()
      state_json_terminal = state.to_json()
      self.assertIsInstance(state_json_terminal, str)
      try:
        data_terminal = json.loads(state_json_terminal)
      except json.JSONDecodeError as e:
        self.fail(f"Failed to parse JSON {state_json_terminal}, error: {e}")
      self.assertTrue(data_terminal["is_terminal"], True)
      self.assertEqual(
          state_struct_terminal.is_terminal, data_terminal["is_terminal"]
      )
      self.assertEqual(
          state_struct_terminal.hand_returns, [[50.0, -50.0], [-50.0, 50.0]]
      )
      self.assertEqual(
          data_terminal["hand_returns"], [[50.0, -50.0], [-50.0, 50.0]]
      )
      self.assertEqual(state_struct_terminal.stacks, [1000, 1000])
      self.assertEqual(data_terminal["stacks"], [1000, 1000])

      # Max hands was 2, so it shouldn't have progressed past hand 1 and the
      # players shouldn't have rotated.
      self.assertEqual(state_struct_terminal.hand_number, 1)
      self.assertEqual(data_terminal["hand_number"], 1)
      self.assertEqual(state_struct_terminal.seat_to_player, {0: 1, 1: 0})
      self.assertEqual(data_terminal["seat_to_player"], [[0, 1], [1, 0]])
      self.assertEqual(state_struct_terminal.player_to_seat, {0: 1, 1: 0})
      self.assertEqual(data_terminal["player_to_seat"], [[0, 1], [1, 0]])


if __name__ == "__main__":
  if IMPORTED_ALL_LIBRARIES:
    absltest.main()
  else:
    logging.warning("Skipping test because not all libraries were imported.")
