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
import random

from absl.testing import absltest
from absl.testing import parameterized
import pokerkit

from open_spiel.python.games import pokerkit_wrapper
from open_spiel.python.games import repeated_pokerkit
import pyspiel

ACTION_CHECK_OR_CALL = pokerkit_wrapper.ACTION_CHECK_OR_CALL
ACTION_FOLD = pokerkit_wrapper.ACTION_FOLD

Card = pokerkit.Card
ACE = pokerkit.Rank.ACE
DEUCE = pokerkit.Rank.DEUCE
KING = pokerkit.Rank.KING
SEVEN = pokerkit.Rank.SEVEN
SIX = pokerkit.Rank.SIX
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
    for params, reset_stacks in itertools.product(_ALL_VARIANTS, [True, False])
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
    blind_schedule_str = "1,1,2;2,2,3"
    blind_schedule = repeated_pokerkit.parse_blind_schedule(blind_schedule_str)
    self.assertLen(blind_schedule, 2)
    self.assertEqual(blind_schedule[0].num_hands, 1)
    self.assertEqual(blind_schedule[0].small_blind, 1)
    self.assertEqual(blind_schedule[0].big_blind, 2)
    self.assertEqual(blind_schedule[1].num_hands, 2)
    self.assertEqual(blind_schedule[1].small_blind, 2)
    self.assertEqual(blind_schedule[1].big_blind, 3)

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
    blind_schedule_str = "1,10,20;2,20,40;1,50,100;1,1000,1000"
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
        action = random.choice([o for o, p in state.chance_outcomes()])
      else:
        legal_actions = state.legal_actions()
        # Choose to check/call in all situations to ensure we reach the final
        # blind schedule level (at which point everyone will be forced all-in).
        if ACTION_CHECK_OR_CALL in legal_actions:
          action = ACTION_CHECK_OR_CALL
        else:
          action = random.choice(state.legal_actions())

      state.apply_action(action)
      # apply_action handles the transition to the next hand when the
      # pokerkit_wrapper state is terminal.
      self.assertBetween(state._hand_number, current_hand, current_hand + 1)
      current_hand = state._hand_number

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
    state.apply_action(state.pokerkit_wrapper_game.card_to_int[Card(ACE, CLUB)])
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
    state.apply_action(state.pokerkit_wrapper_game.card_to_int[Card(SIX, CLUB)])
    state.apply_action(
        state.pokerkit_wrapper_game.card_to_int[Card(SIX, DIAMOND)]
    )
    assert not state.is_terminal()

    # P0 wins the pot. P0 stack: 100 + 100 = 200. P1 stack: 100 - 100 = 0.
    # P2 stack is unchanged. It's now the next hand and only P0 and P2 are still
    # playing (ie in the underlying pokerkit game).
    self.assertEqual(state._hand_number, 1)
    self.assertEqual(state._stacks, [200, 0, 100])
    # Button rotates from P2 to P0 since P0 is still active. But, this is heads
    # up now, meaning:
    # - P0 (former SB) now SB/BTN: 200 - 10 => 190
    # - P2 (former BTN) now BB: 100 - 20 => 80
    self.assertEqual(
        state.pokerkit_wrapper_state._wrapped_state.stacks,
        # NOTE: This is the pokerkit's stacks, meaning that the
        # last element is the BTN (or in this case, SB/BTN). So
        # this makes sense given the rotation from the prior hand.
        [80, 190],
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

      # Double check again that nothing unexpectedly changes out from underneath
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


# TODO: b/444333187 - Add tests for the following behaviors:
# - blind schedule becoming higher than all players' remaining stacks. (This was
#   a known issue previously, though may be fixed now that we no longer are
#   eliminating players with less than one Big Blind.)
# - reaching a low max_num_hands limit
# - multiple players busting out (especially edge cases)
# - returns()'s values being correct when reaching terminal state


if __name__ == "__main__":
  absltest.main()
