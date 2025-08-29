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
"""Tests for OpenSpiel game wrapper for Pokerkit."""

import random
import unittest

from absl.testing import absltest
import numpy as np
import pokerkit

from open_spiel.python.games import pokerkit_wrapper
import pyspiel

PokerkitWrapper = pokerkit_wrapper.PokerkitWrapper
PokerkitWrapperAcpcStyle = pokerkit_wrapper.PokerkitWrapperAcpcStyle

Card = pokerkit.Card

ACE = pokerkit.Rank.ACE
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
DEUCE = pokerkit.Rank.DEUCE

SPADE = pokerkit.Suit.SPADE
CLUB = pokerkit.Suit.CLUB
HEART = pokerkit.Suit.HEART
DIAMOND = pokerkit.Suit.DIAMOND

ACTION_FOLD = pokerkit_wrapper.ACTION_FOLD
ACTION_CHECK_OR_CALL = pokerkit_wrapper.ACTION_CHECK_OR_CALL


class PokerkitWrapperTest(absltest.TestCase):
  """Test the OpenSpiel game wrapper for Pokerkit."""

  # Lightweight testing to verify that the 'usage' directories are at least
  # _somewhat_ correct. (Not intended to be exhaustive!)
  def test_min_bet_usage(self):
    variants = pokerkit_wrapper._VARIANT_PARAM_USAGE["min_bet"]
    self.assertIn(pokerkit.NoLimitTexasHoldem.__name__, variants)
    self.assertNotIn(pokerkit.FixedLimitTexasHoldem.__name__, variants)

  def test_bring_in_usage(self):
    variants = pokerkit_wrapper._VARIANT_PARAM_USAGE["bring_in"]
    self.assertIn(pokerkit.FixedLimitRazz.__name__, variants)
    self.assertIn(pokerkit.FixedLimitSevenCardStud.__name__, variants)
    self.assertNotIn(pokerkit.NoLimitTexasHoldem.__name__, variants)

  def test_small_bet_usage(self):
    variants = pokerkit_wrapper._VARIANT_PARAM_USAGE["small_bet"]
    self.assertIn(pokerkit.FixedLimitTexasHoldem.__name__, variants)
    self.assertIn(pokerkit.FixedLimitSevenCardStud.__name__, variants)
    self.assertIn(pokerkit.FixedLimitRazz.__name__, variants)
    self.assertNotIn(pokerkit.NoLimitTexasHoldem.__name__, variants)

  def test_big_bet_usage(self):
    variants = pokerkit_wrapper._VARIANT_PARAM_USAGE["big_bet"]
    self.assertIn(pokerkit.FixedLimitTexasHoldem.__name__, variants)
    self.assertIn(pokerkit.FixedLimitSevenCardStud.__name__, variants)
    self.assertNotIn(pokerkit.NoLimitTexasHoldem.__name__, variants)

  def test_action_space_equals_max_stack_size(self):
    obj: PokerkitWrapper = PokerkitWrapper(
        params={
            "variant": "NoLimitTexasHoldem",
            "stack_sizes": "400 300 10 20",
            "num_players": 4,
        }
    )
    game_info: pyspiel.GameInfo = obj.game_info()
    # 0 => fold
    # 1 => check
    # 2, 3, ... 400 => raise
    self.assertEqual(game_info.num_distinct_actions, 401)

  def test_constructs_with_default_values_if_empty_params(self):
    obj = PokerkitWrapper(params=None)
    self.assertEqual(obj.params["num_players"], 2)
    self.assertEqual(obj.params["variant"], "NoLimitTexasHoldem")

  def test_deck_size_is_correct_for_nolimit_holdem(self):
    obj = PokerkitWrapper(params={"variant": "NoLimitTexasHoldem"})
    self.assertEqual(obj.deck_size, 52)

  def test_card_to_int_lookups_match_for_nolimit_holdem(self):
    obj = PokerkitWrapper(params={"variant": "NoLimitTexasHoldem"})
    self.assertLen(obj.card_to_int, 52)
    self.assertLen(obj.int_to_card, 52)

    self.assertEqual(obj.card_to_int[obj.int_to_card[0]], 0)
    self.assertEqual(obj.card_to_int[obj.int_to_card[51]], 51)
    self.assertEqual(
        obj.int_to_card[obj.card_to_int[pokerkit.Card(DEUCE, CLUB)]],
        pokerkit.Card(DEUCE, CLUB),
    )
    self.assertEqual(
        obj.int_to_card[obj.card_to_int[pokerkit.Card(ACE, SPADE)]],
        pokerkit.Card(ACE, SPADE),
    )

    # Technically is depending on pokerkit implementation details, in practice
    # should be fine since pokerkit is highly unlikely to change the default
    # ordering in future (which happens to have Spades last due to 's' coming
    # after 'd', 'h' and 'c' when sorted in alphabetical order).
    self.assertEqual(obj.int_to_card[51], pokerkit.Card(ACE, SPADE))

  def test_assumes_big_blind_as_min_bet_for_nolimit_holdem(self):
    obj = PokerkitWrapper(
        params={
            "variant": "NoLimitTexasHoldem",
            "num_players": 2,
            "blinds": "17 17",
        }
    )
    self.assertEqual(obj.params["blinds"], "17 17")
    self.assertEqual(obj.params["min_bet"], 17)

  def test_does_not_assume_big_blind_as_min_bet_or_fixed_bets_for_limit_holdem(
      self,
  ):
    obj_default_fixed_bets = PokerkitWrapper(
        params={
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "17 17",
        }
    )
    self.assertEqual(obj_default_fixed_bets.params["blinds"], "17 17")
    self.assertNotIn("min_bet", obj_default_fixed_bets.params)
    self.assertIn("small_bet", obj_default_fixed_bets.params)
    self.assertIn("big_bet", obj_default_fixed_bets.params)

    obj_set_fixed_bets = PokerkitWrapper(
        params={
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "17 17",
            "small_bet": 7,
            "big_bet": 8,
        }
    )
    self.assertEqual(obj_set_fixed_bets.params["blinds"], "17 17")
    self.assertNotIn("min_bet", obj_set_fixed_bets.params)
    self.assertEqual(obj_set_fixed_bets.params["small_bet"], 7)
    self.assertEqual(obj_set_fixed_bets.params["big_bet"], 8)

  def test_card_burning_is_disabled_everywhere(self):
    obj = PokerkitWrapper()
    fresh_isolated_state: pokerkit.State = obj.wrapped_state_factory()
    for street in fresh_isolated_state.streets:
      self.assertFalse(street.card_burning_status)

    wrapped_state_copy = obj.new_initial_state().deepcopy_wrapped_state()
    for street in wrapped_state_copy.streets:
      self.assertFalse(street.card_burning_status)

  # TODO: b/437724266 - Make sure we've added at least one test exercising the
  # 'returns' function once we flesh out the base class a bit more.


class PokerkitWrapperAcpcStyleTest(unittest.TestCase):
  """Test the OpenSpiel game wrapper for Pokerkit."""

  # TODO: b/437724266 - port over more OpenSpiel universal_poker tests to verify
  # that we have identical (or, at least mostly identical) behavior + are at
  # least as "correct" in general.

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_playthrough_robustness(self):
    """Runs full random playthroughs for various games to ensure stability."""
    scenarios = [
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 3,
            "blinds": "50 100",
            "stack_sizes": "1000 2000 3000",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 4,
            "blinds": "500 1000",
            "stack_sizes": "200 200 1000 4000",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 5,
            "blinds": "500 1000",
            "stack_sizes": "4000 1000 10000 3000 4000",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 4,
            "blinds": "500 1000",
            "stack_sizes": "200 200 1000 4000",
        },
        # Both players forced all-in from blinds
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 2,
            "blinds": "500 1000",
            "stack_sizes": "200 200",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "500 1000",
            "stack_sizes": "200 200",
        },
    ]

    for params in scenarios:
      with self.subTest(variant=params["variant"]):
        game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
        state = game.new_initial_state()
        while not state.is_terminal():
          if state.is_chance_node():
            action = random.choice([o for o, p in state.chance_outcomes()])
          else:
            action = random.choice(state.legal_actions())
          state.apply_action(action)
        self.assertAlmostEqual(sum(state.returns()), 0.0)

  def test_utility_calculation_in_all_in(self):
    """Verifies correct raw chip payoffs in a deterministic all-in."""
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 2,
        "blinds": "50 100",
        "stack_sizes": "10000 10000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()

    state.apply_action(game.card_to_int[Card(SIX, CLUB)])  # P0 card 1
    state.apply_action(game.card_to_int[Card(SEVEN, SPADE)])  # P1 card 1
    state.apply_action(game.card_to_int[Card(SEVEN, HEART)])  # P0 card 2
    state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])  # P1 card 2

    state.apply_action(300)  # P0 raises to 300
    state.apply_action(900)  # P1 re-raises to 900
    state.apply_action(10000)  # P0 shoves
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls

    # Progress through all remaining chance nodes, i.e. deal the flop, turn, and
    # river.
    while not state.is_terminal():
      state.apply_action(state.chance_outcomes()[0][0])

    returns = state.returns()
    self.assertIn(10000.0, returns)
    self.assertIn(-10000.0, returns)

  def test_legal_action_space(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "10 20",
        "stack_sizes": "2000 2000 2000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    state.apply_action(game.card_to_int[Card(ACE, CLUB)])
    state.apply_action(game.card_to_int[Card(DEUCE, SPADE)])
    state.apply_action(game.card_to_int[Card(SEVEN, HEART)])
    state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])
    state.apply_action(game.card_to_int[Card(ACE, SPADE)])
    state.apply_action(game.card_to_int[Card(DEUCE, CLUB)])

    state.apply_action(60)  # P2 raises
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 folds

    # P1 can call (0), fold (1), or reraise
    expected_actions = [0, 1] + list(range(100, 2001))
    self.assertEqual(state.legal_actions(), expected_actions)

  def test_information_state_tensor_content(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "10 20",
        "stack_sizes": "1000 1000 1000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()

    # Manually set a deterministic hand for verification
    # P0: As Ac, P1: Kd Kh, P2: Qd Qh
    # Board: 2s 7h 8d

    # Deal first hole card
    state.apply_action(game.card_to_int[Card(ACE, SPADE)])  # P0
    state.apply_action(game.card_to_int[Card(KING, DIAMOND)])  # P1
    state.apply_action(game.card_to_int[Card(QUEEN, DIAMOND)])  # P2

    # Deal second hole card
    state.apply_action(game.card_to_int[Card(ACE, CLUB)])  # P0
    state.apply_action(game.card_to_int[Card(KING, HEART)])  # P1
    state.apply_action(game.card_to_int[Card(QUEEN, HEART)])  # P2

    # Pre-flop betting
    state.apply_action(60)  # P2 raises
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls
    state.apply_action(ACTION_FOLD)  # P1 folds

    # Deal flop
    state.apply_action(game.card_to_int[Card(DEUCE, SPADE)])
    state.apply_action(game.card_to_int[Card(SEVEN, HEART)])
    state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])

    # --- Verification for Player 0 ---
    tensor = state.information_state_tensor(0)
    deck_size = game.deck_size

    # 0. Current player is player 0
    player_encoding = tensor[0 : game.num_players()]
    self.assertEqual(player_encoding[0], 1.0)  # Tensor is P0's
    self.assertEqual(player_encoding[1], 0.0)  # Tensor is not for P1
    self.assertEqual(player_encoding[2], 0.0)  # Tensor is not for P2
    self.assertEqual(np.sum(player_encoding), 1)
    offset = 3  # num_players

    # 1. Private hole cards (As, Ac)
    private_cards = tensor[offset : deck_size + offset]
    self.assertEqual(private_cards[game.card_to_int[Card(ACE, SPADE)]], 1.0)
    self.assertEqual(private_cards[game.card_to_int[Card(ACE, CLUB)]], 1.0)
    self.assertEqual(np.sum(private_cards), 2)
    offset += deck_size

    # 2. Board cards (2s, 7h, 8d)
    public_cards = tensor[offset : offset + deck_size]
    self.assertEqual(public_cards[game.card_to_int[Card(DEUCE, SPADE)]], 1.0)
    self.assertEqual(public_cards[game.card_to_int[Card(SEVEN, HEART)]], 1.0)
    self.assertEqual(public_cards[game.card_to_int[Card(EIGHT, DIAMOND)]], 1.0)
    self.assertEqual(np.sum(public_cards), 3)
    offset += deck_size

    # 3. Action sequence
    action_sequence = tensor[offset : offset + 2 * game.max_game_length()]
    offset += 2 * game.max_game_length()
    for i in range(12):
      self.assertEqual(action_sequence[i], 0.0)  # Deals get value 0 0
    # Raise gets [0 1]
    self.assertEqual(action_sequence[12], 0.0)
    self.assertEqual(action_sequence[13], 1.0)
    # Call gets [1 0]
    self.assertEqual(action_sequence[14], 1.0)
    self.assertEqual(action_sequence[15], 0.0)
    # Fold gets [0 0]
    self.assertEqual(action_sequence[16], 0.0)
    self.assertEqual(action_sequence[17], 0.0)
    for i in range(18, 24):
      self.assertEqual(action_sequence[i], 0.0)  # Deals get value 0 0
    self.assertEqual(np.sum(action_sequence), 2)  # Only [1 0] and [0 1] => 2

    # 4. Action sequence sizings
    action_sequence_sizings = tensor[offset : offset + game.max_game_length()]
    offset += game.max_game_length()
    for i in range(6):
      self.assertEqual(action_sequence_sizings[i], 0.0)  # Deals get value 0
    self.assertEqual(action_sequence_sizings[6], 60.0)  # Raise's exact value
    self.assertEqual(action_sequence_sizings[7], 0.0)  # Calls get value 0
    self.assertEqual(action_sequence_sizings[8], 0.0)  # Folds get value 0
    self.assertEqual(np.sum(action_sequence_sizings), 60)

  def test_match_universal_poker_test_tensors_records_sizings(
      self,
  ):
    """Verifies that the tensor accurately records action sizings."""
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "1 2",  # Note: P0 != SB, see below for detials.
        # WARNING: STACK_SIZES ORDER MUST DIFFER FROM THE VALUES PROVIDED IN THE
        # SETUP FOR CORRESPONDING UNIVERSAL_POKER TEST! (since we don't have an
        # equivalent 'first_player' here to force certain stacks to be sb bb or
        # dealer)
        "stack_sizes": "100 50 100",  # Button, SB, BB
        "antes": "0",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    actions = [
        # Deal-hole-cards chance nodes
        game.card_to_int[Card(ACE, SPADE)],  # P0
        game.card_to_int[Card(KING, DIAMOND)],  # P1
        game.card_to_int[Card(QUEEN, DIAMOND)],  # P2
        game.card_to_int[Card(ACE, CLUB)],  # P0
        game.card_to_int[Card(KING, HEART)],  # P1
        game.card_to_int[Card(QUEEN, HEART)],  # P2
        # Preflop choices are all call or check
        ACTION_CHECK_OR_CALL,
        ACTION_CHECK_OR_CALL,
        ACTION_CHECK_OR_CALL,
        # Flop chance nodes
        game.card_to_int[Card(DEUCE, SPADE)],
        game.card_to_int[Card(SEVEN, HEART)],
        game.card_to_int[Card(EIGHT, DIAMOND)],
        # Flop choice nodes
        ACTION_CHECK_OR_CALL,  # SB checks
        ACTION_CHECK_OR_CALL,  # BB checks
        20,  # Button bets 20
        40,  # SB reraises to 40
        ACTION_CHECK_OR_CALL,  # BB calls 40
        98,  # Button raises to 98 (all-in for 100 total)
        ACTION_CHECK_OR_CALL,  # SB calls 29 (all-in for 50 total)
        ACTION_CHECK_OR_CALL,  # BB calls 98 (all-in for 100 total)
    ]
    for action in actions:
      state.apply_action(action)

    # We have to choose a player since the no-arg default would result in an
    # error due to the game being 'over'... but the choice is arbitrary since
    # the information we're checking is all public knowledge.
    tensor = state.information_state_tensor(1)
    tensor_size = len(tensor)

    # TODO: b/437724266 - implement this method and test it
    # self.assertEqual(tensor_size, game.information_state_tensor_shape())

    number_preceding_actions = 11
    offset = tensor_size - game.max_game_length() + number_preceding_actions

    # Pre-choices on flop: All actions are deal or check
    self.assertEqual(tensor[offset], 0)  # Deal final card of the flop
    self.assertEqual(tensor[offset + 1], 0)  # SB Check
    self.assertEqual(tensor[offset + 2], 0)  # BB Check
    self.assertEqual(tensor[offset + 3], 20)  # Button raise 20
    self.assertEqual(tensor[offset + 4], 40)  # SB reraise 40
    self.assertEqual(tensor[offset + 5], 0)  # BB call 40
    self.assertEqual(tensor[offset + 6], 98)  # Button all-in 98
    self.assertEqual(tensor[offset + 7], 0)  # SB call for 50 (side-pot)
    self.assertEqual(tensor[offset + 8], 0)  # BB call 100

    # No action taken yet, so should default 0
    self.assertEqual(tensor[offset + 10], 0)

    # Verify the final call sizes can instead be obtained from the Observation
    # Tensor (especially the SB's, since it's a side-pot!)

    observer: pokerkit_wrapper.PokerkitWrapperAcpcStyleObserver = (
        game.make_py_observer()
    )
    # Note: player choice doesn't matter since we're looking for public
    # information only (which would be set on the returend tensor for all 3)
    # Note: this function actually _deliberately mutates_ the tensor since
    # PyObserver is designed this way (rather than returning tensors)
    observer.set_from(state, 0)
    ob_tensor = observer.tensor
    ob_tensor_size = len(ob_tensor)

    self.assertEqual(ob_tensor_size, game.observation_tensor_shape()[0])
    # The last game.num_players() elements of the observation tensor are the
    # total bets of each player.
    # NOTE: ORDER DOESN'T MATCH UNIVERSAL_POKER'S! (Button first instead of SB).
    self.assertEqual(ob_tensor[ob_tensor_size - 3], 100)  # P0 Button
    self.assertEqual(ob_tensor[ob_tensor_size - 2], 50)  # P1 SB
    self.assertEqual(ob_tensor[ob_tensor_size - 1], 100)  # P2 BB

    # --- Quick additional tests NOT in the original universal_poker ones ---
    # TODO: b/437724266 - Extract out into another test function
    info_state_string0, info_state_string1, info_state_string2 = (
        state.information_state_string(0),
        state.information_state_string(1),
        state.information_state_string(2),
    )
    [ob_string0, ob_string1, ob_string2] = (
        state.observation_string(0),
        state.observation_string(1),
        state.observation_string(2),
    )
    # Verify each info and observation string contains only the expected
    # player's private cards (i.e. isn't leaking other players' cards)
    self.assertIn("[Player: 0]", info_state_string0)
    self.assertIn("[Player: 0]", ob_string0)
    self.assertIn("[Private:As Ac]", info_state_string0)
    self.assertIn("[Private:As Ac]", ob_string0)
    self.assertNotIn("Kd", info_state_string0)
    self.assertNotIn("Kd", ob_string0)
    self.assertNotIn("Kh", info_state_string0)
    self.assertNotIn("Kh", ob_string0)
    self.assertNotIn("Qd", info_state_string0)
    self.assertNotIn("Qd", ob_string0)
    self.assertNotIn("Qh", info_state_string0)
    self.assertNotIn("Qh", ob_string0)

    self.assertIn("[Player: 1]", info_state_string1)
    self.assertIn("[Player: 1]", ob_string1)
    self.assertIn("[Private:Kd Kh]", info_state_string1)
    self.assertIn("[Private:Kd Kh]", ob_string1)
    self.assertNotIn("As", info_state_string1)
    self.assertNotIn("As", ob_string1)
    self.assertNotIn("Ac", info_state_string1)
    self.assertNotIn("Ac", ob_string1)
    self.assertNotIn("Qd", info_state_string1)
    self.assertNotIn("Qd", ob_string1)
    self.assertNotIn("Qh", info_state_string1)
    self.assertNotIn("Qh", ob_string1)

    self.assertIn("[Player: 2]", info_state_string2)
    self.assertIn("[Player: 2]", ob_string2)
    self.assertIn("[Private:Qd Qh]", info_state_string2)
    self.assertIn("[Private:Qd Qh]", ob_string2)
    self.assertNotIn("As", info_state_string2)
    self.assertNotIn("As", ob_string2)
    self.assertNotIn("Ac", info_state_string2)
    self.assertNotIn("Ac", ob_string2)
    self.assertNotIn("Kd", info_state_string2)
    self.assertNotIn("Kd", ob_string2)
    self.assertNotIn("Kh", info_state_string2)
    self.assertNotIn("Kh", ob_string2)


if __name__ == "__main__":
  absltest.main()
