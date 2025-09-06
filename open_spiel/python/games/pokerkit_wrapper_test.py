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
FOLD_AND_CHECK_OR_CALL_ACTIONS = [ACTION_FOLD, ACTION_CHECK_OR_CALL]


class PokerkitWrapperTest(absltest.TestCase):
  """Test the OpenSpiel game wrapper for Pokerkit."""

  # TODO: b/437724266 - add additional tests for the following:
  # - sidepot handling in general
  # - sidepots where the next bet will be exactly 1 chip (if possible?)
  # - all-ins with exactly 1 chip left in stack
  # - all-ins in simple multiway situations with varying stack sizes
  # - all-ins in fixed-limit games with sidepots
  # - shortdeck ("6+") poker card dealing
  # - razz poker hand evaluation working as expected / differing from seven card
  #   stud
  # - seven card stud gameplay actually different from texas holdem, including
  #   hand evaluation
  # - hand mucking / displaying unmucked hands that reach showdown properly
  # - censoring private hole cards in the observer strings

  # --- Lightweight testing to verify that the 'usage' directories are at least
  # _somewhat_ correct. (Not intended to be exhaustive!) ---

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

  # --- More intensive / 'typical' testing ---
  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_pokerkit_wrapper")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_playthrough_robustness(self):
    """Runs full random playthroughs for various games to ensure stability."""
    scenarios = [
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 3,
            "blinds": "5 10",
            "stack_sizes": "100 200 300",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 2,
            "blinds": "50 100",
            "stack_sizes": "20000 20000",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 2,
            "blinds": "1 2",
            "stack_sizes": "20 100",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 4,
            "blinds": "50 100",
            "stack_sizes": "20 20 100 400",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 2,
            "blinds": "50 100",  # Both players forced all-in from blinds
            "stack_sizes": "20 20",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 5,
            "blinds": "50 100",
            "stack_sizes": "400 100 1000 300 400",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 4,
            "blinds": "50 100",
            "stack_sizes": "20 20 100 400",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "50 100",
            "stack_sizes": "20 20",
        },
        # Encourage formation of side-pots
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 4,
            "blinds": "5 10",
            "stack_sizes": "200 200 30 80",
        },
        # --- Additional games not supported in universal_poker ---
        {
            "variant": "FixedLimitRazz",
            "num_players": 4,
            "stack_sizes": "200 200 183 190",
            "bring_in": 5,
            "small_bet": 10,
            "big_bet": 20,
        },
        {
            "variant": "FixedLimitRazz",
            "num_players": 4,
            "stack_sizes": "200 200 183 190",
            "bring_in": 500,
            "small_bet": 1000,
            "big_bet": 2000,
        },
        {
            "variant": "FixedLimitSevenCardStud",
            "num_players": 4,
            "stack_sizes": "200 200 189 164",
            "bring_in": 5,
            "small_bet": 10,
            "big_bet": 20,
        },
        {
            "variant": "FixedLimitSevenCardStud",
            "num_players": 4,
            "stack_sizes": "200 600 49 54",
            "bring_in": 50,
            "small_bet": 100,
            "big_bet": 200,
        },
        {
            "variant": "PotLimitOmahaHoldem",
            "num_players": 4,
            "blinds": "5 10",
            "stack_sizes": "198 189 200 200",
        },
        {
            "variant": "PotLimitOmahaHoldem",
            "num_players": 4,
            "blinds": "500 1000",
            "stack_sizes": "2098 189 600 200",
        },
        {
            "variant": "NoLimitShortDeckHoldem",
            "num_players": 3,
            "blinds": "5 10",
            "stack_sizes": "200 200 200",
        },
        {
            "variant": "NoLimitShortDeckHoldem",
            "num_players": 3,
            "blinds": "190 380",
            "stack_sizes": "200 200 200",
        },
        # --- Bet size edge cases ---
        # Encourage situations where player are likely to be betting 1 chip
        # to verify that we gracefully handle any edge cases.
        # TODO: b/437724266 - split out into separate tests just for this edge
        # case. Also run multiple times (fail on any failyres) and/or
        # hand-choose bet sizes to force these edge cases.
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 9,
            "blinds": "1 2",
            "stack_sizes": "100 100 1 2 1 3 3 3 3",
        },
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 9,
            "blinds": "1 2",
            "stack_sizes": "3 3 3 3 3 3 3 3 3",
        },
    ]

    for params in scenarios:
      with self.subTest(variant=params["variant"]):
        game = pyspiel.load_game("python_pokerkit_wrapper", params)
        state = game.new_initial_state()
        while not state.is_terminal():
          if state.is_chance_node():
            action = random.choice([o for o, p in state.chance_outcomes()])
          else:
            action = random.choice(state.legal_actions())
          state.apply_action(action)
        self.assertAlmostEqual(sum(state.returns()), 0.0)

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

  def test_min_bet_and_check_call_amounts_nolimit_holdem(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 2,
        "blinds": "9 17",
    }
    game = PokerkitWrapper(params)
    state = game.new_initial_state()
    wrapped_state: pokerkit.State = state._wrapped_state

    while state.is_chance_node():
      state.apply_action(random.choice([o for o, _ in state.chance_outcomes()]))
    # Minimum bet should be two big blinds
    self.assertEqual(
        wrapped_state.min_completion_betting_or_raising_to_amount, 34
    )
    # Verify that this player was SB and needs to contribute more to call.
    self.assertEqual(wrapped_state.checking_or_calling_amount, 8)
    state.apply_action(ACTION_CHECK_OR_CALL)
    # Again, minimum bet should be two big blinds
    self.assertEqual(
        wrapped_state.min_completion_betting_or_raising_to_amount, 34
    )
    # Verify that this player was BB and can check back.
    self.assertEqual(wrapped_state.checking_or_calling_amount, 0)

  def test_observer_string_contains_phh_for_all_game_variants_iff_perfect_recall(
      self,
  ):
    scenarios = [
        {
            "variant": "NoLimitTexasHoldem",
            "num_players": 3,
            "blinds": "5 10",
            "stack_sizes": "100 200 300",
        },
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 5,
            "blinds": "50 100",
            "stack_sizes": "400 100 1000 300 400",
        },
        {
            "variant": "FixedLimitRazz",
            "num_players": 4,
            "stack_sizes": "200 200 183 190",
            "bring_in": 5,
            "small_bet": 10,
            "big_bet": 20,
        },
        {
            "variant": "FixedLimitSevenCardStud",
            "num_players": 4,
            "stack_sizes": "200 200 189 164",
            "bring_in": 5,
            "small_bet": 10,
            "big_bet": 20,
        },
        {
            "variant": "PotLimitOmahaHoldem",
            "num_players": 4,
            "blinds": "5 10",
            "stack_sizes": "198 189 200 200",
        },
        {
            "variant": "NoLimitShortDeckHoldem",
            "num_players": 3,
            "blinds": "5 10",
            "stack_sizes": "200 200 200",
        },
    ]

    for params in scenarios:
      with self.subTest(variant=params["variant"]):
        game = PokerkitWrapper(params)
        state = game.new_initial_state()
        while not state.is_terminal():
          if state.is_chance_node():
            action = random.choice([o for o, _ in state.chance_outcomes()])
          else:
            action = random.choice(state.legal_actions())
          state.apply_action(action)
        information_state_observer = game.make_py_observer(
            pyspiel.IIGObservationType(perfect_recall=True)
        )
        for player in range(game.num_players()):
          information_state_string = information_state_observer.string_from(
              state, player
          )
          pieces = information_state_string.split("||")
          # pylint: disable=line-too-long
          #
          # Assumed to look roughly like one of the following:
          #
          # PHH Actions: d dh p1 6hKs,d dh p2 ????,d dh p3 ????,p3 cbr 254,p1 cc,p2 f,p3 sm Jh5s,p1 sm 6hKs,d db 6cJdKd4sQs
          # PHH Actions: d dh p1 3dJcKd,d dh p2 ??????,d dh p3 ??????,d dh p4 ??????,p4 cbr 10,p1 cbr 20,p2 cc,p3 cbr 30,p4 f,p1 f,p2 cc,d dh p2 ??,d dh p3 ??,p3 cbr 10,p2 cc,d dh p2 ??,d dh p3 ??,p3 cbr 20,p2 f
          #
          # pylint: enable=line-too-long
          possible_phh = [s for s in pieces if s.startswith("PHH Actions:")]
          self.assertLen(possible_phh, 1)
          phh = possible_phh[0]
          self.assertIn("d dh p1", phh)
          self.assertIn(",d dh p2", phh)

          # Make sure that the observer doesn't include PHH actions in the
          # observation string if perfect_recall is False.
          #
          # (Naming is a little weird here due to how pyspiel expects the
          # information state AND the observation tensors/strings to come out of
          # the same 'Observer' class/methods, as opposed to the C++ code which
          # has this all  more separated out.)
          observer_observer = game.make_py_observer(
              pyspiel.IIGObservationType(perfect_recall=False)
          )
          observation_string = observer_observer.string_from(state, player)
          self.assertEmpty([
              s
              for s in observation_string.split("||")
              if s.startswith("PHH Actions:")
          ])

  def test_card_burning_is_disabled_everywhere(self):
    obj = PokerkitWrapper()
    fresh_isolated_state: pokerkit.State = obj.wrapped_state_factory()
    for street in fresh_isolated_state.streets:
      self.assertFalse(street.card_burning_status)

    wrapped_state_copy = obj.new_initial_state().deepcopy_wrapped_state()
    for street in wrapped_state_copy.streets:
      self.assertFalse(street.card_burning_status)

  def test_utility_equal_stacks_multiway(self):
    scenarios = [
        {
            "params": {
                "variant": "NoLimitTexasHoldem",
                "num_players": 3,
                "blinds": "50 100",
                "stack_sizes": "200 200 200",
            },
            "expected_max_utility": 400.0,
            "expected_min_utility": -200.0,
        },
        {
            "params": {
                "variant": "NoLimitTexasHoldem",
                "num_players": 6,
                "blinds": "50 100",
                "stack_sizes": "6 6 6 6 6 6",
            },
            "expected_max_utility": 30.0,
            "expected_min_utility": -6.0,
        },
        {
            "params": {
                "variant": "NoLimitTexasHoldem",
                "num_players": 6,
                "blinds": "50 100",
                "stack_sizes": "10000 10000 10000 10000 10000 10000",
            },
            "expected_max_utility": 50000.0,
            "expected_min_utility": -10000.0,
        },
    ]

    for scenario in scenarios:
      with self.subTest(params=scenario["params"]):
        game = PokerkitWrapper(params=scenario["params"])
        self.assertEqual(game.max_utility(), scenario["expected_max_utility"])
        self.assertEqual(game.min_utility(), scenario["expected_min_utility"])

  def test_utility_unequal_stacks_headsup(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 2,
        "blinds": "50 100",
        "stack_sizes": "300 200",
    }
    game = PokerkitWrapper(params=params)
    self.assertEqual(game.max_utility(), 200.0)
    # min_utility is technically just upper bound anyways (in magnitude), so it
    # doesn't have to be exact. But, whatever this bound is, it at least needs
    # to avoid *underestimating* how many chips could be lost here.
    self.assertLessEqual(game.min_utility(), -200.0)
    # Even if it technically *could* be even less tight (e.g. -500 is still
    # 'correct'), in practice it'd be concerning for this to be any lower given
    # how easy it is to reason that the minimum utility will be no 'larger' (in
    # magnitude) than the largest stack's size.
    self.assertGreaterEqual(game.min_utility(), -300.0)

  def test_utility_unequal_stacks_multiway(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 4,
        "blinds": "50 100",
        "stack_sizes": "300 200 300 400",
    }
    game = PokerkitWrapper(params=params)
    self.assertEqual(game.max_utility(), 800.0)
    # min_utility is technically just upper bound anyways (in magnitude), so it
    # doesn't have to be exact. But, whatever this bound is, it at least needs
    # to avoid *underestimating* how many chips could be lost here.
    self.assertLessEqual(game.min_utility(), -300.0)
    # Even if it technically *could* be even less tight (e.g. -500 is still
    # 'correct'), in practice it'd be concerning for this to be any lower given
    # how easy it is to reason that the minimum utility will be no 'larger' (in
    # magnitude) than the largest stack's size.
    self.assertGreaterEqual(game.min_utility(), -400.0)

  def test_returns_correct_legal_actions(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "10 20",
        "stack_sizes": "2000 2000 2000",
    }
    game = PokerkitWrapper(params)
    state = game.new_initial_state()
    state.apply_action(game.card_to_int[Card(ACE, CLUB)])
    state.apply_action(game.card_to_int[Card(DEUCE, SPADE)])
    state.apply_action(game.card_to_int[Card(SEVEN, HEART)])
    state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])
    state.apply_action(game.card_to_int[Card(ACE, SPADE)])
    state.apply_action(game.card_to_int[Card(DEUCE, CLUB)])
    # NOTE: 40, not 20, because 20 would be a *call* (not a bet). And the
    # minimum raise size indeed should be one big-blind over the call amount,
    # i.e. 20 + 20 => 40.
    expected_actions = [0, 1] + list(range(40, 2001))
    self.assertEqual(state.legal_actions(), expected_actions)
    # Double check that the call action was indeed one big blind less
    self.assertIn(
        "(20)",
        state._action_to_string(state.current_player(), ACTION_CHECK_OR_CALL),
    )
    # Check the wrapped pokerkit state agrees with the returned legal actions.
    # (Accessing the ._wrapped_state directly is a bit of a hack, but in
    # practice this is unlikely to actually be a problem).
    for a in expected_actions:
      if a not in FOLD_AND_CHECK_OR_CALL_ACTIONS:
        self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to(a))

    state.apply_action(60)  # P2 raises
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls

    # P1 can call (0), fold (1), or reraise. The reraise increase the bet by at
    # least the amount of the last raise, which was (60-20) = 40, meaning it
    # must be at least a raise to 60 + 40 = 100.
    expected_actions = [0, 1] + list(range(100, 2001))
    self.assertEqual(state.legal_actions(), expected_actions)
    for a in expected_actions:
      if a not in FOLD_AND_CHECK_OR_CALL_ACTIONS:
        self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to(a))

  def test_single_chip_shove_is_mapped_to_next_nonreserved_action(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "10 20",
        "stack_sizes": "21 21 21",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper", params)
    state = game.new_initial_state()
    wrapped_state: pokerkit.State = state._wrapped_state
    while state.is_chance_node():
      state.apply_action(random.choice([o for o, _ in state.chance_outcomes()]))
    state.apply_action(ACTION_CHECK_OR_CALL)
    state.apply_action(ACTION_CHECK_OR_CALL)
    state.apply_action(ACTION_CHECK_OR_CALL)
    while state.is_chance_node():
      state.apply_action(random.choice([o for o, _ in state.chance_outcomes()]))
    self.assertEqual(wrapped_state.stacks, [1, 1, 1])
    # P0 can either check or shove for 1 chip, hitting an edge case that needs
    # to be handled correctly (mapping 1 chip shove to action 2).
    # (P0 cannot fold since pokerkit doesn't allow folding when not actually
    # facing an opponent's bet.)
    self.assertEqual(
        state._legal_actions(state._wrapped_state.actor_index),
        [ACTION_CHECK_OR_CALL, 2],
    )
    self.assertIn("[ALL-IN EDGECASE]", state._action_to_string(0, 2))
    self.assertIn("Bet/Raise to 1", state._action_to_string(0, 2))
    state.apply_action(2)

    # Verify that the 2 actually mapped to a bet of 1 chip / that the next
    # players get entirely normal fold/check_or_call actions.
    self.assertEqual(wrapped_state.stacks, [0, 1, 1])
    self.assertEqual(
        state._legal_actions(state._wrapped_state.actor_index),
        [ACTION_FOLD, ACTION_CHECK_OR_CALL],
    )
    self.assertEqual(
        state._action_to_string(
            wrapped_state.actor_index, ACTION_CHECK_OR_CALL
        ),
        "Call(1)",
    )
    state.apply_action(ACTION_FOLD)

    self.assertEqual(wrapped_state.stacks, [0, 1, 1])
    self.assertEqual(
        state._legal_actions(state._wrapped_state.actor_index),
        [ACTION_FOLD, ACTION_CHECK_OR_CALL],
    )
    self.assertEqual(
        state._action_to_string(
            wrapped_state.actor_index, ACTION_CHECK_OR_CALL
        ),
        "Call(1)",
    )


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
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls

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
