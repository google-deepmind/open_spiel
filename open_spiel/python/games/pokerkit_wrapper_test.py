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
    variants = pokerkit_wrapper.VARIANT_PARAM_USAGE["bring_in"]
    self.assertIn(pokerkit.FixedLimitRazz.__name__, variants)
    self.assertIn(pokerkit.FixedLimitSevenCardStud.__name__, variants)
    self.assertNotIn(pokerkit.NoLimitTexasHoldem.__name__, variants)

  def test_small_bet_usage(self):
    variants = pokerkit_wrapper.VARIANT_PARAM_USAGE["small_bet"]
    self.assertIn(pokerkit.FixedLimitTexasHoldem.__name__, variants)
    self.assertIn(pokerkit.FixedLimitSevenCardStud.__name__, variants)
    self.assertIn(pokerkit.FixedLimitRazz.__name__, variants)
    self.assertNotIn(pokerkit.NoLimitTexasHoldem.__name__, variants)

  def test_big_bet_usage(self):
    variants = pokerkit_wrapper.VARIANT_PARAM_USAGE["big_bet"]
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

    # NOTE: Because this is preflop the ACPC style actions are identical to the
    # Pokerkit style actions.
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

  def test_universal_poker_port_player_action_mapping_acpc_style(self):
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        "blinds": "10 20",
        "stack_sizes": "2000 2000 2000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    # NOTE: Chance action ints would not actually match universal_poker since
    # pokerkit deals hole cards more 'realistically' than universal_poker. For
    # more details see the # pokerkit_wrapper_acpc_style docstring. (That said,
    # this fact is irrelevant for this test.)
    state.apply_action(game.card_to_int[Card(ACE, CLUB)])
    state.apply_action(game.card_to_int[Card(DEUCE, SPADE)])
    state.apply_action(game.card_to_int[Card(SEVEN, HEART)])
    state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])
    state.apply_action(game.card_to_int[Card(ACE, SPADE)])
    state.apply_action(game.card_to_int[Card(DEUCE, CLUB)])

    state.apply_action(60)  # P2 raises
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls

    # This is the last spot where the allowed actions > 1 should match
    # normal PokerkitWrapper - since on all following streets both players will
    # have contributed chips on this street (and thus ACPC style actions will be
    # different).
    expected_actions = [ACTION_FOLD, ACTION_CHECK_OR_CALL] + list(
        range(100, 2001)
    )
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls

    # Deal flop
    state.apply_action(game.card_to_int[Card(JACK, SPADE)])
    state.apply_action(game.card_to_int[Card(JACK, CLUB)])
    state.apply_action(game.card_to_int[Card(JACK, DIAMOND)])

    # Unlike pokerkit, the actions should start from 60+20=80 instead of 20
    # since it's factoring in all contributions on prior streets - in this case,
    # the preflop contribution of 60 chips from each player.
    # (The +20 on both is because the minimum bet size is one Big Blind.)
    expected_actions = [ACTION_CHECK_OR_CALL] + list(range(80, 2001))
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 checks
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(90)  # P2 bets 30. The ACPC-style action is 90 (30 + 60
    # from pre-flop).

    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls

    # Again, now including the 60 from preflop + 30 from flop, + another 20 for
    # the minimum bet size of one big blind.
    expected_actions = [ACTION_CHECK_OR_CALL] + list(range(110, 2001))
    state.apply_action(game.card_to_int[Card(JACK, HEART)])
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 checks
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P2 checks

    # No additional bets, so same as the turn at first...
    state.apply_action(game.card_to_int[Card(TREY, CLUB)])
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 checks
    self.assertEqual(state.legal_actions(), expected_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
    self.assertEqual(state.legal_actions(), expected_actions)

    # ... Until P2 goes nearly all-in
    state.apply_action(1999)  # P2 bets everything except for one chip

    # Now P0 can only fold, call, or shove all-in.
    self.assertEqual(
        state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 2000]
    )
    state.apply_action(2000)  # P0 shoves

    # And so finally P1 and P2 can only fold or call.
    self.assertEqual(state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL])
    state.apply_action(ACTION_FOLD)  # P1 folds
    self.assertEqual(state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL])
    state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls

    self.assertTrue(state.is_terminal())
    # P0 wins with JJJJ Ace-kicker vs P3 JJJJ 7-kicker
    # - P0 gained 90 from P1 + 2000 from P2.
    # - P1 lost the 90 they had contributed before folding.
    # - P2 lost all 2000.
    self.assertEqual(state.returns(), [2090.0, -90.0, -2000.0])

  def test_universal_poker_port_full_nl_betting_test1(self):
    """Ported from universal_poker_test.cc FullNLBettingTest1."""
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 2,
        "blinds": "1 2",
        "stack_sizes": "20 20",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    self.assertEqual(game.num_distinct_actions(), 21)

    # Deal hole cards
    # NOTE: Will not actually match universal_poker since pokerkit deals hole
    # cards more 'realistically' than universal_poker (see the
    # pokerkit_wrapper_acpc_style docstring for more details. That said, this
    # shouldn't matter much for the test - both players are still going to end
    # up getting 2s, just with different suits, which isn't relevant given the
    # board runout being four 3s and one 4).
    for _ in range(4):
      state.apply_action(state.chance_outcomes()[0][0])

    # Check valid raise actions, smallest valid raise is double the big blind.
    legal_actions = state.legal_actions()
    self.assertNotIn(3, legal_actions)
    for i in range(4, 21):
      self.assertIn(i, legal_actions)
    self.assertNotIn(21, legal_actions)

    state.apply_action(ACTION_CHECK_OR_CALL)  # call big blind
    state.apply_action(ACTION_CHECK_OR_CALL)  # check big blind

    # Deal flop
    for _ in range(3):
      state.apply_action(state.chance_outcomes()[0][0])

    # Check valid raise actions again.
    legal_actions = state.legal_actions()
    self.assertNotIn(3, legal_actions)
    for i in range(4, 21):
      self.assertIn(i, legal_actions)
    self.assertNotIn(21, legal_actions)

    # Each player keeps min raising until one is all in.
    for i in range(4, 21, 2):
      state.apply_action(i)

    state.apply_action(ACTION_CHECK_OR_CALL)  # call last raise

    # Deal turn and river
    state.apply_action(state.chance_outcomes()[0][0])
    state.apply_action(state.chance_outcomes()[0][0])

    # Hand is a draw with deterministic card deal.
    self.assertEqual(state.returns()[0], 0.0)
    self.assertEqual(state.returns()[1], 0.0)

    # TODO: b/444333187 - test the PHH looks as expected to mimic the C++ test
    # this was ported from.

  def test_universal_poker_port_full_nl_betting_test2(self):
    """Ported from universal_poker_test.cc FullNLBettingTest2."""
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 2,
        "blinds": "50 100",
        "stack_sizes": "10000 10000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    self.assertEqual(game.num_distinct_actions(), 10001)

    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])  # Deal hole cards

    # Check valid raise actions
    legal_actions = state.legal_actions()
    self.assertNotIn(199, legal_actions)
    for i in range(200, 10001):
      self.assertIn(i, legal_actions)
    self.assertNotIn(10001, legal_actions)

    state.apply_action(5100)  # Bet just over half stack.
    # Raise must double the size of the bet.
    # Only legal actions are fold, call, all-in.
    self.assertCountEqual(
        state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 10000]
    )

    state.apply_action(ACTION_CHECK_OR_CALL)

    # Deal flop
    for _ in range(3):
      state.apply_action(state.chance_outcomes()[0][0])

    # New round of betting, min bet is big blind.
    # Total commitment pre-flop was 5100. Min-bet is 100 on this street, so
    # ACPC-style action is 5100 + 100 = 5200.
    legal_actions = state.legal_actions()
    self.assertNotIn(5199, legal_actions)
    for i in range(5200, 10001):
      self.assertIn(i, legal_actions)

    state.apply_action(5200)  # Min bet

    # Now we can raise by at least the big blind.
    legal_actions = state.legal_actions()
    for i in range(5300, 10001):
      self.assertIn(i, legal_actions)

    state.apply_action(ACTION_CHECK_OR_CALL)  # Call.

    # Deal turn.
    state.apply_action(state.chance_outcomes()[0][0])

    state.apply_action(5400)  # Bet 2 big blinds (200). 5200 + 200 = 5400
    state.apply_action(5600)  # Raise to 4 big blinds (400).
    state.apply_action(5900)  # Reraise to 7 big blinds.
    # Now a reraise must increase by at least 3 more big blinds.

    legal_actions = state.legal_actions()
    self.assertNotIn(6199, legal_actions)
    for i in range(6200, 10001):
      self.assertIn(i, legal_actions)
    state.apply_action(ACTION_CHECK_OR_CALL)  # Call.

    # Deal river
    state.apply_action(state.chance_outcomes()[0][0])

    # New round of betting so we can bet as small as one BB (100).
    legal_actions = state.legal_actions()
    self.assertNotIn(5999, legal_actions)
    for i in range(6000, 10001):
      self.assertIn(i, legal_actions)

    state.apply_action(10000)  # All-in
    state.apply_action(ACTION_FOLD)
    self.assertEqual(state.returns(), [5900.0, -5900.0])

  def test_universal_poker_port_full_nl_betting_test3(self):
    """Ported from universal_poker_test.cc FullNLBettingTest3."""
    # NOTE: universal_poker uses atypical turn order and blind structures
    # that pokerkit does not support. This test is adapted to standard poker
    # rules.
    params = {
        "variant": "NoLimitTexasHoldem",
        "num_players": 3,
        # NOTE: Cannot match C++ test which was backwards (BB SB 2111 order).
        # But, in practice this doesn't actaully matter.
        "blinds": "50 100",
        "stack_sizes": "500 1000 2000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()

    # Pokerkit deals players more 'realistically' than universal_poker, i.e.
    # each player gets their first hole card before any player gets their second
    # hole card. As such, to match universal_poker we have to deal the cards in
    # a specific order ourselves (rather than 'all 2c 2h 2d 2s 3c 3d').
    expected_hole_cards = [
        Card(DEUCE, CLUB),
        Card(DEUCE, HEART),
        Card(TREY, CLUB),
        Card(DEUCE, DIAMOND),
        Card(DEUCE, SPADE),
        Card(TREY, DIAMOND),
    ]
    for card in expected_hole_cards:
      state.apply_action(game.card_to_int[card])
      self.assertLess(game.card_to_int[card], 7)

    # Preflop
    state.apply_action(ACTION_CHECK_OR_CALL)  # P2 (BTN) calls big blind
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 (SB) calls big blind
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 (BB) checks

    # Here+below, additionally verifying that card actions match the C++ test.
    # (Which they should since )
    # Deal flop
    expected_flop_cards = [
        Card(TREY, HEART),
        Card(TREY, SPADE),
        Card(FOUR, CLUB),
    ]
    while state.is_chance_node():
      flop_card = state.legal_actions()[0]
      self.assertEqual(flop_card, game.card_to_int[expected_flop_cards.pop(0)])
      state.apply_action(flop_card)

    def check_valid_actions_range_from(min_raise, max_raise_exclusive):
      self.assertNotIn(min_raise - 1, state.legal_actions())
      for i in range(min_raise, max_raise_exclusive):
        self.assertIn(i, state.legal_actions())
      self.assertNotIn(max_raise_exclusive, state.legal_actions())

    # Postflop assert all raise increments are valid
    check_valid_actions_range_from(200, 501)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P0 (SB) checks

    check_valid_actions_range_from(200, 1001)
    state.apply_action(ACTION_CHECK_OR_CALL)  # P1 (BB) checks

    check_valid_actions_range_from(200, 2001)
    state.apply_action(200)  # P2 (BTN) min raises

    check_valid_actions_range_from(300, 501)
    state.apply_action(500)  # SB short stack goes all-in

    check_valid_actions_range_from(800, 1001)
    state.apply_action(800)  # BB min raises

    check_valid_actions_range_from(1000, 2001)
    state.apply_action(2000)  # BTN all-in

    # Can only check or call
    self.assertEqual(state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL])
    state.apply_action(ACTION_CHECK_OR_CALL)  # call

    turn_and_river_cards = [
        Card(FOUR, DIAMOND),
        Card(FOUR, HEART),
    ]
    while state.is_chance_node():
      card = state.legal_actions()[0]
      self.assertEqual(card, game.card_to_int[turn_and_river_cards.pop(0)])
      state.apply_action(card)

    self.assertEqual(state.returns(), [-500.0, -1000.0, 1500.0])

  def test_to_string_at_chance_nodes(self):
    params = {
        "variant": "FixedLimitTexasHoldem",
        "num_players": 2,
        "blinds": "5 10",
        "small_bet": 10,
        "big_bet": 20,
        "stack_sizes": "20000 20000",
    }
    game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
    state = game.new_initial_state()
    # NOTE: order is shuffled to match universal_poker since pokerkit deals hole
    # cards more 'realistically' than universal_poker. (See the
    # pokerkit_wrapper_acpc_style docstring for more details.)
    hole_card_action_sequence = [10, 12, 11, 13]
    for action in hole_card_action_sequence:
      action_string = state.action_to_string(pyspiel.PlayerId.CHANCE, action)
      # Print mainly to match universal_poker_test.cc. (Though, it IS also nice
      # to prove that this did actually give us something we can actually use
      # here.)
      print("Applying action" f" ({action_string})")
      state.apply_action(action)

    # If we've reached this point without crashing / without the game breaking
    # then we've succeeded! (So this asssert isn't technically necessary ...
    # though it is nice to have to verify the game is still working fine here.)
    self.assertFalse(state.is_terminal())


if __name__ == "__main__":
  absltest.main()
