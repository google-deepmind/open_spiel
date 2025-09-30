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

  def test_action_mapping_acpc_style(self):
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

if __name__ == "__main__":
  absltest.main()
