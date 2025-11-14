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
  import pyspiel
  # pylint: disable=g-import-not-at-top
  IMPORTED_ALL_LIBRARIES = True
except ImportError as e:
  logging.error("Failed to import needed pokerkit libraries: %s", e)
  pokerkit_wrapper = None
  pokerkit = None
  pyspiel = None

if IMPORTED_ALL_LIBRARIES:
  assert pokerkit is not None
  assert pokerkit_wrapper is not None
  assert pyspiel is not None

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

  class PokerkitWrapperTest(parameterized.TestCase):
    """Test the OpenSpiel game wrapper for Pokerkit."""

    # --- Lightweight testing to verify that the 'usage' directories are at
    # least _somewhat_ correct. (Not intended to be exhaustive!) ---
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
          pyspiel.random_sim_test(
              game, num_sims=3, serialize=False, verbose=True
          )

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

    def test_shortdeck_deck(self):
      obj = PokerkitWrapper(params={"variant": "NoLimitShortDeckHoldem"})
      self.assertEqual(obj.deck_size, 36)
      self.assertLen(obj.card_to_int, 36)
      self.assertLen(obj.int_to_card, 36)
      self.assertIn(Card(SIX, CLUB), obj.card_to_int)
      self.assertNotIn(Card(FIVE, CLUB), obj.card_to_int)
      self.assertNotIn(Card(DEUCE, CLUB), obj.card_to_int)

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
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
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
            # information state AND the observation tensors/strings to come out
            # of the same 'Observer' class/methods, as opposed to the C++ code
            # which has this all  more separated out.)
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

    def test_default_state_struct_values(self):
      game = pyspiel.load_game("python_pokerkit_wrapper")
      state = game.new_initial_state()
      # The first state is a chance node dealing cards.
      self.assertTrue(state.is_chance_node())
      state_struct = state.to_struct()
      self.assertIsInstance(
          state_struct, pyspiel.pokerkit_wrapper.PokerkitStateStruct
      )
      self.assertEqual(state_struct.current_player, pyspiel.PlayerId.CHANCE)
      self.assertFalse(state_struct.is_terminal)
      self.assertIsInstance(state_struct.observation, list)
      self.assertLen(state_struct.observation, game.num_players())
      self.assertIn("Player: 0", state_struct.observation[0])
      self.assertTrue(state_struct.legal_actions)
      self.assertLen(state_struct.stacks, game.num_players())
      self.assertLen(state_struct.bets, game.num_players())
      self.assertIsInstance(state_struct.board_cards, list)
      self.assertIsInstance(state_struct.hole_cards, list)
      self.assertIsInstance(state_struct.pots, list)
      self.assertIsInstance(state_struct.burn_cards, list)
      self.assertIsInstance(state_struct.mucked_cards, list)
      self.assertEqual(state_struct.per_player_acpc_logs, [[], []])
      self.assertEqual(state_struct.betting_history, "")
      self.assertLen(state_struct.blinds, 2)
      self.assertLen(state_struct.player_contributions, game.num_players())
      self.assertGreaterEqual(state_struct.pot_size, 0)
      self.assertLen(state_struct.starting_stacks, game.num_players())

    def test_create_struct_and_access_members(self):
      """Tests direct creation of the underlying pyspiel PokerkitStateStruct.

      This test could technically go elsewhere as it's not really testing the
      actual pokerkit_wrapper code, but this is a very convenient place to put
      it.
      (And, it is useful to have our tests clearly show if the underlying
      StateStruct we're using has any problems.)
      """
      state_struct = pyspiel.pokerkit_wrapper.PokerkitStateStruct()

      # Test default values (or some initial state if constructor sets them)
      self.assertEqual(
          state_struct.current_player, 0
      )  # Or whatever the C++ default is
      self.assertFalse(state_struct.is_terminal)
      self.assertEqual(state_struct.observation, [])

      # Modify members from Python
      state_struct.observation = ["obs1", "obs2"]
      state_struct.legal_actions = [0, 2, 4]
      state_struct.current_player = 1
      state_struct.is_terminal = True
      state_struct.stacks = [100, 200]
      state_struct.bets = [10, 0]
      state_struct.board_cards = [5, 6, 7]
      state_struct.hole_cards = [[1, 2], [3, 4]]
      state_struct.per_player_phh_actions = [["phh1"], ["phh2"]]
      state_struct.per_player_acpc_logs = [[["S->", "MATCHSTATE:blah"]]]
      state_struct.blinds = [1, 2]
      state_struct.betting_history = "betting_history_test"
      state_struct.player_contributions = [10, 0]
      state_struct.pot_size = 10
      state_struct.starting_stacks = [100, 200]

      # Verify modifications
      self.assertEqual(state_struct.observation, ["obs1", "obs2"])
      self.assertEqual(state_struct.legal_actions, [0, 2, 4])
      self.assertEqual(state_struct.current_player, 1)
      self.assertTrue(state_struct.is_terminal)
      self.assertEqual(state_struct.stacks, [100, 200])
      self.assertEqual(state_struct.bets, [10, 0])
      self.assertEqual(state_struct.board_cards, [5, 6, 7])
      self.assertEqual(state_struct.hole_cards, [[1, 2], [3, 4]])
      self.assertEqual(
          state_struct.per_player_phh_actions, [["phh1"], ["phh2"]]
      )
      self.assertEqual(
          state_struct.per_player_acpc_logs, [[["S->", "MATCHSTATE:blah"]]]
      )
      self.assertEqual(state_struct.blinds, [1, 2])
      self.assertEqual(state_struct.betting_history, "betting_history_test")
      self.assertEqual(state_struct.player_contributions, [10, 0])
      self.assertEqual(state_struct.pot_size, 10)
      self.assertEqual(state_struct.starting_stacks, [100, 200])

    def test_pokerkit_wrapper_state_to_struct_and_json(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 2,
          "blinds": "50 100",
          "stack_sizes": "1000 1000",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper", params)
      state = game.new_initial_state()

      # Deal hole cards
      state.apply_action(game.card_to_int[Card(ACE, CLUB)])  # P0 c1
      state.apply_action(game.card_to_int[Card(ACE, DIAMOND)])  # P1 c1
      state.apply_action(game.card_to_int[Card(KING, CLUB)])  # P0 c2
      state.apply_action(game.card_to_int[Card(KING, DIAMOND)])  # P1 c2
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Deal flop
      state.apply_action(game.card_to_int[Card(QUEEN, CLUB)])
      state.apply_action(game.card_to_int[Card(JACK, CLUB)])
      state.apply_action(game.card_to_int[Card(TEN, CLUB)])

      state.apply_action(200)  # BB raises to 200 post-flop
      state.apply_action(500)  # SB/BTN re-raises to 500
      state.apply_action(800)  # BB re-re-raises to 800

      state_struct = state.to_struct()
      json_str = state.to_json()
      self.assertEqual(json_str, state_struct.to_json())
      self.assertIsInstance(json_str, str)

      try:
        data = json.loads(json_str)
      except json.JSONDecodeError as e:
        self.fail(f"Failed to parse JSON {json_str}, error was: {e}")

      self.assertEqual(state_struct.current_player, 1)
      self.assertEqual(data["current_player"], 1)

      self.assertEqual(state_struct.is_terminal, False)
      self.assertFalse(data["is_terminal"])

      self.assertEqual(state_struct.stacks, [100, 400])
      self.assertEqual(data["stacks"], [100, 400])

      self.assertEqual(state_struct.bets, [800, 500])
      self.assertEqual(data["bets"], [800, 500])

      self.assertEqual(
          state_struct.board_cards,
          [
              game.card_to_int[c]
              for c in [Card(QUEEN, CLUB), Card(JACK, CLUB), Card(TEN, CLUB)]
          ],
      )
      self.assertLen(data["board_cards"], 3)

      board_cards_json = [game.int_to_card[c] for c in data["board_cards"]]
      board_cards_struct = [
          game.int_to_card[c] for c in state_struct.board_cards
      ]
      self.assertEqual(
          board_cards_json,
          [Card(QUEEN, CLUB), Card(JACK, CLUB), Card(TEN, CLUB)],
      )
      self.assertEqual(board_cards_struct, board_cards_json)

      self.assertEqual(
          state_struct.hole_cards,
          [
              [
                  game.card_to_int[c]
                  for c in [Card(ACE, CLUB), Card(KING, CLUB)]
              ],
              [
                  game.card_to_int[c]
                  for c in [Card(ACE, DIAMOND), Card(KING, DIAMOND)]
              ],
          ],
      )
      self.assertLen(data["hole_cards"], 2)
      p0_hole_cards = [game.int_to_card[c] for c in data["hole_cards"][0]]
      p1_hole_cards = [game.int_to_card[c] for c in data["hole_cards"][1]]
      self.assertEqual(p0_hole_cards, [Card(ACE, CLUB), Card(KING, CLUB)])
      self.assertEqual(p1_hole_cards, [Card(ACE, DIAMOND), Card(KING, DIAMOND)])

      self.assertEqual(state_struct.pots, [200])
      self.assertLen(data["pots"], 1)
      self.assertEqual(data["pots"][0], 200)

      self.assertEqual(state_struct.blinds, [50, 100])
      self.assertEqual(data["blinds"], [50, 100])
      self.assertEqual(state_struct.starting_stacks, [1000, 1000])
      self.assertEqual(data["starting_stacks"], [1000, 1000])
      self.assertEqual(state_struct.player_contributions, [900, 600])
      self.assertEqual(data["player_contributions"], [900, 600])
      self.assertEqual(state_struct.pot_size, 1500)
      self.assertEqual(data["pot_size"], 1500)

      self.assertLen(state_struct.per_player_phh_actions, 2)
      # Per-player view of P1's hole cards (P1 uncensored, P2 censored)
      self.assertEqual(
          state_struct.per_player_phh_actions[0][0], "d dh p1 AcKc"
      )
      self.assertEqual(
          state_struct.per_player_phh_actions[1][0], "d dh p1 ????"
      )
      # Per-player view of P2's hole cards (P1 censored, P2 uncensored)
      self.assertEqual(
          state_struct.per_player_phh_actions[0][1], "d dh p2 ????"
      )
      self.assertEqual(
          state_struct.per_player_phh_actions[1][1], "d dh p2 AdKd"
      )

      self.assertNotEmpty(state_struct.per_player_acpc_logs)
      self.assertNotEmpty(data["per_player_acpc_logs"])
      self.assertEqual(
          state_struct.per_player_acpc_logs, data["per_player_acpc_logs"]
      )
      self.assertLen(state_struct.per_player_acpc_logs, 2)

      # 200 500 800 for total contributions of 300 600 and 900 (after including
      # the preflop big-blind both players put in when check_or_call-ing)
      self.assertEqual(state_struct.betting_history, "cc/r200r500r800")

      # P0's view, right before the last bet. This uses the total contributions
      # so the numbers are expected to differ from above.
      self.assertEqual(
          state_struct.per_player_acpc_logs[0][-3:],
          [
              ["S->", "MATCHSTATE:0:0:cc/r300r600:AcKc|/QcJcTc\r\n"],
              ["<-C", "MATCHSTATE:0:0:cc/r300r600:AcKc|/QcJcTc:r900\r\n"],
              ["S->", "MATCHSTATE:0:0:cc/r300r600r900:AcKc|/QcJcTc\r\n"],
          ],
      )

      # P1's view, right before the last bet
      self.assertEqual(
          state_struct.per_player_acpc_logs[1][-3:],
          [
              ["<-C", "MATCHSTATE:1:0:cc/r300:|AdKd/QcJcTc:r600\r\n"],
              ["S->", "MATCHSTATE:1:0:cc/r300r600:|AdKd/QcJcTc\r\n"],
              ["S->", "MATCHSTATE:1:0:cc/r300r600r900:|AdKd/QcJcTc\r\n"],
          ],
      )

      # -- Make sure that postflop call actions are reported correctly --
      # NOTE: This differs from how the actual ACPC spec works. Since here they
      # should be position independent + always show the actions immediately.
      state.apply_action(ACTION_CHECK_OR_CALL)
      state_struct = state.to_struct()
      # (i.e. this *correctly* differs from what would have been inside the
      # ACPC logs at this moment / immediately reflects the call action.)
      self.assertEqual(state_struct.betting_history, "cc/r200r500r800c/")

      # Deal turn, no change
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(state_struct.betting_history, "cc/r200r500r800c/")
      # Check
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(state.to_struct().betting_history, "cc/r200r500r800c/c")
      # Check-back
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(
          state.to_struct().betting_history, "cc/r200r500r800c/cc/"
      )
      # Deal river, no change
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(
          state.to_struct().betting_history, "cc/r200r500r800c/cc/"
      )
      # Check
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(
          state.to_struct().betting_history, "cc/r200r500r800c/cc/c"
      )
      # Check-back
      state.apply_action(state.legal_actions()[0])
      self.assertEqual(
          state.to_struct().betting_history, "cc/r200r500r800c/cc/cc"
      )
      self.assertTrue(state.is_terminal())

    def test_pokerkit_wrapper_state_to_struct_and_json_when_terminal(self):
      """Tests the ToJson() method inherited from StateStruct."""
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 2,
          "blinds": "50 100",
          "stack_sizes": "1000 1000",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper", params)
      state = game.new_initial_state()

      while not state.is_terminal():
        if not state.is_chance_node():
          state.apply_action(ACTION_FOLD)  # SB/BTN will immediately fold
        else:
          state.apply_action(state.legal_actions()[0])

      state_struct = state.to_struct()
      json_str = state.to_json()
      self.assertEqual(json_str, state_struct.to_json())
      self.assertIsInstance(json_str, str)

      try:
        data = json.loads(json_str)
      except json.JSONDecodeError as e:
        self.fail(f"Failed to parse JSON {json_str}, error was: {e}")

      self.assertEqual(data["current_player"], pyspiel.PlayerId.TERMINAL)
      self.assertEqual(state_struct.current_player, pyspiel.PlayerId.TERMINAL)

      self.assertTrue(data["is_terminal"])
      self.assertTrue(state_struct.is_terminal)

      self.assertEmpty(data["legal_actions"])
      self.assertEmpty(state_struct.legal_actions)

      self.assertEqual(data["stacks"], [1050, 950])
      self.assertEqual(state_struct.stacks, [1050, 950])

      self.assertEqual(data["bets"], [0, 0])
      self.assertEqual(state_struct.bets, [0, 0])

      self.assertEmpty(data["board_cards"])
      self.assertEqual(state_struct.board_cards, [])

      # Only BB's cards; SB/BTN mucked upon folding.
      self.assertEqual(data["hole_cards"], [[0, 2], []])
      self.assertEqual(state_struct.hole_cards, [[0, 2], []])

      self.assertEqual(data["pots"], [0])
      self.assertEqual(state_struct.pots, [0])

      self.assertEmpty(data["burn_cards"])
      self.assertEqual(state_struct.burn_cards, [])

      # As mentioned above, SB's cards were mucked upon folding.
      self.assertEqual(data["mucked_cards"], [1, 3])
      self.assertEqual(state_struct.mucked_cards, [1, 3])

      self.assertEqual(state_struct.blinds, [50, 100])
      self.assertEqual(data["blinds"], [50, 100])
      self.assertEqual(state_struct.starting_stacks, [1000, 1000])
      self.assertEqual(data["starting_stacks"], [1000, 1000])
      self.assertEqual(state_struct.player_contributions, [-50, 50])
      self.assertEqual(data["player_contributions"], [-50, 50])
      self.assertEqual(state_struct.pot_size, 0)
      self.assertEqual(data["pot_size"], 0)

      # p1 has 2c2h since first + third card in unshuffled deck
      # p2 has 2d2s since second + fourth card in unshuffled deck
      self.assertEqual(
          state_struct.per_player_phh_actions,
          [
              ["d dh p1 2c2h", "d dh p2 ????", "p2 f"],
              ["d dh p1 ????", "d dh p2 2d2s", "p2 f"],
          ],
      )
      self.assertNotEmpty(state_struct.per_player_acpc_logs)
      self.assertNotEmpty(data["per_player_acpc_logs"])
      self.assertEqual(
          state_struct.per_player_acpc_logs, data["per_player_acpc_logs"]
      )
      self.assertLen(state_struct.per_player_acpc_logs, 2)
      self.assertEqual(
          state_struct.per_player_acpc_logs,
          [
              [
                  ["S->", "MATCHSTATE:0:0::2c2h|\r\n"],
                  ["S->", "MATCHSTATE:0:0:f:2c2h|\r\n"],
              ],
              [
                  ["S->", "MATCHSTATE:1:0::|2d2s\r\n"],
                  ["<-C", "MATCHSTATE:1:0::|2d2s:f\r\n"],
                  ["S->", "MATCHSTATE:1:0:f:|2d2s\r\n"],
              ],
          ],
      )

    def test_side_pot_nlhe_one_all_in_player(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 3,
          "blinds": "5 10",
          "stack_sizes": "50 200 200",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, SPADE), Card(ACE, CLUB)]
      p1_cards = [Card(KING, SPADE), Card(KING, CLUB)]
      p2_cards = [Card(QUEEN, SPADE), Card(QUEEN, CLUB)]
      board = [
          Card(DEUCE, HEART),
          Card(TREY, HEART),
          Card(FOUR, CLUB),
          Card(FIVE, CLUB),
          Card(EIGHT, SPADE),
      ]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p2_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])
      state.apply_action(game.card_to_int[p2_cards[1]])

      # Preflop betting
      # P0 SB 50, P1 BB 200, P2 BTN 200
      # P0 posts 5 (45 left), P1 posts 10 (190 left)
      # P2 acts first
      self.assertEqual(state.current_player(), 2)
      state.apply_action(50)  # P2 raises to 50
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 45 all-in
      self.assertEqual(state.current_player(), 1)
      state.apply_action(200)  # P1 raises to 200
      self.assertEqual(state.current_player(), 2)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 200

      # Deal flop
      state.apply_action(game.card_to_int[board[0]])
      state.apply_action(game.card_to_int[board[1]])
      state.apply_action(game.card_to_int[board[2]])

      # P0 is all in for 50. P1 and P2 are all in for 200.
      # Main pot: 150. Side pot: 300.
      state.apply_action(game.card_to_int[board[3]])  # Turn
      state.apply_action(game.card_to_int[board[4]])  # River

      self.assertTrue(state.is_terminal())
      # P0 wins main pot 150 with AA. Payoff 100.
      # P1 wins side pot 300 with KK vs QQ. Payoff 100.
      # P2 loses 200. Payoff -200.
      self.assertEqual(state.returns(), [100.0, 100.0, -200.0])

    def test_side_pot_nlhe_two_all_in_players_different_streets(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 3,
          "blinds": "5 10",
          "stack_sizes": "100 100 20",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, HEART), Card(ACE, DIAMOND)]
      p1_cards = [Card(KING, HEART), Card(KING, DIAMOND)]
      p2_cards = [Card(DEUCE, SPADE), Card(DEUCE, CLUB)]
      board = [
          Card(TREY, HEART),
          Card(FOUR, HEART),
          Card(FIVE, HEART),
          Card(EIGHT, HEART),
          Card(NINE, DIAMOND),
      ]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p2_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])
      state.apply_action(game.card_to_int[p2_cards[1]])

      # Preflop betting
      # P0 SB 100, P1 BB 100, P2 BTN 20
      # P0 posts 5 (95 left), P1 posts 10 (90 left)
      # P2 acts first
      self.assertEqual(state.current_player(), 2)
      state.apply_action(20)  # P2 raises all-in to 20
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 20
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls 20

      # Flop: 3h 4h 5h
      state.apply_action(game.card_to_int[board[0]])
      state.apply_action(game.card_to_int[board[1]])
      state.apply_action(game.card_to_int[board[2]])

      self.assertEqual(state.current_player(), 0)
      state.apply_action(80)  # P0 bets 80 all-in
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls 80 all-in

      # Turn: 6h
      state.apply_action(game.card_to_int[board[3]])
      # River: 8d
      state.apply_action(game.card_to_int[board[4]])

      self.assertTrue(state.is_terminal())

      # P0 flush wins main pot 60 and side pot 160 with AA. Payoff 120.
      # P1 loses 100. Payoff -100.
      # P2 loses 20. Payoff -20.
      self.assertEqual(state.returns(), [120.0, -100.0, -20.0])

    def test_side_pot_with_one_chip_bet(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 3,
          "blinds": "10 20",
          "stack_sizes": "21 22 100",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, CLUB), Card(ACE, DIAMOND)]
      p1_cards = [Card(KING, CLUB), Card(KING, DIAMOND)]
      p2_cards = [Card(QUEEN, CLUB), Card(QUEEN, DIAMOND)]
      board = [
          Card(FOUR, CLUB),
          Card(FIVE, CLUB),
          Card(SIX, SPADE),
          Card(EIGHT, DIAMOND),
          Card(TEN, HEART),
      ]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p2_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])
      state.apply_action(game.card_to_int[p2_cards[1]])

      # Preflop betting
      # P0 SB 21, P1 BB 22, P2 BTN 100
      # P0 posts 10 (11 left), P1 posts 20 (2 left)
      # P2 acts first
      self.assertEqual(state.current_player(), 2)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 20, 80 left
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 10, 1 left
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks, 2 left

      # Stacks: P0=1, P1=2, P2=80
      self.assertEqual(state._wrapped_state.stacks, [1, 2, 80])

      # Deal flop
      state.apply_action(game.card_to_int[board[0]])
      state.apply_action(game.card_to_int[board[1]])
      state.apply_action(game.card_to_int[board[2]])

      # Flop betting
      # P0 acts first
      self.assertEqual(state.current_player(), 0)
      # P0 can check or bet 1.
      # If P0 bets 1, it's an all-in mapped to action '2'.
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 2])
      state.apply_action(2)  # P0 shoves for 1.
      self.assertEqual(state._wrapped_state.stacks, [0, 2, 80])

      # P1 to act.
      self.assertEqual(state.current_player(), 1)
      # P1 needs to call 1 chip.
      self.assertEqual(state._wrapped_state.checking_or_calling_amount, 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls 1. 1 left.
      self.assertEqual(state._wrapped_state.stacks, [0, 1, 80])

      # P2 to act.
      self.assertEqual(state.current_player(), 2)
      # P2 needs to call 1 chip.
      self.assertEqual(state._wrapped_state.checking_or_calling_amount, 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 1. 79 left.
      self.assertEqual(state._wrapped_state.stacks, [0, 1, 79])

      # P0 is all-in. When P2 calls, flop betting round ends because P0 is
      # all-in and P1/P2 have called.
      # This moves to a chance node for turn dealing.
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[board[3]])  # Deal turn

      # Turn betting round begins. P1 should be next to act.
      self.assertEqual(state.current_player(), 1)
      # P1 has 1 chip and can bet it.
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 2])
      state.apply_action(2)  # P1 bets 1, all-in
      self.assertEqual(state._wrapped_state.stacks, [0, 0, 79])

      # P2 to act.
      self.assertEqual(state.current_player(), 2)
      # P2 needs to call 1 chip to be in side pot.
      self.assertEqual(state._wrapped_state.checking_or_calling_amount, 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 1.
      self.assertEqual(state._wrapped_state.stacks, [0, 0, 78])

      # Deal river
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[board[4]])

      # Hand should immediately proceded to showdown since everyone is all-in.

      self.assertTrue(state.is_terminal())
      # P0 wins main pot of 21*3 => 63 with AA for 42 profit.
      # P1 wins side pot of (22-21)*2 => 2 with KK vs QQ for -20 profit.
      # P2 loses 22.
      self.assertEqual(state.returns(), [42.0, -20.0, -22.0])

    def test_side_pot_nlhe_three_all_in_players(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 4,
          "blinds": "5 10",
          "stack_sizes": "50 120 150 200",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, SPADE), Card(ACE, CLUB)]
      p1_cards = [Card(KING, SPADE), Card(KING, CLUB)]
      p2_cards = [Card(QUEEN, SPADE), Card(QUEEN, CLUB)]
      p3_cards = [Card(JACK, SPADE), Card(JACK, CLUB)]
      board = [
          Card(DEUCE, HEART),
          Card(TREY, HEART),
          Card(SIX, DIAMOND),
          Card(EIGHT, DIAMOND),
          Card(TEN, SPADE),
      ]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p2_cards[0]])
      state.apply_action(game.card_to_int[p3_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])
      state.apply_action(game.card_to_int[p2_cards[1]])
      state.apply_action(game.card_to_int[p3_cards[1]])

      # Preflop betting
      # P0 SB 50, P1 BB 120, P2 UTG 150, P3 BTN 200
      # P0 posts 5 (45 left), P1 posts 10 (110 left)
      # P2 acts first
      self.assertEqual(state.current_player(), 2)
      state.apply_action(150)  # P2 raises all-in to 150
      self.assertEqual(state.current_player(), 3)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P3 calls 150
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 45 all-in
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls 110 all-in

      # Deal flop
      state.apply_action(game.card_to_int[board[0]])
      state.apply_action(game.card_to_int[board[1]])
      state.apply_action(game.card_to_int[board[2]])

      # Deal turn
      state.apply_action(game.card_to_int[board[3]])
      # Deal river
      state.apply_action(game.card_to_int[board[4]])

      if not state.is_terminal():
        state._wrapped_state.end_hand()
      self.assertTrue(state.is_terminal())
      # P0 wins main pot 200 with AA. Payoff 150.
      # P1 wins side pot 1 (210) with KK. Payoff 90.
      # P2 wins side pot 2 (60) with QQ. Payoff -90.
      # P3 loses 150. Payoff -150.
      self.assertEqual(state.returns(), [150.0, 90.0, -90.0, -150.0])

    def test_side_pot_flhe_one_all_in_player(self):
      params = {
          "variant": "FixedLimitTexasHoldem",
          "num_players": 3,
          "blinds": "5 10",
          "small_bet": 10,
          "big_bet": 20,
          "stack_sizes": "25 100 100",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, CLUB), Card(ACE, DIAMOND)]
      p1_cards = [Card(KING, CLUB), Card(KING, DIAMOND)]
      p2_cards = [Card(QUEEN, CLUB), Card(QUEEN, DIAMOND)]
      board = [
          Card(DEUCE, HEART),
          Card(TREY, HEART),
          Card(SIX, DIAMOND),
          Card(EIGHT, DIAMOND),
          Card(TEN, SPADE),
      ]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p2_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])
      state.apply_action(game.card_to_int[p2_cards[1]])

      # Preflop betting
      # P0 SB 25, P1 BB 100, P2 BTN 100
      # P0 posts 5 (20 left), P1 posts 10 (90 left)
      # P2 acts first
      self.assertEqual(state.current_player(), 2)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 10
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 5, 15 left
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks

      # Deal flop
      state.apply_action(game.card_to_int[board[0]])
      state.apply_action(game.card_to_int[board[1]])
      state.apply_action(game.card_to_int[board[2]])

      # Flop betting. small bet = 10.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 checks
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
      self.assertEqual(state.current_player(), 2)
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 10])
      state.apply_action(10)  # P2 bets 10.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 10. 5 left.
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls 10.

      # Deal turn
      state.apply_action(game.card_to_int[board[3]])

      # Turn betting. big bet = 20.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 checks
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
      self.assertEqual(state.current_player(), 2)
      self.assertEqual(
          state._wrapped_state.betting_structure,
          pokerkit.state.BettingStructure.FIXED_LIMIT,
      )
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 20])
      state.apply_action(20)  # P2 bets 20.
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL]
      )
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls 5 all-in.
      self.assertEqual(state.current_player(), 1)
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 40]
      )
      state.apply_action(40)  # P1 raises to 40.
      self.assertEqual(state.current_player(), 2)
      self.assertEqual(state._wrapped_state.checking_or_calling_amount, 20)
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 60]
      )
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 calls 20.

      # Deal river
      state.apply_action(game.card_to_int[board[4]])

      # River betting.
      self.assertEqual(state.current_player(), 1)
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 20])
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 checks
      self.assertEqual(state.current_player(), 2)
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 20])
      state.apply_action(ACTION_CHECK_OR_CALL)  # P2 checks

      if not state.is_terminal():
        state._wrapped_state.end_hand()
      self.assertTrue(state.is_terminal())
      # P0 wins main pot 75 with AA. Payoff 50.
      # P1 wins side pot 70 with KK vs QQ. Payoff 10.
      # P2 loses 60. Payoff -60.
      self.assertEqual(state.returns(), [50.0, 10.0, -60.0])

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
      # min_utility is technically just upper bound anyways (in magnitude), so
      # it doesn't have to be exact. But, whatever this bound is, it at least
      # needs to avoid *underestimating* how many chips could be lost here.
      self.assertLessEqual(game.min_utility(), -200.0)
      # Even if it technically *could* be even less tight (e.g. -500 is still
      # 'correct'), in practice it'd be concerning for this to be any lower
      # given how easy it is to reason that the minimum utility will be no
      # 'larger' (in magnitude) than the largest stack's size.
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
      # min_utility is technically just upper bound anyways (in magnitude), so
      # it doesn't have to be exact. But, whatever this bound is, it at least
      # needs to avoid *underestimating* how many chips could be lost here.
      self.assertLessEqual(game.min_utility(), -300.0)
      # Even if it technically *could* be even less tight (e.g. -500 is still
      # 'correct'), in practice it'd be concerning for this to be any lower
      # given how easy it is to reason that the minimum utility will be no
      # 'larger' (in magnitude) than the largest stack's size.
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
          state.action_to_string(state.current_player(), ACTION_CHECK_OR_CALL),
      )
      # Check the wrapped pokerkit state agrees with the returned legal actions.
      # (Accessing the ._wrapped_state directly is a bit of a hack, but in
      # practice this is unlikely to actually be a problem).
      for a in expected_actions:
        if a not in FOLD_AND_CHECK_OR_CALL_ACTIONS:
          self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to(a))

      state.apply_action(60)  # P2 raises
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls

      # P1 can call (0), fold (1), or reraise. The reraise increase the bet by
      # at least the amount of the last raise, which was (60-20) = 40, meaning
      # it must be at least a raise to 60 + 40 = 100.
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
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)
      while state.is_chance_node():
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
      self.assertEqual(wrapped_state.stacks, [1, 1, 1])
      # P0 can either check or shove for 1 chip, hitting an edge case that needs
      # to be handled correctly (mapping 1 chip shove to action 2).
      # (P0 cannot fold since pokerkit doesn't allow folding when not actually
      # facing an opponent's bet.)
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 2])
      self.assertIn("[ALL-IN EDGECASE]", state.action_to_string(2))
      self.assertIn("Bet/Raise to 1", state.action_to_string(2))
      state.apply_action(2)

      # Verify that the 2 actually mapped to a bet of 1 chip / that the next
      # players get entirely normal fold/check_or_call actions.
      self.assertEqual(wrapped_state.stacks, [0, 1, 1])
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL]
      )
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(1)")
      state.apply_action(ACTION_FOLD)

      self.assertEqual(wrapped_state.stacks, [0, 1, 1])
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(1)")

    def test_postflop_smaller_effective_stack_size_does_not_limit_bet_actions(
        self,
    ):
      # Proves that we don't need to consider the 'single chip shoves' edge
      # case as a result of *other* players having only one chip in their stack.
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 3,
          "blinds": "10 20",
          # NOTE: Remember that the first player to act preflop will be the
          # third stack, but the first player to act *postflop* will be the 1st
          # stack (25 chips).
          "stack_sizes": "25 21 21",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper", params)
      state = game.new_initial_state()
      wrapped_state: pokerkit.State = state._wrapped_state
      while state.is_chance_node():
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
      # BTN acts first preflop
      self.assertEqual(state.current_player(), 2)
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)
      while state.is_chance_node():
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )

      # SB acts first postflop
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(wrapped_state.stacks, [5, 1, 1])
      self.assertEqual(
          state.legal_actions(), [ACTION_CHECK_OR_CALL, 2, 3, 4, 5]
      )
      self.assertIn("Check", state.action_to_string(ACTION_CHECK_OR_CALL))
      for i in range(2, 6):
        self.assertEndsWith(state.action_to_string(i), f"Bet/Raise to {i}")

      # Proof that we can actually bet values that aren't either a shove or
      # the minimum action.
      state.apply_action(3)

      # Verify that the 3 actually mapped to a bet of 3 chips / that the next
      # players get entirely normal fold/check_or_call actions.
      self.assertEqual(wrapped_state.stacks, [2, 1, 1])
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL]
      )
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(1)")
      state.apply_action(ACTION_FOLD)

      self.assertEqual(wrapped_state.stacks, [2, 1, 1])
      self.assertEqual(
          state.legal_actions(),
          [ACTION_FOLD, ACTION_CHECK_OR_CALL],
      )
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(1)")
      state.apply_action(ACTION_FOLD)
      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [40, -20, -20])

    def test_preflop_smaller_effective_stack_size_does_not_limit_bet_actions(
        self,
    ):
      # Proves that we don't need to consider the 'single chip shoves' edge
      # case as a result of *other* players having only one chip in their stack.
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 3,
          "blinds": "10 20",
          # NOTE: Remember that the first player to act preflop will be the
          # third stack, but the first player to act *postflop* will be the 1st
          # stack (25 chips).
          "stack_sizes": "11 21 25",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper", params)
      state = game.new_initial_state()
      wrapped_state: pokerkit.State = state._wrapped_state
      while state.is_chance_node():
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
      # BTN acts first preflop
      self.assertEqual(state.current_player(), 2)
      self.assertEqual(wrapped_state.stacks, [1, 1, 25])
      self.assertEqual(
          state.legal_actions(),
          [ACTION_FOLD, ACTION_CHECK_OR_CALL, 21, 22, 23, 24, 25],
      )
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(20)")
      for i in range(21, 26):
        self.assertEndsWith(state.action_to_string(i), f"Bet/Raise to {i}")
      state.apply_action(ACTION_CHECK_OR_CALL)

      # ... and only now do we loop back around to the SB / player 0, ...
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(wrapped_state.stacks, [1, 1, 5])
      self.assertEqual(
          state.legal_actions(),
          # NOTE: 11 chips, not 21! Meaning they cannot re-raise - just calling
          # will already be putting them all-in.
          [ACTION_FOLD, ACTION_CHECK_OR_CALL],
      )
      self.assertEqual(state.action_to_string(ACTION_CHECK_OR_CALL), "Call(1)")
      self.assertEqual(state._wrapped_state.checking_or_calling_amount, 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      self.assertEqual(state.current_player(), 1)
      self.assertEqual(wrapped_state.stacks, [0, 1, 5])
      self.assertEqual(
          state.action_to_string(ACTION_CHECK_OR_CALL), "Player 1: Check"
      )
      # Note that BB by definition cannot fold since nobody has raised over
      # their Big-blind yet. So they can only check or re-raise.
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 21])
      # Extra asserts just to make extra sure that the underlying pokerkit state
      # is behaving as expected.
      self.assertFalse(state._wrapped_state.can_complete_bet_or_raise_to(1))
      self.assertFalse(state._wrapped_state.can_complete_bet_or_raise_to(20))
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to(21))
      self.assertEqual(state.action_to_string(21), "Player 1: Bet/Raise to 21")
      state.apply_action(21)

      self.assertEqual(wrapped_state.stacks, [0, 0, 5])
      self.assertEqual(
          state.legal_actions(),
          [ACTION_FOLD, ACTION_CHECK_OR_CALL],
      )
      state.apply_action(ACTION_FOLD)  # BTN folds

      # Both SB and BB are already all-in so the hand should be over once we go
      # deal all the board cards.
      self.assertFalse(state.is_terminal())
      self.assertTrue(state.is_chance_node())
      while state.is_chance_node():
        state.apply_action(
            random.choice([o for o, _ in state.chance_outcomes()])
        )
      self.assertTrue(state.is_terminal())

    def test_seven_card_stud_betting_history_with_immediate_all_in(self):
      # Test with Seven Card Stud
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "20 20",
          "bring_in": 5,
          "small_bet": 10,
          "big_bet": 20,
      }
      game = PokerkitWrapper(stud_params)
      state = game.new_initial_state()
      for _ in range(15):
        state.apply_action(max(state.legal_actions()))
      self.assertEqual(state.to_struct().betting_history, "r10r20c////")

    def test_seven_card_stud_betting_history_with_second_street_allin(self):
      # Test with Seven Card Stud
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "50 50",
          "bring_in": 5,
          "small_bet": 10,
          "big_bet": 20,
      }
      game = PokerkitWrapper(stud_params)
      state = game.new_initial_state()
      for _ in range(15):
        state.apply_action(max(state.legal_actions()))
      self.assertEqual(
          state.to_struct().betting_history, "r10r20r30r40c/r10c///"
      )

    def test_razz_vs_seven_card_stud_eval(self):
      card_sequence = [
          Card(ACE, HEART),
          Card(DEUCE, SPADE),
          Card(DEUCE, HEART),
          Card(TREY, SPADE),
          Card(TREY, HEART),
          Card(FOUR, SPADE),  # 3rd street
          Card(FOUR, HEART),
          Card(FIVE, SPADE),  # 4th street
          Card(FIVE, HEART),
          Card(SIX, SPADE),  # 5th street
          Card(SIX, HEART),
          Card(SEVEN, SPADE),  # 6th street
          Card(SEVEN, HEART),
          Card(EIGHT, SPADE),  # 7th street
      ]

      def play_to_showdown(game, state, cards):
        card_iterator = iter(cards)
        while not state.is_terminal():
          if state.is_chance_node():
            card_to_deal = next(card_iterator)
            state.apply_action(game.card_to_int[card_to_deal])
          else:
            legal_actions = state.legal_actions()
            if ACTION_CHECK_OR_CALL in legal_actions:
              state.apply_action(ACTION_CHECK_OR_CALL)
            else:
              min_bet_action = min(
                  a for a in legal_actions if a > ACTION_CHECK_OR_CALL
              )
              state.apply_action(min_bet_action)
        return state.returns()

      # Test with Razz
      razz_params = {
          "variant": "FixedLimitRazz",
          "num_players": 2,
          "stack_sizes": "2000 2000",
          "bring_in": 5,
          "small_bet": 10,
          "big_bet": 20,
      }
      razz_game = PokerkitWrapper(razz_params)
      razz_state = razz_game.new_initial_state()
      razz_returns = play_to_showdown(razz_game, razz_state, card_sequence)
      # P0 has 5432A, P1 has 65432. P0 wins in Razz.
      self.assertGreater(razz_returns[0], 0)
      self.assertLess(razz_returns[1], 0)
      self.assertEqual(
          razz_state.to_struct().betting_history, "b5c/cc/cc/cc/cc"
      )

      # Test with Seven Card Stud
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "2000 2000",
          "bring_in": 5,
          "small_bet": 10,
          "big_bet": 20,
      }
      stud_game = PokerkitWrapper(stud_params)
      stud_state = stud_game.new_initial_state()
      stud_returns = play_to_showdown(stud_game, stud_state, card_sequence)
      # P0 has 7h SF, P1 has 8s SF. P1 wins in 7-card stud.
      self.assertLess(stud_returns[0], 0)
      self.assertGreater(stud_returns[1], 0)
      self.assertEqual(
          stud_state.to_struct().betting_history, "b5c/cc/cc/cc/cc"
      )

    def test_hand_mucking_at_showdown(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 2,
          "blinds": "5 10",
          "stack_sizes": "200 200",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(EIGHT, HEART), Card(EIGHT, DIAMOND)]
      p1_cards = [Card(NINE, SPADE), Card(NINE, CLUB)]
      board = [
          Card(DEUCE, CLUB),
          Card(TREY, DIAMOND),
          Card(FOUR, SPADE),
          Card(SIX, HEART),
          Card(TEN, CLUB),
      ]
      board_iter = iter(board)

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])

      # Preflop: P1 (SB / BTN) calls, P0 (BB) checks.
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Flop
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[next(board_iter)])
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[next(board_iter)])
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[next(board_iter)])
      # P0 checks, P1 checks.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Turn
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[next(board_iter)])
      # P0 checks, P1 checks.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # River
      self.assertTrue(state.is_chance_node())
      state.apply_action(game.card_to_int[next(board_iter)])
      # P0 checks, P1 checks.
      self.assertEqual(state.current_player(), 0)
      state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(state.current_player(), 1)
      state.apply_action(ACTION_CHECK_OR_CALL)

      self.assertTrue(state.is_terminal())
      # P1 loses with 99, P0 loses with 88 and should muck.
      self.assertEqual(state.returns(), [-10.0, 10.0])
      self.assertCountEqual(state._wrapped_state.mucked_cards, p0_cards)
      # Check hole cards to see who mucked.
      # P0 mucked, P1 showed.
      self.assertEqual(state._wrapped_state.hole_cards[0], [])
      self.assertCountEqual(state._wrapped_state.hole_cards[1], p1_cards)

    def test_observer_string_censors_opponent_hole_cards(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 2,
          "blinds": "5 10",
          "stack_sizes": "200 200",
      }
      game = PokerkitWrapper(params)
      state = game.new_initial_state()

      # Cards
      p0_cards = [Card(ACE, CLUB), Card(KING, CLUB)]
      p1_cards = [Card(QUEEN, DIAMOND), Card(JACK, DIAMOND)]

      # Deal hole cards
      state.apply_action(game.card_to_int[p0_cards[0]])
      state.apply_action(game.card_to_int[p1_cards[0]])
      state.apply_action(game.card_to_int[p0_cards[1]])
      state.apply_action(game.card_to_int[p1_cards[1]])

      # Preflop: P0 calls, P1 checks to reach flop and have some actions.
      state.apply_action(ACTION_CHECK_OR_CALL)
      state.apply_action(ACTION_CHECK_OR_CALL)

      # Flop
      state.apply_action(game.card_to_int[Card(DEUCE, HEART)])
      state.apply_action(game.card_to_int[Card(TREY, HEART)])
      state.apply_action(game.card_to_int[Card(FOUR, HEART)])

      observer = game.make_py_observer(
          pyspiel.IIGObservationType(perfect_recall=True)
      )

      # Check observation string for player 0
      obs_str_p0 = observer.string_from(state, 0)
      phh_p0 = [
          s for s in obs_str_p0.split("||") if s.startswith("PHH Actions:")
      ][0]
      self.assertIn("d dh p1 AcKc", phh_p0)
      self.assertIn("d dh p2 ????", phh_p0)

      # Check observation string for player 1
      obs_str_p1 = observer.string_from(state, 1)
      phh_p1 = [
          s for s in obs_str_p1.split("||") if s.startswith("PHH Actions:")
      ][0]
      self.assertIn("d dh p1 ????", phh_p1)
      self.assertIn("d dh p2 QdJd", phh_p1)

    def test_regression_includes_action_1_to_post_bring_in(self):
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "2000 2000",
          # Normally each of these is double the smaller, e.g. 5 10 20. But in
          # this case it's useful to offset them a little more so that in the
          # event of any bugs we can clearly determine which of these parameters
          # are controlling what behavior in the game logic.
          "bring_in": 30,
          "small_bet": 70,
          "big_bet": 175,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()

      # Unlike NLHE, 7 card stud has down cards and up cards. Where each player
      # should gets *three* total cards instead of two.
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p0_up2 = Card(QUEEN, CLUB)

      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      p1_up2 = Card(EIGHT, SPADE)

      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      card_sequence_street4 = [p0_up2, p1_up2]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      # P0 has 2s upcard (lowest), P1 has 3s upcard. So by the rules of 7 card
      # stud, P0 must act first: either posting the bring-in or betting the
      # small bet, and is *not* allowed to simply fold/check/call here.
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 70])
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 30")
      self.assertEqual(state.action_to_string(70), "Player 0: Bet/Raise to 70")
      state.apply_action(1)

      # P1 should be able to fold, call, or 'complete' the small bet (since
      # P0 only posted the bring-in, not the full small bet of 70)
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 70]
      )
      state.apply_action(70)

      # P0 can only fold or re-raise by the small bet (70 + 70 => 140). Notably
      # this is adding more chips than earlier when only 'completing' the small
      # bet was allowed (as opposed to raising by it).
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 140]
      )
      state.apply_action(140)

      # And finally P1 can either fold or (re-)re-raise by the small bet 140
      # + 70 => 210. Which is again larger than earlier when they were
      # 'completing' the small bet.
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 210]
      )
      state.apply_action(210)

      state.apply_action(ACTION_CHECK_OR_CALL)

      for card in card_sequence_street4:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Double check that bring_in only affects the prior street (i.e. the
      # 3rd street) and not betting on later streets like this one.
      self.assertEqual(state.current_player(), 0)
      self.assertFalse(state.is_terminal())
      # small_bet => 70, not 30 (bring_in)
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 70])
      self.assertEqual(state._action_to_string(0, 1), "Player 0: Check")
      state.apply_action(70)

      self.assertEqual(state.current_player(), 1)
      self.assertFalse(state.is_terminal())
      # small_bet + small_bet => 140
      self.assertEqual(
          state.legal_actions(), [ACTION_FOLD, ACTION_CHECK_OR_CALL, 140]
      )
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [210, -210])

    @parameterized.parameters(1, 50)
    def test_two_actions_same_result_if_bringin_equals_stack(
        self, chosen_action_at_state_allowing_bring_in
    ):
      """Verifies our 1 'Post Bring-in' does the same as betting 50 chips here.

      Context: If the bring-in is equal to (or less than) their stack, this hits
      an edge case: pokerkit allows for *either* calling post_bring_in() OR
      betting for the same result here. But, they technically are different
      function calls in pokerkit / will be getting mapped to different actions.
      The goal of this test is to verify that both actions actually end up doing
      the same thing.

      Args:
        chosen_action_at_state_allowing_bring_in: The action chosen at the state
          where both "Post Bring-in" and "Bet/Raise to" are legal and *should*
          result in the same outcome.
      """
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "50 50",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()

      # Unlike NLHE, 7 card stud has down cards and up cards. Where each player
      # should gets *three* total cards instead of two. Additionally, the up
      # cards control the player betting order, so it's useful to explicitly
      # define these all.
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      # P0 has 2s upcard (lowest), P1 has 3s upcard. So by the rules of 7 card
      # stud, P0 must act first.
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 50])
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 50")
      self.assertEqual(state.action_to_string(50), "Player 0: Bet/Raise to 50")
      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 50)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 50
      )
      # **** This is where the two parameterized test cases diverge! Notably,
      # despite choosing different actions the *results* should be exactly the
      # same past this point. ****
      state.apply_action(chosen_action_at_state_allowing_bring_in)

      # P1 should only be able to fold or call
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

    @parameterized.parameters(1, 10)
    def test_two_actions_same_result_if_bringin_greater_than_stack(
        self, chosen_action_at_state_allowing_bring_in
    ):
      """Verifies our 1 'Post Bring-in' does the same as betting 10 chips here.

      Context: If the bring-in is less than (or equal to) their stack, this hits
      an edge case: pokerkit allows for *either* calling post_bring_in() OR
      betting for the same result here. But, they technically are different
      function calls in pokerkit / will be getting mapped to different actions.
      The goal of this test is to verify that both actions actually end up doing
      the same thing.

      Args:
        chosen_action_at_state_allowing_bring_in: The action chosen at the state
          where both "Post Bring-in" and "Bet/Raise to" are legal and *should*
          result in the same outcome.
      """
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "10 10",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()

      # Unlike NLHE, 7 card stud has down cards and up cards. Where each player
      # should gets *three* total cards instead of two. Additionally, the up
      # cards control the player betting order, so it's useful to explicitly
      # define these all.
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      # P0 has 2s upcard (lowest), P1 has 3s upcard. So by the rules of 7 card
      # stud, P0 must act first.
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 10])
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 10")
      self.assertEqual(state.action_to_string(10), "Player 0: Bet/Raise to 10")
      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 10)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 10
      )
      # **** This is where the two parameterized test cases diverge! Notably,
      # despite choosing different actions the *results* should be exactly the
      # same past this point. ****
      state.apply_action(chosen_action_at_state_allowing_bring_in)

      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

    @parameterized.parameters(1, 10)
    def test_two_actions_same_result_if_bringin_between_stack_and_effective_stack(
        self, chosen_action_at_state_allowing_bring_in
    ):
      """Verifies our 1 'Post Bring-in' does the same as betting 50 chips here.

      For more context, see the above 'test_two_actions_same_result...'
      functions. This test verifies an additional edge case beyond that in the
      above tests about whether the same results hold true if only the
      *effective* stack is less than the bring_in, but the actual player's stack
      is larger. (And, while it does, also cleanly demonstrates how the action
      mapping doesn't care about the first player's actual stack size being
      larger; only the effective stack matters.)

      Args:
        chosen_action_at_state_allowing_bring_in: The action chosen at the state
          where both "Post Bring-in" and "Bet/Raise to" are legal and *should*
          result in the same outcome.
      """
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "500 10",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()

      # Unlike NLHE, 7 card stud has down cards and up cards. Where each player
      # should gets *three* total cards instead of two. Additionally, the up
      # cards control the player betting order, so it's useful to explicitly
      # define these all.
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      # P0 has 2s upcard (lowest), P1 has 3s upcard. So by the rules of 7 card
      # stud, P0 must act first.
      self.assertEqual(state.current_player(), 0)
      # Double checking that we didn't mix up which player had which stack above
      # and that their effective stack is indeed much smaller.
      self.assertEqual(state._wrapped_state.stacks[0], 500)
      self.assertEqual(state._wrapped_state.get_effective_stack(0), 10)

      self.assertEqual(state.legal_actions(), [1, 10])
      # NOTE: Interestingly the value of the *effective_bring_in* in pokerkit
      # actually doesn't reflect the effective stack - which differs from
      # everything else, including the betting values! (And yet, despite this
      # all being different, in practice they should still always produce the
      # exact same results here.)
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 50")
      self.assertEqual(state.action_to_string(10), "Player 0: Bet/Raise to 10")

      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      # NOTE: As above, this is counterintuitive but expected and harmless (so
      # long as e.g. we don't put in any overly-aggressive asserts ourselves).
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 50)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 10
      )
      # **** This is where the two parameterized test cases diverge! Notably,
      # despite choosing different actions the *results* should be exactly the
      # same past this point. ****
      state.apply_action(chosen_action_at_state_allowing_bring_in)

      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

    @parameterized.parameters(1, 2)
    def test_bring_in_action_still_1_when_mapping_stack_size_1_shove_to_2(
        self, chosen_action_at_state_allowing_bring_in
    ):
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "1 1",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      # P0 has 2s upcard (lowest), P1 has 3s upcard. So by the rules of 7 card
      # stud, P0 must act first.
      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 2])
      # ** Key part of this test - even if bet/raises get remapped bring-in
      # should not be affected **
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 1")
      self.assertEqual(
          state.action_to_string(2),
          "Player 0: Bet/Raise to 1 [ALL-IN EDGECASE]",
      )
      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 1)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 1
      )

      # **** This is where the two parameterized test cases diverge! Notably,
      # despite choosing different actions the *results* should be exactly the
      # same past this point. ****
      state.apply_action(chosen_action_at_state_allowing_bring_in)

      # P1 should only be able to fold or call since they also have 1 chip.
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

    @parameterized.parameters(1, 2)
    def test_regression_bring_in_1_when_only_effective_stack_is_1(
        self, chosen_action_at_state_allowing_bring_in
    ):
      """Only effective stack is 1 => the player's actual stack is NOT 1."""
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "300 1",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 2])
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 50")

      # "Remapped" action 2 avoids conflicting with the bring_in action just
      # like how it would normally avoid conflicting with ACTION_CHECK_OR_CALL.
      # (See # the relevant logic / comments in pokerkit_wrapper.py for more
      # detail on how this works + why it's correct.)
      self.assertEqual(
          state.action_to_string(2),
          "Player 0: Bet/Raise to 1 [ALL-IN EDGECASE]",
      )

      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 50)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 1
      )
      # **** This is where the two parameterized test cases diverge! Notably,
      # despite choosing different actions the *results* should be exactly the
      # same past this point. ****
      state.apply_action(chosen_action_at_state_allowing_bring_in)

      # P1 should only be able to fold or call since they also have 1 chip.
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

    def test_bring_in_2_does_not_map_to_bet_1_if_effective_stack_is_2(self):
      stud_params = {
          "variant": "FixedLimitSevenCardStud",
          "num_players": 2,
          "stack_sizes": "300 2",
          "bring_in": 50,
          "small_bet": 100,
          "big_bet": 200,
      }
      stud_game = PokerkitWrapper(stud_params)
      state = stud_game.new_initial_state()
      p0_down1 = Card(ACE, DIAMOND)
      p0_down2 = Card(KING, DIAMOND)
      p0_up1 = Card(DEUCE, SPADE)
      p1_down1 = Card(ACE, HEART)
      p1_down2 = Card(KING, HEART)
      p1_up1 = Card(TREY, SPADE)
      card_sequence_street3 = [
          p0_down1,
          p1_down1,
          p0_down2,
          p1_down2,
          p0_up1,
          p1_up1,
      ]
      for card in card_sequence_street3:
        self.assertTrue(state.is_chance_node())
        state.apply_action(stud_game.card_to_int[card])

      # Check no board cards are dealt in stud
      self.assertEqual(state._wrapped_state.board_cards, [])

      self.assertEqual(state.current_player(), 0)
      self.assertEqual(state.legal_actions(), [1, 2])
      # NOTE: effective_bring_in doesn't care about effective stacks!
      self.assertEqual(state.action_to_string(1), "Player 0: Post Bring-in 50")

      # ** First important main part of the test - making sure this action 2
      # isn't being billed as a 'shove for 1 chip' action in its string **
      self.assertEqual(state.action_to_string(2), "Player 0: Bet/Raise to 2")

      # Double checking that we're indeed in the edge case we were worried about
      self.assertTrue(state._wrapped_state.can_post_bring_in())
      self.assertTrue(state._wrapped_state.can_complete_bet_or_raise_to())
      # NOTE: effective_bring_in doesn't care about effective stacks!
      self.assertEqual(state._wrapped_state.effective_bring_in_amount, 50)
      self.assertEqual(
          state._wrapped_state.min_completion_betting_or_raising_to_amount, 2
      )

      # ** Second important main part of the test - making sure this action 2
      # is actually being _interpreted_ properly when applied. **
      state.apply_action(2)

      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)

      self.assertTrue(state.is_terminal())
      self.assertEqual(state.returns(), [0, 0])

  class PokerkitWrapperAcpcStyleTest(parameterized.TestCase):
    """Test the OpenSpiel game wrapper for Pokerkit."""

    # TODO: b/437724266 - port over more OpenSpiel universal_poker tests to
    # verify that we have identical (or, at least mostly identical) behavior +
    # are at least as "correct" in general.

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
          # --- Bet size edge cases ---
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
          # Encourage situations where player are likely to be betting 1 chip
          # to verify that we gracefully handle any edge cases.
          # TODO: b/437724266 - split out into separate tests just for this edge
          # case. Also run multiple times (fail on any failures) and/or
          # hand-choose bet sizes to force these edge cases.
          {
              "variant": "NoLimitTexasHoldem",
              "num_players": 4,
              "blinds": "1 2",
              "stack_sizes": "7 7 5 5",
          },
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
          {
              "variant": "NoLimitTexasHoldem",
              "num_players": 9,
              "blinds": "1 2",
              "stack_sizes": "3 3 1 1 2 2 2 1 1",
          },
      ]

      for params in scenarios:
        with self.subTest(variant=params["variant"]):
          game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
          pyspiel.random_sim_test(
              game, num_sims=3, serialize=False, verbose=True
          )

    def test_regression_single_chip_shove_action_to_string(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 9,
          "blinds": "1 2",
          "stack_sizes": "100 100 1 2 1 3 3 3 3",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
      state = game.new_initial_state()
      player_actions = [0, 0, 0, 1, 0, 0, 1, 31, 85, 0, 0, 1, 96, 1, 99, 1]
      while state.is_chance_node() or player_actions:
        # Specific cards don't matter since the error hapened before showdown.
        if state.is_chance_node():
          action = random.choice([o for o, _ in state.chance_outcomes()])
        else:
          action = player_actions.pop(0)
        state.apply_action(action)

      # Now we've reached the specific edge case. The player has one chip behind
      # and is able to either check or shove all-in with this one last chip.
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 100])
      self.assertEqual(
          state.action_to_string(ACTION_CHECK_OR_CALL), "Player 0: Check"
      )
      self.assertEqual(
          state.action_to_string(100),
          "Player 0: Bet/Raise to 1 [ALL-IN EDGECASE]",
      )
      # Shove all-in
      state.apply_action(100)

    def test_single_chip_alllin_headsup(self):
      params = {
          "variant": "NoLimitTexasHoldem",
          "num_players": 2,
          "blinds": "1 2",
          "stack_sizes": "100 100",
      }
      game = pyspiel.load_game("python_pokerkit_wrapper_acpc_style", params)
      state = game.new_initial_state()
      player_actions = [1, 1, 99, 1]
      while state.is_chance_node() or player_actions:
        # Specific cards don't matter since the error hapened before showdown.
        if state.is_chance_node():
          action = state.chance_outcomes()[0][0]
        else:
          action = player_actions.pop(0)
        state.apply_action(action)

      # Now we've reached the specific edge case. The player has one chip behind
      # and is able to either check or shove all-in with this one last chip.
      self.assertEqual(state.legal_actions(), [ACTION_CHECK_OR_CALL, 100])
      self.assertEqual(
          state.action_to_string(ACTION_CHECK_OR_CALL), "Player 0: Check"
      )
      self.assertEqual(
          state.action_to_string(100),
          "Player 0: Bet/Raise to 1 [ALL-IN EDGECASE]",
      )
      # Shove all-in
      state.apply_action(100)
      state.apply_action(ACTION_CHECK_OR_CALL)
      while not state.is_terminal():
        state.apply_action(state.chance_outcomes()[0][0])

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

      # NOTE: Because this is preflop the ACPC style actions are identical to
      # the Pokerkit style actions.
      state.apply_action(300)  # P0 raises to 300
      state.apply_action(900)  # P1 re-raises to 900
      state.apply_action(10000)  # P0 shoves
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls

      # Progress through all remaining chance nodes, i.e. deal the flop, turn,
      # and river.
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
      # more details see the # pokerkit_wrapper_acpc_style docstring. (That
      # said, this fact is irrelevant for this test.)
      state.apply_action(game.card_to_int[Card(ACE, CLUB)])
      state.apply_action(game.card_to_int[Card(DEUCE, SPADE)])
      state.apply_action(game.card_to_int[Card(SEVEN, HEART)])
      state.apply_action(game.card_to_int[Card(EIGHT, DIAMOND)])
      state.apply_action(game.card_to_int[Card(ACE, SPADE)])
      state.apply_action(game.card_to_int[Card(DEUCE, CLUB)])

      state.apply_action(60)  # P2 raises
      state.apply_action(ACTION_CHECK_OR_CALL)  # P0 calls

      # This is the last spot where the allowed actions > 1 should match normal
      # PokerkitWrapper - since on all following streets both players will have
      # contributed chips on this street (and thus ACPC style actions will be
      # different).
      expected_actions = FOLD_AND_CHECK_OR_CALL_ACTIONS + list(range(100, 2001))
      self.assertEqual(state.legal_actions(), expected_actions)
      state.apply_action(ACTION_CHECK_OR_CALL)  # P1 calls

      # Deal flop
      state.apply_action(game.card_to_int[Card(JACK, SPADE)])
      state.apply_action(game.card_to_int[Card(JACK, CLUB)])
      state.apply_action(game.card_to_int[Card(JACK, DIAMOND)])

      # Unlike pokerkit, the actions should start from 60+20=80 instead of 20
      # since it's factoring in all contributions on prior streets - in this
      # case, the preflop contribution of 60 chips from each player.
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

      # Again, now including the 60 from preflop + 30 from flop, + another 20
      # for the minimum bet size of one big blind.
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
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
      state.apply_action(ACTION_FOLD)  # P1 folds
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
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
      # each player gets their first hole card before any player gets their
      # second hole card. As such, to match universal_poker we have to deal the
      # cards in a specific order ourselves (rather than 'all 2c 2h 2d 2s 3c
      # 3d').
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
        self.assertEqual(
            flop_card, game.card_to_int[expected_flop_cards.pop(0)]
        )
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
      self.assertEqual(state.legal_actions(), FOLD_AND_CHECK_OR_CALL_ACTIONS)
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
      # NOTE: order is shuffled to match universal_poker since pokerkit deals
      # hole cards more 'realistically' than universal_poker. (See the
      # pokerkit_wrapper_acpc_style docstring for more details.)
      hole_card_action_sequence = [10, 12, 11, 13]
      for action in hole_card_action_sequence:
        action_string = state.action_to_string(pyspiel.PlayerId.CHANCE, action)
        # Print mainly to match universal_poker_test.cc. (Though, it IS also
        # nice to prove that this did actually give us something we can actually
        # use here.)
        print(f"Applying action ({action_string})")
        state.apply_action(action)

      # If we've reached this point without crashing / without the game breaking
      # then we've succeeded! (So this asssert isn't technically necessary ...
      # though it is nice to have to verify the game is still working fine
      # here.)
      self.assertFalse(state.is_terminal())

    # TODO: b/437724266 - Parameterize or add a similar test for limit holdem.
    @parameterized.parameters(
        "100 100",
        "25 100",
        "100 25",
        "99 100",
        "100 99",
        "98 100",
        "100 98",
        "97 100",
        "100 97",
        # Testing extremely small post-blind stacks (e.g. 11-10 => 1 chip left)
        "100 11",
        "11 100",
        "100 12",
        "12 100",
        "100 13",
        "13 100",
        # Test having exactly one BB left in stacks post-blind
        "100 20",
        "20 100",
    )
    def test_headsup_random_hands_match_universal_poker_gameplay(self, stacks):
      # universal_poker is an optional dependency, so we need to check it's
      # registered before trying to use it.
      if "universal_poker" not in pyspiel.registered_names():
        self.skipTest("universal_poker is not registered.")
      universal_poker_game_string = (
          "universal_poker("
          "betting=nolimit,"
          "bettingAbstraction=fullgame,"
          "blind=10 5,"
          "firstPlayer=2 1 1 1,"
          "numBoardCards=0 3 1 1,"
          "numHoleCards=2,"
          "numPlayers=2,"
          "numRanks=13,"
          "numRounds=4,"
          "numSuits=4,"
          f"stack={stacks})"
      )
      universal_poker_game = pyspiel.load_game(universal_poker_game_string)
      pokerkit_game = PokerkitWrapperAcpcStyle(
          params={
              "variant": "NoLimitTexasHoldem",
              "num_players": 2,
              "blinds": "5 10",
              "stack_sizes": " ".join(stacks.split(" ")[::-1]),
          }
      )
      min_stack_size = min([int(s) for s in stacks.split(" ")])
      max_stack_size = max([int(s) for s in stacks.split(" ")])

      # The two games represent shoving over effective stack differently, and in
      # ways that aren't easy to account for otherwise. So instead it's easier
      # to just relax the strictness with which we do the assert when comparing
      # betting histories.
      def replace_all_shoves_with_min_stack(betting_history):
        for size in range(min_stack_size, max_stack_size + 1):
          if f"r{size}" in betting_history:
            betting_history = betting_history.replace(
                f"r{size}", f"r{min_stack_size}"
            )
        return betting_history

      # Each hand is quick so we can run a decent number of them without the
      # test taking too long.
      number_hands_to_play = 5
      for _ in range(number_hands_to_play):
        pokerkit_state = pokerkit_game.new_initial_state()
        universal_poker_state = universal_poker_game.new_initial_state()
        # pokerkit_wrapper and universal_poker deal hole cards differently -
        # pokerkit_wrapper rotates, universal_poker does all for each player
        # one at a time.
        hole_cards = random.sample(range(52), 4)
        for card in hole_cards:
          pokerkit_state.apply_action(card)
        for per_player_pair in zip(hole_cards[0:2], hole_cards[2:4]):
          universal_poker_state.apply_action(per_player_pair[0])
          universal_poker_state.apply_action(per_player_pair[1])

        step = 0
        while True:
          step += 1
          if step >= 200:  # Mostly-arbitrary limit.
            self.fail(
                "Detected probable infinite loop. Hand had a very large number"
                " of actions."
            )

          self.assertEqual(
              universal_poker_state.current_player(),
              pokerkit_state.current_player(),
          )
          self.assertEqual(
              universal_poker_state.returns(), pokerkit_state.returns()
          )

          up_struct = universal_poker_state.to_struct()
          pk_struct = pokerkit_state.to_struct()
          self.assertEqual(up_struct.current_player, pk_struct.current_player)
          # Blinds and stacks should be in reverse order since otherwise the
          # player number won't line up.
          self.assertEqual(up_struct.blinds, pk_struct.blinds[::-1])
          self.assertEqual(
              up_struct.starting_stacks, pk_struct.starting_stacks[::-1]
          )

          self.assertEqual(
              replace_all_shoves_with_min_stack(up_struct.betting_history),
              replace_all_shoves_with_min_stack(pk_struct.betting_history),
              f"universal_poker betting_history: {up_struct.betting_history},"
              f" pokerkit_wrapper betting_history: {pk_struct.betting_history},"
              " underlying operations were"
              f" {list(pokerkit_state._wrapped_state.operations)}",
          )
          if pokerkit_state.is_terminal():
            self.assertTrue(universal_poker_state.is_terminal())
            break

          self.assertEqual(
              universal_poker_state.is_chance_node(),
              pokerkit_state.is_chance_node(),
          )
          if pokerkit_state.is_chance_node():
            self.assertEqual(
                universal_poker_state.chance_outcomes(),
                pokerkit_state.chance_outcomes(),
            )

            random_action = random.choice(
                [o for o, _ in pokerkit_state.chance_outcomes()]
            )
            universal_poker_state.apply_action(random_action)
            pokerkit_state.apply_action(random_action)
          else:
            universal_poker_actions = universal_poker_state.legal_actions()
            pokerkit_actions = pokerkit_state.legal_actions()
            # Inside the assert, replacing actions that are larger than the
            # smallest stack size with said value. Since doing so is
            # strategically identical to shoving all-in, but the games handle
            # them very differently and in ways that aren't easy to account for
            # otherwise here.
            self.assertEqual(
                set([
                    action if action <= min_stack_size else min_stack_size
                    for action in universal_poker_actions
                ]),
                set([
                    action if action <= min_stack_size else min_stack_size
                    for action in pokerkit_actions
                ]),
            )
            random_action = random.choice(pokerkit_actions)
            pokerkit_state.apply_action(random_action)
            if random_action not in universal_poker_actions:
              if min_stack_size in universal_poker_actions:
                universal_poker_state.apply_action(min_stack_size)
              else:
                self.assertGreaterEqual(random_action, min_stack_size)
                self.assertIn(max_stack_size, universal_poker_actions)
                universal_poker_state.apply_action(max_stack_size)
            else:
              universal_poker_state.apply_action(random_action)

    @parameterized.parameters(
        "10 10",
        "100 10",
        "10 100",
        "10 20",
        "20 10",
        "10 11",
        "11 10",
    )
    def test_headsup_immediate_all_in_matches_universal_poker(self, stacks):
      # universal_poker is an optional dependency, so we need to check it's
      # registered before trying to use it.
      if "universal_poker" not in pyspiel.registered_names():
        self.skipTest("universal_poker is not registered.")
      universal_poker_game_string = (
          "universal_poker("
          "betting=nolimit,"
          "bettingAbstraction=fullgame,"
          "blind=10 5,"
          "firstPlayer=2 1 1 1,"
          "numBoardCards=0 3 1 1,"
          "numHoleCards=2,"
          "numPlayers=2,"
          "numRanks=13,"
          "numRounds=4,"
          "numSuits=4,"
          f"stack={stacks})"
      )
      universal_poker_game = pyspiel.load_game(universal_poker_game_string)
      pokerkit_game = PokerkitWrapperAcpcStyle(
          params={
              "variant": "NoLimitTexasHoldem",
              "num_players": 2,
              "blinds": "5 10",
              "stack_sizes": " ".join(stacks.split(" ")[::-1]),
          }
      )
      pokerkit_state = pokerkit_game.new_initial_state()
      universal_poker_state = universal_poker_game.new_initial_state()
      # pokerkit_wrapper and universal_poker deal hole cards differently -
      # pokerkit_wrapper rotates, universal_poker does all for each player
      # one at a time.
      hole_cards = range(4)
      for card in hole_cards:
        pokerkit_state.apply_action(card)
      for per_player_pair in zip(hole_cards[0:2], hole_cards[2:4]):
        universal_poker_state.apply_action(per_player_pair[0])
        universal_poker_state.apply_action(per_player_pair[1])

      # Ensure both players immediately contribute one big blind if they haven't
      # already (forcing immediate showdown).
      if (
          not universal_poker_state.is_chance_node()
          or not pokerkit_state.is_chance_node()
      ):
        universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
        pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertTrue(pokerkit_state.is_chance_node())
      # NOTE: unlike pokerkit_wrapper which will automatically determine that
      # the hand can procede immediately to showdown, universal_poker requires
      # you to explicitly apply a second check action in certain cases (i.e.
      # where the Big Blind has > 1 BB but the Small Blind does not). This also
      # affects the betting history below too.
      # TODO: b/459073855 - Consider whether this difference is actually
      # problematic or not. And whether we want the ACPC style variant to start
      # simulating this need for a check action...
      if not universal_poker_state.is_chance_node():
        universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)

      for board_card in range(5, 10):
        self.assertTrue(pokerkit_state.is_chance_node())
        self.assertTrue(universal_poker_state.is_chance_node())
        pokerkit_betting_history = pokerkit_state.to_struct().betting_history
        universal_poker_betting_history = (
            universal_poker_state.to_struct().betting_history
        )
        # As discussed above, the extra forced check by universal_poker does
        # indeed show up here, causing a small discrepancy in betting history
        # in certain cases.
        if not (
            pokerkit_betting_history == "c///"
            and universal_poker_betting_history == "cc///"
        ):
          self.assertEqual(
              universal_poker_betting_history, pokerkit_betting_history
          )
        self.assertEqual(
            pokerkit_state.chance_outcomes(),
            universal_poker_state.chance_outcomes(),
        )
        pokerkit_state.apply_action(board_card)
        universal_poker_state.apply_action(board_card)

      self.assertTrue(pokerkit_state.is_terminal())
      self.assertTrue(universal_poker_state.is_terminal())
      self.assertEqual(
          universal_poker_state.returns(), pokerkit_state.returns()
      )

    def test_4p_hand_generation_with_forced_check_call_until_turn(self):
      """Tests a few edge cases involving multiple players + check actions.

      Particularly useful for verifying that we handle the betting history
      properly / do not over-rely on the ACPC logs in a way that delays check
      or call actions from showing up in time.

      Also demonstrates that if we don't go our of our way to enable fractional
      pot splitting (see test below) the discrepency vs Universal Poker is still
      bounded properly, ie in a 4 player NLHE game no more than 4/3 BBs.
      """
      # universal_poker is an optional dependency, so we need to check it's
      # registered before trying to use it.
      if "universal_poker" not in pyspiel.registered_names():
        self.skipTest("universal_poker is not registered.")
      universal_poker_game_string = (
          "universal_poker("
          "betting=nolimit,"
          "bettingAbstraction=fullgame,"
          "blind=5 10 0 0,"
          "firstPlayer=3 1 1 1,"
          "numBoardCards=0 3 1 1,"
          "numHoleCards=2,"
          "numPlayers=4,"
          "numRanks=13,"
          "numRounds=4,"
          "numSuits=4,"
          "stack=100 100 100 100)"
      )
      universal_poker_game = pyspiel.load_game(universal_poker_game_string)
      pokerkit_game = PokerkitWrapperAcpcStyle(
          params={
              "variant": "NoLimitTexasHoldem",
              "num_players": 4,
              "blinds": "5 10",
              "stack_sizes": "100 100 100 100",
          }
      )
      # These hands aren't too too slow, so we can run a couple of them in a
      # row here without the test taking too long.
      for _ in range(25):
        pokerkit_state = pokerkit_game.new_initial_state()
        universal_poker_state = universal_poker_game.new_initial_state()
        # pokerkit_wrapper and universal_poker deal hole cards differently -
        # pokerkit_wrapper rotates, universal_poker does all for each player
        # one at a time.
        hole_cards = random.sample(range(52), 8)
        for card in hole_cards:
          pokerkit_state.apply_action(card)
        for per_player_pair in zip(hole_cards[0:4], hole_cards[4:8]):
          universal_poker_state.apply_action(per_player_pair[0])
          universal_poker_state.apply_action(per_player_pair[1])

        step = 0
        while True:
          step += 1
          if step >= 400:  # Mostly-arbitrary limit.
            self.fail(
                "Detected probable infinite loop. Hand had a very large number"
                " of actions."
            )

          self.assertEqual(
              universal_poker_state.current_player(),
              pokerkit_state.current_player(),
          )

          # Universal Poker returns fractional BigBlinds in cases where
          # pokerkit would 'unfairly' award the BigBlind to one specific player
          # (in a way that would be difficult to exactly account for) when
          # we passed in integer stack sizes instead of floats. But we
          # can assume that even in the worst case this only results in a
          # 1-(1/N) discrepency once per N-2 splits, totalling:
          #   ((1-(1/N)) * N-2) => 1.333...
          # delta for N=4.
          self.assertSequenceAlmostEqual(
              universal_poker_state.returns(),
              pokerkit_state.returns(),
              delta=1.34,
          )

          up_struct = universal_poker_state.to_struct()
          pk_struct = pokerkit_state.to_struct()
          self.assertEqual(up_struct.current_player, pk_struct.current_player)
          self.assertEqual(up_struct.starting_stacks, pk_struct.starting_stacks)
          self.assertEqual(
              up_struct.betting_history,
              pk_struct.betting_history,
              f"up betting_history: {up_struct.betting_history},"
              f" pk betting_history: {pk_struct.betting_history},"
              " underlying operations were"
              f" {list(pokerkit_state._wrapped_state.operations)}",
          )
          if pokerkit_state.is_terminal():
            self.assertTrue(universal_poker_state.is_terminal())
            break

          self.assertEqual(
              universal_poker_state.is_chance_node(),
              pokerkit_state.is_chance_node(),
          )
          if pokerkit_state.is_chance_node():
            self.assertEqual(
                universal_poker_state.chance_outcomes(),
                pokerkit_state.chance_outcomes(),
            )
            random_action = random.choice(
                [o for o, _ in pokerkit_state.chance_outcomes()]
            )
            universal_poker_state.apply_action(random_action)
            pokerkit_state.apply_action(random_action)
          else:
            self.assertEqual(
                universal_poker_state.legal_actions(),
                pokerkit_state.legal_actions(),
            )
            is_preflop_or_flop_betting = (
                len(pokerkit_state._wrapped_state.board_cards) <= 3
            )
            if (
                is_preflop_or_flop_betting
                and ACTION_CHECK_OR_CALL in pokerkit_state.legal_actions()
            ):
              action = ACTION_CHECK_OR_CALL
            else:
              action = random.choice(pokerkit_state.legal_actions())
            universal_poker_state.apply_action(action)
            pokerkit_state.apply_action(action)

    def test_can_have_fractional_pot_splitting_via_4p_post_turn_randomize(self):
      """Tests a few edge cases involving multiple players + check actions.

      Specifically: forces all 4 players to check/call until the turn, i.e.
      preflop and on the flop, to get a wider variety of spots tested. And then
      causes the players to play randomly while checking betting history and
      the resulting returns() at the end of the game.

      Particularly useful for making sure that:
      - we can handle fractional pot splitting exactly like Universal Poker
      - we handle check/call actions properly, and specifically aren't
      over-relying on the ACPC logs (which as per the spec don't immediately
      update for all players instantly upon check/call actions)

      """
      # universal_poker is an optional dependency, so we need to check it's
      # registered before trying to use it.
      if "universal_poker" not in pyspiel.registered_names():
        self.skipTest("universal_poker is not registered.")
      universal_poker_game_string = (
          "universal_poker("
          "betting=nolimit,"
          "bettingAbstraction=fullgame,"
          "blind=5 10 0 0,"
          "firstPlayer=3 1 1 1,"
          "numBoardCards=0 3 1 1,"
          "numHoleCards=2,"
          "numPlayers=4,"
          "numRanks=13,"
          "numRounds=4,"
          "numSuits=4,"
          "stack=100 100 100 100)"
      )
      universal_poker_game = pyspiel.load_game(universal_poker_game_string)
      pokerkit_game = PokerkitWrapperAcpcStyle(
          params={
              "variant": "NoLimitTexasHoldem",
              "num_players": 4,
              "blinds": "5.0 10.0",
              # Floats to ensure fractional returns instead of floor division.
              # For more details see
              # https://pokerkit.readthedocs.io/en/stable/simulation.html#optional-divmod
              "stack_sizes": "100.0 100.0 100.0 100.0",
          }
      )
      # These hands aren't too too slow, so we can run a couple of them in a
      # row here without the test taking too long.
      for _ in range(25):
        pokerkit_state = pokerkit_game.new_initial_state()
        universal_poker_state = universal_poker_game.new_initial_state()
        # pokerkit_wrapper and universal_poker deal hole cards differently -
        # pokerkit_wrapper rotates, universal_poker does all for each player
        # one at a time.
        hole_cards = random.sample(range(52), 8)
        for card in hole_cards:
          pokerkit_state.apply_action(card)
        for per_player_pair in zip(hole_cards[0:4], hole_cards[4:8]):
          universal_poker_state.apply_action(per_player_pair[0])
          universal_poker_state.apply_action(per_player_pair[1])

        step = 0
        while True:
          step += 1
          if step >= 400:  # Mostly-arbitrary limit.
            self.fail(
                "Detected probable infinite loop. Hand had a very large number"
                " of actions."
            )

          self.assertEqual(
              universal_poker_state.current_player(),
              pokerkit_state.current_player(),
          )

          # Major crux of this test - when using float stack inputs we expect
          # the returns here to *exactly* match Universal Poker's.
          self.assertSequenceEqual(
              universal_poker_state.returns(),
              pokerkit_state.returns(),
          )

          up_struct = universal_poker_state.to_struct()
          pk_struct = pokerkit_state.to_struct()
          self.assertEqual(up_struct.current_player, pk_struct.current_player)
          self.assertEqual(up_struct.starting_stacks, pk_struct.starting_stacks)
          self.assertEqual(
              up_struct.betting_history,
              pk_struct.betting_history,
              f"up betting_history: {up_struct.betting_history},"
              f" pk betting_history: {pk_struct.betting_history},"
              " underlying operations were"
              f" {list(pokerkit_state._wrapped_state.operations)}",
          )
          if pokerkit_state.is_terminal():
            self.assertTrue(universal_poker_state.is_terminal())
            break

          self.assertEqual(
              universal_poker_state.is_chance_node(),
              pokerkit_state.is_chance_node(),
          )
          if pokerkit_state.is_chance_node():
            self.assertEqual(
                universal_poker_state.chance_outcomes(),
                pokerkit_state.chance_outcomes(),
            )
            random_action = random.choice(
                [o for o, _ in pokerkit_state.chance_outcomes()]
            )
            universal_poker_state.apply_action(random_action)
            pokerkit_state.apply_action(random_action)
          else:
            self.assertEqual(
                universal_poker_state.legal_actions(),
                pokerkit_state.legal_actions(),
            )
            is_preflop_or_flop_betting = (
                len(pokerkit_state._wrapped_state.board_cards) <= 3
            )
            if (
                is_preflop_or_flop_betting
                and ACTION_CHECK_OR_CALL in pokerkit_state.legal_actions()
            ):
              action = ACTION_CHECK_OR_CALL
            else:
              action = random.choice(pokerkit_state.legal_actions())
            universal_poker_state.apply_action(action)
            pokerkit_state.apply_action(action)

    def test_acpc_logs_match_spec_when_differing_from_universal_poker(self):
      """Verifies that delayed check/calls in pokerkit ACPC logs are all WAI.

      See https://pokerkit.readthedocs.io/en/stable/_static/protocol.pdf,
      specifically the examples at the end of the document.

      (This specifically verifies that we recreate the exact
      "Two player no-limit Texas Hold'em" example.)
      """
      # universal_poker is an optional dependency, so we need to check it's
      # registered before trying to use it.
      if "universal_poker" not in pyspiel.registered_names():
        self.skipTest("universal_poker is not registered.")

      # Taken from the "Two player no-limit Texas Hold'em" example given at the
      # end of https://pokerkit.readthedocs.io/en/stable/_static/protocol.pdf,
      # except marked as hand 0 instead of hand 30 (0:0 not 0:30).
      expected_acpc_logs_p0 = [
          ["S->", "MATCHSTATE:0:0::9s8h|\r\n"],
          ["S->", "MATCHSTATE:0:0:c:9s8h|\r\n"],
          ["<-C", "MATCHSTATE:0:0:c:9s8h|:c\r\n"],
          ["S->", "MATCHSTATE:0:0:cc/:9s8h|/8c8d5c\r\n"],
          ["<-C", "MATCHSTATE:0:0:cc/:9s8h|/8c8d5c:r250\r\n"],
          ["S->", "MATCHSTATE:0:0:cc/r250:9s8h|/8c8d5c\r\n"],
          ["S->", "MATCHSTATE:0:0:cc/r250c/:9s8h|/8c8d5c/6s\r\n"],
          ["<-C", "MATCHSTATE:0:0:cc/r250c/:9s8h|/8c8d5c/6s:r500\r\n"],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500:9s8h|/8c8d5c/6s\r\n"],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500c/:9s8h|/8c8d5c/6s/2d\r\n"],
          [
              "<-C",
              "MATCHSTATE:0:0:cc/r250c/r500c/:9s8h|/8c8d5c/6s/2d:r1250\r\n",
          ],
          [
              "S->",
              "MATCHSTATE:0:0:cc/r250c/r500c/r1250:9s8h|/8c8d5c/6s/2d\r\n",
          ],
          [
              "S->",
              # NOTE: Since tournament rules, P1's hand actually just gets
              # mucked instead of being shown. Causing this to correctly differ
              # from what's written in the spec example (no 9c6h)
              # "MATCHSTATE:0:0:cc/r250c/r500c/r1250c:9s8h|9c6h/8c8d5c/6s/2d\r\n",
              "MATCHSTATE:0:0:cc/r250c/r500c/r1250c:9s8h|/8c8d5c/6s/2d\r\n",
          ],
      ]

      universal_poker_game_string = (
          "universal_poker("
          "betting=nolimit,"
          "bettingAbstraction=fullgame,"
          "blind=10 5,"
          "firstPlayer=2 1 1 1,"
          "numBoardCards=0 3 1 1,"
          "numHoleCards=2,"
          "numPlayers=2,"
          "numRanks=13,"
          "numRounds=4,"
          "numSuits=4,"
          "stack=2000 2000)"
      )
      universal_poker_game = pyspiel.load_game(universal_poker_game_string)
      pokerkit_game = PokerkitWrapperAcpcStyle(
          params={
              "variant": "NoLimitTexasHoldem",
              "num_players": 2,
              "blinds": "5 10",
              "stack_sizes": "2000 2000",
          }
      )
      pokerkit_state = pokerkit_game.new_initial_state()
      universal_poker_state = universal_poker_game.new_initial_state()

      hole_cards = [
          pokerkit_game.card_to_int[card]
          for card in [
              pokerkit.Card(NINE, SPADE),  # p0 card 0
              pokerkit.Card(NINE, CLUB),  # p1 card 0
              pokerkit.Card(EIGHT, HEART),  # p0 card 1
              pokerkit.Card(SIX, HEART),  # p1 card 1
          ]
      ]
      for card in hole_cards:
        pokerkit_state.apply_action(card)
      for per_player_pair in zip(hole_cards[0:2], hole_cards[2:4]):
        universal_poker_state.apply_action(per_player_pair[0])
        universal_poker_state.apply_action(per_player_pair[1])
      self.assertSequenceEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:1],
      )

      # NOTE: This is counterintuitive, but because it's a headsup game the
      # first player to act is actually P1 (which is indeed SB/BTN despite the
      # blind order being <SB BB>).
      self.assertEqual(pokerkit_state.current_player(), 1)

      # P1 (SB/BTN) checks, immediately shows up since it's not closing out the
      # betting round / since P0 (BB) needs to act next...
      pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertSequenceEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:2],
      )
      # (redundant check to be DAMP + certain this is correct)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:c:9s8h|\r\n"],
      )

      # ...and we can confirm this is correct by observing that the other
      self.assertSequenceEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[1],
          [
              ["S->", "MATCHSTATE:1:0::|9c6h\r\n"],
              ["<-C", "MATCHSTATE:1:0::|9c6h:c\r\n"],
              ["S->", "MATCHSTATE:1:0:c:|9c6h\r\n"],
          ],
      )
      # (and that it's now P0 BB's turn)
      self.assertEqual(pokerkit_state.current_player(), 0)

      # Now, P0 checking will finish the betting round, but because we're
      # looking at P0's view we'll still see the message to the server go out.
      pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertSequenceEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:3],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["<-C", "MATCHSTATE:0:0:c:9s8h|:c\r\n"],
      )

      # Dealing Flop
      for action in [
          pokerkit_game.card_to_int[card]
          for card in [
              pokerkit.Card(EIGHT, CLUB),
              pokerkit.Card(EIGHT, DIAMOND),
              pokerkit.Card(FIVE, CLUB),
          ]
      ]:
        pokerkit_state.apply_action(action)
        universal_poker_state.apply_action(action)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:4],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/:9s8h|/8c8d5c\r\n"],
      )

      # (proof that now it's P0 BB's turn first postflop)
      self.assertEqual(pokerkit_state.current_player(), 0)

      # Now, since we are looking via the perspective of P0 BB (first to act
      # player) as we proved directly above, we'll see this raise 250's message
      # to the server + that it's reflected immediately.
      pokerkit_state.apply_action(250)
      universal_poker_state.apply_action(250)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:6],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250:9s8h|/8c8d5c\r\n"],
      )

      # Now, naively we might expect to see the call already in the log (ie
      # r250c). But, in reality with how ACPC spec is defined the opponent P1's
      # call will not be reflected until the next street's card has been dealt!
      pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:6],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250:9s8h|/8c8d5c\r\n"],
      )

      # Dealing Turn
      action = pokerkit_game.card_to_int[Card(SIX, SPADE)]
      pokerkit_state.apply_action(action)
      universal_poker_state.apply_action(action)
      # NOW finally the flop's call action shows up! (along with the 6s turn)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:7],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250c/:9s8h|/8c8d5c/6s\r\n"],
      )

      # P0 bets 500 - rinse and repeat what happened on the flop
      pokerkit_state.apply_action(500)
      universal_poker_state.apply_action(500)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:9],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500:9s8h|/8c8d5c/6s\r\n"],
      )

      # P1 calls 500, but again doesn't show up in the logs until the river.
      pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:9],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500:9s8h|/8c8d5c/6s\r\n"],
      )

      # Which similarly doesn't match universal poker (along with other stuff)
      # TODO: b/457647698 - Uncomment after fixing universal_poker. This
      # currently results in
      # ```
      # AssertionError:
      # - STATE:0:cc/r250c/r500c/:9s8h|9c6h/8c8d5c/6s/2c
      # + STATE:0:cc/r250c/r500c/:9s8h|9c6h/8c8d5c/6s/
      # ```
      # We should investigate why 2c shows up here. (This seems like a potential
      # bug in universal_poker's acpc_state and/or its to_string(), especially
      # since this fake river card vanishes as expected imediately below when we
      # deal the actual 2d river.)
      # self.assertEqual(
      #     universal_poker_state.to_struct().acpc_state,
      #     "STATE:0:cc/r250c/r500c/:9s8h|9c6h/8c8d5c/6s/\n"
      #     "Spent: [P0: 500  P1: 500  ]\n",
      # )

      # Dealing River
      action = pokerkit_game.card_to_int[Card(DEUCE, DIAMOND)]
      pokerkit_state.apply_action(action)
      universal_poker_state.apply_action(action)
      # Finally we see r500c instead of r500 (only along with the 2d river)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:10],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500c/:9s8h|/8c8d5c/6s/2d\r\n"],
      )

      # Double check universal_poker lines up as expected still too just in case
      self.assertEqual(
          universal_poker_state.to_struct().acpc_state,
          "STATE:0:cc/r250c/r500c/:9s8h|9c6h/8c8d5c/6s/2d\n"
          "Spent: [P0: 500  P1: 500  ]\n",
      )

      # And yet again, large bet => call. But difference is that now the hand
      # ends after the raise is called, so we immediatley see the call show up!
      pokerkit_state.apply_action(1250)
      universal_poker_state.apply_action(1250)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:12],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          ["S->", "MATCHSTATE:0:0:cc/r250c/r500c/r1250:9s8h|/8c8d5c/6s/2d\r\n"],
      )

      pokerkit_state.apply_action(ACTION_CHECK_OR_CALL)
      universal_poker_state.apply_action(ACTION_CHECK_OR_CALL)
      # (i.e. this actually differs from above, unlike before!)
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0],
          expected_acpc_logs_p0[0:13],
      )
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[0][-1],
          [
              "S->",
              # Note: despite being showdown, the other player's hand gets
              # mucked instead of shown. This is correct given pokerkit's
              # stricter 'tournament' rule set, but differs from the example
              # given in the pokerkit docs.
              "MATCHSTATE:0:0:cc/r250c/r500c/r1250c:9s8h|/8c8d5c/6s/2d\r\n",
              # i.e. normally this would instead be:
              # "MATCHSTATE:0:0:cc/r250c/r500c/r1250c:9s8h|9c6h/8c8d5c/6s/2d\r\n",
          ],
      )

      # Verify that P1 on the other hand DOES get to see P0's hand as well at
      # showdown.
      self.assertEqual(
          pokerkit_state.to_struct().per_player_acpc_logs[1][-1],
          [
              "S->",
              "MATCHSTATE:1:0:cc/r250c/r500c/r1250c:9s8h|9c6h/8c8d5c/6s/2d\r\n",
          ],
      )

      # Finally, prove that this all aligns with the Universal Poker's ToStruct
      # results despite not literally matching exactly.
      self.assertEqual(
          universal_poker_state.to_struct().acpc_state,
          "STATE:0:cc/r250c/r500c/r1250c:9s8h|9c6h/8c8d5c/6s/2d\n"
          "Spent: [P0: 1250  P1: 1250  ]\n",
      )


if __name__ == "__main__":
  if IMPORTED_ALL_LIBRARIES:
    absltest.main()
  else:
    logging.warning("Skipping test because not all libraries were imported.")
