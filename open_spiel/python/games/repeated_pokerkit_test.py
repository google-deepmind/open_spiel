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

import random

from absl.testing import absltest
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


class RepeatedPokerkitTest(absltest.TestCase):
  """Test the OpenSpiel game 'Repeated Pokerkit'."""

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

  def test_random_sim_cash_game_6max(self):
    # Note the reset_stacks=True
    game = pyspiel.load_game(
        "python_repeated_pokerkit("
        "max_num_hands=20,reset_stacks=True,rotate_dealer=True,"
        "pokerkit_game_params=python_pokerkit_wrapper("
        "num_players=6,blinds=5 10,"
        "stack_sizes=1000 1000 1000 1000 1000 1000))"
    )
    pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=True)

  def test_random_sim_tournament_6max(self):
    # Note the reset_stacks=False
    game = pyspiel.load_game(
        "python_repeated_pokerkit("
        "max_num_hands=20,reset_stacks=False,rotate_dealer=True,"
        "pokerkit_game_params=python_pokerkit_wrapper("
        "num_players=6,blinds=5 10,"
        "stack_sizes=1000 1000 1000 1000 1000 1000))"
    )
    pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=True)

  def test_blind_schedule_progression(self):
    blind_schedule_str = "1,10,20;2,20,40;1,50,100;1,1000,1000"
    params = {
        "max_num_hands": 1000,
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


# TODO: b/444333187 - Add tests for the following behaviors:
# - blind schedule becoming higher than all players' remaining stacks. (THIS IS
#   A KNOWN ISSUE - in that case, all players are currently marked as eliminated
#   instead of being forced all-in)
# - having reset_stacks=True
# - reaching a low max_num_hands limit
# - multiple players busting out (especially edge cases)
# - other games besides no limit texas hold'em
# - returns()'s values being correct when reaching terminal state


if __name__ == "__main__":
  absltest.main()
