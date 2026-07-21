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
"""Tests for Python Belote."""

# Tests reach into BeloteState's private fields to set up and verify
# specific deals/tricks deterministically.
# pylint: disable=protected-access

import pickle

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import belote
import pyspiel


def _deal_hands(state):
  """Applies chance actions until the initial 5-card deal + turned card."""
  while state.current_player(
  ) == pyspiel.PlayerId.CHANCE and state._phase == "deal":
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(int(action))


def _finish_dealing(state):
  """Applies chance actions until a bidding phase is reached again."""
  while state.current_player() == pyspiel.PlayerId.CHANCE:
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(int(action))


class BeloteTest(absltest.TestCase):
  """Tests for the BeloteGame and BeloteState classes."""

  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertEqual(state.hands, [[] for _ in range(4)])

  def test_deal_and_turn_card(self):
    """Checks the initial deal gives 5 cards to each player and turns one up."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)

    for hand in state.hands:
      self.assertLen(hand, 5)
    self.assertIsNotNone(state._turned_card)
    self.assertEqual(state._phase, "bid1")
    # Bidding starts to the left of the dealer.
    self.assertEqual(state.current_player(), (state._dealer + 1) % 4)
    self.assertCountEqual(state.legal_actions(),
                          [belote.PASS_ACTION, belote.TAKE_ACTION])

  def test_round1_take_completes_deal_to_eight_cards(self):
    """If a player takes in round 1, everyone ends up with 8 cards."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)

    taker = state.current_player()
    turned_suit = belote.card_suit(state._turned_card)
    state.apply_action(belote.TAKE_ACTION)
    _finish_dealing(state)

    for hand in state.hands:
      self.assertLen(hand, 8)
    self.assertEqual(state._taker, taker)
    self.assertEqual(state._trump_suit, turned_suit)
    self.assertEqual(state._phase, "play")
    # Total cards in play must still be the full deck.
    all_cards = sorted(c for hand in state.hands for c in hand)
    self.assertEqual(all_cards, list(range(32)))

  def test_round2_choice_completes_deal_to_eight_cards(self):
    """If everyone passes round 1, round 2 lets players pick another suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    turned_suit = belote.card_suit(state._turned_card)

    for _ in range(4):
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._phase, "bid2")
    legal = state.legal_actions()
    self.assertIn(belote.PASS_ACTION, legal)
    self.assertNotIn(belote.CHOOSE_SUIT_ACTION_BASE + turned_suit, legal)
    self.assertLen(legal, 4)  # pass + 3 remaining suits

    chosen_suit = next(s for s in range(4) if s != turned_suit)
    chooser = state.current_player()
    state.apply_action(belote.CHOOSE_SUIT_ACTION_BASE + chosen_suit)
    _finish_dealing(state)

    for hand in state.hands:
      self.assertLen(hand, 8)
    self.assertEqual(state._taker, chooser)
    self.assertEqual(state._trump_suit, chosen_suit)
    self.assertEqual(state._phase, "play")
    all_cards = sorted(c for hand in state.hands for c in hand)
    self.assertEqual(all_cards, list(range(32)))

  def test_all_pass_twice_redeals_and_rotates_dealer(self):
    """If everyone passes both rounds, the deal restarts with next dealer."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    original_dealer = state._dealer
    _deal_hands(state)
    for _ in range(8):  # 4 passes in round 1, 4 in round 2.
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._phase, "deal")
    self.assertEqual(state._dealer, (original_dealer + 1) % 4)
    self.assertEqual(state.hands, [[] for _ in range(4)])

  def test_must_follow_suit(self):
    """A player holding the led suit must play a card of that suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._phase = "play"
    state._trump_suit = 3  # Spades trump; leading suit below is Clubs.
    state._trick = [(0, 0)]  # Player 0 led the 7 of Clubs.
    state.hands[1] = [1, 8, 16]  # 8 of Clubs, 7 of Diamonds, 7 of Hearts.
    legal = state._legal_card_plays(1)
    self.assertEqual(legal, [1])  # Must follow with the Clubs card.

  def test_must_trump_when_void_and_opponent_winning(self):
    """A player void in the led suit must cut with trump if opponent leads."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    state._trick = [(0, 0)]  # Player 0 led 7 of Clubs; player 2 is partner.
    state.hands[1] = [16, 24]  # 7 of Hearts, 7 of Spades (trump).
    legal = state._legal_card_plays(1)
    self.assertEqual(legal, [24])  # Must cut with the only trump held.

  def test_no_obligation_to_trump_when_partner_winning(self):
    """No obligation to cut or overtrump while the partner holds the trick."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    state._trick = [(3, 0)]  # Partner (player 3) led and is winning.
    state.hands[1] = [16, 24]  # 7 of Hearts, 7 of Spades (trump).
    legal = state._legal_card_plays(1)
    self.assertCountEqual(legal, [16, 24])

  def test_must_overtrump_when_trump_led_even_if_partner_winning(self):
    """When trump itself is led, must overtrump even if partner is winning."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    # Player 3 (partner of player 1) led the 10 of Spades and currently wins.
    state._trick = [(3, 3 * 8 + 3)]
    state.hands[1] = [3 * 8 + 1, 3 * 8 + 4]  # 8 of Spades, Jack of Spades.
    legal = state._legal_card_plays(1)
    # Only the Jack of Spades outranks the 10 of Spades in the trump order;
    # the 8 of Spades does not, so it is forbidden despite the partner leading.
    self.assertEqual(legal, [3 * 8 + 4])

  def test_trick_winner_trump_beats_lead_suit(self):
    """A trump card always beats a non-trump card, regardless of lead suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    trick = [(0, 7), (1, 24), (2, 6),
             (3, 16)]  # Ace of Clubs vs 7 of Spades, etc.
    self.assertEqual(state._trick_winner(trick), 1)  # Trump always wins.

  def test_full_random_game_scores_correctly(self):
    """Plays a full random game and sanity-checks final scoring invariants."""
    game = belote.BeloteGame()
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes_with_probs = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes_with_probs)
          action = np.random.choice(action_list, p=prob_list)
        else:
          legal = state.legal_actions()
          action = np.random.choice(legal)
        state.apply_action(int(action))

      returns = state.returns()
      self.assertAlmostEqual(sum(returns), 0.0)
      self.assertEqual(returns[0], returns[2])
      self.assertEqual(returns[1], returns[3])
      # 162 normally, or 252 in the (rare, random-play) case of a capot.
      self.assertIn(sum(state._team_points), (162, 252))

  def test_belote_rebelote_not_applied_by_default(self):
    """Without the parameter, holding K+Q of trump earns no bonus."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, -1)

    state._declarer_team = 0
    state._team_points = [100, 62]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 100 - 62)

  def test_belote_rebelote_bonus_awarded_to_declarer_when_enabled(self):
    """With the parameter on, the holder's team gets a 20-point bonus."""
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, belote.team_of(0))

    state._declarer_team = 0
    state._team_points = [100, 62]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], (100 + 20) - 62)

  def test_belote_rebelote_bonus_awarded_to_defenders_when_enabled(self):
    """The bonus goes to whichever team holds K+Q, even if defending."""
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[1] = [6, 5]  # King and Queen of Clubs held by player 1.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, belote.team_of(1))

    state._declarer_team = 0
    state._team_points = [100, 62]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 100 - (62 + 20))

  def test_belote_rebelote_bonus_survives_failed_contract(self):
    """The bonus is paid even if the holder's team scores 0 trick points."""
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, belote.team_of(0))

    state._declarer_team = 0
    # Declarer team fails its contract outright: 0 trick points, so the
    # 162 trick points all go to the defenders, but the 20-point bonus
    # for holding K+Q of trump is unaffected by the failed contract.
    state._team_points = [0, 162]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 20 - 162)
    self.assertEqual(state.returns()[1], 162 - 20)

  def test_belote_rebelote_bonus_can_flip_a_failed_contract_to_success(self):
    """Official rule: the belote total counts toward contract success.

    75 raw trick points would fail the plain >81 threshold, but 75+20=95
    exceeds the defenders' 87, so the contract must succeed and the
    declarer keeps its 75 points (plus the 20-point bonus).
    """
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, belote.team_of(0))

    state._declarer_team = 0
    state._team_points = [75, 87]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], (75 + 20) - 87)

  def test_belote_rebelote_bonus_can_flip_a_made_contract_to_failure(self):
    """Symmetric case: the defenders' belote can sink an otherwise-made contract.

    The declarer's raw 85 trick points clear the plain >81 threshold, but
    the defenders hold belote (77+20=97 > 85), so the contract fails and
    the declarer keeps nothing.
    """
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[1] = [6, 5]  # King and Queen of Clubs held by player 1.
    state._enter_play_phase()
    self.assertEqual(state._belote_team, belote.team_of(1))

    state._declarer_team = 0
    state._team_points = [85, 77]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 0 - (162 + 20))

  def test_belote_rebelote_requires_same_player_to_hold_both_cards(self):
    """Splitting K and Q of trump across partners does not earn the bonus."""
    game = belote.BeloteGame({"use_belote_rebelote": True})
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6]  # King of Clubs held by player 0.
    state.hands[2] = [5]  # Queen of Clubs held by partner (player 2).
    state._enter_play_phase()
    self.assertEqual(state._belote_team, -1)

  def test_full_random_game_with_belote_rebelote_scores_correctly(self):
    """Same sanity checks as above, with the belote/rebelote bonus enabled."""
    game = belote.BeloteGame({"use_belote_rebelote": True})
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes_with_probs = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes_with_probs)
          action = np.random.choice(action_list, p=prob_list)
        else:
          legal = state.legal_actions()
          action = np.random.choice(legal)
        state.apply_action(int(action))

      returns = state.returns()
      self.assertAlmostEqual(sum(returns), 0.0)
      self.assertEqual(returns[0], returns[2])
      self.assertEqual(returns[1], returns[3])
      trick_total = sum(state._team_points)
      self.assertIn(trick_total, (162, 252))
      bonus = belote._BELOTE_REBELOTE_BONUS if state._belote_team >= 0 else 0
      self.assertLessEqual(abs(returns[0]), trick_total + bonus)

  def test_capot_awards_100_point_last_trick_bonus(self):
    """If one team wins all 8 tricks, the last trick is worth 100, not 10."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs; the final trick below is all Hearts.
    state._declarer_team = 0
    state._tricks_played = 7
    state._trick_winners = [0, 2, 0, 2, 0, 2, 0]  # Team 0 won every trick.
    state._team_points = [140, 12]

    # Final trick: player 0's Ace of Hearts (23) beats 8/9/10 of Hearts.
    state._trick = []
    state._trick_leader = 0
    state._current_player_play = 0
    state.hands[0] = [23]
    state.hands[1] = [17]
    state.hands[2] = [18]
    state.hands[3] = [19]
    state._apply_play_action(23, 0)
    state._apply_play_action(17, 1)
    state._apply_play_action(18, 2)
    state._apply_play_action(19, 3)

    # Card points in the trick: A=11, 8=0, 9=0, 10=10, i.e. 21, plus the
    # 100-point capot bonus (instead of the usual 10).
    self.assertEqual(state._team_points[0], 140 + 21 + 100)
    self.assertEqual(sum(state._team_points), 140 + 12 + 21 + 100)

  def test_non_capot_last_trick_keeps_10_point_bonus(self):
    """If tricks were split between teams, the last trick is worth only 10."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs; the final trick below is all Hearts.
    state._declarer_team = 0
    state._tricks_played = 7
    state._trick_winners = [0, 2, 0, 2, 0, 2, 1]  # Team 1 won one trick.
    state._team_points = [130, 22]

    state._trick = []
    state._trick_leader = 0
    state._current_player_play = 0
    state.hands[0] = [23]
    state.hands[1] = [17]
    state.hands[2] = [18]
    state.hands[3] = [19]
    state._apply_play_action(23, 0)
    state._apply_play_action(17, 1)
    state._apply_play_action(18, 2)
    state._apply_play_action(19, 3)

    self.assertEqual(state._team_points[0], 130 + 21 + 10)
    self.assertEqual(sum(state._team_points), 130 + 22 + 21 + 10)

  def test_capot_gives_defenders_252_points_on_failed_contract(self):
    """A capot changes the trick total used when the contract fails."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._declarer_team = 0
    # Team 1 (the defenders) achieved a capot: all 252 points are theirs.
    state._team_points = [0, 252]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 0 - 252)
    self.assertEqual(state.returns()[1], 252 - 0)

  def test_game_from_cc(self):
    """Runs the standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_belote")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=False)

  def test_pickle(self):
    """Checks pickling and unpickling of game and state."""
    game = pyspiel.load_game("python_belote")
    pickled_game = pickle.dumps(game)
    unpickled_game = pickle.loads(pickled_game)
    self.assertEqual(str(game), str(unpickled_game))

    state = game.new_initial_state()
    _deal_hands(state)
    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))

  def test_cloned_state_matches_original_state(self):
    """Check we can clone states successfully."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    clone = state.clone()

    self.assertEqual(state.history(), clone.history())
    self.assertEqual(state.num_players(), clone.num_players())
    self.assertEqual(state.move_number(), clone.move_number())
    self.assertEqual(state.num_distinct_actions(), clone.num_distinct_actions())
    self.assertEqual(state.hands, clone.hands)
    self.assertEqual(state._turned_card, clone._turned_card)


if __name__ == "__main__":
  absltest.main()
