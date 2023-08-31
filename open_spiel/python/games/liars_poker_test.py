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
"""Tests for Python Liar's Poker."""

import pickle

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import liars_poker
import pyspiel


class LiarsPokerTest(absltest.TestCase):

  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()
    # Ensure no moves have been made.
    expected_hands = [[] for _ in range(game.num_players())]
    expected_bidder = -1
    expected_current_player = pyspiel.PlayerId.CHANCE
    expected_current_count = "None"
    expected_current_number = "None"
    expected_rebid = False
    expected = (
        "Hands: {}, Bidder: {}, Current Player: {}, Current Bid: {} of {},"
        " Rebid: {}".format(
            expected_hands,
            expected_bidder,
            expected_current_player,
            expected_current_count,
            expected_current_number,
            expected_rebid,
        )
    )
    self.assertEqual(str(state), expected)

  def test_draw_hands(self):
    """Tests hand drawing functions."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()
    expected_hands = [[] for _ in range(game.num_players())]
    for i in range(game.num_players() * game.hand_length):
      # Verify we have chance nodes until all player hands are filled.
      self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)

      # Draw a digit.
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)

      # Verify players' hands are filled correctly.
      cur_player = i % game.num_players()
      expected_hands[cur_player].append(action)
      state.apply_action(action)
      self.assertEqual(state.hands, expected_hands)
    # Assert after all hands are filled, we have non-chance nodes.
    cur_player = state.current_player()
    self.assertNotEqual(cur_player, pyspiel.PlayerId.CHANCE)
    self.assertEqual(cur_player, 0)

  def _populate_game_hands(self, game, state):
    """Populates players hands for testing."""
    for _ in range(game.num_players() * game.hand_length):
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)

  def test_basic_bid(self):
    """Tests a single bid."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()
    expected_bid_history = np.zeros(
        (state.total_possible_bids, state.num_players())
    )

    # Fill players hands.
    self._populate_game_hands(game, state)
    # After all hands are filled, have player 0 bid.
    cur_player = state.current_player()
    action = 2
    state.apply_action(action)

    # Verify bid history is updated correctly.
    bid_offset = liars_poker.BID_ACTION_OFFSET
    expected_bid_history[action - bid_offset][cur_player] = 1
    self.assertTrue((state.bid_history == expected_bid_history).all())

    # Verify next set of legal bids is greater than the current bid.
    for next_action in state.legal_actions():
      if next_action == liars_poker.CHALLENGE_ACTION:
        continue
      self.assertGreater(next_action, action)

  def _verify_returns(self, game, state):
    self.assertTrue(state.winner() != -1 or state.loser() != -1)
    actual_returns = state.returns()
    if state.winner() != -1:
      expected_returns = [-1.0 for _ in range(game.num_players())]
      expected_returns[state.winner()] = game.num_players() - 1
    else:
      expected_returns = [1.0 for _ in range(game.num_players())]
      expected_returns[state.loser()] = -1.0 * (game.num_players() - 1)
    self.assertEqual(actual_returns, expected_returns)

  def test_single_random_round(self):
    """Runs a single round of bidding followed by a challenge."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()
    expected_challenge_history = np.zeros(
        (state.total_possible_bids, state.num_players())
    )

    # Fill players hands.
    self._populate_game_hands(game, state)
    # Have player 0 bid.
    action = 2
    state.apply_action(action)
    # Verify challenge action is available to the next player.
    challenge = liars_poker.CHALLENGE_ACTION
    self.assertIn(challenge, state.legal_actions())
    # Player 1 challenges.
    cur_player = state.current_player()
    state.apply_action(challenge)
    bid_offset = liars_poker.BID_ACTION_OFFSET
    expected_challenge_history[action - bid_offset][cur_player] = 1
    # Verify challenge history is updated correctly.
    self.assertTrue(
        (state.challenge_history == expected_challenge_history).all()
    )
    # Original bidder challenges, thus agreeing to a count.
    cur_player = state.current_player()
    state.apply_action(challenge)
    expected_challenge_history[action - bid_offset][cur_player] = 1
    # Verify challenge history is updated correctly.
    self.assertTrue(
        (state.challenge_history == expected_challenge_history).all()
    )

    # Verify game is over.
    self.assertTrue(state.is_terminal())
    # Verify returns.
    self._verify_returns(game, state)

  def test_single_deterministic_round(self):
    """Runs a single round where cards are dealt deterministically."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()

    # Deal player 0 all "1" cards and player 1 all "2" cards.
    for i in range(game.num_players() * game.hand_length):
      if i % 2 == 0:
        # Deal card to player 0
        state.apply_action(1)
      else:
        # Deal card to player 1
        state._apply_action(2)

    # Have player 0 bid that there are four 1's.
    state.apply_action(state.encode_bid(4, 1) + liars_poker.BID_ACTION_OFFSET)
    # Player 1 challenges.
    state.apply_action(liars_poker.CHALLENGE_ACTION)
    # Player 0 accepts the challenge.
    state.apply_action(liars_poker.CHALLENGE_ACTION)
    # Verify game ends with player 0 losing.
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.loser(), 0)
    expected_returns = [1.0 for _ in range(game.num_players())]
    expected_returns[state.loser()] = -1.0 * (game.num_players() - 1)
    self.assertEqual(state.returns(), expected_returns)

  def test_single_rebid(self):
    """Runs a 2 player game where a rebid is enacted."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()

    # Fill players hands.
    self._populate_game_hands(game, state)
    # Have player 0 bid.
    state.apply_action(2)
    # Player 1 challenges.
    state.apply_action(liars_poker.CHALLENGE_ACTION)
    # Original bidder rebids.
    state.apply_action(3)
    # Verify game is not over.
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.returns(), [0.0 for _ in range(game.num_players())])
    # Player 1 challenges again.
    state.apply_action(liars_poker.CHALLENGE_ACTION)

    # Verify game is now over.
    self.assertTrue(state.is_terminal())
    self._verify_returns(game, state)

  def test_rebid_then_new_bid(self):
    """Runs a 2 player game where a rebid is enacted."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()

    # Fill players hands.
    self._populate_game_hands(game, state)
    # Have player 0 bid.
    state.apply_action(2)
    # Player 1 challenges.
    state.apply_action(liars_poker.CHALLENGE_ACTION)
    # Original bidder rebids.
    state.apply_action(3)
    # Verify game is not over.
    self.assertFalse(state.is_terminal())
    self.assertEqual(state.returns(), [0.0 for _ in range(game.num_players())])
    # Player 1 bids.
    state.apply_action(4)
    # Verify game is not over.
    self.assertFalse(state.is_terminal())
    # Player 0 challenges.
    state.apply_action(liars_poker.CHALLENGE_ACTION)
    # Verify we're not rebidding and counts is only called once both players
    # challenge.
    self.assertFalse(state.is_terminal())
    # Player 1 challenges and ends the game with a counts.
    state.apply_action(liars_poker.CHALLENGE_ACTION)

    # Verify game is now over.
    self.assertTrue(state.is_terminal())
    self._verify_returns(game, state)

  def test_game_from_cc(self):
    """Runs the standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_liars_poker", {"players": 2})
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_pickle(self):
    """Checks pickling and unpickling of game and state."""
    game = pyspiel.load_game("python_liars_poker")
    pickled_game = pickle.dumps(game)
    unpickled_game = pickle.loads(pickled_game)
    self.assertEqual(str(game), str(unpickled_game))
    state = game.new_initial_state()
    for a in [2, 3, 4, 5]:
      state.apply_action(a)
    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))

  def test_cloned_state_matches_original_state(self):
    """Check we can clone states successfully."""
    game = liars_poker.LiarsPoker({"hand_length": 3, "num_digits": 3})
    state = game.new_initial_state()
    state.apply_action(1)
    state.apply_action(2)
    clone = state.clone()

    self.assertEqual(state.history(), clone.history())
    self.assertEqual(state.num_players(), clone.num_players())
    self.assertEqual(state.move_number(), clone.move_number())
    self.assertEqual(state.num_distinct_actions(), clone.num_distinct_actions())

    self.assertEqual(state._current_player, clone._current_player)
    self.assertEqual(state._current_action, clone._current_action)
    np.testing.assert_array_equal(state.bid_history, clone.bid_history)
    np.testing.assert_array_equal(
        state.challenge_history, clone.challenge_history
    )


if __name__ == "__main__":
  absltest.main()
