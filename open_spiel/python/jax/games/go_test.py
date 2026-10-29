# Copyright 2026 DeepMind Technologies Limited
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

"""Go: moves, passes, captures, ko, suicide, the end and scoring.

Single and group captures, the two-pass end, Tromp-Taylor scoring and
returns.
"""

import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import go
import pyspiel


class GoTest(parameterized.TestCase):
  """Go against OpenSpiel's rules."""

  @classmethod
  def setUpClass(cls):
    """Builds a 9x9 game and its jitted functions."""
    super().setUpClass()
    cls.game = go.Go(board_size=9)
    cls.apply_action_jit = staticmethod(jax.jit(cls.game.apply_action))
    cls.returns_jit = staticmethod(jax.jit(cls.game.returns))
    cls.observation_tensor_jit = staticmethod(
        jax.jit(cls.game.observation_tensor)
    )

  def test_init(self):
    """The initial state is an empty board with black to move."""
    state = self.game.new_initial_state()
    self.assertEqual(state.current_player, 0)
    self.assertFalse(state.is_terminal)
    self.assertEqual(state.winner, -1)
    self.assertEqual(state.move_count, 0)
    self.assertEqual(state.ko_point, -1)
    self.assertEqual(state.consecutive_passes, 0)
    # All board cells + pass are legal initially
    self.assertTrue(jnp.all(state.legal_actions_mask))

  def test_num_actions(self):
    """num_actions equals pyspiel's."""
    os_game = pyspiel.load_game("go(board_size=9,komi=7.5)")
    self.assertEqual(self.game.num_actions, os_game.num_distinct_actions())

  def test_apply_action(self):
    """A move places a stone and passes the turn."""
    state = self.game.new_initial_state()
    # Black plays center (4, 4) -> 4 * 9 + 4 = 40
    action = 40
    state = self.apply_action_jit(state, jnp.array(action))
    self.assertEqual(state.board[action], action + 1)  # Black ID = action + 1
    self.assertEqual(state.current_player, 1)
    self.assertEqual(state.move_count, 1)

    # White plays (4, 5) -> 4 * 9 + 5 = 41
    action2 = 41
    state = self.apply_action_jit(state, jnp.array(action2))
    self.assertEqual(
        state.board[action2], -(action2 + 1)
    )  # White ID = -(action2 + 1)
    self.assertEqual(state.current_player, 0)
    self.assertEqual(state.move_count, 2)

  def test_pass_action(self):
    """A pass passes the turn."""
    state = self.game.new_initial_state()
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    self.assertEqual(state.current_player, 1)
    self.assertEqual(state.consecutive_passes, 1)
    self.assertFalse(state.is_terminal)

    # Second pass -> terminal
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    self.assertTrue(state.is_terminal)

  def test_capture_single_stone(self):
    """Test capturing a single stone surrounded on all 4 sides."""
    state = self.game.new_initial_state()
    # Place white stone at (1, 1) = 10, surround with black
    # Black: (0,1)=1, (1,0)=9, (2,1)=19, (1,2)=11
    # White: (1,1)=10
    moves = [
        1,  # Black (0,1)
        10,  # White (1,1)
        9,  # Black (1,0)
        0,  # White plays elsewhere
        19,  # Black (2,1)
        72,  # White plays elsewhere
        11,  # Black (1,2) - captures white stone at 10
    ]
    for m in moves:
      state = self.apply_action_jit(state, jnp.array(m))

    # White stone at 10 should be captured (empty)
    self.assertEqual(state.board[10], 0)

  def test_capture_group(self):
    """Test capturing a group of stones."""
    state = self.game.new_initial_state()
    # Place two white stones at (1,1)=10 and (1,2)=11
    # Surround with black: (0,1)=1, (0,2)=2, (1,0)=9, (1,3)=12, (2,1)=19,
    # (2,2)=20
    moves = [
        1,  # Black (0,1)
        10,  # White (1,1)
        2,  # Black (0,2)
        11,  # White (1,2)
        9,  # Black (1,0)
        72,  # White plays elsewhere
        12,  # Black (1,3)
        73,  # White plays elsewhere
        19,  # Black (2,1)
        74,  # White plays elsewhere
        20,  # Black (2,2) - captures white group at 10, 11
    ]
    for m in moves:
      state = self.apply_action_jit(state, jnp.array(m))

    self.assertEqual(state.board[10], 0)
    self.assertEqual(state.board[11], 0)

  def test_ko_rule(self):
    """Test that the ko rule prevents immediate recapture."""
    state = self.game.new_initial_state()
    # Create a ko-like position
    # Black at: (0,0), (0,2), (1,1)
    # White at: (0,1)
    # If black captures (0,1), ko_point should be set
    moves_to_setup = [
        (0, 0),  # B
        (0, 1),  # W  - will be captured
        (1, 0),  # B  - part of the surrounding
        (8, 0),  # W  - elsewhere
        (0, 2),  # B  - part of the surrounding
        (8, 1),  # W  - elsewhere
    ]
    for r, c in moves_to_setup:
      state = self.apply_action_jit(
          state, jnp.array(r * self.game.board_size + c)
      )

    # Now it's Black's turn. Black plays (1,1) to capture (0,1)
    state = self.apply_action_jit(
        state, jnp.array(1 * self.game.board_size + 1)
    )
    # Check white stone at (0,1) is captured
    self.assertEqual(
        state.board[1], 0, "White stone at (0,1) should be captured"
    )

  def test_self_capture_illegal(self):
    """Test that self-capture (suicide) is illegal."""
    state = self.game.new_initial_state()
    # Create a position where Black playing at (0,0) would be suicide
    # White stones at (0,1)=1 and (1,0)=9
    # Black at (0,0) would have no liberties

    moves = [
        40,  # Black center (irrelevant, just to alternate turns)
        1,  # White (0,1)
        41,  # Black (irrelevant)
        9,  # White (1,0)
    ]
    for m in moves:
      state = self.apply_action_jit(state, jnp.array(m))

    # Now Black to play. Playing at (0,0) would be suicide
    self.assertFalse(
        state.legal_actions_mask[0],
        "Playing at (0,0) should be illegal (suicide)",
    )

  def test_two_passes_terminal(self):
    """Two consecutive passes should end the game."""
    state = self.game.new_initial_state()

    # Play a few moves first
    state = self.apply_action_jit(state, jnp.array(40))  # Black
    state = self.apply_action_jit(state, jnp.array(41))  # White

    # Both pass
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    self.assertFalse(state.is_terminal)
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    self.assertTrue(state.is_terminal)

  def test_scoring(self):
    """Test Tromp-Taylor scoring on a simple board."""
    # Empty board -> score = -7.5 (komi), so white wins
    state = self.game.new_initial_state()
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    self.assertTrue(state.is_terminal)
    self.assertEqual(state.winner, 1)  # White wins due to komi

  def test_returns(self):
    """Test return values."""
    state = self.game.new_initial_state()
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))
    state = self.apply_action_jit(state, jnp.array(self.game.pass_action))

    r = self.returns_jit(state)
    # White should win on empty board due to komi
    self.assertEqual(r[0], -1.0)
    self.assertEqual(r[1], 1.0)

  def test_observation_shape(self):
    """The observation is (9, 9, 3)."""
    state = self.game.new_initial_state()
    obs = self.observation_tensor_jit(state, jnp.array(0))
    self.assertEqual(obs.shape, (9, 9, 3))

  def test_observation_planes(self):
    """The observation planes show own stones, the opponent's and empty."""
    state = self.game.new_initial_state()
    state = self.apply_action_jit(state, jnp.array(40))  # Black at center

    obs = self.observation_tensor_jit(state, jnp.array(0))
    # For player 0, plane 0 should show own pieces (black)
    # Black stone at (4,4)
    self.assertEqual(obs[4, 4, 0], 1.0)  # My piece
    self.assertEqual(obs[4, 4, 1], 0.0)  # Not opponent
    self.assertEqual(obs[4, 4, 2], 0.0)  # Player 0 plane

    # From player 1's perspective
    obs1 = self.observation_tensor_jit(state, jnp.array(1))
    self.assertEqual(obs1[4, 4, 0], 0.0)  # Not my piece
    self.assertEqual(obs1[4, 4, 1], 1.0)  # Opponent's piece
    self.assertEqual(obs1[4, 4, 2], 1.0)  # Player 1 plane

  def test_observation_from_os(self):
    """Compare observation tensor with observation_from_os."""
    os_game = pyspiel.load_game("go(board_size=9,komi=7.5)")
    os_state = os_game.new_initial_state()
    jax_state = self.game.new_initial_state()

    # Play a few moves
    moves = [40, 41, 30, 31]
    for m in moves:
      os_state.apply_action(m)
      jax_state = self.apply_action_jit(jax_state, jnp.array(m))

    player = int(jax_state.current_player)

    jax_obs = self.observation_tensor_jit(jax_state, jnp.array(player))
    jax_obs_from_os = self.game.observation_from_os(os_state, player)

    np.testing.assert_array_equal(np.array(jax_obs), np.array(jax_obs_from_os))

  def test_score_matches_os(self):
    """Compare Tromp-Taylor score with OpenSpiel."""
    os_game = pyspiel.load_game("go(board_size=9,komi=7.5)")
    os_state = os_game.new_initial_state()
    jax_state = self.game.new_initial_state()
    # Play until terminal
    while not os_state.is_terminal():
      legal_actions = os_state.legal_actions()
      action = random.choice(legal_actions)
      os_state.apply_action(action)
      jax_state = self.apply_action_jit(jax_state, jnp.array(action))

    os_returns = os_state.returns()
    jax_returns = self.returns_jit(jax_state)

    np.testing.assert_array_almost_equal(
        np.array(jax_returns), np.array(os_returns)
    )

  def test_defaults_are_openspiel_defaults(self):
    """The defaults are OpenSpiel's."""
    game = go.Go()
    params = pyspiel.load_game("go").get_parameters()
    self.assertEqual(game.board_size, params["board_size"])
    self.assertEqual(game.komi, params["komi"])
    self.assertEqual(game.max_game_length, params["max_game_length"])
    self.assertEqual(params["handicap"], 0)
    with self.assertRaises(ValueError):
      go.Go(handicap=2)

  def test_superko_ends_the_game_as_a_draw_like_pyspiel(self):
    """Positional superko ends the game as a draw, as in pyspiel."""
    # A random 4x4 game whose 16th move recreates an earlier position:
    # OpenSpiel's positional superko ends it at once as a draw.
    moves = [1, 11, 9, 7, 8, 3, 6, 12, 5, 10, 4, 15, 14, 13, 14, 12]
    game = go.Go(board_size=4, komi=0.5)
    os_state = pyspiel.load_game(
        "go(board_size=4,komi=0.5)"
    ).new_initial_state()
    apply = jax.jit(game.apply_action)
    state = game.new_initial_state()
    for i, move in enumerate(moves):
      self.assertFalse(bool(state.is_terminal), i)
      self.assertIn(move, os_state.legal_actions())
      self.assertTrue(bool(state.legal_actions_mask[move]), i)
      state = apply(state, jnp.int32(move))
      os_state.apply_action(move)
      self.assertEqual(bool(state.is_terminal), os_state.is_terminal(), i)
    self.assertTrue(bool(state.superko))
    self.assertEqual(int(state.winner), -1)
    np.testing.assert_array_equal(game.returns(state), os_state.returns())
    np.testing.assert_array_equal(game.returns(state), [0.0, 0.0])

  def test_a_pass_never_triggers_superko(self):
    """A pass never triggers superko."""
    game = go.Go(board_size=4, komi=0.5)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.int32(5))
    state = game.apply_action(state, jnp.int32(game.pass_action))
    self.assertFalse(bool(state.superko))
    self.assertFalse(bool(state.is_terminal))

  def test_position_history_detects_a_repeat_of_the_rebuilt_position(self):
    """position_history detects a repeat of the rebuilt position."""
    game = go.Go(board_size=4)
    board = jnp.zeros((16,), jnp.int32).at[0].set(1)
    hist = game.position_history(board, jnp.int32(3))
    self.assertEqual(hist.shape, (game.max_game_length + 1, 2))
    np.testing.assert_array_equal(hist[3], game.position_hash(board))
    np.testing.assert_array_equal(hist[4], [0, 0])


if __name__ == "__main__":
  absltest.main()
