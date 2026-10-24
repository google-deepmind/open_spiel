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

"""Breakthrough plays in OpenSpiel's action ids.

Its egocentric view reproduces the same game in compact per-player actions.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import breakthrough

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False

DOWN, UP = 1, 4  # Straight-ahead directions of black and white.
DOWN_RIGHT = 2


def _id(row, col, direction, capture=0):
  """The action id of a move on the 8-column board.

  Args:
    row: The row.
    col: The column.
    direction: The direction.
    capture: Whether it captures.

  Returns:
    The action id.
  """
  return breakthrough.action_id(row, col, direction, capture, columns=8)


class BreakthroughTest(parameterized.TestCase):
  """Breakthrough against OpenSpiel's rules."""

  @classmethod
  def setUpClass(cls):
    """Builds the default game and its jitted step."""
    super().setUpClass()
    cls.game = breakthrough.Breakthrough()
    cls.apply = staticmethod(jax.jit(cls.game.apply_action))

  def test_initial_state(self):
    """The initial position is OpenSpiel's."""
    state = self.game.new_initial_state()
    self.assertEqual(int(state.current_player), 0)
    self.assertFalse(bool(state.is_terminal))
    self.assertEqual(state.pieces.tolist(), [16, 16])
    self.assertEqual(self.game.num_actions, 8 * 8 * 12)
    self.assertTrue(bool(state.legal_actions_mask[_id(1, 0, DOWN)]))
    self.assertFalse(bool(state.legal_actions_mask[_id(1, 0, DOWN, 1)]))

  def test_move_and_capture(self):
    """Moves and a capture."""
    state = self.game.new_initial_state()
    for action in (
        _id(1, 0, DOWN),
        _id(6, 1, UP),
        _id(2, 0, DOWN),
        _id(5, 1, UP),
    ):
      state = self.apply(state, jnp.int32(action))
    capture = _id(3, 0, DOWN_RIGHT, 1)
    self.assertTrue(bool(state.legal_actions_mask[capture]))
    self.assertFalse(bool(state.legal_actions_mask[capture - 1]))
    state = self.apply(state, jnp.int32(capture))
    self.assertEqual(int(state.board[3 * 8]), -1)
    self.assertEqual(int(state.board[4 * 8 + 1]), 0)
    self.assertEqual(state.pieces.tolist(), [16, 15])

  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_legal_actions_equal_pyspiel_along_random_games(self):
    """Legal actions equal pyspiel's along random games."""
    os_game = pyspiel.load_game("breakthrough")
    rng = np.random.default_rng(0)
    for _ in range(3):
      os_state = os_game.new_initial_state()
      state = self.game.new_initial_state()
      while not os_state.is_terminal():
        mask = np.asarray(self.game.legal_actions_mask(state))
        self.assertEqual(
            np.flatnonzero(mask).tolist(), sorted(os_state.legal_actions())
        )
        for player in (0, 1):
          np.testing.assert_array_equal(
              self.game.observation_tensor(state, jnp.int32(player)),
              self.game.observation_from_os(os_state, player),
          )
        action = int(rng.choice(np.flatnonzero(mask)))
        os_state.apply_action(action)
        state = self.apply(state, jnp.int32(action))
      self.assertTrue(bool(state.is_terminal))
      np.testing.assert_array_equal(
          self.game.returns(state), os_state.returns()
      )

  def test_egocentric_view_plays_the_same_game(self):
    """The egocentric view plays the same game."""
    view = breakthrough.egocentric_view(self.game)
    self.assertEqual(view.num_actions, 8 * 8 * 3)
    rng = np.random.default_rng(1)
    state = self.game.new_initial_state()
    while not bool(state.is_terminal):
      mask = np.asarray(view.legal_actions_mask(state))
      self.assertEqual(
          mask.sum(), int(np.asarray(state.legal_actions_mask).sum())
      )
      action = int(rng.choice(np.flatnonzero(mask)))
      player = int(state.current_player)
      obs = np.asarray(view.observation_tensor(state, jnp.int32(player)))
      base_obs = np.asarray(
          self.game.observation_tensor(state, jnp.int32(player))
      )
      np.testing.assert_array_equal(
          obs, base_obs[::-1, ::-1] if player == 1 else base_obs
      )
      state = view.apply_action(state, jnp.int32(action))

  def test_egocentric_actions_move_forward_for_both_players(self):
    """Egocentric actions move forward for both players."""
    view = breakthrough.egocentric_view(self.game)
    state = self.game.new_initial_state()
    # Compact action (row 1, col 0, straight) in each player's own view.
    state = view.apply_action(state, jnp.int32((1 * 8 + 0) * 3 + 1))
    self.assertEqual(int(state.board[2 * 8 + 0]), 0)
    state = view.apply_action(state, jnp.int32((1 * 8 + 0) * 3 + 1))
    # White's own (1, 0) is the board's (6, 7); it moves up to (5, 7).
    self.assertEqual(int(state.board[5 * 8 + 7]), 1)


if __name__ == "__main__":
  absltest.main()
