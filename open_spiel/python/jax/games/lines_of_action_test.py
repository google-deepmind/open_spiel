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

"""Lines of Action: the initial position, a move, and pyspiel observations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import lines_of_action

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


class LinesOfActionTest(parameterized.TestCase):
  """Lines of Action against OpenSpiel's rules.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = lines_of_action.LinesOfAction()

  def test_init(self):
    """The initial position is OpenSpiel's."""
    state = self.game.new_initial_state()
    self.assertEqual(state.current_player, 0)
    self.assertFalse(state.is_terminal)
    self.assertEqual(state.winner, -1)

    # Check initial board setup
    self.assertEqual(state.board[1], 0)  # (0,1) Black
    self.assertEqual(state.board[8], 1)  # (1,0) White
    self.assertEqual(state.board[9], -1)  # (1,1) Empty

    self.assertTrue(jnp.any(state.legal_actions_mask))

  def test_apply_action(self):
    """A legal move moves a piece."""
    state = self.game.new_initial_state()

    legal_indices = jnp.where(state.legal_actions_mask)[0]
    self.assertNotEmpty(legal_indices)
    action = legal_indices[0]

    state = self.game.apply_action(state, action)
    self.assertEqual(state.current_player, 1)
    self.assertEqual(state.move_count, 1)

  @parameterized.parameters(0, 1)
  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_observation_from_os(self, player):
    """observation_from_os equals the JAX observation.

    Args:
      player: The observing player.
    """
    os_game = pyspiel.load_game("lines_of_action")
    os_state = os_game.new_initial_state()
    jax_state = self.game.new_initial_state()

    legal_indices = jnp.where(jax_state.legal_actions_mask)[0]
    action = legal_indices[0]

    os_state.apply_action(int(action))
    jax_state = self.game.apply_action(jax_state, action)

    jax_obs = self.game.observation_tensor(jax_state, jnp.array(player))
    jax_obs_from_os = self.game.observation_from_os(os_state, player)

    np.testing.assert_array_almost_equal(jax_obs, jax_obs_from_os)

  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_default_move_limit_is_openspiels(self):
    """The default move limit is OpenSpiel's."""
    self.assertEqual(
        self.game.max_game_length,
        pyspiel.load_game("lines_of_action").max_game_length(),
    )

  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_recreating_a_seen_board_is_a_draw(self):
    """Recreating a seen board is a draw."""
    # Both sides move out and back: the fourth move recreates the initial
    # board, which counts as seen.
    moves = [
        _action(0, 1, 2, 1),
        _action(1, 0, 1, 2),
        _action(2, 1, 0, 1),
        _action(1, 2, 1, 0),
    ]
    os_state = pyspiel.load_game("lines_of_action").new_initial_state()
    state = self.game.new_initial_state()
    for i, move in enumerate(moves):
      self.assertFalse(bool(state.is_terminal), msg=f"move {i}")
      self.assertTrue(bool(state.legal_actions_mask[move]), msg=f"move {i}")
      os_state.apply_action(move)
      state = self.game.apply_action(state, jnp.int32(move))
    self.assertTrue(os_state.is_terminal())
    self.assertTrue(bool(state.is_terminal))
    self.assertEqual(int(state.winner), 2)
    np.testing.assert_array_equal(self.game.returns(state), os_state.returns())
    self.assertFalse(bool(jnp.any(state.legal_actions_mask)))

  def test_board_history_from_holds_only_that_board(self):
    """board_history_from holds only the given board."""
    state = _position(
        self.game, black=[(0, 0), (0, 3)], white=[(7, 7), (5, 4)], move_count=6
    )
    seen = self.game.board_history_from(state.board, state.move_count)
    self.assertEqual(seen.shape[0], self.game.max_game_length + 1)
    self.assertTrue(bool(jnp.any(seen[6] != 0)))
    self.assertFalse(bool(jnp.any(seen[:6] != 0)))

  def test_a_group_longer_than_eight_steps_is_connected(self):
    """A group longer than eight steps is connected."""
    # Twelve stones in one chain: the far end is eleven king steps from the
    # first stone.
    chain = [(0, c) for c in range(8)] + [(r, 7) for r in range(1, 5)]
    board = _board(self.game, black=chain, white=[(7, 0)])
    # Reaches private _is_connected of the code under test.
    self.assertTrue(bool(self.game._is_connected(board, jnp.int32(0))))  # pylint: disable=protected-access
    broken = _board(self.game, black=chain[:-2] + [(4, 7)], white=[(7, 0)])
    # Reaches private _is_connected of the code under test.
    self.assertFalse(bool(self.game._is_connected(broken, jnp.int32(0))))  # pylint: disable=protected-access

  def test_connecting_move_wins(self):
    """A connecting move wins."""
    state = _position(
        self.game, black=[(0, 0), (0, 3)], white=[(7, 7), (5, 5)], move_count=10
    )
    move = _action(0, 3, 0, 1)
    self.assertTrue(bool(state.legal_actions_mask[move]))
    after = self.game.apply_action(state, jnp.int32(move))
    self.assertTrue(bool(after.is_terminal))
    np.testing.assert_array_equal(self.game.returns(after), [1.0, -1.0])

  def test_the_move_limit_overrides_a_connection(self):
    """The move limit overrides a connection."""
    # OpenSpiel checks the move limit before the connection: the last move
    # draws even when it connects.
    game = lines_of_action.LinesOfAction(max_game_length=11)
    state = _position(
        game, black=[(0, 0), (0, 3)], white=[(7, 7), (5, 5)], move_count=10
    )
    after = game.apply_action(state, jnp.int32(_action(0, 3, 0, 1)))
    self.assertTrue(bool(after.is_terminal))
    self.assertEqual(int(after.winner), 2)
    np.testing.assert_array_equal(game.returns(after), [0.0, 0.0])

  def test_repetition_overrides_a_connection(self):
    """A repetition overrides a connection."""
    # The connecting move recreates a board seen earlier: a draw.
    game = self.game
    state = _position(
        game, black=[(0, 0), (0, 3)], white=[(7, 7), (5, 5)], move_count=10
    )
    connected = _board(game, black=[(0, 0), (0, 1)], white=[(7, 7), (5, 5)])
    history = state.board_history.at[4].set(
        game.board_history_from(connected, jnp.int32(4))[4]
    )
    state = state._replace(board_history=history)
    after = game.apply_action(state, jnp.int32(_action(0, 3, 0, 1)))
    self.assertTrue(bool(after.is_terminal))
    self.assertEqual(int(after.winner), 2)


def _action(from_row, from_col, to_row, to_col, capture=0):
  """The action id of a move.

  Args:
    from_row: The from row.
    from_col: The from column.
    to_row: The to row.
    to_col: The to column.
    capture: Whether it captures.

  Returns:
    The action id.
  """
  return (((from_row * 8 + from_col) * 8 + to_row) * 8 + to_col) * 2 + capture


def _board(game, black, white):
  """A board holding the given black and white pieces.

  Args:
    game: The game.
    black: (row, col) of each black piece.
    white: (row, col) of each white piece.

  Returns:
    i32[num_cells], the board.
  """
  board = np.full(game.num_cells, -1, dtype=np.int32)
  for r, c in black:
    board[r * game.num_cols + c] = 0
  for r, c in white:
    board[r * game.num_cols + c] = 1
  return jnp.asarray(board)


def _position(game, black, white, move_count):
  """The position with `black` to move after `move_count` moves.

  Only the position itself is seen.

  Args:
    game: The game.
    black: Black's stones, (row, col) pairs.
    white: White's stones, (row, col) pairs.
    move_count: The moves played so far.

  Returns:
    The GameState.
  """
  board = _board(game, black, white)
  player = jnp.int32(0)
  return lines_of_action.GameState(
      board=board,
      current_player=player,
      is_terminal=jnp.bool_(False),
      winner=jnp.int32(-1),
      # Reaches private _compute_legal_actions of the code under test.
      legal_actions_mask=game._compute_legal_actions(board, player),  # pylint: disable=protected-access
      move_count=jnp.int32(move_count),
      board_history=game.board_history_from(board, jnp.int32(move_count)),
  )


if __name__ == "__main__":
  absltest.main()
