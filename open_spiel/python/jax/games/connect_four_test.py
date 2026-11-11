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

"""Connect Four: drops, wins in each direction, full columns and terminals.

Terminal states are left unchanged by further actions; pyspiel states are
observed as the JAX game observes its own.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import connect_four

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


class ConnectFourTest(parameterized.TestCase):
  """Connect Four against OpenSpiel's rules.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = connect_four.ConnectFour()

  def test_init(self):
    """The initial state is empty with player 0 to move."""
    state = self.game.new_initial_state()
    self.assertEqual(state.current_player, 0)
    self.assertFalse(state.is_terminal)
    self.assertEqual(state.winner, -1)
    self.assertTrue(jnp.all(state.legal_actions_mask))

  def test_column_drop(self):
    """A piece drops to the bottom of its column."""
    state = self.game.new_initial_state()
    # Drop piece in column 0
    state = self.game.apply_action(state, jnp.array(0))
    # Should be at the bottom of column 0
    # Col 0, Row 5 is index 35 in flat array.
    self.assertEqual(state.board[35], 0)
    self.assertEqual(state.current_player, 1)

    # Player 1 drops in column 0
    state = self.game.apply_action(state, jnp.array(0))
    # Should be at row 4 col 0 = 28.
    self.assertEqual(state.board[28], 1)
    self.assertEqual(state.current_player, 0)

  def test_win_horizontal(self):
    """Four in a row wins."""
    state = self.game.new_initial_state()
    moves = [0, 0, 1, 1, 2, 2, 3]
    for m in moves:
      state = self.game.apply_action(state, jnp.array(m))

    self.assertTrue(state.is_terminal)
    self.assertEqual(state.winner, 0)

  def test_redundant_apply_action_preserves_terminal_winner(self):
    """An action at a terminal state keeps the winner."""
    state = self.game.new_initial_state()
    moves = [0, 0, 1, 1, 2, 2, 3]
    for m in moves:
      state = self.game.apply_action(state, jnp.array(m))

    redundant_state = self.game.apply_action(state, jnp.array(4))

    self.assertTrue(redundant_state.is_terminal)
    self.assertEqual(redundant_state.winner, state.winner)

  def test_win_vertical(self):
    """Four in a column wins."""
    state = self.game.new_initial_state()
    moves = [0, 1, 0, 1, 0, 1, 0]
    for m in moves:
      state = self.game.apply_action(state, jnp.array(m))

    self.assertTrue(state.is_terminal)
    self.assertEqual(state.winner, 0)

  def test_full_column_disables_move(self):
    """A full column is not a legal move."""
    state = self.game.new_initial_state()
    # Play 6 moves in column 0
    for _ in range(6):
      state = self.game.apply_action(state, jnp.array(0))

    # Column 0 should be full
    self.assertFalse(state.legal_actions_mask[0])

  @parameterized.parameters(0, 1)
  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_observation_from_os(self, player):
    """observation_from_os equals the JAX observation.

    Args:
      player: The observing player.
    """
    os_game = pyspiel.load_game("connect_four")
    os_state = os_game.new_initial_state()
    os_state.apply_action(0)
    os_state.apply_action(1)

    # Create corresponding JAX state
    jax_state = self.game.new_initial_state()
    jax_state = self.game.apply_action(jax_state, jnp.array(0))
    jax_state = self.game.apply_action(jax_state, jnp.array(1))

    jax_obs = self.game.observation_tensor(jax_state, jnp.array(player))
    jax_obs_from_os = self.game.observation_from_os(os_state, player)

    np.testing.assert_array_almost_equal(jax_obs, jax_obs_from_os)


if __name__ == "__main__":
  absltest.main()
