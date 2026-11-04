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

"""Gomoku: moves, horizontal and vertical wins, and pyspiel observations."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gomoku

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


class GomokuTest(parameterized.TestCase):
  """Gomoku against OpenSpiel's rules.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = gomoku.Gomoku()

  def test_init(self):
    """The initial state is empty with player 0 to move."""
    state = self.game.new_initial_state()
    self.assertEqual(state.current_player, 0)
    self.assertFalse(state.is_terminal)
    self.assertEqual(state.winner, -1)
    self.assertTrue(jnp.all(state.legal_actions_mask))

  def test_apply_action(self):
    """A move places a stone and passes the turn."""
    state = self.game.new_initial_state()
    # Player 0 plays in center (approx)
    action = 15 * 7 + 7
    state = self.game.apply_action(state, jnp.array(action))
    self.assertEqual(state.board[action], 0)
    self.assertEqual(state.current_player, 1)

    # Player 1 plays next to it
    action2 = action + 1
    state = self.game.apply_action(state, jnp.array(action2))
    self.assertEqual(state.board[action2], 1)
    self.assertEqual(state.current_player, 0)

  def test_win_horizontal(self):
    """Five in a row wins."""
    state = self.game.new_initial_state()
    # P0 plays 0, 1, 2, 3, 4
    # P1 plays 15, 16, 17, 18
    moves = [0, 15, 1, 16, 2, 17, 3, 18, 4]
    for m in moves:
      state = self.game.apply_action(state, jnp.array(m))

    self.assertTrue(state.is_terminal)
    self.assertEqual(state.winner, 0)

  def test_win_vertical(self):
    """Five in a column wins."""
    state = self.game.new_initial_state()
    # P0 plays in col 0: 0, 15, 30, 45, 60
    # P1 plays in col 1: 1, 16, 31, 46
    moves = [0, 1, 15, 16, 30, 31, 45, 46, 60]
    for m in moves:
      state = self.game.apply_action(state, jnp.array(m))

    self.assertTrue(state.is_terminal)
    self.assertEqual(state.winner, 0)

  @parameterized.parameters(0, 1)
  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_observation_from_os(self, player):
    """observation_from_os equals the JAX observation.

    Args:
      player: The observing player.
    """
    os_game = pyspiel.load_game("gomoku")
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
