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

"""Hex plays in OpenSpiel's cell ids with OpenSpiel's defaults (swap off).

The swap rule matches `hex(swap=True)`, and the canonical view reproduces
the same game transposed for white.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import hex as hex_game

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False

_HEX_NEIGHBOURS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


def _reach(pieces: np.ndarray, seed: np.ndarray) -> np.ndarray:
  """The cells of `pieces` connected (hex adjacency) to the cells of `seed`.

  Args:
    pieces: bool[n, n], a player's pieces.
    seed: bool[n, n], the cells to grow from.

  Returns:
    bool[n, n], the connected cells.
  """
  n = pieces.shape[0]
  mask = seed & pieces
  while True:
    grown = mask.copy()
    for r, c in zip(*np.nonzero(mask)):
      for dr, dc in _HEX_NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and pieces[nr, nc]:
          grown[nr, nc] = True
    if np.array_equal(grown, mask):
      return mask
    mask = grown


def to_os_obs(jax_obs, observing_player: int) -> np.ndarray:
  """Returns OpenSpiel's hex observation of the position in a JAX observation.

  OpenSpiel's nine cell states record, for each stone, whether it is connected
  to its player's first edge, second edge, both or neither.

  Args:
    jax_obs: The JAX observation.
    observing_player: The observing player.

  Returns:
    f32[9 * n * n].
  """
  mine = np.asarray(jax_obs[:, :, 0]).astype(bool)
  theirs = np.asarray(jax_obs[:, :, 1]).astype(bool)
  p0, p1 = (mine, theirs) if observing_player == 0 else (theirs, mine)

  def edge_mask(index):
    """Returns a board mask set at `index`.

    Args:
      index: The edge cells.

    Returns:
      The mask.
    """
    m = np.zeros_like(p0)
    m[index] = True
    return m

  north = _reach(p0, edge_mask((0, slice(None))))
  south = _reach(p0, edge_mask((-1, slice(None))))
  west = _reach(p1, edge_mask((slice(None), 0)))
  east = _reach(p1, edge_mask((slice(None), -1)))
  obs = np.zeros((9,) + p0.shape, np.float32)
  b_win, w_win = north & south, west & east
  obs[0], obs[1], obs[2], obs[3] = (
      w_win,
      west & ~w_win,
      east & ~w_win,
      p1 & ~west & ~east,
  )
  obs[4] = ~p0 & ~p1
  obs[5], obs[6], obs[7], obs[8] = (
      p0 & ~north & ~south,
      south & ~b_win,
      north & ~b_win,
      b_win,
  )
  return obs.reshape(-1)


class HexTest(parameterized.TestCase):
  """Hex against OpenSpiel's hex.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the game with the swap rule."""
    super().setUp()
    self.game = hex_game.Hex(swap=True)

  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_defaults_match_pyspiel_defaults(self):
    """The defaults are pyspiel's (swap off)."""
    game = hex_game.Hex()
    os_game = pyspiel.load_game("hex")
    self.assertFalse(game.swap)
    self.assertEqual(game.num_actions, os_game.num_distinct_actions())
    self.assertEqual(game.num_actions, 121)
    state = game.new_initial_state()
    self.assertEqual(state.legal_actions_mask.shape, (121,))
    state = game.apply_action(state, jnp.int32(25))
    os_state = os_game.new_initial_state()
    os_state.apply_action(25)
    self.assertEqual(
        np.flatnonzero(np.asarray(state.legal_actions_mask)).tolist(),
        os_state.legal_actions(),
    )

  @parameterized.parameters((False, 121), (True, 122))
  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_num_actions_match_pyspiel(self, swap, expected):
    """num_actions equals pyspiel's.

    Args:
      swap: The swap rule.
      expected: The expected number of actions.
    """
    os_game = pyspiel.load_game(f"hex(swap={str(swap).lower()})")
    self.assertEqual(os_game.num_distinct_actions(), expected)
    self.assertEqual(hex_game.Hex(swap=swap).num_actions, expected)

  def test_canonical_view_without_swap(self):
    """The canonical view works without the swap rule."""
    game = hex_game.Hex(board_size=5)
    view = hex_game.canonical_view(game)
    state = view.apply_action(game.new_initial_state(), jnp.int32(7))
    self.assertEqual(view.legal_actions_mask(state).shape, (25,))

  def test_initial_state(self):
    """The initial state is empty with player 0 to move."""
    state = self.game.new_initial_state()
    self.assertEqual(int(state.current_player), 0)
    self.assertFalse(bool(state.is_terminal))
    expected_mask = [True] * self.game.num_cells + [False]
    self.assertEqual(state.legal_actions_mask.tolist(), expected_mask)

  def test_apply_action_uses_board_cells_for_both_players(self):
    """Both players' actions are board cells."""
    state = self.game.new_initial_state()
    state = self.game.apply_action(state, jnp.int32(0))
    self.assertEqual(int(state.board[0]), 0)
    self.assertTrue(bool(state.legal_actions_mask[self.game.num_cells]))
    state = self.game.apply_action(state, jnp.int32(25))  # Row 2, col 3.
    self.assertEqual(int(state.board[25]), 1)
    self.assertEqual(int(state.move_count), 2)

  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_swap_rule_matches_pyspiel(self):
    """The swap rule matches pyspiel's."""
    os_game = pyspiel.load_game("hex(board_size=11,swap=True)")
    os_state = os_game.new_initial_state()
    state = self.game.new_initial_state()
    os_state.apply_action(25)
    state = self.game.apply_action(state, jnp.int32(25))
    swap = self.game.num_cells
    self.assertIn(swap, os_state.legal_actions())
    self.assertTrue(bool(state.legal_actions_mask[swap]))
    os_state.apply_action(swap)
    state = self.game.apply_action(state, jnp.int32(swap))
    self.assertEqual(int(state.current_player), os_state.current_player())
    self.assertEqual(int(state.current_player), 0)
    self.assertEqual(int(state.board[35]), 1)  # (2, 3) mirrored to (3, 2).
    self.assertEqual(int(state.board[25]), -1)
    player = int(state.current_player)
    obs = self.game.observation_tensor(state, jnp.int32(player))
    np.testing.assert_array_equal(
        to_os_obs(obs, player),
        np.asarray(os_state.observation_tensor(), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        obs, self.game.observation_from_os(os_state, player)
    )

  @parameterized.parameters((5, True), (11, True), (5, False), (11, False))
  @absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
  def test_legal_actions_equal_pyspiel_along_random_games(self, size, swap):
    """Legal actions equal pyspiel's along random games.

    Args:
      size: The board size.
      swap: The swap rule.
    """
    game = hex_game.Hex(board_size=size, swap=swap)
    os_game = pyspiel.load_game(
        f"hex(board_size={size},swap={str(swap).lower()})"
    )
    apply = jax.jit(game.apply_action)
    rng = np.random.default_rng(size)
    for _ in range(3):
      os_state = os_game.new_initial_state()
      state = game.new_initial_state()
      while not os_state.is_terminal():
        mask = np.asarray(game.legal_actions_mask(state))
        self.assertEqual(
            np.flatnonzero(mask).tolist(), sorted(os_state.legal_actions())
        )
        action = int(rng.choice(np.flatnonzero(mask)))
        os_state.apply_action(action)
        state = apply(state, jnp.int32(action))
      self.assertTrue(bool(state.is_terminal))
      np.testing.assert_array_equal(game.returns(state), os_state.returns())

  def test_canonical_view_transposes_for_white(self):
    """The canonical view transposes the board for white."""
    view = hex_game.canonical_view(self.game)
    state = self.game.new_initial_state()
    state = view.apply_action(state, jnp.int32(25))
    self.assertEqual(int(state.board[25]), 0)
    # White's own (2, 3) is the board's (3, 2).
    mask = np.asarray(view.legal_actions_mask(state))
    self.assertFalse(mask[35])
    self.assertTrue(mask[25])
    state = view.apply_action(state, jnp.int32(25))
    self.assertEqual(int(state.board[35]), 1)
    obs = np.asarray(view.observation_tensor(state, jnp.int32(1)))
    base = np.asarray(self.game.observation_tensor(state, jnp.int32(1)))
    np.testing.assert_array_equal(obs, base.transpose(1, 0, 2))


if __name__ == "__main__":
  absltest.main()
