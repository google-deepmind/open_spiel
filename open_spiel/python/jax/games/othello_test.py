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

"""Othello: registration, the opening position and its moves, and flips.

Also compiled and batched steps, and random playthroughs against OpenSpiel.
"""

import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import othello
from open_spiel.python.jax.games import registry

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


class OthelloBasicTest(parameterized.TestCase):
  """Property tests that do not require pyspiel."""

  def test_registered(self):
    """The othello game is registered."""
    self.assertIn("othello", registry.registered_names())

  def test_load_game(self):
    """load_game builds the two-player game."""
    game = registry.load_game("othello")
    self.assertIsInstance(game, othello.Othello)
    self.assertEqual(game.num_players, 2)
    self.assertEqual(game.num_actions, 65)
    self.assertEqual(game.obs_shape, (8, 8, 3))

  def test_initial_state(self):
    """The four center discs are placed."""
    game = registry.load_game("othello")
    state = game.new_initial_state()
    # Four center discs placed, everything else empty.
    self.assertEqual(int(jnp.sum(state.board == 1)), 2)
    self.assertEqual(int(jnp.sum(state.board == 2)), 2)
    self.assertEqual(int(state.current_player), 0)
    self.assertFalse(bool(state.is_terminal))
    self.assertEqual(int(state.winner), -1)

  def test_initial_legal_actions(self):
    """Black has OpenSpiel's four opening moves."""
    game = registry.load_game("othello")
    state = game.new_initial_state()
    mask = np.array(game.legal_actions_mask(state))
    # Black has exactly 4 opening moves: d3, c4, f5, e6 in (row, col).
    expected = np.zeros(65, dtype=bool)
    for r, c in [(2, 3), (3, 2), (4, 5), (5, 4)]:
      expected[r * 8 + c] = True
    np.testing.assert_array_equal(mask, expected)

  def test_placement_flips(self):
    """A placement flips the outflanked disc."""
    game = registry.load_game("othello")
    state = game.new_initial_state()
    # Black plays d3 -> (row=2, col=3).
    action = jnp.array(2 * 8 + 3, dtype=jnp.int32)
    next_state = game.apply_action(state, action)
    # Black should now have 4 discs, white 1.
    self.assertEqual(int(jnp.sum(next_state.board == 1)), 4)
    self.assertEqual(int(jnp.sum(next_state.board == 2)), 1)
    self.assertEqual(int(next_state.current_player), 1)

  def test_jit_step(self):
    """A step and the observations run under jit."""
    game = registry.load_game("othello")

    @jax.jit
    def one_step():
      """Plays the first legal move and observes for both players.

      Returns:
        (next state, player 0's observation, player 1's observation).
      """
      state = game.new_initial_state()
      mask = game.legal_actions_mask(state)
      action = jnp.argmax(mask.astype(jnp.int32))
      next_state = game.apply_action(state, action.astype(jnp.int32))
      obs0 = game.observation_tensor(next_state, jnp.array(0))
      obs1 = game.observation_tensor(next_state, jnp.array(1))
      return next_state, obs0, obs1

    next_state, obs0, obs1 = one_step()
    self.assertEqual(obs0.shape, (8, 8, 3))
    self.assertEqual(obs1.shape, (8, 8, 3))
    self.assertFalse(bool(next_state.is_terminal))

  def test_vmap_step(self):
    """A step runs under vmap."""
    game = registry.load_game("othello")
    initial = game.new_initial_state()

    def batched_apply(state, action):
      """Applies one action.

      Args:
        state: The state.
        action: The action.

      Returns:
        The next state.
      """
      return game.apply_action(state, action)

    batch_states = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (3,) + x.shape), initial
    )
    actions = jnp.array([2 * 8 + 3, 3 * 8 + 2, 4 * 8 + 5], dtype=jnp.int32)
    results = jax.vmap(batched_apply)(batch_states, actions)
    self.assertEqual(results.board.shape, (3, 8, 8))

  def test_random_rollout_runs_to_terminal(self):
    """Pure-JAX rollout terminates and scores are consistent."""
    game = registry.load_game("othello")
    state = game.new_initial_state()
    rng = random.Random(0)
    for _ in range(200):
      if bool(state.is_terminal):
        break
      mask = np.array(game.legal_actions_mask(state))
      legal = np.flatnonzero(mask)
      self.assertNotEmpty(legal)
      action = int(rng.choice(legal))
      state = game.apply_action(state, jnp.array(action, dtype=jnp.int32))
    self.assertTrue(bool(state.is_terminal))
    returns = np.array(game.returns(state))
    self.assertEqual(returns.tolist().count(0.0) % 2, 0)  # paired
    np.testing.assert_allclose(returns.sum(), 0.0)


@absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not available")
class OthelloEquivalenceTest(parameterized.TestCase):
  """Equivalence vs pyspiel across random playthroughs."""

  @parameterized.parameters(range(3))
  def test_random_playthrough(self, seed):
    """A random playthrough agrees with pyspiel.

    Args:
      seed: The seed.
    """
    os_game = pyspiel.load_game("othello")
    jax_game = registry.load_game("othello")
    num_actions = os_game.num_distinct_actions()

    os_state = os_game.new_initial_state()
    jax_state = jax_game.new_initial_state()
    rng = random.Random(seed)

    while not os_state.is_terminal():
      self.assertEqual(bool(jax_state.is_terminal), os_state.is_terminal())
      self.assertEqual(int(jax_state.current_player), os_state.current_player())

      os_legal = os_state.legal_actions()
      os_mask = [False] * num_actions
      for a in os_legal:
        os_mask[a] = True
      jax_mask = np.array(jax_game.legal_actions_mask(jax_state)).tolist()
      self.assertEqual(jax_mask, os_mask)

      action = rng.choice(os_legal)
      os_state.apply_action(action)
      jax_state = jax_game.apply_action(
          jax_state, jnp.array(action, dtype=jnp.int32)
      )

    self.assertEqual(bool(jax_state.is_terminal), os_state.is_terminal())
    np.testing.assert_array_almost_equal(
        np.array(jax_game.returns(jax_state)), os_state.returns()
    )


if __name__ == "__main__":
  absltest.main()
