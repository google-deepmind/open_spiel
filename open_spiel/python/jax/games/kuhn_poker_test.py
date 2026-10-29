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

"""Kuhn poker against OpenSpiel.

Kuhn's game tree is small enough (58 reachable states) to walk
exhaustively — every node is compared against pyspiel.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import kuhn_poker
from open_spiel.python.jax.games import registry

try:
  # pyspiel is optional: the comparisons skip without it.
  import pyspiel  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


class KuhnPokerBasicTest(parameterized.TestCase):
  """Property tests that do not require pyspiel."""

  def test_registered(self):
    """kuhn_poker is registered."""
    self.assertIn("kuhn_poker", registry.registered_names())

  def test_load_game(self):
    """load_game builds the two-player game."""
    game = registry.load_game("kuhn_poker")
    self.assertIsInstance(game, kuhn_poker.KuhnPoker)
    self.assertEqual(game.num_players, 2)
    self.assertEqual(game.num_actions, 2)
    self.assertEqual(game.obs_shape, (7,))

  def test_initial_state_is_chance(self):
    """The initial state is a chance node."""
    game = registry.load_game("kuhn_poker")
    state = game.new_initial_state()
    self.assertTrue(bool(game.is_chance_node(state)))
    self.assertEqual(int(game.current_player(state)), -1)
    self.assertFalse(bool(state.is_terminal))

  def test_initial_chance_outcomes(self):
    """The first deal is uniform over the three cards."""
    game = registry.load_game("kuhn_poker")
    state = game.new_initial_state()
    probs = game.chance_outcomes(state)
    self.assertEqual(probs.shape, (3,))
    self.assertEqual(probs.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.array(probs), [1 / 3, 1 / 3, 1 / 3], atol=1e-6
    )

  def test_second_deal_excludes_first_card(self):
    """The second deal excludes the first card."""
    game = registry.load_game("kuhn_poker")
    state = game.apply_action(game.new_initial_state(), jnp.int32(1))
    self.assertTrue(bool(game.is_chance_node(state)))
    probs = game.chance_outcomes(state)
    # Dense, indexed by card: P0 holds card 1, so it has probability 0.
    np.testing.assert_allclose(np.array(probs), [0.5, 0.0, 0.5], atol=1e-6)

  def test_betting_legal_mask(self):
    """Both actions are legal while betting."""
    game = registry.load_game("kuhn_poker")
    s = game.new_initial_state()
    s = game.apply_action(s, jnp.int32(0))  # deal 0 to P0
    s = game.apply_action(s, jnp.int32(1))  # deal 1 to P1
    self.assertFalse(bool(game.is_chance_node(s)))
    mask = np.array(game.legal_actions_mask(s))
    np.testing.assert_array_equal(mask, [True, True])
    self.assertEqual(int(game.current_player(s)), 0)

  def test_bp_terminal_p0_wins(self):
    """Bet, pass: player 0 wins the pot."""
    game = registry.load_game("kuhn_poker")
    s = game.new_initial_state()
    s = game.apply_action(s, jnp.int32(0))  # P0 gets J
    s = game.apply_action(s, jnp.int32(2))  # P1 gets K
    s = game.apply_action(s, jnp.int32(1))  # P0 bets
    s = game.apply_action(s, jnp.int32(0))  # P1 folds
    self.assertTrue(bool(s.is_terminal))
    np.testing.assert_array_equal(np.array(game.returns(s)), [1.0, -1.0])

  def test_pp_showdown(self):
    """Pass, pass: the higher card wins the showdown."""
    game = registry.load_game("kuhn_poker")
    s = game.new_initial_state()
    s = game.apply_action(s, jnp.int32(0))
    s = game.apply_action(s, jnp.int32(2))
    s = game.apply_action(s, jnp.int32(0))
    s = game.apply_action(s, jnp.int32(0))
    np.testing.assert_array_equal(np.array(game.returns(s)), [-1.0, 1.0])

  def test_jit_step(self):
    """A deal and the tensors run under jit."""
    game = registry.load_game("kuhn_poker")

    @jax.jit
    def one_step():
      """Deals one card and computes the tensors.

      Returns:
        (next state, observation, information state) of player 0.
      """
      state = game.new_initial_state()
      probs = game.chance_outcomes(state)
      next_state = game.apply_action(state, jnp.argmax(probs))
      obs0 = game.observation_tensor(next_state, jnp.array(0))
      info0 = game.information_state_tensor(next_state, jnp.array(0))
      return next_state, obs0, info0

    next_state, obs0, info0 = one_step()
    self.assertEqual(obs0.shape, (7,))
    self.assertEqual(info0.shape, (11,))
    self.assertEqual(int(next_state.p0_card), 0)
    self.assertEqual(int(next_state.p1_card), -1)


@absltest.skipUnless(_HAS_PYSPIEL, "pyspiel not installed")
class KuhnPokerEquivalenceTest(absltest.TestCase):
  """Exhaustively compares the JAX kuhn_poker to pyspiel at every state."""

  def _assert_chance_matches(self, os_state, jax_state, game):
    os_outcomes = dict(os_state.chance_outcomes())  # action -> prob
    jax_probs = np.array(game.chance_outcomes(jax_state))
    jax_outcomes = {a: p for a, p in enumerate(jax_probs.tolist()) if p > 0.0}
    self.assertEqual(set(jax_outcomes), set(os_outcomes))
    for a, p in os_outcomes.items():
      self.assertAlmostEqual(jax_outcomes[a], p, places=6)

  def _assert_nodes_match(self, os_state, jax_state, game):
    """Asserts the JAX state matches the OpenSpiel state at one node.

    Args:
      os_state: The OpenSpiel state.
      jax_state: The JAX state.
      game: The JAX game.
    """
    # Terminal flag.
    self.assertEqual(bool(jax_state.is_terminal), os_state.is_terminal())

    # Chance vs player.
    self.assertEqual(
        bool(game.is_chance_node(jax_state)), os_state.is_chance_node()
    )

    if os_state.is_terminal():
      np.testing.assert_array_equal(
          np.array(game.returns(jax_state)),
          np.array(os_state.returns(), dtype=np.float32),
      )
      return

    # Observation and information state tensors match pyspiel exactly.
    for p in range(2):
      np.testing.assert_array_equal(
          np.array(game.observation_tensor(jax_state, jnp.array(p))),
          np.array(os_state.observation_tensor(p), dtype=np.float32),
      )
      np.testing.assert_array_equal(
          np.array(game.information_state_tensor(jax_state, jnp.array(p))),
          np.array(os_state.information_state_tensor(p), dtype=np.float32),
      )

    if os_state.is_chance_node():
      self._assert_chance_matches(os_state, jax_state, game)
    else:
      self.assertEqual(
          int(game.current_player(jax_state)), os_state.current_player()
      )
      os_legal = os_state.legal_actions()
      jax_mask = np.array(game.legal_actions_mask(jax_state))
      expected_mask = np.zeros_like(jax_mask)
      for a in os_legal:
        expected_mask[a] = True
      np.testing.assert_array_equal(jax_mask, expected_mask)

  def _walk(self, os_state, jax_state, game, visited):
    """Walks the whole tree, matching every node against OpenSpiel.

    Args:
      os_state: The OpenSpiel state.
      jax_state: The JAX state.
      game: The JAX game.
      visited: A one-element list counting the visited nodes.
    """
    # Plain recursion: Kuhn is acyclic and shallow.
    self._assert_nodes_match(os_state, jax_state, game)
    visited[0] += 1
    if os_state.is_terminal():
      return
    actions = (
        [a for a, _ in os_state.chance_outcomes()]
        if os_state.is_chance_node()
        else list(os_state.legal_actions())
    )
    for a in actions:
      child_os = os_state.child(a)
      child_jax = game.apply_action(jax_state, jnp.int32(a))
      self._walk(child_os, child_jax, game, visited)

  def test_exhaustive_tree_walk(self):
    """The whole game tree agrees with pyspiel."""
    os_game = pyspiel.load_game("kuhn_poker")
    game = registry.load_game("kuhn_poker")
    visited = [0]
    self._walk(
        os_game.new_initial_state(), game.new_initial_state(), game, visited
    )
    # Kuhn's reachable game tree has a fixed size; pinning the exact node
    # count catches accidental early exits and missed branches.
    self.assertEqual(visited[0], 58)


if __name__ == "__main__":
  absltest.main()
