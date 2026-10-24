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

"""Leduc poker: every method compiles, and equal hands split the pot."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import leduc_poker
from open_spiel.python.jax.games import registry


class LeducPokerTest(parameterized.TestCase):
  """Leduc poker's methods compile and follow OpenSpiel's rules.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = leduc_poker.LeducPoker()

  def test_openspiel_default_parameters_load(self):
    """The game string OpenSpiel accepts at its defaults loads; others raise."""
    for game_string in ("leduc_poker(players=2)", "kuhn_poker(players=2)"):
      self.assertEqual(registry.load_game(game_string).num_players, 2)
    with self.assertRaises(ValueError):
      registry.load_game("leduc_poker(players=3)")
    with self.assertRaises(ValueError):
      registry.load_game("kuhn_poker(players=3)")

  def test_jit_apply_action_chance(self):
    """apply_action compiles for chance actions."""
    apply_jit = jax.jit(self.game.apply_action)
    state = self.game.new_initial_state()
    # Action 0 is dealing card 0.
    next_state = apply_jit(state, jnp.array(0))
    self.assertEqual(int(next_state.private_cards_dealt), 1)

  def test_jit_apply_action_player(self):
    """apply_action compiles for player actions."""
    # Reaches private _CALL of the code under test.
    # pylint: disable=protected-access
    apply_jit = jax.jit(self.game.apply_action)
    state = self.game.new_initial_state()
    # Deal cards to both players.
    state = apply_jit(state, jnp.array(0))
    state = apply_jit(state, jnp.array(1))
    # Now it should be player 0's turn.
    self.assertEqual(int(state.current_player), 0)
    # Player 0 calls.
    state = apply_jit(state, jnp.array(leduc_poker._CALL))
    self.assertEqual(int(state.current_player), 1)

  def test_jit_legal_actions_mask(self):
    """legal_actions_mask compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    mask_jit = jax.jit(self.game.legal_actions_mask)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    state = apply_jit(state, jnp.array(1))
    mask = mask_jit(state)
    self.assertEqual(mask.shape, (self.game.num_actions,))
    # Player 0 can Call or Raise, but not Fold (not facing a bet).
    np.testing.assert_array_equal(mask, [False, True, True])

  def test_jit_observation_tensor(self):
    """observation_tensor compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    obs_jit = jax.jit(self.game.observation_tensor)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    state = apply_jit(state, jnp.array(1))
    obs = obs_jit(state, jnp.array(0))
    self.assertEqual(obs.shape, self.game.obs_shape)

  def test_jit_information_state_tensor(self):
    """information_state_tensor compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    is_jit = jax.jit(self.game.information_state_tensor)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    state = apply_jit(state, jnp.array(1))
    info_state = is_jit(state, jnp.array(0))
    self.assertEqual(info_state.shape, self.game.info_state_shape)

  def test_jit_returns(self):
    """The returns method compiles."""
    # Reaches private _FOLD, _RAISE of the code under test.
    # pylint: disable=protected-access
    apply_jit = jax.jit(self.game.apply_action)
    returns_jit = jax.jit(self.game.returns)
    state = self.game.new_initial_state()
    # Deal cards, then fold.
    state = apply_jit(state, jnp.array(0))
    state = apply_jit(state, jnp.array(1))
    # P0 raises.
    state = apply_jit(state, jnp.array(leduc_poker._RAISE))
    # P1 folds.
    state = apply_jit(state, jnp.array(leduc_poker._FOLD))
    self.assertTrue(state.is_terminal)
    ret = returns_jit(state)
    self.assertEqual(ret.shape, (2,))
    # P0 wins the pot (ante 1 each + P0 raise 2 = 4).
    # P0 profit = 4 - 3 = 1. P1 loss = -1.
    np.testing.assert_array_almost_equal(ret, [1.0, -1.0])

  def test_jit_chance_outcomes(self):
    """chance_outcomes compiles."""
    chance_jit = jax.jit(self.game.chance_outcomes)
    state = self.game.new_initial_state()
    probs = chance_jit(state)
    self.assertEqual(probs.shape, (self.game.max_chance_outcomes,))
    # Uniform over 6 cards.
    np.testing.assert_array_almost_equal(probs, [1 / 6] * 6)

  def test_draw(self):
    """Verify correct payoffs after a draw."""
    # Reaches private _CALL of the code under test.
    # pylint: disable=protected-access
    apply_jit = jax.jit(self.game.apply_action)
    returns_jit = jax.jit(self.game.returns)
    state = self.game.new_initial_state()

    # Deal cards of same rank but different suits to both players.
    # 0 = Jack1, 1 = Jack2.
    state = apply_jit(state, jnp.array(0))  # P0 gets Jack1
    state = apply_jit(state, jnp.array(1))  # P1 gets Jack2

    # Both check/call in round 1.
    state = apply_jit(state, jnp.array(leduc_poker._CALL))
    state = apply_jit(state, jnp.array(leduc_poker._CALL))

    # Deal public card of different rank. 2 = Queen1.
    state = apply_jit(state, jnp.array(2))

    # Both check/call in round 2.
    state = apply_jit(state, jnp.array(leduc_poker._CALL))
    state = apply_jit(state, jnp.array(leduc_poker._CALL))

    self.assertTrue(state.is_terminal)
    ret = returns_jit(state)
    np.testing.assert_array_almost_equal(ret, [0.0, 0.0])


if __name__ == "__main__":
  absltest.main()
