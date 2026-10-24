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

"""Bargaining: every method compiles, and the turn structure follows the rules.

The instance chance node comes first, then player 0; no accept before an
offer.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bargaining


class BargainingTest(parameterized.TestCase):
  """Bargaining's methods and turn structure.

  Attributes:
    game: The game.
  """

  def setUp(self):
    """Builds the default game."""
    super().setUp()
    self.game = bargaining.Bargaining()

  def _apply_initial_chance(self, state, instance_idx=0):
    """Helper: apply initial chance action to resolve instance.

    Args:
      state: The state.
      instance_idx: The instance.

    Returns:
      The state after the instance chance action.
    """
    return self.game.apply_action(state, jnp.array(instance_idx))

  # --- Jittability tests ---

  def test_jit_apply_action_chance(self):
    """apply_action compiles for chance actions."""
    apply_jit = jax.jit(self.game.apply_action)
    state = self.game.new_initial_state()
    next_state = apply_jit(state, jnp.array(0))
    self.assertEqual(int(next_state.current_player), 0)

  def test_jit_apply_action_player(self):
    """apply_action compiles for player actions."""
    apply_jit = jax.jit(self.game.apply_action)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))  # chance
    state = apply_jit(state, jnp.array(0))  # player 0 offers
    self.assertEqual(int(state.current_player), 1)

  def test_jit_legal_actions_mask(self):
    """legal_actions_mask compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    mask_jit = jax.jit(self.game.legal_actions_mask)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    mask = mask_jit(state)
    self.assertEqual(mask.shape, (self.game.num_actions,))

  def test_jit_observation_tensor(self):
    """observation_tensor compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    obs_jit = jax.jit(self.game.observation_tensor)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    obs = obs_jit(state, jnp.array(0))
    self.assertEqual(obs.shape, self.game.obs_shape)

  def test_jit_information_state_tensor(self):
    """information_state_tensor compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    is_jit = jax.jit(self.game.information_state_tensor)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    info_state = is_jit(state, jnp.array(0))
    self.assertEqual(info_state.shape, self.game.info_state_shape)

  def test_jit_returns(self):
    """The returns method compiles."""
    apply_jit = jax.jit(self.game.apply_action)
    returns_jit = jax.jit(self.game.returns)
    state = self.game.new_initial_state()
    state = apply_jit(state, jnp.array(0))
    ret = returns_jit(state)
    self.assertEqual(ret.shape, (2,))

  def test_jit_chance_outcomes(self):
    """chance_outcomes compiles."""
    chance_jit = jax.jit(self.game.chance_outcomes)
    state = self.game.new_initial_state()
    probs = chance_jit(state)
    self.assertEqual(probs.shape, (self.game.max_chance_outcomes,))

  def test_jit_full_game_loop(self):
    """JIT-compile a full game trajectory using lax.while_loop."""
    game = self.game
    apply_jit = jax.jit(game.apply_action)

    # Play: chance -> P0 offers [0,0,0] -> P1 agrees
    state = game.new_initial_state()
    state = apply_jit(state, jnp.array(0))  # chance: instance 0
    state = apply_jit(state, jnp.array(0))  # P0: offer (0,0,0)

    agree_action = jnp.array(game.num_actions - 1)
    state = apply_jit(state, agree_action)  # P1: agree
    self.assertTrue(state.is_terminal)
    self.assertTrue(state.agreement_reached)

  def test_tensor_header_bits(self):
    """The tensor header bits are OpenSpiel's."""
    # Reaches private _max_turns of the code under test.
    # pylint: disable=protected-access
    # Bit 0 says whether an agreement was reached; bits 1 .. max_turns + 1
    # one-hot the number of offers made, as in OpenSpiel.
    game = self.game
    state = game.apply_action(game.new_initial_state(), jnp.int32(0))
    for tensor in (game.observation_tensor, game.information_state_tensor):
      # Reaches private _max_turns of the code under test.
      head = np.asarray(tensor(state, jnp.int32(0)))[: game._max_turns + 2]  # pylint: disable=protected-access
      np.testing.assert_array_equal(np.flatnonzero(head), [1])
    state = game.apply_action(state, jnp.int32(0))  # player 0 offers
    head = np.asarray(game.observation_tensor(state, jnp.int32(1)))
    np.testing.assert_array_equal(
        # Reaches private _max_turns of the code under test.
        np.flatnonzero(head[: game._max_turns + 2]),
        [2],
    )  # pylint: disable=protected-access
    state = game.apply_action(state, jnp.int32(game.num_actions - 1))
    for tensor in (game.observation_tensor, game.information_state_tensor):
      # Reaches private _max_turns of the code under test.
      head = np.asarray(tensor(state, jnp.int32(0)))[: game._max_turns + 2]  # pylint: disable=protected-access
      np.testing.assert_array_equal(np.flatnonzero(head), [0, 2])

  # --- Game logic tests ---

  def test_initial_state_is_chance(self):
    """The initial state is a chance node."""
    state = self.game.new_initial_state()
    self.assertEqual(int(state.current_player), -1)
    self.assertFalse(state.is_terminal)
    # No player actions should be legal at a chance node.
    mask = self.game.legal_actions_mask(state)
    self.assertFalse(jnp.any(mask))

  def test_after_chance_is_player_0(self):
    """Player 0 moves after the chance node."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    self.assertEqual(int(state.current_player), 0)
    # Pool and values should now be set.
    self.assertTrue(jnp.any(state.pool > 0))

  def test_agree_unavailable_first_turn(self):
    """Agree action is not legal until at least one offer exists."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    mask = self.game.legal_actions_mask(state)
    agree_idx = self.game.num_actions - 1
    self.assertFalse(mask[agree_idx])

  def test_agree_available_after_offer(self):
    """Agree action is legal after the first offer."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    state = self.game.apply_action(state, jnp.array(0))  # P0 offers
    mask = self.game.legal_actions_mask(state)
    agree_idx = self.game.num_actions - 1
    self.assertTrue(mask[agree_idx])

  def test_legal_actions_pool_filter(self):
    """Offers exceeding pool quantities are masked out."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    pool = np.array(state.pool)
    mask = np.array(self.game.legal_actions_mask(state))

    for i, offer in enumerate(self.game.all_offers):
      if np.any(offer > pool):
        self.assertFalse(
            mask[i], f"Offer {offer} should be illegal for pool {pool}"
        )
      else:
        self.assertTrue(
            mask[i], f"Offer {offer} should be legal for pool {pool}"
        )

  def test_agree_returns(self):
    """Verify correct general-sum payoffs after agreement."""
    # Instance 0: pool=[1,2,3], P0 vals=[8,1,0], P1 vals=[4,0,2]
    state = self.game.new_initial_state()
    state = self.game.apply_action(state, jnp.array(0))  # instance 0

    # P0 offers (0,0,0) - proposer keeps nothing
    state = self.game.apply_action(state, jnp.array(0))

    # P1 agrees - P1 (other) gets items: pool - offer = [1,2,3] - [0,0,0] =
    # [1,2,3]
    # Proposer (P0) gets offer [0,0,0]: value = 8*0+1*0+0*0 = 0
    # Other (P1) gets pool-offer [1,2,3]: value = 4*1+0*2+2*3 = 10
    agree_action = jnp.array(self.game.num_actions - 1)
    state = self.game.apply_action(state, agree_action)

    ret = self.game.returns(state)
    np.testing.assert_array_almost_equal(ret, [0.0, 10.0])

  def test_max_turns_terminal(self):
    """Game terminates after max_turns offers with no agreement."""
    game = bargaining.Bargaining(max_turns=4)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.array(0))  # chance

    for _ in range(4):
      self.assertFalse(state.is_terminal)
      state = game.apply_action(state, jnp.array(0))  # offer (0,0,0)

    self.assertTrue(state.is_terminal)
    ret = game.returns(state)
    np.testing.assert_array_almost_equal(ret, [0.0, 0.0])

  def test_discount(self):
    """Discount is applied correctly after turn 2."""
    game = bargaining.Bargaining(discount=0.9)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.array(0))  # chance: instance 0

    # Make 4 offers (P0, P1, P0, P1), then P0 agrees
    for _ in range(4):
      state = game.apply_action(state, jnp.array(0))  # offer (0,0,0)

    agree_action = jnp.array(game.num_actions - 1)
    state = game.apply_action(state, agree_action)  # P0 agrees

    # P0 agrees to P1's last offer of (0,0,0).
    # P1 proposed, P0 is the other player.
    # P0 gets pool - offer = pool items, P1 gets offer items.
    # Discount should be 0.9^3 (applied at moves 3, 4, 5 = 3 times after move 2)
    ret = game.returns(state)
    # P0 gets all items at their values, discounted.
    # Instance 0: pool=[1,2,3], P0 vals=[8,1,0], P1 vals=[4,0,2]
    # P1 proposed (0,0,0). P0 (other) gets pool-offer = [1,2,3].
    # P0 return = (8*1 + 1*2 + 0*3) * discount = 10 * discount
    # P1 (proposer) gets offer [0,0,0]: val = 0
    # Discount: applied at move 3, 4 and 5 (when state.num_offers >= 2),
    # so discount = 0.9^3 = 0.729.
    np.testing.assert_array_almost_equal(ret, [10.0 * 0.9**3, 0.0])

  def test_observation_shape(self):
    """The observation has obs_shape."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    obs = self.game.observation_tensor(state, jnp.array(0))
    expected_size = 1 + 11 + 24 + 33 + 24  # = 93
    self.assertEqual(obs.shape, (expected_size,))

  def test_information_state_shape(self):
    """The information state has info_state_shape."""
    state = self.game.new_initial_state()
    state = self._apply_initial_chance(state, 0)
    info = self.game.information_state_tensor(state, jnp.array(0))
    expected_size = 1 + 11 + 24 + 33 + 10 * 24  # = 309
    self.assertEqual(info.shape, (expected_size,))

  # --- prob_end tests ---

  def test_prob_end_chance_nodes(self):
    """Mid-game chance nodes appear with prob_end > 0 after 2 offers."""
    game = bargaining.Bargaining(prob_end=0.1)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.array(0))  # chance: instance

    # P0 offers
    state = game.apply_action(state, jnp.array(0))
    self.assertEqual(int(state.current_player), 1)  # no chance yet

    # P1 offers
    state = game.apply_action(state, jnp.array(0))
    # After 2 offers with prob_end > 0, should be a chance node
    self.assertEqual(int(state.current_player), -1)

  def test_prob_end_continue(self):
    """Game continues when mid-game chance says continue."""
    game = bargaining.Bargaining(prob_end=0.1)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.array(0))  # chance: instance
    state = game.apply_action(state, jnp.array(0))  # P0 offers
    state = game.apply_action(state, jnp.array(0))  # P1 offers

    # Chance node: OpenSpiel's continue outcome, after the 1000 instances.
    self.assertEqual(game.continue_outcome, 1000)
    probs = np.asarray(game.chance_outcomes(state))
    self.assertEqual(probs.shape, (1002,))
    np.testing.assert_allclose(probs[1000:], [0.9, 0.1], rtol=1e-6)
    self.assertEqual(np.flatnonzero(probs).tolist(), [1000, 1001])
    state = game.apply_action(state, jnp.array(game.continue_outcome))
    self.assertFalse(state.is_terminal)
    self.assertNotEqual(int(state.current_player), -1)

  def test_prob_end_game_ends(self):
    """Game terminates when mid-game chance says end."""
    game = bargaining.Bargaining(prob_end=0.1)
    state = game.new_initial_state()
    state = game.apply_action(state, jnp.array(0))  # chance: instance
    state = game.apply_action(state, jnp.array(0))  # P0 offers
    state = game.apply_action(state, jnp.array(0))  # P1 offers

    # Chance node: OpenSpiel's end outcome.
    self.assertEqual(game.end_outcome, 1001)
    state = game.apply_action(state, jnp.array(game.end_outcome))
    self.assertTrue(state.is_terminal)
    self.assertTrue(state.game_ended)
    ret = game.returns(state)
    np.testing.assert_array_almost_equal(ret, [0.0, 0.0])


if __name__ == "__main__":
  absltest.main()
