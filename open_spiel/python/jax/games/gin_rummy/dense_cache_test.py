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

"""The dense-cache gin rummy backend computes what the reference backend does.

Each dense helper equals its lookup-table counterpart on random hands, and
1000 random games stepped with both backends agree on every state field the
game reads, every legal mask and every return.
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import registry
from open_spiel.python.jax.games.gin_rummy import dense_cache
from open_spiel.python.jax.games.gin_rummy import dense_cache_game
from open_spiel.python.jax.games.gin_rummy import game
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    calculate_deadwood_all_discards_exact,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    calculate_deadwood_lut,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    compute_legal_knocker_melds,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_run_score_lut,
)


def _hand_from_cards(cards: np.ndarray) -> jnp.ndarray:
  """A 52-card hand mask holding `cards`.

  Args:
    cards: The card indices.

  Returns:
    int8[52], the hand.
  """
  hand = np.zeros(52, dtype=np.int8)
  hand[cards] = 1
  return jnp.asarray(hand)


class DenseCacheBackendTest(absltest.TestCase):
  """The dense-cache backend equals the reference backend."""

  def test_backend_is_a_constructor_option(self):
    """The backend is a constructor option and a game-string parameter."""
    for game_obj in (
        registry.load_game("gin_rummy(backend=dense_cache)"),
        registry.load_game("gin_rummy", backend="dense_cache"),
    ):
      self.assertEqual(game_obj.backend, "dense_cache")
      # Reaches private _step of the code under test.
      self.assertIs(game_obj._step, dense_cache_game.step)  # pylint: disable=protected-access
    self.assertEqual(registry.load_game("gin_rummy").backend, "reference")

  def test_dp_run_scorer_matches_lut_for_all_suit_masks(self):
    """The DP run scorer equals the LUT on every suit mask."""
    masks = jnp.arange(1 << 13, dtype=jnp.int16)
    expected = np.asarray(get_run_score_lut())
    actual = np.asarray(dense_cache.dp_run_scores_from_suit_bits(masks))
    np.testing.assert_array_equal(actual, expected)

  def test_dense_deadwood_matches_reference_lut_on_random_hands(self):
    """Dense deadwood equals the reference on random hands."""
    rng = np.random.default_rng(1793)
    for _ in range(1000):
      hand = _hand_from_cards(rng.choice(52, size=10, replace=False))
      self.assertEqual(
          int(dense_cache.dense_deadwood_lut(hand)),
          int(calculate_deadwood_lut(hand)),
      )

  def test_dense_discard_deadwoods_match_reference_exact_on_random_hands(self):
    """Dense discard deadwoods equal the reference on random hands."""
    rng = np.random.default_rng(1794)
    dense_fn = jax.jit(dense_cache.dense_deadwood_held_discards_delta)
    reference_fn = jax.jit(calculate_deadwood_all_discards_exact)
    for _ in range(1000):
      hand = _hand_from_cards(rng.choice(52, size=11, replace=False))
      held_indices, held_deadwoods = dense_fn(hand)
      dense_deadwoods = jnp.full((52,), 999, dtype=jnp.int16)
      dense_deadwoods = dense_deadwoods.at[held_indices].set(held_deadwoods)
      reference_deadwoods = reference_fn(hand)
      np.testing.assert_array_equal(
          np.asarray(dense_deadwoods), np.asarray(reference_deadwoods)
      )

  def test_selection_matrix_legal_knocker_melds_match_reference(self):
    """Selection-matrix knocker melds equal the reference."""
    rng = np.random.default_rng(1811)
    dense_fn = jax.jit(dense_cache.compute_legal_knocker_melds_selection_matrix)
    reference_fn = jax.jit(compute_legal_knocker_melds)
    for hand_size in (10, 11):
      for _ in range(1000):
        hand = _hand_from_cards(rng.choice(52, size=hand_size, replace=False))
        np.testing.assert_array_equal(
            np.asarray(dense_fn(hand)), np.asarray(reference_fn(hand))
        )

  def test_dense_backend_matches_reference_on_1000_random_games(self):
    """Both backends play 1000 random games identically."""

    @jax.jit
    def run_trajectories(keys):
      """Plays the games in both backends in lockstep.

      Args:
        keys: One deal key per game.

      Returns:
        bool[]: whether every state and mask agreed.
      """
      reference_states = jax.vmap(lambda key: game.init_dealt_state(key)[0])(
          keys
      )
      dense_states = jax.vmap(
          lambda key: dense_cache_game.init_dealt_state(key)[0]
      )(keys)

      def tree_equal(left, right):
        """Returns whether two states agree outside the knocker-meld cache.

        Args:
          left: A state.
          right: Another state.

        Returns:
          bool[].
        """
        # The dense backend fills the knocker-meld cache only in KNOCK, where
        # it is compared below; no legal mask, step or return reads it
        # otherwise.
        ignored_cache_fields = frozenset(["cached_legal_knocker_melds"])
        comparable_keys = sorted(set(left) - ignored_cache_fields)
        return jnp.all(
            jnp.asarray(
                [jnp.all(left[key] == right[key]) for key in comparable_keys]
            )
        )

      def sample_from_mask(key, mask):
        """Samples uniformly from a mask; from every action when it is empty.

        Args:
          key: The PRNG key.
          mask: The mask.

        Returns:
          i32[], the action.
        """
        mask = mask.astype(jnp.int32)
        count = jnp.sum(mask)
        safe_mask = jnp.where(count > 0, mask, jnp.ones_like(mask))
        rank = jax.random.randint(key, (), 0, jnp.sum(safe_mask))
        return jnp.argmax(jnp.cumsum(safe_mask) > rank).astype(jnp.int32)

      def split_key(key):
        """Splits a key in two.

        Args:
          key: The key.

        Returns:
          (key, subkey).
        """
        split = jax.random.split(key, 2)
        return split[0], split[1]

      def body(carry, xs):
        """Compares the backends and plays one step in both.

        Args:
          carry: (reference states, dense states, keys, ok).
          xs: The scan's xs, None (unused).

        Returns:
          (carry, None) for `jax.lax.scan`.
        """
        del xs
        reference_states, dense_states, keys, ok = carry
        reference_masks = jax.vmap(game.legal_actions_mask)(reference_states)
        dense_masks = dense_cache_game.batch_legal_actions_mask(dense_states)
        ok = ok & tree_equal(reference_states, dense_states)
        ok = ok & jnp.all(reference_masks == dense_masks)
        in_knock = dense_states["phase"] == game.PHASE_KNOCK
        ok = ok & jnp.all(
            jnp.where(
                in_knock[:, None],
                reference_states["cached_legal_knocker_melds"]
                == dense_states["cached_legal_knocker_melds"],
                True,
            )
        )

        keys, action_keys = jax.vmap(split_key)(keys)
        actions = jax.vmap(sample_from_mask)(action_keys, reference_masks)

        def reference_step(state, action):
          """Steps the reference state; a no-op when it is done.

          Args:
            state: The state.
            action: The action.

          Returns:
            The next state.
          """
          return jax.lax.cond(
              state["done"], lambda: state, lambda: game.step(state, action)
          )

        reference_states = jax.vmap(reference_step)(reference_states, actions)
        dense_stepped = dense_cache_game.batch_step(dense_states, actions)
        dense_states = jax.tree.map(
            lambda old, stepped: jnp.where(
                dense_states["done"].reshape((-1,) + (1,) * (old.ndim - 1)),
                old,
                stepped,
            ),
            dense_states,
            dense_stepped,
        )
        return (reference_states, dense_states, keys, ok), None

      (reference_states, dense_states, _, ok), _ = jax.lax.scan(
          body,
          (reference_states, dense_states, keys, jnp.bool_(True)),
          None,
          length=1000,
      )
      final_ok = ok & tree_equal(reference_states, dense_states)
      final_ok = final_ok & jnp.all(reference_states["done"])
      final_ok = final_ok & jnp.all(
          jax.vmap(game.get_returns)(reference_states)
          == jax.vmap(dense_cache_game.get_returns)(dense_states)
      )
      return final_ok

    keys = jax.random.split(jax.random.PRNGKey(1795), 1000)
    self.assertTrue(bool(run_trajectories(keys)))


if __name__ == "__main__":
  absltest.main()
