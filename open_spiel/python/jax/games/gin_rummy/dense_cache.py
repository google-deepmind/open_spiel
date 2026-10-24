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

"""Dense computation of the gin rummy meld caches, for accelerators.

The reference backend fills a hand's meld caches with large dynamic gathers from
lookup tables. These functions compute the same values with dense array code,
which runs well on TPUs; `dense_cache_game` builds the dense-cache backend from
them. Each function returns exactly what its reference counterpart in `deadwood`
returns.
"""

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    KNOCK_THRESHOLD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import MAX_HAND_SIZE
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_RANKS
from open_spiel.python.jax.games.gin_rummy.deadwood import (
    get_optimal_meld_indices,
)
from open_spiel.python.jax.games.gin_rummy.deadwood import hand_to_4x13
from open_spiel.python.jax.games.gin_rummy.deadwood import hand_total_points
from open_spiel.python.jax.games.gin_rummy.deadwood import valid_melds_mask
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_popcounts,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_powers_of_2_16,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_rank_values_16,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_scenario_idx_a,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_scenario_idx_b,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_scenario_idx_c,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_selection_matrix,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_subset_table
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_suit_powers

# The tables below are built where they are used, not at import: a module
# makes no JAX call when imported. Inside a jitted function each is a constant.


def _rank_value_prefix_16() -> jax.Array:
  """Prefix sums of the rank values (ace 1, faces 10).

  Returns:
    int16[14], 0 then the cumulative sums.
  """
  rank_values = jnp.array(
      [min(rank + 1, 10) for rank in range(NUM_RANKS)], dtype=jnp.int16
  )
  return jnp.concatenate(
      [jnp.array([0], dtype=jnp.int16), jnp.cumsum(rank_values)]
  )


def _meld_masks_padded() -> jax.Array:
  """The meld masks with one all-zero padding row.

  Returns:
    The padded meld masks.
  """
  return jnp.pad(jnp.asarray(get_meld_masks()), ((0, 1), (0, 0)))


def _meld_points_padded() -> jax.Array:
  """The meld points with their padding entry.

  Returns:
    The padded meld points.
  """
  return jnp.asarray(get_meld_points_padded())


def _selection_matrix_24() -> jax.Array:
  """The 24-entry selection matrix as int8.

  Returns:
    The selection matrix.
  """
  return jnp.asarray(get_selection_matrix(), dtype=jnp.int8)


@jax.jit
def dp_run_scores_from_suit_bits(suit_bits: jax.Array) -> jax.Array:
  """Returns the exact optimal run points via a small dense DP over 13 ranks.

  Args:
    suit_bits: The 13-bit rank masks per suit.

  Returns:
    i16 run points, shaped like `suit_bits`.
  """
  rank_value_prefix = _rank_value_prefix_16()
  ranks = jnp.arange(NUM_RANKS, dtype=jnp.int16)
  held = ((suit_bits[..., None].astype(jnp.int16) >> ranks) & 1).astype(
      jnp.bool_
  )
  zero = jnp.zeros(suit_bits.shape, dtype=jnp.int16)
  dp = [zero] * (NUM_RANKS + 1)
  dp[NUM_RANKS] = zero

  for start in range(NUM_RANKS - 1, -1, -1):
    best = dp[start + 1]
    for length in range(3, NUM_RANKS - start + 1):
      valid = jnp.all(held[..., start : start + length], axis=-1)
      points = rank_value_prefix[start + length] - rank_value_prefix[start]
      best = jnp.maximum(
          best, jnp.where(valid, points + dp[start + length], jnp.int16(-1))
      )
    dp[start] = best
  return dp[0]


@jax.jit
def dense_deadwood_lut(hand: jax.Array) -> jax.Array:
  """Returns the deadwood using exact DP run scoring instead of LUT gathers.

  Args:
    hand: The 52-card mask.

  Returns:
    The deadwood.
  """
  suit_powers = jnp.asarray(get_suit_powers())
  nibble_popcounts = jnp.asarray(get_nibble_popcounts())
  subset_table = jnp.asarray(get_subset_table())
  scenario_idx_a = jnp.asarray(get_scenario_idx_a())
  scenario_idx_b = jnp.asarray(get_scenario_idx_b())
  scenario_idx_c = jnp.asarray(get_scenario_idx_c())
  rank_values_16 = jnp.asarray(get_rank_values_16())
  powers_of_2_16 = jnp.asarray(get_powers_of_2_16())

  hand_4x13 = hand_to_4x13(hand)
  rank_nibbles = jnp.dot(suit_powers, hand_4x13.astype(jnp.int32))
  rank_counts = nibble_popcounts[rank_nibbles]
  set_indices = jnp.argsort(~(rank_counts >= 3))[:3]
  set_nibbles = rank_nibbles[set_indices]
  options = subset_table[set_nibbles]

  mask_a = options[0, scenario_idx_a]
  mask_b = options[1, scenario_idx_b]
  mask_c = options[2, scenario_idx_c]

  scenario_set_points = (
      nibble_popcounts[mask_a].astype(jnp.int16)
      * rank_values_16[set_indices[0]]
      + nibble_popcounts[mask_b].astype(jnp.int16)
      * rank_values_16[set_indices[1]]
      + nibble_popcounts[mask_c].astype(jnp.int16)
      * rank_values_16[set_indices[2]]
  )

  def to_suit_removal(mask, rank_idx):
    """Returns the per-suit rank bits a set removes.

    Args:
      mask: The suit nibble of the set.
      rank_idx: The set's rank.

    Returns:
      i16[..., 4] rank bits per suit.
    """
    bits = jnp.stack([((mask >> i) & 1) for i in range(4)], axis=-1).astype(
        jnp.int16
    )
    return bits * (jnp.int16(1) << rank_idx)

  total_removal = (
      to_suit_removal(mask_a, set_indices[0])
      | to_suit_removal(mask_b, set_indices[1])
      | to_suit_removal(mask_c, set_indices[2])
  )
  base_suits = jnp.dot(hand_4x13.astype(jnp.int16), powers_of_2_16)
  scenario_suits = base_suits[None, :] & (~total_removal)
  run_total = jnp.sum(dp_run_scores_from_suit_bits(scenario_suits), axis=1)
  total_meld_points = scenario_set_points + run_total

  set_card_count = (
      nibble_popcounts[mask_a]
      + nibble_popcounts[mask_b]
      + nibble_popcounts[mask_c]
  ).astype(jnp.int32)
  score_with_tiebreak = (
      total_meld_points.astype(jnp.int32) * 100 + set_card_count
  )
  best_points = total_meld_points[jnp.argmax(score_with_tiebreak)]
  total_hand_value = jnp.sum(
      hand_4x13.astype(jnp.int16) * rank_values_16[None, :]
  )
  return total_hand_value - best_points


@jax.jit
def dense_deadwood_held_discards_delta(
    hand: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes held-card discard deadwoods with shared scenario work.

  Args:
    hand: The 52-card mask.

  Returns:
    (held card indices, i16 deadwood after discarding each; 999 where not
    held).
  """
  suit_powers = jnp.asarray(get_suit_powers())
  nibble_popcounts = jnp.asarray(get_nibble_popcounts())
  subset_table = jnp.asarray(get_subset_table())
  scenario_idx_a = jnp.asarray(get_scenario_idx_a())
  scenario_idx_b = jnp.asarray(get_scenario_idx_b())
  scenario_idx_c = jnp.asarray(get_scenario_idx_c())
  rank_values_16 = jnp.asarray(get_rank_values_16())
  powers_of_2_16 = jnp.asarray(get_powers_of_2_16())

  hand_4x13 = hand_to_4x13(hand)
  rank_nibbles = jnp.dot(suit_powers, hand_4x13.astype(jnp.int32))
  rank_counts = nibble_popcounts[rank_nibbles]
  set_indices = jnp.argsort(~(rank_counts >= 3))[:3]
  set_nibbles = rank_nibbles[set_indices]
  options = subset_table[set_nibbles]

  mask_a = options[0, scenario_idx_a]
  mask_b = options[1, scenario_idx_b]
  mask_c = options[2, scenario_idx_c]

  scenario_set_points = (
      nibble_popcounts[mask_a].astype(jnp.int16)
      * rank_values_16[set_indices[0]]
      + nibble_popcounts[mask_b].astype(jnp.int16)
      * rank_values_16[set_indices[1]]
      + nibble_popcounts[mask_c].astype(jnp.int16)
      * rank_values_16[set_indices[2]]
  )

  def to_suit_removal(mask, rank_idx):
    """Returns the per-suit rank bits a set removes.

    Args:
      mask: The suit nibble of the set.
      rank_idx: The set's rank.

    Returns:
      i16[..., 4] rank bits per suit.
    """
    bits = jnp.stack([((mask >> i) & 1) for i in range(4)], axis=-1).astype(
        jnp.int16
    )
    return bits * (jnp.int16(1) << rank_idx)

  total_removal = (
      to_suit_removal(mask_a, set_indices[0])
      | to_suit_removal(mask_b, set_indices[1])
      | to_suit_removal(mask_c, set_indices[2])
  )
  base_suits = jnp.dot(hand_4x13.astype(jnp.int16), powers_of_2_16)
  scenario_suits = base_suits[None, :] & (~total_removal)
  base_run_scores = dp_run_scores_from_suit_bits(scenario_suits)
  total_meld_points = scenario_set_points + jnp.sum(base_run_scores, axis=1)

  held_indices = jnp.argsort(~(hand > 0))[:MAX_HAND_SIZE]
  held_mask = jnp.take(hand, held_indices) > 0
  d_suit = held_indices // NUM_RANKS
  d_rank = held_indices % NUM_RANKS

  fail_a = (d_rank[:, None] == set_indices[0]) & (
      ((mask_a[None, :] >> d_suit[:, None]) & 1).astype(jnp.bool_)
  )
  fail_b = (d_rank[:, None] == set_indices[1]) & (
      ((mask_b[None, :] >> d_suit[:, None]) & 1).astype(jnp.bool_)
  )
  fail_c = (d_rank[:, None] == set_indices[2]) & (
      ((mask_c[None, :] >> d_suit[:, None]) & 1).astype(jnp.bool_)
  )
  is_invalid = fail_a | fail_b | fail_c

  selected_suit_bits = scenario_suits[:, d_suit].T
  new_suit_bits = selected_suit_bits & (~(jnp.int16(1) << d_rank[:, None]))
  new_run_score = dp_run_scores_from_suit_bits(new_suit_bits)
  old_run_score = base_run_scores[:, d_suit].T

  final_meld_points = total_meld_points[None, :] - old_run_score + new_run_score
  valid_meld_points = jnp.where(is_invalid, jnp.int16(-1), final_meld_points)
  best_meld_points = jnp.max(valid_meld_points, axis=1)

  total_hand_value = jnp.sum(
      hand_4x13.astype(jnp.int16) * rank_values_16[None, :]
  )
  deadwoods = total_hand_value - rank_values_16[d_rank] - best_meld_points
  return held_indices, jnp.where(held_mask, deadwoods, jnp.int16(999))


@jax.jit
def compute_all_meld_info_dense_full(
    hand: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Dense equivalent of `deadwood.compute_all_meld_info`.

  Args:
    hand: The 52-card mask.

  Returns:
    (bool[185] optimal melds, i16[52] deadwood per discard, 999 where not
    held).
  """
  optimal_melds = get_optimal_meld_indices(hand)
  held_indices, held_deadwoods = dense_deadwood_held_discards_delta(hand)
  all_deadwoods = jnp.full((NUM_CARDS,), 999, dtype=jnp.int16)
  all_deadwoods = all_deadwoods.at[held_indices].set(
      held_deadwoods.astype(jnp.int16)
  )
  return optimal_melds, all_deadwoods


@jax.jit
def compute_all_meld_info_dense_discard_only(
    hand: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Like `compute_all_meld_info_dense_full`, but the optimal melds are zero.

  Args:
    hand: The 52-card mask.

  Returns:
    (bool[185] zeros, i16[52] deadwood per discard, 999 where not held).
  """
  held_indices, held_deadwoods = dense_deadwood_held_discards_delta(hand)
  all_deadwoods = jnp.full((NUM_CARDS,), 999, dtype=jnp.int16)
  all_deadwoods = all_deadwoods.at[held_indices].set(
      held_deadwoods.astype(jnp.int16)
  )
  # No legal mask, step or return reads the optimal melds, so this backend
  # returns zeros for them.
  return jnp.zeros(NUM_MELDS, dtype=jnp.bool_), all_deadwoods


@jax.jit
def compute_legal_knocker_melds_selection_matrix(
    hand: jax.Array, threshold=KNOCK_THRESHOLD
) -> jax.Array:
  """Returns the exact legal-knocker melds via the 1/2/3-meld selection matrix.

  Args:
    hand: The 52-card mask.
    threshold: The knock threshold.

  Returns:
    bool[185].
  """
  selection = _selection_matrix_24()
  total_points = hand_total_points(hand)
  valid = valid_melds_mask(hand)
  valid_indices = jnp.where(valid, size=24, fill_value=NUM_MELDS)[0]
  slot_valid = valid_indices < NUM_MELDS

  top_masks = _meld_masks_padded()[valid_indices]
  top_points = _meld_points_padded()[valid_indices]
  card_usage = jnp.dot(selection, top_masks)
  disjoint = jnp.all(card_usage <= 1, axis=1)
  selected_valid = jnp.dot(selection, slot_valid.astype(jnp.int8)) == jnp.sum(
      selection, axis=1
  )
  combo_points = jnp.dot(selection.astype(jnp.int32), top_points)
  legal_combo = (
      disjoint & selected_valid & ((total_points - combo_points) <= threshold)
  )
  legal_slots = jnp.any(
      legal_combo[:, None] & selection.astype(jnp.bool_), axis=0
  )

  legal_melds_extended = jnp.zeros(NUM_MELDS + 1, dtype=jnp.bool_)
  legal_melds_extended = legal_melds_extended.at[valid_indices].set(
      legal_slots & slot_valid
  )
  return legal_melds_extended[:NUM_MELDS]
