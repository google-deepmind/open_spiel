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

"""Deadwood and optimal melds of gin rummy hands.

A hand's deadwood is the point total of the cards outside its best set of
non-overlapping melds. The functions here compute it exactly for 10- and
11-card hands (for 11 cards, after each possible discard), the optimal
melds, and the melds a knocker may lay, all jittable with fixed shapes.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_points,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_overlap_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    KNOCK_THRESHOLD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_popcounts,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_powers_of_2_16,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_rank_values
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_rank_values_16,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_run_decomp_lut,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_run_score_lut,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_run_score_lut_8,
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
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_subset_table
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_suit_powers

__all__ = [
    "calculate_deadwood_all_discards_exact",
    "calculate_deadwood_compressed",
    "calculate_deadwood_lut",
    "compute_all_meld_info",
    "compute_legal_knocker_melds",
    "get_optimal_meld_indices",
    "hand_to_4x13",
    "hand_total_points",
    "optimal_melds_mask",
    "valid_melds_mask",
]


@jax.jit
def hand_to_4x13(hand):
  """Converts a 52-card binary mask to a (4, 13) suit x rank matrix.

  Args:
    hand: The 52-card mask.

  Returns:
    The (4, 13) matrix.
  """
  return hand.reshape(4, 13)


class _Scenarios(NamedTuple):
  """The 216 set scenarios of a hand.

  Every choice, for each of the top three set candidate ranks, of no set, one
  of its 3-card sets or its 4-card set.

  Attributes:
    mask_a: [216]: suit nibble of candidate 0's set per scenario.
    mask_b: [216].
    mask_c: [216].
    set_indices: i32[3]: the candidate ranks.
    set_pts: i16[216]: the scenario's set points.
    suits: i16[216, 4]: per-suit rank masks left for runs.
  """

  set_indices: jax.Array
  mask_a: jax.Array
  mask_b: jax.Array
  mask_c: jax.Array
  set_pts: jax.Array
  suits: jax.Array


def _set_scenarios(hand_4x13) -> _Scenarios:
  """The scenarios of a (4, 13) hand (the one enumeration every solver uses).

  Args:
    hand_4x13: int[4, 13], the hand by suit and rank.

  Returns:
    The hand's _Scenarios.
  """
  suit_powers = jnp.asarray(get_suit_powers())
  nibble_popcounts = jnp.asarray(get_nibble_popcounts())
  subset_table = jnp.asarray(get_subset_table())
  rank_values_16 = jnp.asarray(get_rank_values_16())
  powers_of_2_16 = jnp.asarray(get_powers_of_2_16())

  rank_nibbles = jnp.dot(suit_powers, hand_4x13.astype(jnp.int32))  # (13,)
  rank_counts = nibble_popcounts[rank_nibbles]
  is_candidate = rank_counts >= 3
  set_indices = jnp.argsort(~is_candidate)[:3]  # Top 3 ranks
  set_nibbles = rank_nibbles[set_indices]  # (3,) held cards per candidate

  # Options per candidate (6 each: skip, the four 3-card subsets, the 4-card
  # set).
  options = subset_table[set_nibbles]  # (3, 6)
  mask_a = options[0, jnp.asarray(get_scenario_idx_a())]  # (216,)
  mask_b = options[1, jnp.asarray(get_scenario_idx_b())]
  mask_c = options[2, jnp.asarray(get_scenario_idx_c())]

  def pts(mask, k):
    """Returns the set points of nibble `mask` at candidate `k`.

    Args:
      mask: The suit nibble of the set.
      k: The candidate index.

    Returns:
      i16 points.
    """
    return (
        nibble_popcounts[mask].astype(jnp.int16)
        * rank_values_16[set_indices[k]]
    )

  set_pts = pts(mask_a, 0) + pts(mask_b, 1) + pts(mask_c, 2)  # (216,)

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
  )  # (216, 4)
  base_suits = jnp.dot(hand_4x13.astype(jnp.int16), powers_of_2_16)  # (4,)
  suits = base_suits[None, :] & (~total_removal)  # (216, 4)
  return _Scenarios(set_indices, mask_a, mask_b, mask_c, set_pts, suits)


def _invalid_after_discard(sc: _Scenarios, d_suit, d_rank):
  """bool[216]: the scenarios whose sets use the discarded card.

  Args:
    sc: The hand's scenarios.
    d_suit: The discard's suit.
    d_rank: The discard's rank.

  Returns:
    bool[216], True where a set uses the discarded card.
  """
  fail = [
      (d_rank == sc.set_indices[k]) & ((m >> d_suit) & 1).astype(jnp.bool_)
      for k, m in enumerate((sc.mask_a, sc.mask_b, sc.mask_c))
  ]
  return fail[0] | fail[1] | fail[2]


def _best_scenario(sc: _Scenarios, total_meld_pts):
  """The index of the scenario with the most meld points.

  On a tie, the one with more cards in sets (sets before runs, as OpenSpiel's
  AllMelds).

  Args:
    sc: The scenarios.
    total_meld_pts: (216,) each scenario's meld points.

  Returns:
    The index of the best scenario.
  """
  nibble_popcounts = jnp.asarray(get_nibble_popcounts())
  set_card_count = (
      nibble_popcounts[sc.mask_a]
      + nibble_popcounts[sc.mask_b]
      + nibble_popcounts[sc.mask_c]
  ).astype(jnp.int32)
  score_with_tiebreak = total_meld_pts.astype(jnp.int32) * 100 + set_card_count
  return jnp.argmax(score_with_tiebreak)


def _scenario_meld_mask(sc: _Scenarios, best_idx):
  """The melds of scenario best_idx.

  Its sets and the runs left in each suit decomposed into melds.

  Args:
    sc: The scenarios.
    best_idx: The index of the scenario.

  Returns:
    bool[185] the scenario's melds.
  """
  nibble_popcounts = jnp.asarray(get_nibble_popcounts())
  run_decomp_lut = jnp.asarray(get_run_decomp_lut())

  # A 3-card set's meld id is rank * 5 + missing suit; a 4-card set's rank * 5 +
  # 4.
  def get_set_meld_id(nibble_mask, rank):
    """Returns the meld id of a set.

    Args:
      nibble_mask: The suit nibble of the set.
      rank: The set's rank.

    Returns:
      The 4-card or 3-card set's meld id, or -1 for no set.
    """
    popcount = nibble_popcounts[nibble_mask]
    missing_suit = jnp.where(
        nibble_mask & 1,
        jnp.where(nibble_mask & 2, jnp.where(nibble_mask & 4, 3, 2), 1),
        0,
    )
    meld_id_3 = rank * 5 + missing_suit
    meld_id_4 = rank * 5 + 4
    return jnp.where(
        popcount >= 4,
        meld_id_4,
        jnp.where(popcount >= 3, meld_id_3, jnp.int32(-1)),
    )

  optimal_meld_mask = jnp.zeros(185, dtype=jnp.bool_)
  for k, m in enumerate((sc.mask_a, sc.mask_b, sc.mask_c)):
    set_meld = get_set_meld_id(m[best_idx], sc.set_indices[k])
    optimal_meld_mask = jnp.where(
        set_meld >= 0,
        optimal_meld_mask.at[set_meld].set(True),
        optimal_meld_mask,
    )

  # Runs: run_decomp_lut holds suit 0's meld ids (runs of three cards from
  # id 65, of four from id 109, of five from id 149); suit s adds s * 11,
  # s * 10 or s * 9.
  best_scenario_suits = sc.suits[best_idx]
  for suit in range(4):
    run_meld_ids = run_decomp_lut[best_scenario_suits[suit]]  # (5,)
    for j in range(5):
      base_meld_id = run_meld_ids[j]  # suit 0's meld id, or -1
      is_3card = (base_meld_id >= 65) & (base_meld_id < 109)
      is_4card = (base_meld_id >= 109) & (base_meld_id < 149)
      is_5card = (base_meld_id >= 149) & (base_meld_id < 185)
      suit_offset = jnp.where(
          is_3card,
          suit * 11,
          jnp.where(is_4card, suit * 10, jnp.where(is_5card, suit * 9, 0)),
      )
      meld_id = base_meld_id + suit_offset
      is_valid = base_meld_id >= 0
      optimal_meld_mask = jnp.where(
          is_valid, optimal_meld_mask.at[meld_id].set(True), optimal_meld_mask
      )
  return optimal_meld_mask


def _discard_deadwoods(
    sc: _Scenarios, hand_4x13, run_scores, run_lut, discards
):
  """Deadwood after each discard in `discards`.

  Args:
    sc: The scenarios.
    hand_4x13: The hand as a suit-by-rank grid.
    run_scores: i16[216, 4] the scenarios' per-suit run scores under run_lut.
    run_lut: The run score lookup table run_scores was read from.
    discards: i32[k] the discarded cards.

  Returns:
    i16[k] the deadwood after each discard.
  """
  rank_values_16 = jnp.asarray(get_rank_values_16())
  base_total = sc.set_pts + jnp.sum(run_scores, axis=1)  # (216,)
  total_hand_value = jnp.sum(
      hand_4x13.astype(jnp.int16) * rank_values_16[None, :]
  )

  def process_discard(discard_idx):
    """Returns the deadwood after discarding `discard_idx`.

    Args:
      discard_idx: The discarded card.

    Returns:
      The deadwood.
    """
    d_suit = discard_idx // 13
    d_rank = discard_idx % 13
    is_invalid = _invalid_after_discard(sc, d_suit, d_rank)
    # Only the discarded suit's runs change.
    new_suit_bits = sc.suits[:, d_suit] & (~(jnp.int16(1) << d_rank))
    new_run_score = run_lut[new_suit_bits].astype(jnp.int16)
    final_meld_pts = base_total - run_scores[:, d_suit] + new_run_score
    valid_meld_pts = jnp.where(is_invalid, jnp.int16(-1), final_meld_pts)
    new_hand_value = total_hand_value - rank_values_16[d_rank]
    return new_hand_value - jnp.max(valid_meld_pts)

  return jax.vmap(process_discard)(discards)


@jax.jit
def _compute_optimal_meld_info(hand_4x13):
  """Compute optimal meld decomposition and return the optimal melds.

  Args:
    hand_4x13: The hand as a suit-by-rank grid.

  Returns:
      max_meld_points: Total points from optimal melds
      optimal_meld_mask: (185,) bool mask of the melds in it
  """
  sc = _set_scenarios(hand_4x13)
  run_scores = jnp.asarray(get_run_score_lut())[sc.suits]  # (216, 4)
  total_meld_pts = sc.set_pts + jnp.sum(run_scores, axis=1)  # (216,)
  best_idx = _best_scenario(sc, total_meld_pts)
  return total_meld_pts[best_idx], _scenario_meld_mask(sc, best_idx)


@jax.jit
def calculate_deadwood_lut(hand):
  """Calculates exact deadwood using LUT + 3-set enumeration.

  Handles up to 3 set candidates using the 216-scenario approach.
  This is the fast version using lookup tables.

  Args:
    hand: The 52-card mask.

  Returns:
    The deadwood.
  """
  rank_values = jnp.asarray(get_rank_values())
  hand_4x13 = hand_to_4x13(hand)
  max_meld_points, _ = _compute_optimal_meld_info(hand_4x13)
  total_hand_value = jnp.sum(hand_4x13.astype(jnp.int32) * rank_values[None, :])
  return total_hand_value - max_meld_points


@jax.jit
def get_optimal_meld_indices(hand):
  """Returns the melds of the optimal decomposition.

  Args:
    hand: The 52-card mask.

  Returns:
    (185,) bool array where True = meld is in the optimal decomposition.
  """
  hand_4x13 = hand_to_4x13(hand)
  _, optimal_meld_mask = _compute_optimal_meld_info(hand_4x13)
  return optimal_meld_mask


@jax.jit
def calculate_deadwood_all_discards_exact(hand):
  """Exact deadwood after each of the 52 discards.

  Over the full 216-scenario enumeration: up to 3 set candidates, a 4-card
  set may be split (3 in the set, 1 left for runs), and a scenario whose
  sets use the discarded card is invalid.

  Args:
    hand: The 52-card mask.

  Returns:
    i16[52] deadwood per discard; 999 for a card not held.
  """
  run_lut = jnp.asarray(get_run_score_lut())
  hand_4x13 = hand_to_4x13(hand)
  sc = _set_scenarios(hand_4x13)
  dw = _discard_deadwoods(
      sc, hand_4x13, run_lut[sc.suits], run_lut, jnp.arange(52)
  )
  return jnp.where(hand > 0, dw, jnp.int16(999))


@jax.jit
def compute_all_meld_info(hand):
  """Optimal melds and all discard deadwoods in one 216-scenario pass.

  Args:
    hand: The 52-card mask.

  Returns:
      optimal_meld_mask: (185,) bool - which melds are in optimal decomposition
      all_discard_deadwoods: (52,) int16 - deadwood after discarding each card
  """
  run_lut = jnp.asarray(get_run_score_lut())
  hand_4x13 = hand_to_4x13(hand)
  sc = _set_scenarios(hand_4x13)
  run_scores = run_lut[sc.suits]  # (216, 4)
  total_meld_pts = sc.set_pts + jnp.sum(run_scores, axis=1)
  optimal_meld_mask = _scenario_meld_mask(
      sc, _best_scenario(sc, total_meld_pts)
  )
  dw = _discard_deadwoods(sc, hand_4x13, run_scores, run_lut, jnp.arange(52))
  return optimal_meld_mask, jnp.where(hand > 0, dw, jnp.int16(999))


@jax.jit
def calculate_deadwood_compressed(hand):
  """Calculate deadwood ONLY for cards actually in hand (11 vs 52 iterations).

  Args:
    hand: The 52-card mask.

  Returns:
      held_indices: (11,) int32 - indices of cards in hand (padded with 0)
      deadwoods: (11,) int16 - deadwood if that card is discarded
  """
  run_lut_8 = jnp.asarray(get_run_score_lut_8())
  held_indices = jnp.argsort(~(hand > 0))[:11]  # Non-zero first
  held_mask = jnp.take(hand, held_indices) > 0  # (11,) bool
  hand_4x13 = hand_to_4x13(hand)
  sc = _set_scenarios(hand_4x13)
  run_scores = run_lut_8[sc.suits].astype(jnp.int16)
  deadwoods = _discard_deadwoods(
      sc, hand_4x13, run_scores, run_lut_8, held_indices
  )
  return held_indices, jnp.where(held_mask, deadwoods, jnp.int16(999))


@jax.jit
def hand_total_points(hand):
  """Calculates the total point value of all cards in hand.

  card = suit * 13 + rank, so rank = card % 13.

  Args:
    hand: The 52-card mask.

  Returns:
    int32, to prevent dtype promotion issues in downstream computations.
  """
  card_points = jnp.asarray(get_card_points())
  card_vals = jnp.zeros(52, dtype=jnp.int32)  # Use literal 52 for array shape
  for i in range(52):  # Use literal for range
    rank = i % 13
    card_vals = card_vals.at[i].set(card_points[rank])
  return jnp.int32(jnp.sum(hand * card_vals))


@jax.jit
def valid_melds_mask(hand):
  """Returns which melds the hand holds.

  Args:
    hand: The 52-card mask.

  Returns:
    bool[185].
  """
  meld_masks = jnp.asarray(get_meld_masks())
  hand_expanded = hand[None, :]
  meld_card_counts = jnp.sum(meld_masks * hand_expanded, axis=1)
  meld_sizes = jnp.sum(meld_masks, axis=1)
  return meld_card_counts == meld_sizes


@jax.jit
def optimal_melds_mask(hand):
  """Return mask of melds that are part of the optimal decomposition.

  A meld is "optimal" if it's one of the EXACT melds used in the
  optimal meld decomposition, as OpenSpiel computes it.

  Args:
    hand: The 52-card mask.

  Returns:
      Boolean array of shape (185,) where True means the meld can be laid.
  """
  meld_masks = jnp.asarray(get_meld_masks())
  # Get the mask of which meld indices are in the optimal decomposition
  optimal_indices = get_optimal_meld_indices(hand)

  # Check if all meld cards are present in hand
  has_all = jnp.all(hand[None, :] >= meld_masks, axis=1)  # (185,)

  # Meld is optimal if it's in the optimal decomposition AND all cards are in
  # hand
  return has_all & optimal_indices


# Valid-meld slots compute_legal_knocker_melds enumerates: the worst case is
# 11 consecutive cards of one suit, 24 run melds.
KNOCKER_MELD_SLOTS = 24


def _compute_legal_knocker_melds_impl(hand, threshold):
  """compute_legal_knocker_melds over KNOCKER_MELD_SLOTS valid-meld slots.

  Args:
      hand: (52,) int8 array of card counts
      threshold: max deadwood allowed

  Returns:
      (185,) boolean mask of legal melds for knocker
  """
  max_slots = KNOCKER_MELD_SLOTS
  # Use pre-padded versions to avoid creating temporaries
  meld_points_padded = jnp.asarray(
      get_meld_points_padded()
  )  # (186,) - index 185 = 0 points
  meld_overlap_padded = jnp.asarray(
      get_meld_overlap_padded()
  )  # (186, 186) - row/col 185 = True

  total_points = hand_total_points(hand)
  valid = valid_melds_mask(hand)  # (185,) bool

  # Find valid meld indices (up to max_slots)
  valid_indices = jnp.where(valid, size=max_slots, fill_value=NUM_MELDS)[0]
  n_valid = jnp.minimum(jnp.sum(valid.astype(jnp.int32)), max_slots)

  # Index directly into padded arrays
  v_pts = meld_points_padded[valid_indices]
  v_overlap = meld_overlap_padded[valid_indices][:, valid_indices]

  # Mask for which slots are actually valid melds
  slot_valid = jnp.arange(max_slots) < n_valid

  # === Singles: deadwood after laying meld i ===
  single_deadwood = total_points - v_pts
  single_valid_combo = slot_valid & (single_deadwood <= threshold)

  # === Pairs: deadwood after laying melds i and j ===
  pair_non_overlap = ~v_overlap & slot_valid[:, None] & slot_valid[None, :]
  pair_deadwood = total_points - (v_pts[:, None] + v_pts[None, :])
  pair_valid_combo = pair_non_overlap & (pair_deadwood <= threshold)

  # === Triples: deadwood after laying melds i, j, k ===
  ov = v_overlap
  triple_non_overlap = (
      ~ov[:, None, :]
      & ~ov[None, :, :]
      & ~ov[:, :, None]
      & slot_valid[:, None, None]
      & slot_valid[None, :, None]
      & slot_valid[None, None, :]
  )
  triple_deadwood = total_points - (
      v_pts[:, None, None] + v_pts[None, :, None] + v_pts[None, None, :]
  )
  triple_valid_combo = triple_non_overlap & (triple_deadwood <= threshold)

  # === Find which melds appear in ANY valid combo ===
  legal_from_single = single_valid_combo
  legal_from_pair = jnp.any(pair_valid_combo, axis=1) | jnp.any(
      pair_valid_combo, axis=0
  )
  legal_from_triple = (
      jnp.any(triple_valid_combo, axis=(1, 2))
      | jnp.any(triple_valid_combo, axis=(0, 2))
      | jnp.any(triple_valid_combo, axis=(0, 1))
  )

  legal_slots = legal_from_single | legal_from_pair | legal_from_triple

  # Map back to original 185 meld indices
  legal_melds_extended = jnp.zeros(NUM_MELDS + 1, dtype=jnp.bool_)
  legal_melds_extended = legal_melds_extended.at[valid_indices].set(
      legal_slots & slot_valid
  )
  legal_melds = legal_melds_extended[:NUM_MELDS]

  return legal_melds


@jax.jit
def compute_legal_knocker_melds(hand, threshold=KNOCK_THRESHOLD):
  """Compute which melds can legally be laid by the knocker.

  A meld is legal iff it appears in at least one non-overlapping meld
  combination that results in deadwood <= threshold.

  Enumerates singles, pairs and triples of up to KNOCKER_MELD_SLOTS (24)
  valid melds, enough for the worst case of 11 consecutive cards of one
  suit (24 run melds): 24^3 = 13,824 combinations.

  Args:
      hand: (52,) int8 array of card counts
      threshold: max deadwood allowed (default 10)

  Returns:
      (185,) boolean mask of legal melds for knocker
  """
  return _compute_legal_knocker_melds_impl(hand, threshold)
