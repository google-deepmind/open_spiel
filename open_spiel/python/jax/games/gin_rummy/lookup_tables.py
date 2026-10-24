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

"""Precomputed tables for gin rummy deadwood.

The tables are run scores per suit bitmask, popcounts, subset and
decomposition tables, and the meld selection matrix. Every builder is cached
and returns numpy arrays (see `docs/cached_constants.md`).
"""

import functools
import itertools

import numpy as np

__all__ = [
    "MAX_MELDS_IN_HAND",
    "NUM_COMBOS",
    "get_all_nibbles",
    "get_grid_a",
    "get_grid_b",
    "get_mask_a_all",
    "get_mask_b_all",
    "get_nibble_popcounts",
    "get_nibble_popcounts_8",
    "get_nibble_to_suits",
    "get_nibble_to_suits_8",
    "get_popcnt_a",
    "get_popcnt_b",
    "get_powers_of_2",
    "get_powers_of_2_16",
    "get_rank_values",
    "get_rank_values_16",
    "get_run_decomp_lut",
    "get_run_member_lut",
    "get_run_score_lut",
    "get_run_score_lut_8",
    "get_scenario_idx_a",
    "get_scenario_idx_b",
    "get_scenario_idx_c",
    "get_selection_matrix",
    "get_subset_table",
    "get_suit_powers",
    "get_suit_powers_8",
]

# =============================================================================
# Lookup tables for exact deadwood computation.
# =============================================================================


def _build_run_lookup_table():
  """Build lookup table: suit bitmask (13 bits) -> max points from runs.

  Uses DP: for each position, either skip or start a run of length 3/4/5/...

  Returns:
    i16[8192] the run points of each suit bitmask.
  """
  card_values = [min(i + 1, 10) for i in range(13)]  # A=1, 2-9, 10/J/Q/K=10
  lut = []

  for bits in range(8192):  # 2^13 configurations
    held = [(bits >> i) & 1 for i in range(13)]

    # DP: dp[i] = max points using cards from position i onwards
    dp = [0] * 14  # dp[13] = 0 (base case)

    for i in range(12, -1, -1):  # Process right to left
      best = dp[i + 1]  # Option 1: skip this position

      # Option 2: start a run at position i (if possible)
      for length in range(3, 14 - i):  # runs of length 3, 4, 5...
        if all(held[i + k] for k in range(length)):
          run_points = sum(card_values[i + k] for k in range(length))
          best = max(best, run_points + dp[i + length])
        else:
          break

      dp[i] = best

    lut.append(dp[0])

  return np.array(lut, dtype=np.int16)


# Precomputed tables for LUT-based deadwood
@functools.cache
def get_run_score_lut():
  """Returns the run-score table of a suit bitmask.

  Returns:
    int16[8192]: the most points a 13-bit suit bitmask scores in runs.
  """
  return _build_run_lookup_table()


@functools.cache
def get_rank_values():
  """Returns each rank's deadwood value.

  Returns:
    int32[13]: A=1, 2-9 face value, 10/J/Q/K=10.
  """
  return np.array([min(i + 1, 10) for i in range(13)], dtype=np.int32)


@functools.cache
def get_rank_values_16():
  """Returns `get_rank_values` as int16.

  Returns:
    int16[13]: each rank's deadwood value.
  """
  return get_rank_values().astype(np.int16)


@functools.cache
def get_powers_of_2():
  """Returns the bit weight of each rank.

  Returns:
    int32[13]: 2**rank.
  """
  return 2 ** np.arange(13, dtype=np.int32)


@functools.cache
def get_powers_of_2_16():
  """Returns `get_powers_of_2` as int16.

  Returns:
    int16[13]: 2**rank.
  """
  return get_powers_of_2().astype(np.int16)


@functools.cache
def get_suit_powers():
  """Returns the bit weight of each suit in a suit nibble.

  Returns:
    int32[4]: [1, 2, 4, 8].
  """
  return np.array([1, 2, 4, 8], dtype=np.int32)


@functools.cache
def get_suit_powers_8():
  """Returns `get_suit_powers` as int8.

  Returns:
    int8[4]: [1, 2, 4, 8].
  """
  return get_suit_powers().astype(np.int8)


# All 16 possible 4-bit masks for set subsets
@functools.cache
def get_all_nibbles():
  """Returns every 4-bit suit nibble.

  Returns:
    int32[16]: 0..15.
  """
  return np.arange(16, dtype=np.int32)


@functools.cache
def get_nibble_popcounts():
  """Returns the number of set bits of each nibble.

  Returns:
    int32[16]: popcount(n) for n in 0..15.
  """
  return np.array([bin(i).count("1") for i in range(16)], dtype=np.int32)


@functools.cache
def get_nibble_popcounts_8():
  """Returns `get_nibble_popcounts` as int8.

  Returns:
    int8[16]: popcount(n) for n in 0..15.
  """
  return get_nibble_popcounts().astype(np.int8)


# Precompute meshgrid for 256 set combinations
@functools.cache
def get_grid_a():
  """Returns the first index of every nibble pair.

  Returns:
    int[256]: the column index of a 16x16 meshgrid, flattened.
  """
  idx_grid = np.arange(16)
  grid_a, _ = np.meshgrid(idx_grid, idx_grid)
  return grid_a.flatten()  # (256,)


@functools.cache
def get_grid_b():
  """Returns the second index of every nibble pair.

  Returns:
    int[256]: the row index of a 16x16 meshgrid, flattened.
  """
  idx_grid = np.arange(16)
  _, grid_b = np.meshgrid(idx_grid, idx_grid)
  return grid_b.flatten()  # (256,)


@functools.cache
def get_mask_a_all():
  """Returns the first nibble of every nibble pair.

  Returns:
    int32[256]: get_all_nibbles()[get_grid_a()].
  """
  return get_all_nibbles()[get_grid_a()]  # (256,)


@functools.cache
def get_mask_b_all():
  """Returns the second nibble of every nibble pair.

  Returns:
    int32[256]: get_all_nibbles()[get_grid_b()].
  """
  return get_all_nibbles()[get_grid_b()]  # (256,)


@functools.cache
def get_popcnt_a():
  """Returns the popcount of the first nibble of every pair.

  Returns:
    int32[256]: the popcounts, indexed like get_grid_a().
  """
  return get_nibble_popcounts()[get_grid_a()]  # (256,)


@functools.cache
def get_popcnt_b():
  """Returns the popcount of the second nibble of every pair.

  Returns:
    int32[256]: the popcounts, indexed like get_grid_b().
  """
  return get_nibble_popcounts()[get_grid_b()]  # (256,)


# Precompute suit expansion for all nibble values (16, 4)
@functools.cache
def get_nibble_to_suits():
  """Returns each nibble's suits as bits.

  Returns:
    int[16, 4]: bit i of nibble n at [n, i].
  """
  return np.stack([((np.arange(16) >> i) & 1) for i in range(4)], axis=-1)


@functools.cache
def get_nibble_to_suits_8():
  """Returns `get_nibble_to_suits` as int8.

  Returns:
    int8[16, 4]: bit i of nibble n at [n, i].
  """
  return get_nibble_to_suits().astype(np.int8)


def _build_subset_table():
  """Build (16, 6) table mapping held nibble -> valid subset options.

  For each nibble (0-15), list valid subsets:
  - 0 (don't use as set)
  - Any subset with popcount >= 3 that fits within held cards

  Returns:
    i32[16, 6] the subset options of each nibble.
  """
  table = np.zeros((16, 6), dtype=np.int32)

  for nibble in range(16):
    options = [0]  # Always can skip
    popcount = bin(nibble).count("1")

    if popcount >= 3:
      for subset in range(1, 16):
        subset_pop = bin(subset).count("1")
        if (nibble & subset) == subset and subset_pop >= 3:
          options.append(subset)

    while len(options) < 6:
      options.append(0)
    table[nibble] = options[:6]

  return np.array(table, dtype=np.int32)


@functools.cache
def get_subset_table():
  """Returns the valid set-meld subsets of each held nibble.

  Returns:
    (16, 6) table: per held suit nibble, its subset options.
  """
  return _build_subset_table()  # (16, 6)


def _build_run_member_table():
  """Build (8192,) table: which ranks are part of a run for each suit config.

  Value is a 13-bit mask indicating which ranks participate in runs.

  Returns:
    i16[8192] the run-member mask of each suit bitmask.
  """
  lut = []
  for bits in range(8192):
    held = [(bits >> i) & 1 for i in range(13)]
    is_member = [0] * 13

    for i in range(11):  # Start positions 0..10
      if held[i] and held[i + 1] and held[i + 2]:
        is_member[i] = 1
        is_member[i + 1] = 1
        is_member[i + 2] = 1
        k = 3
        while i + k < 13 and held[i + k]:
          is_member[i + k] = 1
          k += 1

    mask = 0
    for i, m in enumerate(is_member):
      if m:
        mask |= 1 << i
    lut.append(mask)

  return np.array(lut, dtype=np.int16)


@functools.cache
def get_run_member_lut():
  """Returns the ranks in a run for each suit bitmask.

  Returns:
    (8192,) table: per 13-bit suit bitmask, the bitmask of its ranks that
    are part of a run.
  """
  return _build_run_member_table()  # (8192,)


@functools.cache
def get_run_score_lut_8():
  """Returns `get_run_score_lut` as int8.

  Returns:
    int8[8192]: run points per suit bitmask.
  """
  return get_run_score_lut().astype(np.int8)


def _build_run_decomposition_table():
  """Build (8192, 5) table: optimal run meld IDs for each suit config.

  For suit 0 (spades), returns actual meld IDs. Adjust for other suits at
  runtime. Meld ID encoding:
  - 3-card runs: 65 + suit*11 + start_rank
  - 4-card runs: 109 + suit*10 + start_rank
  - 5-card runs: 149 + suit*9 + start_rank

  Uses greedy decomposition: for each maximal run, split into valid melds.

  Returns:
    i32[8192, 5] the run meld ids of each suit bitmask.
  """
  lut = []
  for bits in range(8192):
    held = [(bits >> i) & 1 for i in range(13)]

    # Find maximal runs
    runs = []  # List of (start, length)
    i = 0
    while i < 13:
      if held[i]:
        start = i
        while i < 13 and held[i]:
          i += 1
        length = i - start
        if length >= 3:
          runs.append((start, length))
      else:
        i += 1

    # Greedy decomposition of each maximal run into 3/4/5-card melds, 3s first.
    meld_ids = []
    for start, length in runs:
      pos = start
      remaining = length
      while remaining >= 3:
        if remaining == 3:
          # 3-card run
          meld_ids.append(65 + pos)  # For suit 0
          pos += 3
          remaining -= 3
        elif remaining == 4:
          # 4-card run
          meld_ids.append(109 + pos)
          pos += 4
          remaining -= 4
        elif remaining == 5:
          # 5-card run
          meld_ids.append(149 + pos)
          pos += 5
          remaining -= 5
        elif remaining == 6:
          # Two 3-card runs
          meld_ids.append(65 + pos)
          meld_ids.append(65 + pos + 3)
          remaining = 0
        elif remaining == 7:
          # 3 + 4
          meld_ids.append(65 + pos)
          meld_ids.append(109 + pos + 3)
          remaining = 0
        elif remaining == 8:
          # 3 + 5 or 4 + 4 (both valid, prefer 3+5 for consistency)
          meld_ids.append(65 + pos)
          meld_ids.append(149 + pos + 3)
          remaining = 0
        elif remaining >= 9:
          # 3 + 3 + ... (take a 3-card run, continue)
          meld_ids.append(65 + pos)
          pos += 3
          remaining -= 3

    # Pad to 5 entries
    while len(meld_ids) < 5:
      meld_ids.append(-1)

    lut.append(meld_ids[:5])

  return np.array(lut, dtype=np.int32)


@functools.cache
def get_run_decomp_lut():
  """Returns the optimal run melds of each suit bitmask.

  Returns:
    (8192, 5) table: the suit-0 run meld ids, padded to 5 entries.
  """
  return _build_run_decomposition_table()  # (8192, 5)


# Precompute the 216 scenario indices
@functools.cache
def get_scenario_idx_a():
  """Returns the first option index of each of the 216 scenarios.

  Returns:
    int[216]: the first axis of a 6x6x6 meshgrid, flattened.
  """
  idx6 = np.arange(6)
  grid_a_216, _, _ = np.meshgrid(idx6, idx6, idx6, indexing="ij")
  return grid_a_216.flatten()  # (216,)


@functools.cache
def get_scenario_idx_b():
  """Returns the second option index of each of the 216 scenarios.

  Returns:
    int[216]: the second axis of a 6x6x6 meshgrid, flattened.
  """
  idx6 = np.arange(6)
  _, grid_b_216, _ = np.meshgrid(idx6, idx6, idx6, indexing="ij")
  return grid_b_216.flatten()  # (216,)


@functools.cache
def get_scenario_idx_c():
  """Returns the third option index of each of the 216 scenarios.

  Returns:
    int[216]: the third axis of a 6x6x6 meshgrid, flattened.
  """
  idx6 = np.arange(6)
  _, _, grid_c_216 = np.meshgrid(idx6, idx6, idx6, indexing="ij")
  return grid_c_216.flatten()  # (216,)


# =============================================================================
# Vectorized meld combination search (K=24 max melds)
# =============================================================================

MAX_MELDS_IN_HAND = 24  # Max possible melds in an 11-card hand


def _generate_selection_matrix(k):
  """Generate all subsets of size 0, 1, 2, 3 from k elements.

  For k=24: 1 + 24 + 276 + 2024 = 2325 rows.

  Args:
    k: The number of elements.

  Returns:
    An int8 binary matrix where each row is a subset selection.
  """
  subsets = []
  # Size 0: empty set
  subsets.append([0] * k)
  # Size 1
  for i in range(k):
    row = [0] * k
    row[i] = 1
    subsets.append(row)
  # Size 2
  for i, j in itertools.combinations(range(k), 2):
    row = [0] * k
    row[i] = 1
    row[j] = 1
    subsets.append(row)
  # Size 3
  for i, j, l in itertools.combinations(range(k), 3):
    row = [0] * k
    row[i] = 1
    row[j] = 1
    row[l] = 1
    subsets.append(row)

  return np.array(subsets, dtype=np.int8)


@functools.cache
def get_selection_matrix():
  """Returns every subset of up to 3 of MAX_MELDS_IN_HAND melds.

  Returns:
    The selection matrix, one row per subset.
  """
  return np.array(_generate_selection_matrix(MAX_MELDS_IN_HAND))


# NUM_COMBOS = 2325 for K=24 (constant, doesn't depend on JAX)
NUM_COMBOS = 2325
