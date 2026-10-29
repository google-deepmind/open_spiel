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

"""Layoffs onto the knocker's melds, and the warm-up of the cached tables.

`init_lazy_constants()` builds every cached constant table of the package
(NumPy arrays) up front; the game module calls it at import.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ALL_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_points,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_card_values,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_cards_bool,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_overlap,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_overlap_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    get_meld_points_padded,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_sizes
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import make_card
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_RANKS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_SUITS
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_all_nibbles
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_grid_a
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_grid_b
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_mask_a_all
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_mask_b_all
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_popcounts,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_popcounts_8,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_to_suits,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_nibble_to_suits_8,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_popcnt_a
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_popcnt_b
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_powers_of_2
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
    get_run_member_lut,
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
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_selection_matrix,
)
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_subset_table
from open_spiel.python.jax.games.gin_rummy.lookup_tables import get_suit_powers
from open_spiel.python.jax.games.gin_rummy.lookup_tables import (
    get_suit_powers_8,
)

__all__ = [
    "compute_layoff_cards",
    "get_is_rank_meld",
    "get_is_suit_meld",
    "get_meld_layoff_cards",
    "init_lazy_constants",
]


@functools.cache
def _compute_meld_info_np():
  """Compute meld info as numpy arrays (for Python control flow).

  Returns:
    (is_rank_meld, is_suit_meld), each bool[NUM_MELDS].
  """
  is_rank_meld = np.zeros(NUM_MELDS, dtype=np.bool_)
  is_suit_meld = np.zeros(NUM_MELDS, dtype=np.bool_)

  for meld_id, meld in enumerate(ALL_MELDS):
    if meld is None:
      continue
    # rank = card % 13
    ranks = [c % NUM_RANKS for c in meld]
    if len(set(ranks)) == 1:
      is_rank_meld[meld_id] = True
    else:
      is_suit_meld[meld_id] = True

  return is_rank_meld, is_suit_meld


@functools.cache
def get_is_rank_meld():
  """Returns which melds are sets (rank melds).

  Returns:
    bool[185].
  """
  is_rank_meld, _ = _compute_meld_info_np()
  return np.array(is_rank_meld)


@functools.cache
def get_is_suit_meld():
  """Returns which melds are runs (suit melds).

  Returns:
    bool[185].
  """
  _, is_suit_meld = _compute_meld_info_np()
  return np.array(is_suit_meld)


def _compute_meld_layoff_info():
  """The cards that can be laid off on each meld.

  A set of three takes its missing suit; a run takes the card just below or
  above it, when that rank exists.

  Returns:
    bool[NUM_MELDS, NUM_CARDS]: whether each card can be laid off on each meld.
  """
  # Use numpy for Python control flow
  is_rank_meld, _ = _compute_meld_info_np()
  layoff_cards = np.zeros((NUM_MELDS, NUM_CARDS), dtype=np.bool_)

  for meld_id, meld in enumerate(ALL_MELDS):
    if meld is None:
      continue

    if is_rank_meld[meld_id]:
      # Rank meld (set): same rank, different suits
      if len(meld) == 3:
        rank = meld[0] % NUM_RANKS
        for suit in range(NUM_SUITS):
          card = make_card(suit, rank)
          if card not in meld:
            layoff_cards[meld_id, card] = True
    else:
      # Suit meld (run): same suit, consecutive ranks
      suit = meld[0] // NUM_RANKS
      ranks = sorted([c % NUM_RANKS for c in meld])
      min_rank, max_rank = ranks[0], ranks[-1]
      if min_rank > 0:
        layoff_cards[meld_id, make_card(suit, min_rank - 1)] = True
      if max_rank < NUM_RANKS - 1:
        layoff_cards[meld_id, make_card(suit, max_rank + 1)] = True

  return np.array(layoff_cards)


@functools.cache
def get_meld_layoff_cards():
  """Returns the cards that can be laid off onto each meld.

  Returns:
    The per-meld layoff cards.
  """
  return _compute_meld_layoff_info()


# True once init_lazy_constants() has completed; callers can assert
# lazy_constants_initialized() before tracing.
_LAZY_CONSTANTS_INITIALIZED = False


def lazy_constants_initialized() -> bool:
  """Returns whether ``init_lazy_constants()`` has run in this process.

  Returns:
    True iff it has.
  """
  return _LAZY_CONSTANTS_INITIALIZED


def init_lazy_constants():
  """Builds every cached constant table of the package now.

  The tables are NumPy arrays, so building them inside or outside a trace is
  equally safe; the game module calls this at import so the first trace does
  not pay for them. Calling it again does nothing.
  """
  global _LAZY_CONSTANTS_INITIALIZED
  # Core constants
  get_card_points()
  get_card_values()
  get_meld_masks()
  get_meld_cards_bool()
  get_meld_sizes()
  get_meld_overlap()
  get_meld_overlap_padded()
  get_meld_points()
  get_meld_points_padded()
  # LUT building blocks
  get_run_score_lut()
  get_rank_values()
  get_rank_values_16()
  get_powers_of_2()
  get_powers_of_2_16()
  get_suit_powers()
  get_suit_powers_8()
  get_all_nibbles()
  get_nibble_popcounts()
  get_nibble_popcounts_8()
  # Grid constants
  get_grid_a()
  get_grid_b()
  get_mask_a_all()
  get_mask_b_all()
  get_popcnt_a()
  get_popcnt_b()
  get_nibble_to_suits()
  get_nibble_to_suits_8()
  # Subset and run tables
  get_subset_table()
  get_run_member_lut()
  get_run_score_lut_8()
  get_run_decomp_lut()
  # Scenario indices
  get_scenario_idx_a()
  get_scenario_idx_b()
  get_scenario_idx_c()
  # Selection matrix
  get_selection_matrix()
  # Meld info
  get_is_rank_meld()
  get_is_suit_meld()
  get_meld_layoff_cards()
  # Flag LAST: only a fully-completed warmup counts as initialized.
  _LAZY_CONSTANTS_INITIALIZED = True


@jax.jit
def compute_layoff_cards(layed_melds_mask, layoffs_mask):
  """Computes the valid layoff cards from the knocker's laid melds.

  Run extensions are included: when a card is laid off onto a run, it extends
  the run, creating new layoff opportunities. For example: if 2-3-4♠ is a
  meld and 5♠ is laid off, now 6♠ is also valid. Sets (rank melds) don't
  extend - once they have 4 cards, they're full.

  Args:
    layed_melds_mask: The knocker's laid melds.
    layoffs_mask: The cards laid off so far.

  Returns:
    bool[52] valid layoff cards.
  """
  meld_layoff_cards = jnp.asarray(get_meld_layoff_cards())
  is_suit_meld = jnp.asarray(get_is_suit_meld())

  # Base layoffs from original melds
  base_layoffs = jnp.any(meld_layoff_cards * layed_melds_mask[:, None], axis=0)

  # Only consider extensions for cards laid off onto RUN melds (suit melds)
  # First, find which base layoffs came from run melds (not set melds)
  run_meld_layoffs = jnp.any(
      meld_layoff_cards * (layed_melds_mask & is_suit_meld)[:, None], axis=0
  )

  def extend_once(carry, xs):
    """One iteration of run extension.

    Args:
      carry: (extended, run reach).
      xs: The loop's xs, None (unused).

    Returns:
      ((extended, run reach), None) for `jax.lax.scan`.
    """
    del xs
    # A newly attached card can itself extend the run on the next scan step.
    # Keep run reach separate so set-only layoffs never create run endpoints.
    extended, run_reach = carry
    just_extended = extended & layoffs_mask & run_reach
    ext_2d = just_extended.reshape(NUM_SUITS, NUM_RANKS)

    # High extension: rank+1 valid if rank was extended (shift right, clear
    # first col)
    high_ext = jnp.roll(ext_2d, 1, axis=1).at[:, 0].set(False)

    # Low extension: rank-1 valid if rank was extended (shift left, clear last
    # col)
    low_ext = jnp.roll(ext_2d, -1, axis=1).at[:, -1].set(False)

    newly_reached = (high_ext | low_ext).flatten()
    return (extended | newly_reached, run_reach | newly_reached), None

  # Iterate extension up to 10 times (max practical run extension depth)
  (extended, _), _ = jax.lax.scan(
      extend_once, (base_layoffs, run_meld_layoffs), None, length=10
  )

  return extended
