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

"""Layoffs extend runs at both ends, chained, and sets never start a run.

The extensions equal an independent interval computation.
"""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ALL_MELDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_MELDS
from open_spiel.python.jax.games.gin_rummy.layoffs import compute_layoff_cards
from open_spiel.python.jax.games.gin_rummy.layoffs import init_lazy_constants


class RunLayoffTest(absltest.TestCase):
  """Run layoffs against an independent interval computation."""

  def test_all_contiguous_run_extensions(self):
    """Every contiguous run extension is a legal layoff."""
    init_lazy_constants()
    meld_masks, laid_masks, occupied_masks, expected = [], [], [], []
    for mid, meld in enumerate(ALL_MELDS):
      suit = meld[0] // 13
      if any(c // 13 != suit for c in meld):
        continue
      low, high = min(meld) % 13, max(meld) % 13
      for left in range(low + 1):
        for right in range(13 - high):
          if left + right > 10:
            continue
          lo, hi = low - left, high + right
          laid = list(range(suit * 13 + lo, suit * 13 + low))
          laid += list(range(suit * 13 + high + 1, suit * 13 + hi + 1))
          mm = np.zeros(NUM_MELDS, bool)
          mm[mid] = True
          lm = np.zeros(52, bool)
          lm[laid] = True
          occupied = lm.copy()
          occupied[list(meld)] = True
          endpoint = np.zeros(52, bool)
          if lo > 0:
            endpoint[suit * 13 + lo - 1] = True
          if hi < 12:
            endpoint[suit * 13 + hi + 1] = True
          meld_masks.append(mm)
          laid_masks.append(lm)
          occupied_masks.append(occupied)
          expected.append(endpoint)
    got = jax.jit(jax.vmap(compute_layoff_cards))(
        jnp.asarray(np.stack(meld_masks)), jnp.asarray(np.stack(laid_masks))
    )
    # Callers intersect the helper's candidate set with cards still in hand.
    # Remove cards already laid or melded before comparing future endpoints.
    got = np.asarray(got) & ~np.stack(occupied_masks)
    np.testing.assert_array_equal(got, np.stack(expected))
    self.assertGreater(len(expected), 1000)

  def test_set_layoff_does_not_start_a_run(self):
    """A layoff on a set never starts a run."""
    init_lazy_constants()
    mid = next(i for i, m in enumerate(ALL_MELDS) if sorted(m) == [5, 18, 31])
    meld = jnp.zeros(NUM_MELDS, bool).at[mid].set(True)
    laid = jnp.zeros(52, bool).at[44].set(True)
    actual = np.asarray(compute_layoff_cards(meld, laid))
    self.assertFalse(actual[43])
    self.assertFalse(actual[45])


if __name__ == '__main__':
  absltest.main()
