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

"""sources.targets: Monte Carlo value targets on a hand-built sequence.

Hand-built sequence, perspective seat 0, gamma 0.5:
  t: 0 own | 1 opp | 2 own | 3 opp, done, r=+3 | 4 terminal | 5 own (new hand)
     | 6 opp | 7 own (segment ends mid-hand)
  ret[2] = 3 (rewards from ply 2 up to the hand end); ret[0] = 0 + 0.5 *
  ret[2] = 1.5; plies 5 and 7 have no finished hand inside the segment ->
  masked.
With a lookahead segment whose ply 0 is an opponent ply that ends the hand
with r=-1: ret[7] = -1, ret[5] = 0.5 * ret[7] = -0.5, both complete.
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.sources import targets
from open_spiel.python.scale.sources.tests import helpers

R = [[0, 0, 0, 3, 0, 0, 0, 0]]
V = [[1, 0, 1, 0, 0, 1, 0, 1]]
D = [[0, 0, 0, 1, 0, 0, 0, 0]]
TO_MOVE = [[0, 1, 0, 1, -1, 0, 1, 0]]


def test_mc_returns_known_answer():
  """Mc returns known answer."""
  g, ok = targets.mc_returns(
      jnp.asarray(R, jnp.float32),
      jnp.asarray(V, bool),
      jnp.asarray(D, bool),
      gamma=0.5,
  )
  assert g.dtype == jnp.float32 and ok.dtype == jnp.bool_
  np.testing.assert_allclose(
      np.asarray(g), [[1.5, 0, 3, 0, 0, 0, 0, 0]], rtol=0, atol=0
  )
  np.testing.assert_array_equal(np.asarray(ok), [[1, 0, 1, 0, 0, 0, 0, 0]])


def test_mc_returns_rejects_non_f32_reward():
  """Mc returns rejects non f32 reward."""
  with pytest.raises(TypeError, match="float32"):
    targets.mc_returns(
        jnp.asarray(R, jnp.float64),
        jnp.asarray(V, bool),
        jnp.asarray(D, bool),
        gamma=0.5,
    )


def test_with_mc_value_targets_writes_seat_column_and_mask():
  """With mc value targets writes seat column and mask."""
  b = helpers.fake_batch(R, V, D, seat=1, to_move=[[1, 0, 1, 0, -1, 1, 0, 1]])
  out = targets.with_mc_value_targets(b, gamma=0.5)
  tv = np.asarray(out.target_value)
  assert out.target_value.dtype == jnp.float32 and tv.shape == (1, 8, 2)
  np.testing.assert_array_equal(
      tv[..., 0], 0.0
  )  # only the perspective seat's column
  np.testing.assert_allclose(tv[..., 1], [[1.5, 0, 3, 0, 0, 0, 0, 0]], atol=0)
  np.testing.assert_array_equal(
      np.asarray(out.labels[targets.TARGET_VALUE_VALID]),
      [[1, 0, 1, 0, 0, 0, 0, 0]],
  )


def test_lookahead_completes_the_unfinished_hand():
  """Lookahead completes the unfinished hand."""
  b = helpers.fake_batch(R, V, D, seat=0, to_move=TO_MOVE)
  nxt = helpers.fake_batch(
      [[-1, 0, 0, 0, 0, 0, 0, 0]], [[0] * 8], [[1, 0, 0, 0, 0, 0, 0, 0]], seat=0
  )
  out = targets.with_mc_value_targets(b, gamma=0.5, lookahead=nxt)
  np.testing.assert_allclose(
      np.asarray(out.target_value[..., 0]),
      [[1.5, 0, 3, 0, 0, -0.5, 0, -1]],
      atol=0,
  )
  np.testing.assert_array_equal(
      np.asarray(out.labels[targets.TARGET_VALUE_VALID]),
      [[1, 0, 1, 0, 0, 1, 0, 1]],
  )
  # Everything but target_value / the mask label is the batch's own.
  helpers.assert_tree_equal(
      out._replace(target_value=b.target_value, labels=b.labels), b
  )


def test_lookahead_must_continue_the_same_rows():
  """Lookahead must continue the same rows."""
  b = helpers.fake_batch(R, V, D, seat=0)
  nxt = helpers.fake_batch(
      np.zeros((2, 8)), np.zeros((2, 8)), np.zeros((2, 8)), seat=0
  )
  with pytest.raises(ValueError, match="same rows"):
    targets.with_mc_value_targets(b, gamma=0.5, lookahead=nxt)


def test_existing_mask_label_is_refused():
  """Existing mask label is refused."""
  b = helpers.fake_batch(
      R,
      V,
      D,
      seat=0,
      labels={targets.TARGET_VALUE_VALID: jnp.ones((1, 8), bool)},
  )
  with pytest.raises(targets.SecondValueWriterError):
    targets.with_mc_value_targets(b, gamma=0.5)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
