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

"""sources.replay.ReplayBuffer.

sources.replay.ReplayBuffer: FIFO over the last K adds, uniform rows from
the retained adds only, deterministic under a key, layout checked on add,
int64 counters.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.sources import replay
from open_spiel.python.scale.sources.tests import helpers


def _it(i, rows=4, t=3):
  """An iteration's batch whose reward is the iteration index everywhere.

  Args:
    i: The iteration.
    rows: The rows.
    t: The plies.

  Returns:
    The batch.
  """
  return helpers.fake_batch(
      np.full((rows, t), float(i)),
      np.ones((rows, t)),
      np.zeros((rows, t)),
      seat=0,
  )


def _iters_of(batch):
  """The iteration index of each row.

  Args:
    batch: The batch.

  Returns:
    int rows.
  """
  r = np.asarray(batch.reward)
  assert (r == r[:, :1]).all()
  return r[:, 0].astype(int)


def test_fifo_keeps_last_k_iterations():
  """Fifo keeps last k iterations."""
  buf = replay.ReplayBuffer(capacity_iterations=3, batch_size=4)
  for i in range(5):
    buf.add(_it(i))
  assert buf.num_iterations == 3 and buf.num_rows == 12
  assert buf.iterations_added == 5 and isinstance(
      buf.iterations_added, np.int64
  )
  np.testing.assert_array_equal(
      _iters_of(buf.contents()), [2] * 4 + [3] * 4 + [4] * 4
  )


def test_fifo_variable_rows_per_iteration():
  """Fifo variable rows per iteration."""
  buf = replay.ReplayBuffer(capacity_iterations=2, batch_size=2)
  buf.add(_it(0, rows=2))
  buf.add(_it(1, rows=6))
  buf.add(_it(2, rows=1))
  assert buf.num_rows == 7
  np.testing.assert_array_equal(_iters_of(buf.contents()), [1] * 6 + [2])


def test_next_samples_only_retained_rows_uniformly():
  """Next samples only retained rows uniformly."""
  buf = replay.ReplayBuffer(capacity_iterations=2, batch_size=8)
  for i in range(4):
    buf.add(_it(i, rows=8))
  seen = np.concatenate(
      [_iters_of(buf.next(jax.random.key(k))) for k in range(200)]
  )
  assert set(seen.tolist()) == {2, 3}
  frac = (seen == 3).mean()
  assert abs(frac - 0.5) < 4 * np.sqrt(0.25 / seen.size), frac


def test_next_without_replacement_and_deterministic():
  """Next without replacement and deterministic."""
  buf = replay.ReplayBuffer(capacity_iterations=2, batch_size=8)
  buf.add(
      helpers.fake_batch(
          np.arange(8.0)[:, None] * np.ones((1, 3)),
          np.ones((8, 3)),
          np.zeros((8, 3)),
          seat=0,
      )
  )
  a = buf.next(jax.random.key(7))
  helpers.assert_tree_equal(a, buf.next(jax.random.key(7)))
  assert sorted(np.asarray(a.reward)[:, 0].tolist()) == list(
      range(8)
  )  # a permutation


def test_with_replacement_when_asked():
  """With replacement when asked."""
  buf = replay.ReplayBuffer(capacity_iterations=1, batch_size=16, replace=True)
  buf.add(_it(0, rows=2))
  assert np.asarray(buf.next(jax.random.key(0)).reward).shape == (16, 3)


def test_errors():
  """Errors."""
  buf = replay.ReplayBuffer(capacity_iterations=2, batch_size=4)
  with pytest.raises(AssertionError):
    buf.next(jax.random.key(0))  # empty
  buf.add(_it(0, rows=2))
  with pytest.raises(AssertionError):
    buf.next(
        jax.random.key(0)
    )  # fewer rows than batch_size without replacement
  with pytest.raises(AssertionError):
    buf.add(_it(1, t=4))  # T changed
  bad = _it(1)
  with pytest.raises(AssertionError):
    buf.add(
        bad._replace(reward=bad.reward.astype(jnp.float64))
    )  # dtype changed
  with pytest.raises(AssertionError):
    buf.add(bad._replace(labels={"x": bad.valid}))  # structure changed
  with pytest.raises(AssertionError):
    replay.ReplayBuffer(capacity_iterations=0, batch_size=4)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
