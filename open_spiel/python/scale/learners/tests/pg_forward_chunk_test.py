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

"""PGLearner forward chunking (pg.py module doc "Forward chunking").

PGLearner forward chunking (pg.py module doc "Forward chunking"): every
learner call of the policy runs at most PGConfig.forward_chunk_rows rows at a
time, and the results are the unchunked ones.
  * config: forward_chunk_rows is an int >= 1, default 16,384;
  * a recurrent policy's [B, T] forward (evaluate_seq from the init carry)
    chunks over B and gives the unchunked logits, values and update.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx


def test_forward_chunk_rows_config():
  """Forward chunk rows config."""
  assert pg.PGConfig(num_updates=1).forward_chunk_rows == 16384
  for bad in (0, -1, 2.0, True, "16"):
    with pytest.raises(ValueError, match="forward_chunk_rows"):
      pg.PGConfig(forward_chunk_rows=bad)


def _param_diffs(a, b):
  """The absolute differences of two param trees, flattened.

  Args:
    a: A param tree.
    b: The other param tree.

  Returns:
    f64 differences.
  """
  return np.concatenate([
      np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).ravel()
      for x, y in zip(
          jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
      )
  ])


def test_recurrent_segments_chunked_over_rows():
  """Recurrent segments chunked over rows."""
  pol = fx.make_policy("leduc_poker", fx.rnn())

  def _learner(chunk):
    """A recurrent-segment learner with forward_chunk_rows `chunk`.

    Args:
      chunk: The forward chunk rows.

    Returns:
      The PGLearner.
    """
    cfg = pg.PGConfig(
        loss="mmd",
        lr=1e-3,
        num_updates=10,
        update_epochs=1,
        num_minibatches=2,
        shuffle="env_axis",
        kl_coef=schedules.Schedule("constant", 0.2),
        forward_chunk_rows=chunk,
    )
    return pg.PGLearner(pol, cfg)

  big, small = _learner(10**9), _learner(3)
  s = big.init(jax.random.key(0), fx.make_env("leduc_poker"))
  src = fx.source(pol, s.params, "leduc_poker", b=8, t=5)
  src.next(jax.random.key(1))
  batch = src.next(jax.random.key(2))  # starts mid-hand: a non-zero init carry
  assert np.abs(np.asarray(batch.init_carry["h"])).max() > 0
  lb, vb, _ = big.forward(s.params, batch.init_carry, batch.steps)
  ls, vs, _ = small.forward(
      s.params, batch.init_carry, batch.steps
  )  # 8 rows: 3 chunks of 3
  np.testing.assert_allclose(np.asarray(ls), np.asarray(lb), atol=1e-6, rtol=0)
  np.testing.assert_allclose(np.asarray(vs), np.asarray(vb), atol=1e-6, rtol=0)
  s2b, _ = big.update(s, batch, jax.random.key(4))
  s2s, _ = small.update(
      small.init_from_params(s.params), batch, jax.random.key(4)
  )
  diffs = _param_diffs(s2b.params, s2s.params)
  assert (
      float(np.mean(diffs <= 1e-6)) >= 0.99
      and diffs.max() <= 2 * 1e-3 * 2 + 1e-9
  ), diffs.max()


def test_an_annealed_step_size_needs_its_horizon():
  """An annealed step size needs its horizon."""
  with pytest.raises(ValueError, match="num_updates"):
    pg.PGConfig()  # anneal_lr is on by default
  assert pg.PGConfig(anneal_lr=False).lr_schedule().is_constant()
  assert not pg.PGConfig(num_updates=10).lr_schedule().is_constant()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
