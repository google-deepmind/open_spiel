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

"""The Arena's row-subset machinery.

on_rows (the deal and the post-action chance resolve run on the rows that need
them) and owned_only with chunk sizes (a NATIVE_JAX slot of a multi-slot Arena
evaluates in the smallest chunk holding its rows). Known answers, masks and
determinism; the bits of whole rollouts are pinned by arena_golden_test.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import frameworks


def test_chunk_ladders_known_answer():
  """row_chunks and slot_chunks on known row counts."""
  assert arena_lib.row_chunks(1024) == (32, 256, 512, 1024)
  assert arena_lib.row_chunks(4096) == (128, 1024, 2048, 4096)
  assert arena_lib.row_chunks(3) == (1, 2, 3)
  assert arena_lib.row_chunks(1) == (1,)
  assert arena_lib.slot_chunks(4096) == (1024, 2560, 4096)
  assert arena_lib.slot_chunks(8) == (2, 5, 8)
  assert arena_lib.slot_chunks(1) == (1,)


def _per_row(x, rows=None):
  """A per-row program with exact float, int and transcendental leaves.

  Args:
    x: The rows.
    rows: The row indices, or None for all.

  Returns:
    {f, i, t} leaves.
  """
  v = x if rows is None else x[rows]
  f = v.astype(jnp.float32)
  return {
      "f": f * 2.0 + 1.0,
      "i": (v * 7 + 1).astype(jnp.int32),
      "t": jnp.sin(f) * 3.0 + 1.0,
  }


@pytest.mark.parametrize(
    "pattern", ["none", "one", "few", "quarter", "half", "most", "all"]
)
def test_on_rows_equals_compute_everywhere_then_select(pattern):
  """on_rows equals computing every row then selecting."""
  r = 64
  x = jnp.arange(r, dtype=jnp.int32) * 5 - 17
  cur = {
      "f": jnp.full((r,), -9.0, jnp.float32),
      "i": jnp.full((r,), -9, jnp.int32),
      "t": jnp.full((r,), -9.0, jnp.float32),
  }
  rng = np.random.default_rng(3)
  k = {
      "none": 0,
      "one": 1,
      "few": 2,
      "quarter": 16,
      "half": 30,
      "most": 50,
      "all": r,
  }[pattern]
  need = np.zeros(r, bool)
  need[rng.permutation(r)[:k]] = True
  got = jax.jit(
      lambda n, c: arena_lib.on_rows(n, lambda rows: _per_row(x, rows), c)
  )(jnp.asarray(need), cur)
  full = _per_row(x)
  for name in ("f", "i", "t"):
    want = np.where(need, np.asarray(full[name]), np.asarray(cur[name]))
    assert got[name].dtype == cur[name].dtype
    if (
        name == "t"
    ):  # XLA:CPU may round a transcendental's tail differently per chunk size
      np.testing.assert_allclose(
          np.asarray(got[name]), want, rtol=1e-6, atol=1e-6
      )
    else:  # exact arithmetic: the same bits whatever chunk a row ran in
      np.testing.assert_array_equal(
          np.asarray(got[name]), want, err_msg=f"{pattern} {name}"
      )


def test_on_rows_runs_only_the_smallest_chunk_that_holds_the_needed_rows():
  """on_rows runs only the smallest chunk that holds the needed rows."""
  r = 64
  seen = []

  def compute(rows):
    """Records the chunk size it runs at.

    Args:
      rows: the rows, or None for every row.

    Returns:
      ones for those rows.
    """
    n = r if rows is None else rows.shape[0]
    jax.debug.callback(lambda: seen.append(n))
    return {"v": jnp.ones((n,), jnp.float32)}

  f = jax.jit(
      lambda need: arena_lib.on_rows(
          need, compute, {"v": jnp.zeros((r,), jnp.float32)}
      )
  )
  for k, chunk in (
      (0, None),
      (1, 2),
      (2, 2),
      (3, 16),
      (16, 16),
      (17, 32),
      (33, 64),
      (64, 64),
  ):
    seen.clear()
    need = np.zeros(r, bool)
    need[:k] = True
    out = jax.block_until_ready(f(jnp.asarray(need)))
    assert seen == ([] if chunk is None else [chunk]), (k, seen)
    np.testing.assert_array_equal(np.asarray(out["v"]), need.astype(np.float32))


def test_owned_only_chunks_never_show_fn_a_row_it_does_not_own():
  """owned_only's chunks never show fn a row the slot does not own."""
  r = 32
  owner = jnp.asarray(np.random.default_rng(0).integers(0, 3, r), jnp.int32)
  x = jnp.arange(r, dtype=jnp.float32)
  for slot in range(3):
    seen = []
    mine = owner == slot

    # Binds this iteration's list now (a closure would bind late).
    def fn(inp, seen=seen):  # pylint: disable=dangerous-default-value
      """Records the owners of the rows it sees.

      Args:
        inp: (owner, value) per row.
        seen: the list the owners are appended to.

      Returns:
        {"y": 2 * value + 1}.
      """
      o, v = inp
      jax.debug.callback(lambda a: seen.append(np.asarray(a).copy()), o)
      return {"y": v * 2.0 + 1.0}

    for sizes in (None, arena_lib.slot_chunks(r)):
      seen.clear()
      out = jax.jit(
          lambda m, fn=fn, sizes=sizes: arena_lib.owned_only(
              m, fn, (owner, x), sizes
          )
      )(mine)
      jax.block_until_ready(out)
      assert seen and all((s == slot).all() for s in seen), (slot, sizes, seen)
      m = np.asarray(mine)
      np.testing.assert_array_equal(
          np.asarray(out["y"])[m], np.asarray(x * 2.0 + 1.0)[m]
      )
      if sizes is not None:
        want = min(c for c in sizes if c >= m.sum())
        assert seen[0].shape == (want,), (seen[0].shape, want)


def test_owned_only_with_no_owned_row_calls_nothing_and_returns_zeros():
  """With no owned row owned_only calls nothing and returns zeros."""
  r = 16
  calls = []

  def fn(v):
    """Records the call.

    Args:
      v: the rows.

    Returns:
      v + 1.
    """
    jax.debug.callback(lambda: calls.append(1))
    return v + 1.0

  for sizes in (None, arena_lib.slot_chunks(r)):
    calls.clear()
    out = jax.jit(
        lambda m, sizes=sizes: arena_lib.owned_only(
            m, fn, jnp.ones((r,), jnp.float32), sizes
        )
    )(jnp.zeros((r,), jnp.bool_))
    np.testing.assert_array_equal(np.asarray(out), np.zeros(r, np.float32))
    assert not calls, sizes


def test_owned_only_rejects_a_bad_ladder():
  """owned_only rejects a chunk ladder not ascending or not ending at R."""
  with pytest.raises(AssertionError):
    arena_lib.owned_only(
        jnp.ones((8,), jnp.bool_), lambda v: v, jnp.ones((8,)), (4, 6)
    )
  with pytest.raises(AssertionError):
    arena_lib.owned_only(
        jnp.ones((8,), jnp.bool_), lambda v: v, jnp.ones((8,)), (8, 4)
    )


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  frameworks.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
