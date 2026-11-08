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

"""interop.host.

numpy row gathers and scatters, copies to numpy, and a registered array backend
handling its own leaves.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.interop import host


def test_take_put_alloc_numpy():
  """take, put and alloc_rows on numpy leaves."""
  full = {"a": np.zeros((5, 2), np.float32), "b": np.arange(5, dtype=np.int64)}
  part = {"a": np.ones((2, 2), np.float32), "b": np.array([7, 8], np.int64)}
  idx = np.array([1, 3])
  out = host.put(full, idx, part)
  assert full["a"].sum() == 0 and full["b"].tolist() == [
      0,
      1,
      2,
      3,
      4,
  ], "put mutated its input"
  np.testing.assert_array_equal(out["a"][:, 0], [0, 1, 0, 1, 0])
  np.testing.assert_array_equal(out["b"], [0, 7, 2, 8, 4])
  back = host.take(out, idx)
  np.testing.assert_array_equal(back["a"], part["a"])
  z = host.alloc_rows(part, 4)
  assert (
      z["a"].shape == (4, 2) and z["a"].dtype == np.float32 and not z["a"].any()
  )
  assert z["b"].dtype == np.int64


def test_to_numpy_copies_jax_leaves_and_unwraps_keys():
  """to_numpy copies jax leaves and unwraps typed keys."""
  x = jnp.arange(3.0)
  out = host.to_numpy({"x": x, "k": jax.random.key(1)})
  assert isinstance(out["x"], np.ndarray) and out["x"].tolist() == [
      0.0,
      1.0,
      2.0,
  ]
  assert out["k"].dtype == np.uint32 and out["k"].shape == (2,)
  assert host.raw_keys(jax.random.split(jax.random.key(0), 3)).shape == (3, 2)


class _Rows:
  """A toy array type the numpy path cannot index.

  Attributes:
    rows: the rows, a list.
  """

  def __init__(self, rows):
    self.rows = list(rows)


def test_a_registered_backend_handles_its_leaves():
  """A registered backend handles the leaves its predicate takes."""
  ops = host.ArrayOps(
      take=lambda x, idx: _Rows([x.rows[i] for i in idx]),
      put=lambda x, idx, y: _Rows([
          y.rows[list(idx).index(i)] if i in idx else r
          for i, r in enumerate(x.rows)
      ]),
      zeros_like_rows=lambda x, n: _Rows([0] * n),
      to_numpy=lambda x: np.asarray(x.rows),
  )
  host.register_array_backend(lambda x: isinstance(x, _Rows), ops)
  full = _Rows([10, 11, 12])
  out = host.put({"r": full}, np.array([2]), {"r": _Rows([99])})
  assert out["r"].rows == [10, 11, 99] and full.rows == [10, 11, 12]
  assert host.take({"r": full}, np.array([0, 2]))["r"].rows == [10, 12]
  assert host.alloc_rows({"r": full}, 2)["r"].rows == [0, 0]
  assert host.to_numpy({"r": full})["r"].tolist() == [10, 11, 12]


def test_registering_a_backend_twice_has_no_effect(monkeypatch):
  """Registering a backend twice has no effect."""
  # Reaches private _BACKENDS of the code under test.
  # pylint: disable=protected-access
  monkeypatch.setattr(host, "_BACKENDS", [])

  def pred(x):
    """Whether `x` is a _Rows.

    Args:
      x: A value.

    Returns:
      True for a _Rows.
    """
    return isinstance(x, _Rows)

  host.register_array_backend(pred, host.NUMPY)
  host.register_array_backend(pred, host.NUMPY)
  assert len(host._BACKENDS) == 1
  other = host.NUMPY._replace(to_numpy=lambda x: np.asarray(x.rows))
  with pytest.raises(ValueError, match="already registered"):
    host.register_array_backend(pred, other)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
