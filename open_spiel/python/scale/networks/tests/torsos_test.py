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

"""Torsos alone: shapes, dtypes, known answers, MLPTorso flattening input."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.networks import torsos


def test_mlp_torso_known_answer():
  """Mlp torso known answer."""
  t = torsos.MLPTorso(widths=(2,), layout="shared", activation="relu")
  p = {
      "params": {
          "shared_0": {
              "kernel": jnp.asarray([[1.0, -1.0], [1.0, 1.0]]),
              "bias": jnp.asarray([0.0, 0.5]),
          }
      }
  }
  f = t.apply(p, jnp.asarray([[2.0, 3.0], [-2.0, -3.0]], jnp.float32))
  np.testing.assert_array_equal(
      np.asarray(f["policy"]), [[5.0, 1.5], [0.0, 0.0]]
  )
  assert f["policy"] is f["value"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("layout", ["shared", "separate"])
def test_mlp_torso_shapes_dtypes(layout, dtype):
  """Mlp torso shapes dtypes."""
  t = torsos.MLPTorso(widths=(8, 4), layout=layout, dtype=dtype)
  x = jnp.ones((3, 5), jnp.float64)  # f64 input is cast to the compute dtype
  p = t.init(jax.random.key(0), x)
  assert all(l.dtype == jnp.float32 for l in jax.tree.leaves(p))
  names = set(p["params"])
  assert names == (
      {"shared_0", "shared_1"}
      if layout == "shared"
      else {"actor_0", "actor_1", "critic_0", "critic_1"}
  )
  f = t.apply(p, x)
  for k in ("policy", "value"):
    assert f[k].shape == (3, 4) and f[k].dtype == jnp.dtype(dtype)
  if layout == "separate":
    assert not np.array_equal(
        np.asarray(f["policy"], np.float32), np.asarray(f["value"], np.float32)
    )


def test_mlp_torso_rejects_bad_layout_and_rank():
  """Mlp torso rejects bad layout and rank."""
  with pytest.raises(ValueError, match="layout"):
    torsos.MLPTorso(layout="tied").init(jax.random.key(0), jnp.ones((1, 2)))
  with pytest.raises(ValueError, match="N, ..."):
    torsos.MLPTorso().init(jax.random.key(0), jnp.ones((3,)))


def test_cnn_torso_known_answer():
  """Identity 3x3 conv, max_pool 3x3/2 SAME on 4x4 -> [x22, x23, x32, x33]."""
  t = torsos.CNNTorso(
      channels=1, relative_stack_sizes=(1.0,), num_blocks=0, dense_hidden=4
  )
  k = np.zeros((3, 3, 1, 1), np.float32)
  k[1, 1, 0, 0] = 1.0
  p = {
      "params": {
          "stack0_conv": {"kernel": jnp.asarray(k), "bias": jnp.zeros((1,))},
          "dense": {"kernel": jnp.eye(4), "bias": jnp.zeros((4,))},
      }
  }
  x = jnp.arange(16.0, dtype=jnp.float32).reshape(1, 4, 4, 1)
  f = t.apply(p, x)
  np.testing.assert_array_equal(
      np.asarray(f["policy"]), [[10.0, 11.0, 14.0, 15.0]]
  )


def test_cnn_torso_shapes_names():
  """Cnn torso shapes names."""
  t = torsos.CNNTorso(
      channels=8, relative_stack_sizes=(0.5, 1.0), num_blocks=2, dense_hidden=16
  )
  x = jnp.ones((2, 6, 7, 3), jnp.float32)
  p = t.init(jax.random.key(0), x)
  names = set(p["params"])
  assert {
      "stack0_conv",
      "stack0_block1_conv1",
      "stack1_conv",
      "stack1_block0_conv0",
      "dense",
  } <= names and len(names) == 2 * (1 + 2 * 2) + 1
  f = t.apply(p, x)
  assert f["policy"].shape == (2, 16) and f["policy"].dtype == jnp.float32


def test_recurrent_torso_carry_and_reset_shape():
  """Recurrent torso carry and reset shape."""
  inner = torsos.MLPTorso(widths=(4,))
  t = torsos.RecurrentTorso(inner=inner, hidden_size=3)
  x, h = jnp.ones((2, 5)), jnp.zeros((2, 3))
  p = t.init(jax.random.key(0), x, h, method=torsos.RecurrentTorso.step)
  h1 = t.apply(p, x, h, method=torsos.RecurrentTorso.step)
  assert h1.shape == (2, 3) and h1.dtype == jnp.float32
  h2 = t.apply(p, x, h1, method=torsos.RecurrentTorso.step)
  # the carry feeds the next step
  assert not np.array_equal(np.asarray(h1), np.asarray(h2))
  with pytest.raises(ValueError, match="shared"):
    bad = torsos.RecurrentTorso(
        inner=torsos.MLPTorso(widths=(4,), layout="separate"), hidden_size=3
    )
    bad.init(jax.random.key(0), x, h, method=torsos.RecurrentTorso.step)


def test_mlp_torso_flattens_planes():
  """Mlp torso flattens planes."""
  t = torsos.MLPTorso(widths=(3,))
  x = jnp.arange(2 * 6 * 7 * 3, dtype=jnp.float32).reshape(2, 6, 7, 3) / 100.0
  p = t.init(jax.random.key(0), x)
  assert p["params"]["shared_0"]["kernel"].shape == (126, 3)
  np.testing.assert_array_equal(
      np.asarray(t.apply(p, x)["policy"]),
      np.asarray(t.apply(p, x.reshape(2, 126))["policy"]),
  )
  with pytest.raises(ValueError, match="N, ..."):
    t.apply(p, jnp.ones((126,), jnp.float32))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
