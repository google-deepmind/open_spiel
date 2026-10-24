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

"""Heads and HL-Gauss alone: shapes, dtypes, known answers (set weights)."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.networks import heads
from open_spiel.python.scale.networks import hl_gauss


def _p(**mods):
  """A params tree from {module: {name: value}}.

  Args:
    **mods: The modules' params.

  Returns:
    {params: ...}, float32 leaves.
  """
  return {
      "params": {
          k: {kk: jnp.asarray(vv, jnp.float32) for kk, vv in v.items()}
          for k, v in mods.items()
      }
  }


def test_dense_policy_head_known_answer():
  """Dense policy head known answer."""
  h = heads.DensePolicyHead(num_actions=2)
  p = _p(dense={"kernel": [[1, 2], [3, 4], [5, 6]], "bias": [0.5, -0.5]})
  out = h.apply(p, {"policy": jnp.asarray([[1.0, 0.0, -1.0]], jnp.float32)})
  assert out.dtype == jnp.float32
  np.testing.assert_array_equal(np.asarray(out), [[-3.5, -4.5]])


def test_dense_policy_head_bf16_compute_f32_params():
  """Dense policy head bf16 compute f32 params."""
  h = heads.DensePolicyHead(num_actions=3, dtype="bfloat16")
  x = {"policy": jnp.ones((2, 4), jnp.bfloat16)}
  p = h.init(jax.random.key(0), x)
  assert all(l.dtype == jnp.float32 for l in jax.tree.leaves(p))
  assert h.apply(p, x).dtype == jnp.bfloat16


def test_value_head_scalar_hidden_known_answer():
  """Value head scalar hidden known answer."""
  h = heads.ValueHead(kind="scalar", hidden=2, hidden_activation="relu")
  p = _p(
      hidden={"kernel": [[1, 0], [0, 1]], "bias": [0, 0]},
      out={"kernel": [[3], [5]], "bias": [1]},
  )
  v, logits = h.apply(p, {"value": jnp.asarray([[1.0, -2.0]], jnp.float32)})
  assert logits is None
  # relu([1,-2]) = [1,0]; 3 + 1
  np.testing.assert_array_equal(np.asarray(v), [4.0])


def test_value_head_tanh_bounds():
  """Value head tanh bounds."""
  h = heads.ValueHead(kind="scalar", tanh=True)
  p = _p(out={"kernel": [[100.0]], "bias": [0.0]})
  v, _ = h.apply(p, {"value": jnp.asarray([[1.0], [-1.0], [0.0]], jnp.float32)})
  np.testing.assert_array_equal(np.asarray(v), [1.0, -1.0, 0.0])


@pytest.mark.parametrize("k", [0, 3, 9])
def test_value_head_hl_gauss_known_answer(k):
  """A bias spike on bin k makes the softmax one-hot: value = center k."""
  h = heads.ValueHead(kind="hl_gauss", num_bins=10, v_min=-5.0, v_max=5.0)
  bias = np.zeros(10, np.float32)
  bias[k] = 1e4
  p = _p(out={"kernel": np.zeros((3, 10), np.float32), "bias": bias})
  v, logits = h.apply(p, {"value": jnp.ones((2, 3), jnp.float32)})
  assert logits.shape == (2, 10) and logits.dtype == jnp.float32
  np.testing.assert_array_equal(np.asarray(v), [k - 4.5, k - 4.5])


def test_value_head_rejects_tanh_with_hl_gauss_and_unknown_kind():
  """Value head rejects tanh with hl gauss and unknown kind."""
  x = {"value": jnp.ones((1, 2), jnp.float32)}
  with pytest.raises(ValueError, match="tanh"):
    heads.ValueHead(kind="hl_gauss", tanh=True).init(jax.random.key(0), x)
  with pytest.raises(ValueError, match="kind"):
    heads.ValueHead(kind="quantile").init(jax.random.key(0), x)


def test_aux_head_known_answer_and_card_tokens_shape():
  """Aux head known answer and card tokens shape."""
  h = heads.AuxHead(features=1, source="value")
  p = _p(dense={"kernel": [[2.0], [0.0], [1.0]], "bias": [-1.0]})
  y = h.apply(p, {"value": jnp.asarray([[1.0, 5.0, 3.0]], jnp.float32)})
  np.testing.assert_array_equal(np.asarray(y), [4.0])
  hc = heads.AuxHead(features=1, source="cards")
  y = hc.apply(p, {"cards": jnp.ones((2, 52, 3), jnp.float32)})
  assert y.shape == (2, 52)
  np.testing.assert_array_equal(np.asarray(y), np.full((2, 52), 2.0))


def test_hl_gauss_support_and_soft_labels():
  """Hl gauss support and soft labels."""
  s = hl_gauss.make_support(-5.0, 5.0, 10, 0.75)
  # f32 linspace arithmetic: centers within 1 ulp of k - 4.5.
  np.testing.assert_allclose(
      np.asarray(s.centers), np.arange(10) - 4.5, rtol=0, atol=6e-8
  )
  assert s.sigma == 0.75
  probs = hl_gauss.target_to_probs(
      jnp.asarray([0.0, 5.0, 99.0], jnp.float32), s
  )
  np.testing.assert_allclose(np.asarray(probs.sum(-1)), 1.0, atol=1e-6)
  # symmetric
  np.testing.assert_allclose(
      np.asarray(probs[0]), np.asarray(probs[0])[::-1], atol=1e-7
  )
  assert abs(float(hl_gauss.logits_to_scalar(jnp.log(probs[0]), s))) < 1e-6
  # clipped to v_max
  np.testing.assert_array_equal(np.asarray(probs[1]), np.asarray(probs[2]))
  ce = hl_gauss.cross_entropy(jnp.log(probs), jnp.asarray([0.0, 5.0, 99.0]), s)
  assert ce.shape == (3,) and ce.dtype == jnp.float32
  for bad in [(1.0, 1.0, 10), (-1.0, 1.0, 1), (-1.0, 1.0, 10, 0.0)]:
    with pytest.raises(ValueError):
      hl_gauss.make_support(*bad)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
