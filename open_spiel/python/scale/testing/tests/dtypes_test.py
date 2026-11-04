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

"""check_forward_dtypes: a flax forward under x64 keeps its declared dtypes.

The declared dtypes are f32 params, bf16 compute and f32 outputs; planted
controls cover each way x64 silently promotes.
"""

import sys
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.testing import check_forward_dtypes
from open_spiel.python.scale.testing.errors import ContractError


class MLP(nn.Module):
  """The declared form: f32 params, compute in `dtype`, f32 out."""

  dtype: Any = jnp.bfloat16
  param_dtype: Any = jnp.float32
  cast_out: bool = True

  @nn.compact
  def __call__(self, x):
    x = x.astype(self.dtype) if self.dtype is not None else x
    x = nn.Dense(8, dtype=self.dtype, param_dtype=self.param_dtype)(x)
    x = nn.relu(x)
    x = nn.Dense(2, dtype=self.dtype, param_dtype=self.param_dtype)(x)
    return x.astype(jnp.float32) if self.cast_out else x


def _check(module, x, compute_dtype=jnp.bfloat16):
  """Checks a module's forward dtypes.

  Args:
    module: The module.
    x: The input.
    compute_dtype: The compute dtype.

  Returns:
    The module's params.
  """
  params = module.init(jax.random.key(0), x)
  check_forward_dtypes.check_forward_dtypes(
      module.apply,
      params,
      (x,),
      param_dtype=jnp.float32,
      out_dtype=jnp.float32,
      compute_dtype=compute_dtype,
  )
  return params


def test_declared_mlp_passes_under_x64():
  """The declared MLP passes under x64."""
  assert jax.config.jax_enable_x64
  x = jnp.ones((4, 3), jnp.float32)
  params = _check(MLP(), x)
  out = MLP().apply(params, x)
  assert out.dtype == jnp.float32
  for leaf in jax.tree_util.tree_leaves(params):
    assert leaf.dtype == jnp.float32


def test_known_answer_forward_in_bf16():
  """W = 1, b = 0, x = [1, 2, 3]: hidden units are 6, outputs 8 * 6 = 48.

  The answer is exact in bf16.
  """
  x = jnp.array([[1.0, 2.0, 3.0]], jnp.float32)
  params = MLP().init(jax.random.key(0), x)
  params = jax.tree_util.tree_map(
      lambda p: jnp.ones_like(p) if p.ndim == 2 else jnp.zeros_like(p), params
  )
  out = MLP().apply(params, x)
  assert out.dtype == jnp.float32
  np.testing.assert_array_equal(np.asarray(out), [[48.0, 48.0]])


def test_f32_compute_passes_when_declared_f32():
  """f32 compute passes when f32 is declared."""
  _check(
      MLP(dtype=jnp.float32),
      jnp.ones((4, 3), jnp.float32),
      compute_dtype=jnp.float32,
  )


def test_planted_undeclared_dtype_with_default_input_is_caught():
  """dtype=None with a dtype-less input (f64 under x64) is caught.

  The forward silently runs in f64.
  """
  x = jnp.ones((4, 3))
  assert x.dtype == jnp.float64
  with pytest.raises(ContractError, match="float64"):
    _check(MLP(dtype=None), x, compute_dtype=None)


def test_planted_f64_params_are_caught():
  """f64 params are caught."""
  with pytest.raises(ContractError, match="param"):
    _check(MLP(param_dtype=jnp.float64), jnp.ones((4, 3), jnp.float32))


def test_planted_uncast_bf16_output_is_caught():
  """An uncast bf16 output is caught."""
  with pytest.raises(ContractError, match="output"):
    _check(MLP(cast_out=False), jnp.ones((4, 3), jnp.float32))


def test_planted_f32_compute_when_bf16_declared_is_caught():
  """f32 compute when bf16 is declared is caught."""
  with pytest.raises(ContractError, match="bfloat16"):
    _check(
        MLP(dtype=jnp.float32),
        jnp.ones((4, 3), jnp.float32),
        compute_dtype=jnp.bfloat16,
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
