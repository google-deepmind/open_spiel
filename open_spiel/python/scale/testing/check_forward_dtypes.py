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

"""check_forward_dtypes: a network forward keeps its declared dtypes under x64.

With jax_enable_x64 on, as the package runs, anything built without an
explicit dtype is 64-bit (jnp.ones(...), jnp.arange(...), jax.nn.one_hot,
python-float arrays), and one such operand silently promotes a whole forward
to f64. Every network states its dtypes (f32 params, bf16 compute where
configured, f32 outputs); this checker verifies that for any
(apply_fn, params, inputs):
  - every floating param leaf is `param_dtype`;
  - every floating output leaf is `out_dtype`;
  - no float64 value appears anywhere in the traced forward: its inputs,
    constants and equations, nested jaxprs included (check_f64.f64_sites);
  - with `compute_dtype`, every dot_general/conv runs on operands of that
    dtype (and at least one exists).

  check_forward_dtypes.check_forward_dtypes(module.apply, params, (x,),
                                            compute_dtype=jnp.bfloat16)
"""

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp

from open_spiel.python.scale.testing import check_f64
from open_spiel.python.scale.testing.errors import require

_MATMUL = ("dot_general", "conv_general_dilated")


def _float_leaves(tree, label):
  for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
    dt = leaf.dtype if hasattr(leaf, "dtype") else jnp.asarray(leaf).dtype
    if jnp.issubdtype(dt, jnp.floating):
      yield f"{label}{jax.tree_util.keystr(path)}", dt


def _aval_dtype(v):
  """The dtype of a jaxpr variable, if any.

  Args:
    v: A jaxpr variable.

  Returns:
    The dtype, or None.
  """
  aval = getattr(v, "aval", None)
  return getattr(aval, "dtype", None)


def check_forward_dtypes(
    apply_fn,
    params: Any,
    inputs: Sequence[Any],
    *,
    param_dtype: Any = jnp.float32,
    out_dtype: Any = jnp.float32,
    compute_dtype: Optional[Any] = None,
) -> None:
  """Raises ContractError on the first declared-dtype violation.

  Args:
    apply_fn: The forward, apply_fn(params, *inputs).
    params: The params.
    inputs: The forward's inputs.
    param_dtype: The declared dtype of every floating param leaf.
    out_dtype: The declared dtype of every floating output leaf.
    compute_dtype: The declared dtype of every dot_general/conv operand, or
      None to skip that check.
  """
  require(
      jax.config.read("jax_enable_x64"),
      "check_forward_dtypes must run with jax_enable_x64 on (the package "
      "setting)",
  )
  param_dtype, out_dtype = jnp.dtype(param_dtype), jnp.dtype(out_dtype)
  for name, dt in _float_leaves(params, "params"):
    require(dt == param_dtype, f"param {name} is {dt}, declared {param_dtype}")

  closed = jax.make_jaxpr(apply_fn)(params, *inputs)
  out = jax.eval_shape(apply_fn, params, *inputs)
  for name, dt in _float_leaves(out, "out"):
    require(dt == out_dtype, f"output {name} is {dt}, declared {out_dtype}")

  sites = check_f64.f64_sites(closed)
  if sites:
    require(
        False,
        f"{len(sites)} float64 value(s) in the forward, first `{sites[0]}`: an "
        "operand was built without an explicit dtype",
    )
  matmuls = 0
  for eqn in check_f64.equations(closed.jaxpr):
    if compute_dtype is not None and eqn.primitive.name in _MATMUL:
      matmuls += 1
      want = jnp.dtype(compute_dtype)
      for v in eqn.invars:
        dt = _aval_dtype(v)
        require(
            dt == want,
            f"{eqn.primitive.name} operand is {dt}, declared compute {want}",
        )
  if compute_dtype is not None:
    require(
        matmuls > 0,
        f"no matmul found to check compute dtype {jnp.dtype(compute_dtype)}",
    )
