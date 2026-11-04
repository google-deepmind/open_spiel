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

"""Shared pieces of every network: dtypes, activations, per-role inits.

Dtype policy: params are float32 (every layer passes param_dtype=float32
explicitly); each layer's compute dtype is a config string ("float32" |
"bfloat16", plus "float64" where a component declares it); outputs that leave
a network (logits, value, aux) are cast to float32 by the composer. Nothing
relies on default dtypes, so jax_enable_x64 changes no dtype silently.

  dense = layers.dense(256, layers.dtype_of("bfloat16"), "trunk_0",
                       **layers.init_kwargs("orthogonal", "trunk"))
"""

from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

PARAM_DTYPE = jnp.float32

DTYPES = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float64": jnp.float64,
}


def dtype_of(name: str) -> Any:
  """Config string -> jnp dtype.

  Args:
    name: "float32", "bfloat16" or "float64".

  Returns:
    The jnp dtype.

  Raises:
    ValueError: any other name.
  """
  if name not in DTYPES:
    raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {name!r}")
  if name == "float64":
    require_x64("compute dtype float64")
  return DTYPES[name]


def require_x64(what: str) -> None:
  """Checks jax_enable_x64: float64 arithmetic is f32 without it.

  Args:
    what: what needs float64, for the message.

  Raises:
    RuntimeError: jax_enable_x64 is off.
  """
  if not jax.config.jax_enable_x64:
    raise RuntimeError(
        f"{what} needs jax_enable_x64=True; "
        "set it at the entry point or JAX_ENABLE_X64=1"
    )


ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "none": lambda x: x,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "swish": nn.swish,
    "gelu": nn.gelu,  # approximate=True (flax default)
}


def activation(name: str, x: jax.Array) -> jax.Array:
  """Applies the named activation.

  Args:
    name: a layers.ACTIVATIONS name.
    x: the input.

  Returns:
    The activation of `x`.

  Raises:
    ValueError: an unknown name.
  """
  if name not in ACTIVATIONS:
    raise ValueError(
        f"activation must be one of {sorted(ACTIVATIONS)}, got {name!r}"
    )
  return ACTIVATIONS[name](x)


# Orthogonal-init scales by layer role: sqrt(2) for ReLU trunks, 0.01 for the
# policy output (near-uniform initial policy), 1.0 for the value output.
_ORTHOGONAL_SCALES = {
    "trunk": float(np.sqrt(2)),
    "actor_head": 0.01,
    "critic_head": 1.0,
}
INIT_SCHEMES = ("lecun", "orthogonal")


def init_kwargs(scheme: str, role: str) -> dict[str, Any]:
  """Kernel and bias initialisers for a Dense of `role` under `scheme`.

  {} is the flax default: lecun_normal kernel, zero bias.

  Args:
    scheme: "lecun" or "orthogonal".
    role: the layer's role (its orthogonal scale).

  Returns:
    nn.Dense keyword arguments.

  Raises:
    ValueError: an unknown scheme or role.
  """
  if scheme not in INIT_SCHEMES:
    raise ValueError(
        f"init_scheme must be one of {INIT_SCHEMES}, got {scheme!r}"
    )
  if role not in _ORTHOGONAL_SCALES:
    raise ValueError(f"unknown init role {role!r}")
  if scheme == "lecun":
    return {}
  return dict(
      kernel_init=nn.initializers.orthogonal(_ORTHOGONAL_SCALES[role]),
      bias_init=nn.initializers.constant(0.0),
  )


def dense(features: int, dtype: Any, name: str, **kw) -> nn.Dense:
  """nn.Dense with an explicit compute dtype and float32 params.

  Args:
    features: the output width.
    dtype: the compute dtype.
    name: the layer name.
    **kw: further nn.Dense arguments (initialisers).

  Returns:
    The nn.Dense.
  """
  return nn.Dense(
      features, dtype=dtype, param_dtype=PARAM_DTYPE, name=name, **kw
  )
