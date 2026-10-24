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

"""Heads: read a named torso feature and emit logits / value / aux.

  DensePolicyHead    Dense(num_actions) on feats[source]
  ValueHead          scalar (optional tanh) or HL-Gauss; optional hidden layer
  AuxHead            Dense(features) on feats[source] (a scalar or per-token
                     prediction)

Heads return their native compute dtype; the composer casts every network
output to float32.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from open_spiel.python.scale.networks import hl_gauss
from open_spiel.python.scale.networks import layers

F32 = jnp.float32


class DensePolicyHead(nn.Module):
  """Dense(num_actions) on feats[source].

  Attributes:
    num_actions: the number of logits.
    dtype: the compute dtype.
    init_scheme: "lecun" or "orthogonal".
    source: the torso feature it reads.
  """

  num_actions: int
  dtype: str = "float32"
  init_scheme: str = "lecun"
  source: str = "policy"

  @nn.compact
  def __call__(self, feats):
    """The logits.

    Args:
      feats: the torso's named features.

    Returns:
      [N, num_actions] logits in the compute dtype.
    """
    return layers.dense(
        self.num_actions,
        layers.dtype_of(self.dtype),
        "dense",
        **layers.init_kwargs(self.init_scheme, "actor_head"),
    )(feats[self.source])


class ValueHead(nn.Module):
  """A scalar or HL-Gauss value head.

  kind "scalar": Dense(1) squeezed, optional tanh. kind "hl_gauss":
  Dense(num_bins) bin logits decoded to the expectation (hl_gauss.py).

  hidden > 0 adds Dense(hidden) + hidden_activation first (`hidden`, compute
  `dtype`, input cast to `dtype`); the output layer `out` computes in
  out_dtype (default: dtype). Returns (value [N], bin logits [N, bins] or
  None).

  Attributes:
    kind: "scalar" or "hl_gauss".
    hidden: the hidden Dense width; 0 for none.
    hidden_activation: the hidden layer's activation name.
    tanh: squash a scalar value with tanh.
    num_bins: the HL-Gauss bin count.
    v_min: the HL-Gauss support's lower end.
    v_max: the HL-Gauss support's upper end.
    sigma_ratio: the HL-Gauss sigma over the bin width.
    dtype: the compute dtype name.
    out_dtype: the output layer's compute dtype name; None for `dtype`.
    init_scheme: the initialiser scheme (layers.INIT_SCHEMES).
    source: the feature key read ("policy" or "value").
  """

  kind: str = "scalar"
  hidden: int = 0
  hidden_activation: str = "relu"
  tanh: bool = False
  num_bins: int = 51
  v_min: float = -1.0
  v_max: float = 1.0
  sigma_ratio: float = 0.75
  dtype: str = "float32"
  out_dtype: str | None = None
  init_scheme: str = "lecun"
  source: str = "value"

  def support(self) -> hl_gauss.Support:
    """The HL-Gauss support of the head's fields.

    Returns:
      The hl_gauss.Support.
    """
    return hl_gauss.make_support(
        self.v_min, self.v_max, self.num_bins, self.sigma_ratio
    )

  @nn.compact
  def __call__(self, feats):
    """The value and, for HL-Gauss, the bin logits.

    Args:
      feats: the torso's named features.

    Returns:
      (value [N], bin logits [N, bins] or None).

    Raises:
      ValueError: an unknown kind, or tanh with hl_gauss.
    """
    if self.kind not in ("scalar", "hl_gauss"):
      raise ValueError(
          f"ValueHead kind must be 'scalar' or 'hl_gauss', got {self.kind!r}"
      )
    if self.kind == "hl_gauss" and self.tanh:
      raise ValueError(
          "hl_gauss is bounded by its support; tanh=True is a config error"
      )
    dt = layers.dtype_of(self.dtype)
    out_dt = layers.dtype_of(self.out_dtype or self.dtype)
    x = feats[self.source]
    if self.hidden > 0:
      x = layers.dense(
          self.hidden,
          dt,
          "hidden",
          **layers.init_kwargs(self.init_scheme, "trunk"),
      )(x.astype(dt))
      x = layers.activation(self.hidden_activation, x)
    out_dim = self.num_bins if self.kind == "hl_gauss" else 1
    raw = layers.dense(
        out_dim,
        out_dt,
        "out",
        **layers.init_kwargs(self.init_scheme, "critic_head"),
    )(x)
    if self.kind == "hl_gauss":
      return hl_gauss.logits_to_scalar(raw, self.support()), raw
    v = jnp.squeeze(raw, axis=-1)
    return (nn.tanh(v) if self.tanh else v), None


class AuxHead(nn.Module):
  """Dense(features) on feats[source].

  squeeze (the default) drops the trailing size-1 axis (a scalar target:
  [N, d] -> [N]; one prediction per token: [N, K, d] -> [N, K]) and needs
  features == 1; a head with more features passes squeeze=False.

  Attributes:
    features: the output width.
    source: the feature key read ("policy" or "value").
    dtype: the compute dtype name.
    init_scheme: the initialiser scheme (layers.INIT_SCHEMES).
    squeeze: drop the trailing size-1 axis.
  """

  features: int = 1
  source: str = "value"
  dtype: str = "float32"
  init_scheme: str = "lecun"
  squeeze: bool = True

  @nn.compact
  def __call__(self, feats):
    """The aux prediction.

    Args:
      feats: the torso's named features.

    Returns:
      The prediction, squeezed when `squeeze`.

    Raises:
      ValueError: squeeze with features != 1.
    """
    y = layers.dense(
        self.features,
        layers.dtype_of(self.dtype),
        "dense",
        **layers.init_kwargs(self.init_scheme, "critic_head"),
    )(feats[self.source])
    if self.squeeze:
      if self.features != 1:
        raise ValueError(
            f"AuxHead: squeeze needs features == 1, got {self.features}; "
            "pass squeeze=False"
        )
      y = y[..., 0]
    return y
