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

"""Torsos: encoded observations -> a dict of named feature arrays.

A torso computes features once; each head reads the feature it names (its
`source`), so one torso serves a shared or a separate actor and critic.

  MLPTorso        {"policy", "value"}   shared or separate dense trunks
  CNNTorso        {"policy", "value"}   IMPALA residual CNN + Dense
  RecurrentTorso  a GRU over an inner torso; the carry goes through
                  Policy.observe

Every layer states its compute dtype; params are float32 (layers.py).
"""

from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from open_spiel.python.scale.networks import layers

F32 = jnp.float32


class MLPTorso(nn.Module):
  """Dense x len(widths) with `activation`.

  layout "shared": one trunk (shared_{i}); policy and value read the same
  features. layout "separate": independent actor_{i} and critic_{i} trunks
  (no shared params). Input [N, ...] is flattened to [N, D] and cast to
  `dtype` first.

  Attributes:
    widths: the Dense widths.
    layout: "shared" or "separate".
    activation: the activation name.
    init_scheme: the initialiser scheme (layers.INIT_SCHEMES).
    dtype: the compute dtype name.
  """

  widths: tuple[int, ...] = (256, 256)
  layout: str = "shared"
  activation: str = "relu"
  init_scheme: str = "lecun"
  dtype: str = "float32"

  def _trunk(self, x, prefix):
    """The dense layers of the trunk named `prefix`.

    Args:
      x: The input.
      prefix: The layer-name prefix.

    Returns:
      The trunk's output.
    """
    dt = layers.dtype_of(self.dtype)
    for i, w in enumerate(self.widths):
      x = layers.dense(
          int(w),
          dt,
          f"{prefix}_{i}",
          **layers.init_kwargs(self.init_scheme, "trunk"),
      )(x)
      x = layers.activation(self.activation, x)
    return x

  @nn.compact
  def __call__(self, x):
    """The trunk features.

    Args:
      x: [N, ...] features, flattened per row.

    Returns:
      {"policy", "value"}: one trunk's output twice, or each trunk's.

    Raises:
      ValueError: an unknown layout or an input without a row axis.
    """
    if x.ndim < 2:
      raise ValueError(f"MLPTorso wants [N, ...] features, got {x.shape}")
    x = x.reshape(x.shape[0], -1).astype(layers.dtype_of(self.dtype))
    if self.layout == "shared":
      h = self._trunk(x, "shared")
      return {"policy": h, "value": h}
    if self.layout == "separate":
      return {
          "policy": self._trunk(x, "actor"),
          "value": self._trunk(x, "critic"),
      }
    raise ValueError(
        f"MLPTorso layout must be 'shared' or 'separate', got {self.layout!r}"
    )


class CNNTorso(nn.Module):
  """IMPALA residual CNN (Espeholt et al. 2018), flatten, Dense + relu.

  Per stack i (channels int(relative_stack_sizes[i] * channels)): [norm if
  i > 0], conv 3x3 (stack{i}_conv), max_pool 3x3 stride 2 SAME, then
  num_blocks residual blocks of (act, norm, conv, act, norm, conv) + skip
  (stack{i}_block{j}_conv{0,1}); a final act; then Dense(dense_hidden) + relu
  (`dense`). The input is cast to `dtype`; convs compute in `conv_dtype`.

  Attributes:
    channels: the base channel count.
    relative_stack_sizes: each stack's channels as a fraction of `channels`.
    num_blocks: residual blocks per stack.
    dense_hidden: the final Dense width.
    activation: the activation name.
    norm: "none" or a normalisation name.
    dtype: the input dtype name.
    conv_dtype: the convs' compute dtype name.
  """

  channels: int = 256
  relative_stack_sizes: tuple[float, ...] = (1 / 8, 1 / 4, 1 / 4)
  num_blocks: int = 2
  dense_hidden: int = 256
  activation: str = "relu"
  norm: str = "none"
  dtype: str = "float32"
  conv_dtype: str = "float32"

  def _norm(self, x, name):
    """The configured normalisation of `x`.

    Args:
      x: The input.
      name: The layer name.

    Returns:
      The normalised input.

    Raises:
      ValueError: The norm is unknown.
    """
    if self.norm == "none":
      return x
    if self.norm == "layer_norm":
      return nn.LayerNorm(
          dtype=layers.dtype_of(self.conv_dtype), param_dtype=F32, name=name
      )(x)
    raise ValueError(
        f"CNNTorso norm must be 'none' or 'layer_norm', got {self.norm!r}"
    )

  def _conv(self, ch, name):
    """A 3x3 SAME convolution.

    Args:
      ch: The output channels.
      name: The layer name.

    Returns:
      The Conv module.
    """
    return nn.Conv(
        features=ch,
        kernel_size=(3, 3),
        strides=1,
        padding="SAME",
        kernel_init=nn.initializers.xavier_uniform(),
        dtype=layers.dtype_of(self.conv_dtype),
        param_dtype=F32,
        name=name,
    )

  @nn.compact
  def __call__(self, x):
    """The IMPALA-CNN features.

    Args:
      x: [N, H, W, C] planes.

    Returns:
      {"policy": x, "value": x} after the dense layer.

    Raises:
      ValueError: the input is not [N, H, W, C].
    """
    if x.ndim != 4:
      raise ValueError(f"CNNTorso wants [N, H, W, C] features, got {x.shape}")
    x = x.astype(layers.dtype_of(self.dtype))
    for i, rel in enumerate(self.relative_stack_sizes):
      ch = int(rel * self.channels)
      if i > 0:
        x = self._norm(x, f"stack{i}_norm")
      x = self._conv(ch, f"stack{i}_conv")(x)
      x = nn.max_pool(x, window_shape=(3, 3), padding="SAME", strides=(2, 2))
      for j in range(self.num_blocks):
        skip = x
        x = layers.activation(self.activation, x)
        x = self._norm(x, f"stack{i}_block{j}_norm0")
        x = self._conv(ch, f"stack{i}_block{j}_conv0")(x)
        x = layers.activation(self.activation, x)
        x = self._norm(x, f"stack{i}_block{j}_norm1")
        x = self._conv(ch, f"stack{i}_block{j}_conv1")(x)
        x = x + skip
    x = layers.activation(self.activation, x)
    x = x.reshape(x.shape[0], -1)
    x = nn.relu(
        layers.dense(self.dense_hidden, layers.dtype_of(self.dtype), "dense")(x)
    )
    return {"policy": x, "value": x}


class RecurrentTorso(nn.Module):
  """A GRU over an inner torso.

  step(x, h, extra) -> h': z = inner(x)["value"] (a shared-layout inner
  torso), [concat extra, e.g. the last-action one-hot], act(proj(z)),
  h' = GRUCell(h, z). features(h) -> {"policy": h, "value": h}.

  The carry h is f32[N, hidden_size]; the policy resets it to zeros on
  new_hand / new_match before step (NetworkPolicy.observe).

  Attributes:
    inner: the inner (shared-layout) torso.
    hidden_size: the GRU's hidden size.
    activation: the activation name.
    dtype: the compute dtype name.
    proj: the input projection (built in setup).
    gru: the GRU cell (built in setup).
  """

  inner: nn.Module
  hidden_size: int = 128
  activation: str = "relu"
  dtype: str = "float32"

  def setup(self):
    """Builds the input projection and the GRU cell."""
    dt = layers.dtype_of(self.dtype)
    self.proj = nn.Dense(self.hidden_size, dtype=dt, param_dtype=F32)
    self.gru = nn.GRUCell(features=self.hidden_size, dtype=dt, param_dtype=F32)

  def step(self, x, h, extra: Optional[jax.Array] = None):
    """One GRU step on the inner torso's features.

    Args:
      x: [N, ...] encoded features.
      h: the carry f32[N, hidden_size].
      extra: extra inputs concatenated to the features, or None.

    Returns:
      The new carry f32[N, hidden_size].

    Raises:
      ValueError: the inner torso is not shared-layout.
    """
    f = self.inner(x)
    if f["policy"] is not f["value"]:
      raise ValueError("RecurrentTorso needs a shared-layout inner torso")
    z = f["value"]
    if extra is not None:
      z = jnp.concatenate([z, extra.astype(z.dtype)], axis=-1)
    z = layers.activation(self.activation, self.proj(z))
    new_h, _ = self.gru(h.astype(F32), z)
    return new_h.astype(F32)

  def features(self, h):
    """The features heads read from the carry.

    Args:
      h: the carry f32[N, hidden_size].

    Returns:
      {"policy": h, "value": h}.
    """
    return {"policy": h, "value": h}
