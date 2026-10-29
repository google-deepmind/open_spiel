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

"""HL-Gauss categorical value (Imani and White 2018; Farebrother et al. 2024).

Regressing a value with cross-entropy over a fixed support of bins, against
a Gaussian-smoothed target, trains more stably than mean squared error. This
module is the one implementation: ValueHead decodes with it and the value
losses train with it.

  support = hl_gauss.make_support(v_min=-1.0, v_max=1.0, num_bins=51)
  value = hl_gauss.logits_to_scalar(logits, support)        # f32[...]
  loss = hl_gauss.cross_entropy(logits, target, support)    # f32[...]

The support is `num_bins` equal bins over [v_min, v_max]; the head emits one
logit per bin and the scalar value is the softmax expectation over the bin
centers. A regression target y becomes a soft label by integrating
N(y, sigma^2), sigma = sigma_ratio * bin width, over each bin, renormalised by
the in-support mass (targets are clipped to the support first).

Bin edges are computed in float64 and rounded to float32 once, centers are
edge midpoints, and the softmax is float32, so a support decodes to the same
bits wherever it is built. make_support requires jax_enable_x64 (it raises
RuntimeError without it).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from open_spiel.python.scale.networks import layers


@dataclasses.dataclass(frozen=True)
class Support:
  """The bins of an HL-Gauss value head.

  Attributes:
    edges: f32[num_bins + 1], the bin edges.
    centers: f32[num_bins], the bin centers.
    sigma: the Gaussian's standard deviation.
    v_min: the lowest edge.
    v_max: the highest edge.
    num_bins: the number of bins.
  """

  edges: jax.Array  # f32[num_bins + 1]
  centers: jax.Array  # f32[num_bins]
  sigma: float
  v_min: float
  v_max: float
  num_bins: int


def make_support(
    v_min: float, v_max: float, num_bins: int, sigma_ratio: float = 0.75
) -> Support:
  """The HL-Gauss support of num_bins bins over [v_min, v_max].

  Args:
    v_min: the lowest value.
    v_max: the highest value.
    num_bins: the number of bins (>= 2).
    sigma_ratio: sigma as a multiple of the bin width.

  Returns:
    The Support.

  Raises:
    ValueError: num_bins < 2, v_max <= v_min or sigma_ratio <= 0.
  """
  if num_bins < 2:
    raise ValueError(f"hl_gauss num_bins must be >= 2, got {num_bins}")
  if not v_max > v_min:
    raise ValueError(f"hl_gauss needs v_max > v_min, got [{v_min}, {v_max}]")
  # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
  # pylint: disable-next=unnecessary-negation
  if not sigma_ratio > 0.0:
    raise ValueError(f"hl_gauss sigma_ratio must be > 0, got {sigma_ratio}")
  # The edges are computed in float64 (the package runs with x64 on) and
  # rounded to float32 once; float32 endpoints give different edges for some
  # supports.
  layers.require_x64("hl_gauss.make_support computes its bin edges in float64")
  edges = jnp.linspace(
      jnp.asarray(v_min, jnp.float64),
      jnp.asarray(v_max, jnp.float64),
      num_bins + 1,
      dtype=jnp.float32,
  )
  centers = 0.5 * (edges[:-1] + edges[1:])
  # sigma_ratio * bin width
  sigma = float(sigma_ratio * ((v_max - v_min) / num_bins))
  return Support(
      edges=edges,
      centers=centers,
      sigma=sigma,
      v_min=float(v_min),
      v_max=float(v_max),
      num_bins=int(num_bins),
  )


def logits_to_scalar(logits: jax.Array, support: Support) -> jax.Array:
  """The f32 expectation sum_i softmax(logits)_i * center_i.

  Args:
    logits: [..., num_bins] bin logits.
    support: the Support.

  Returns:
    f32[...].
  """
  probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
  return jnp.sum(probs * support.centers, axis=-1)


def _normal_cdf(x: jax.Array) -> jax.Array:
  """The standard normal CDF.

  Args:
    x: The values.

  Returns:
    The CDF of `x`.
  """
  return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(jnp.float32(2.0))))


def target_to_probs(target: jax.Array, support: Support) -> jax.Array:
  """The HL-Gauss soft labels of scalar targets.

  Args:
    target: f32[...] scalar targets.
    support: the Support.

  Returns:
    f32[..., num_bins] soft labels summing to 1.
  """
  # f32 bounds: python-float bounds would trace as f64 scalars.
  t = jnp.clip(
      target.astype(jnp.float32),
      jnp.float32(support.v_min),
      jnp.float32(support.v_max),
  )
  cdf = _normal_cdf((support.edges - t[..., None]) / jnp.float32(support.sigma))
  z = cdf[..., -1] - cdf[..., 0]
  return (cdf[..., 1:] - cdf[..., :-1]) / jnp.maximum(
      z[..., None], jnp.float32(1e-12)
  )


def cross_entropy(
    logits: jax.Array, target: jax.Array, support: Support
) -> jax.Array:
  """Per-example HL-Gauss cross-entropy (no reduction).

  Args:
    logits: [..., num_bins] bin logits.
    target: f32[...] scalar targets.
    support: the Support.

  Returns:
    f32[...].
  """
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  return -jnp.sum(target_to_probs(target, support) * log_probs, axis=-1)
