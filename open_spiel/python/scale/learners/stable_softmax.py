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

"""Log-softmax, softmax and cross-entropy without two numerics bugs.

From J. T. Barron, D. Gnanapragasam and M. Tschannen, "A Fix for Softmax
and Cross-Entropy", arXiv:2610.31642 (2026). Their two bugs:

  * softmax collapse: once the top logit leads the rest by a margin past
    about 16.6 in fp32 (5.5 in bf16), the standard log-softmax rounds the
    top logit's gradient to zero, so the gradient no longer sums to zero
    over the classes;
  * softmax cancellation: soft-target cross-entropy computes q - p from two
    separately rounded probabilities, so near z_S = z_T its gradient is
    noise, and not zero at equality.

`log_softmax`, `softmax`, `cross_entropy` and `soft_target_cross_entropy`
are the paper's code (Sections 2 and 3), statement for statement; only the
trailing comments moved above their lines for the 80-column limit, and the
docstrings are ours. `reverse_kl` is NOT in the paper: it is KL(q || p),
the direction the MMD magnet and behaviour terms take, built from the
paper's log-space ratio (its eq. 9) and its stop-gradient construction.

Every function reduces over the last axis. The learners' losses use these
only when a loss config sets `stable_softmax` (learners.losses); off, the
losses are the standard jax.nn ones, unchanged.
"""

import jax
import jax.numpy as jnp

# pylint: disable=invalid-name
# The paper's names (z_S, z_T: student and teacher logits) are kept.

#: reverse_kl's switch from log1p(sum p expm1(d)) to logsumexp(log p + d):
#: expm1 of it is ~1e13, far below the fp32 and bf16 overflow at ~88.7.
_LOG1P_SAFE = 30.0


def log_softmax(z):
  """Log-softmax over the last axis, exact 0 at the argmax's shift.

  Args:
    z: [..., K] logits.

  Returns:
    [..., K] log-probabilities whose gradient sums to 0 over the classes.
  """
  i_max = jnp.argmax(z, axis=-1)
  # Max logit.
  z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)
  # Max mask.
  is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)
  # Exact 0 at argmax.
  shifted = jnp.where(is_max, 0.0, z - z_max)
  # Exclude argmax from sum.
  others = jnp.where(is_max, -jnp.inf, shifted)
  # Logit margin.
  m = -jax.nn.logsumexp(others, axis=-1, keepdims=True)
  # log_softmax(z).
  return shifted - jax.nn.softplus(-m)


softmax = lambda z: jnp.exp(log_softmax(z))  # pylint: disable=unnecessary-lambda-assignment


def cross_entropy(z, t):
  """Hard-target cross-entropy -log_softmax(z)[t].

  Args:
    z: [..., K] logits.
    t: int [...] target classes.

  Returns:
    [...] the loss; its gradient sums to 0 over the classes.
  """
  # Target logit.
  z_t = jnp.take_along_axis(z, t[..., None], axis=-1)
  # Target mask.
  is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)
  # Exclude target from sum.
  others = jnp.where(is_t, -jnp.inf, z - z_t)
  # Logit margin.
  m = -jax.nn.logsumexp(others, axis=-1)
  # -log_softmax(z)_t.
  return jax.nn.softplus(-m)


def soft_target_cross_entropy(z_S, z_T):
  """Soft-target cross-entropy -sum softmax(z_T) log_softmax(z_S).

  The gradient (to z_S only) is q - p computed in log space: accurate as
  z_S -> z_T and exactly 0 at z_S == z_T.

  Args:
    z_S: [..., K] student logits.
    z_T: [..., K] teacher logits. A -inf entry gives NaN; give masked
      actions the same finite fill in z_S and z_T (as reverse_kl).

  Returns:
    [...] the loss.
  """
  log_q = log_softmax(z_S)
  q, p = jnp.exp(log_q), softmax(z_T)
  # Logit difference.
  delta = z_S - z_T
  # Center under p.
  d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)
  # log(q / p).
  s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))
  # Difference q - p.
  u = p * jnp.expm1(s)
  # Ensure sum(g) = 0.
  g = u - q * jnp.sum(u, axis=-1, keepdims=True)
  sg = jax.lax.stop_gradient
  # Forward loss.
  loss = -jnp.sum(p * sg(log_q), axis=-1)
  # Custom gradient g.
  return loss + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)


def reverse_kl(z_S, z_T):
  """KL(softmax(z_S) || softmax(z_T)), gradient to z_S only. Not the paper's.

  s = log(q / p) is the paper's eq. 9 (soft_target_cross_entropy's `s`):
  exact 0 at z_S == z_T. The value is sum q s and the gradient
  q (s - sum q s), recentred so it sums to 0 as the paper's `g` is; both
  are exactly 0 at equality. One addition: where some centred difference d
  exceeds _LOG1P_SAFE, log(1 + sum p expm1(d)) is taken as
  logsumexp(log p + d) instead, since expm1(d) would overflow (0 * inf when
  p underflows); that is far from equality, where log1p's accuracy is moot.
  Masked actions: give both inputs the same large negative fill (the loss's
  NEG); those entries then add nothing.

  Args:
    z_S: [..., K] the policy's logits.
    z_T: [..., K] the reference's logits (stop-gradient by construction).

  Returns:
    [...] the KL.
  """
  log_q = log_softmax(z_S)
  log_p = log_softmax(z_T)
  q, p = jnp.exp(log_q), jnp.exp(log_p)
  delta = z_S - z_T
  d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)
  near = jnp.log1p(
      jnp.sum(
          p * jnp.expm1(jnp.minimum(d, _LOG1P_SAFE)), axis=-1, keepdims=True
      )
  )
  far = jax.nn.logsumexp(log_p + d, axis=-1, keepdims=True)
  s = d - jnp.where(jnp.max(d, axis=-1, keepdims=True) > _LOG1P_SAFE, far, near)
  kl = jnp.sum(q * s, axis=-1, keepdims=True)
  g = q * (s - kl)
  g = g - q * jnp.sum(g, axis=-1, keepdims=True)
  sg = jax.lax.stop_gradient
  return sg(kl[..., 0]) + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)
