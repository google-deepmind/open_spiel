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

"""Advantage estimation over TrajectoryBatch rows: GAE(lambda) and TD(lambda).

One recursion serves every learner: policy-gradient advantages and targets,
and Monte Carlo value targets (lambda 1, value 0). Its conventions follow
core.types.TrajectoryBatch:

  out = advantages.gae(reward, value, valid, done, bootstrap_value,
                       gamma=0.99, lam=0.95)
  out.advantages, out.returns          # f32[B, T]

  * each row is one seat's perspective, so no zero-sum sign flipping is
    needed;
  * valid[t]: the perspective seat acted at ply t. Only valid plies get an
    advantage; every other ply (opponent plies, the observe-only terminal ply)
    gets 0 and its `value` is never read;
  * done[t]: ply t's action ended the hand (often an opponent ply);
  * reward[t]: the seat's reward for the transition out of ply t. Rewards are
    summed (undiscounted) from a valid ply up to the next valid ply of the same
    hand and credited to it; gamma applies once per own decision (a per-seat
    value chain);
  * a hand boundary (done) between a valid ply and the next valid ply cuts the
    bootstrap and the lambda trace;
  * bootstrap_value[b]: the seat's value at `bootstrap_step` (ply T), used for
    the last valid ply when no done follows it inside the segment. It is the
    value of the seat's InfoState at ply T, whoever is to move there; when ply
    T is a terminal or new-hand ply a done inside the segment already cuts it.
    Rewards at plies before the row's first valid ply (a segment that starts
    mid-hand) belong to the previous segment and are dropped.

  delta_k = R_k + gamma * (1 - cut_k) * V_{k+1} - V_k
  A_k     = delta_k + gamma * lam * (1 - cut_k) * A_{k+1}
  G_k     = A_k + V_k            (the TD(lam) / lambda-return target)

TD(lam_td) returns and GAE(lam_gae) advantages with different lambdas (e.g.
TD(0.8) / GAE(0.5)) are two calls.

Arithmetic is float32 (inputs are asserted f32: under jax_enable_x64 a stray
f64 input would silently promote the whole estimate).
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Advantages(NamedTuple):
  """GAE outputs over [B, T] perspective rows.

  Attributes:
    advantages: f32[B, T]; 0 where not valid.
    returns: f32[B, T]; advantages + value where valid, else 0.
  """

  advantages: jax.Array  # f32[B, T]; 0 where not valid
  returns: jax.Array  # f32[B, T]; advantages + value where valid, else 0


def _check_f32(name: str, x: Any) -> jax.Array:
  """Checks `x` is float32.

  Args:
    name: The array's name, for errors.
    x: The array.

  Returns:
    `x` as an array.
  """
  x = jnp.asarray(x)
  assert x.dtype == jnp.float32, f"{name}: want float32, got {x.dtype}"
  return x


def seat_value(value_pred: jax.Array, seat: jax.Array) -> jax.Array:
  """f32[B, T]: value_pred[b, t, seat[b, t]] (the per-seat value head).

  Args:
    value_pred: f32[B, T, num_players] the value head's predictions.
    seat: int[B, T] the perspective seat of each ply.

  Returns:
    f32[B, T] value_pred[b, t, seat[b, t]].
  """
  value_pred = _check_f32("value_pred", value_pred)
  seat = jnp.asarray(seat)
  assert jnp.issubdtype(seat.dtype, jnp.integer), seat.dtype
  assert value_pred.ndim == 3 and seat.shape == value_pred.shape[:2], (
      value_pred.shape,
      seat.shape,
  )
  return jnp.take_along_axis(
      value_pred, seat[..., None].astype(jnp.int32), axis=-1
  )[..., 0]


def gae(
    reward: jax.Array,
    value: jax.Array,
    valid: jax.Array,
    done: jax.Array,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    lam: float,
) -> Advantages:
  """GAE(lam) advantages and TD(lam) returns, [B, T], per perspective row.

  Args:
    reward: f32[B, T] the seat's reward for the transition out of each ply.
    value: f32[B, T] perspective-seat values (read only where valid).
    valid: bool[B, T] plies where the perspective seat acted.
    done: bool[B, T] plies whose action ended the hand.
    bootstrap_value: f32[B] the value after the last ply.
    gamma: discount, applied once per own decision, in [0, 1].
    lam: the GAE / TD lambda, in [0, 1].

  Returns:
    The Advantages (advantages and returns, f32[B, T]).
  """
  reward = _check_f32("reward", reward)
  value = _check_f32("value", value)
  bootstrap_value = _check_f32("bootstrap_value", bootstrap_value)
  valid = jnp.asarray(valid)
  done = jnp.asarray(done)
  assert valid.dtype == jnp.bool_ and done.dtype == jnp.bool_, (
      valid.dtype,
      done.dtype,
  )
  assert reward.ndim == 2, reward.shape
  assert value.shape == reward.shape == valid.shape == done.shape, (
      reward.shape,
      value.shape,
      valid.shape,
      done.shape,
  )
  assert bootstrap_value.shape == reward.shape[:1], (
      bootstrap_value.shape,
      reward.shape,
  )
  gamma = float(gamma)
  lam = float(lam)
  assert 0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0, (gamma, lam)
  return _gae_scan(
      reward, value, valid, done, bootstrap_value, gamma=gamma, lam=lam
  )


@functools.partial(jax.jit, static_argnames=("gamma", "lam"))
def _gae_scan(
    reward, value, valid, done, bootstrap_value, *, gamma: float, lam: float
) -> Advantages:
  """GAE's recursion, jitted once per (shapes, gamma, lam).

  Called eagerly (AZ and BC targets, sources) a lax.scan over a fresh closure
  would compile again on every call; under an outer jit this is inlined.

  Args:
    reward: f32[B, T] the seat's reward for the transition out of each ply.
    value: f32[B, T] perspective-seat values (read only where valid).
    valid: bool[B, T] plies where the perspective seat acted.
    done: bool[B, T] plies whose action ended the hand.
    bootstrap_value: f32[B] the value after the last ply.
    gamma: discount, applied once per own decision, in [0, 1].
    lam: the GAE / TD lambda, in [0, 1].

  Returns:
    The Advantages of the rows.
  """
  f32 = jnp.float32
  g = f32(gamma)
  gl = f32(gamma * lam)

  def step(carry, x):
    """One reverse scan step over ply t.

    Args:
      carry: (next value, next advantage, reward accumulator, cut).
      x: (reward, value, valid, done) at ply t.

    Returns:
      The new carry and ply t's advantage (0 where not valid).
    """
    next_v, next_a, racc, cut = carry
    r, v, ok, d = x
    # The transition out of this ply: a done drops everything later (another
    # hand) and cuts the bootstrap.
    racc = jnp.where(d, r, racc + r)
    cut = d | cut
    live = jnp.where(cut, f32(0.0), f32(1.0))
    delta = racc + g * live * next_v - v
    a = delta + gl * live * next_a
    zero = jnp.zeros_like(racc)
    new = (
        jnp.where(ok, v, next_v),
        jnp.where(ok, a, next_a),
        jnp.where(ok, zero, racc),
        jnp.where(ok, jnp.zeros_like(cut), cut),
    )
    return new, jnp.where(ok, a, zero)

  b = reward.shape[0]
  init = (
      bootstrap_value,
      jnp.zeros((b,), f32),
      jnp.zeros((b,), f32),
      jnp.zeros((b,), jnp.bool_),
  )
  xs = (reward.T, value.T, valid.T, done.T)
  _, adv_t = jax.lax.scan(step, init, xs, reverse=True)
  adv = adv_t.T
  assert adv.dtype == f32, adv.dtype
  return Advantages(
      advantages=adv, returns=jnp.where(valid, adv + value, f32(0.0))
  )


def td_lambda_returns(
    reward, value, valid, done, bootstrap_value, *, gamma: float, lam: float
) -> jax.Array:
  """f32[B, T] TD(lam) (lambda-return) value targets; 0 where not valid.

  Args:
    reward: f32[B, T] the seat's reward for the transition out of each ply.
    value: f32[B, T] perspective-seat values (read only where valid).
    valid: bool[B, T] plies where the perspective seat acted.
    done: bool[B, T] plies whose action ended the hand.
    bootstrap_value: f32[B] the value after the last ply.
    gamma: discount, applied once per own decision, in [0, 1].
    lam: the GAE / TD lambda, in [0, 1].

  Returns:
    f32[B, T] lambda-return targets; 0 where not valid.
  """
  return gae(
      reward, value, valid, done, bootstrap_value, gamma=gamma, lam=lam
  ).returns


def batch_gae(
    batch: Any,
    bootstrap_value: jax.Array,
    *,
    gamma: float,
    lam: float,
    value: jax.Array | None = None,
) -> Advantages:
  """GAE over a TrajectoryBatch.

  value: f32[B, T] perspective-seat values to use (e.g. recomputed by the
  learner); default: batch.value_pred[..., batch.steps.seat] (behaviour).

  Args:
    batch: the TrajectoryBatch.
    bootstrap_value: f32[B] the value after the last ply.
    gamma: discount, applied once per own decision, in [0, 1].
    lam: the GAE / TD lambda, in [0, 1].
    value: f32[B, T] perspective-seat values to use (e.g. recomputed by the
      learner); default: batch.value_pred[..., batch.steps.seat] (behaviour).

  Returns:
    The Advantages of the batch's rows.
  """
  if value is None:
    value = seat_value(batch.value_pred, batch.steps.seat)
  return gae(
      batch.reward,
      value,
      batch.valid,
      batch.done,
      bootstrap_value,
      gamma=gamma,
      lam=lam,
  )
