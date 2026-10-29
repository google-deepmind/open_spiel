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

"""Training diagnostics computed on device beside an update.

These functions only read params, gradients and batches, and no update
reads their results, so they cannot change training.

  module_groups(tree)            {module: subtree}, the top level of a params
                                 tree
  norms(tree, name)              {name: global L2 norm, name/<module>: per
                                 module}
  episode_sums(reward, done)     completed-hand sums of a [B, T] batch
  episode_summary(sums)          host: episode_return_mean / _std,
                                 episode_length_mean,
                                 episode_return_mean/seat<p>
  variance_parts(pred, target, mask) / explained_variance(parts)
                                 1 - Var(target - pred) / Var(target) over
                                 masked rows

Everything returned on device is f32 (counts int64), so a learner update stays
free of f64. The host summaries leave a statistic out when
it is undefined (no completed hand, a target with no variance) instead of
writing a NaN, which the metrics writer refuses.
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

F32 = jnp.float32
I64 = jnp.int64


def module_groups(tree: Any) -> dict[str, Any]:
  """{module name: subtree}: the top-level entries of a params-shaped tree.

  Looks through flax's lone "params" collection ({"params": {"torso": ...,
  "policy_head": ..., "value_head": ...}} -> torso, policy_head, value_head).

  Args:
    tree: a params-shaped tree.

  Returns:
    The modules; {} for a tree that is not a mapping or has a single module (the
      global norm says it all).
  """
  t = tree
  while isinstance(t, Mapping) and len(t) == 1 and "params" in t:
    t = t["params"]
  if not isinstance(t, Mapping) or len(t) < 2:
    return {}
  return {str(k): v for k, v in t.items()}


def tree_norm(tree: Any) -> jax.Array:
  """f32 global L2 norm of a tree of f32 leaves.

  Args:
    tree: a tree of f32 leaves.

  Returns:
    The f32 norm.
  """
  return optax.tree.norm(tree).astype(F32)


def norms(tree: Any, name: str) -> dict[str, jax.Array]:
  """{name: global norm, name/<module>: each module's norm} (f32).

  Args:
    tree: a params-shaped tree.
    name: the metric name.

  Returns:
    The norms dict.
  """
  out = {name: tree_norm(tree)}
  for mod, sub in module_groups(tree).items():
    out[f"{name}/{mod}"] = tree_norm(sub)
  return out


def episode_sums(
    reward: jax.Array, done: jax.Array, seat: Any = None, num_players: int = 0
) -> dict[str, jax.Array]:
  """Completed hands of a [B, T] batch, per row (a perspective seat's row).

  A hand ends at a ply where done is set. Its return is the row's reward from
  the ply after the row's previous done (or the segment's first ply) through the
  done ply: exact for games that pay at the end of a hand (every env game: kuhn,
  leduc, gin, ...), and for a hand that began in an earlier segment only the
  rewards inside this one count. Its length (plies, both seats' and the
  observe-only terminal ply included) is counted only for hands that start
  inside the segment (a previous done in the row), so a hand cut by the segment
  start never shortens the mean.

  Args:
    reward: f32[B, T] rewards.
    done: bool[B, T] hand ends.
    seat: int[B, T] the row's perspective seat (optional).
    num_players: P, with seat.

  Returns:
    n (int64 hands), return_sum / return_sumsq (f32), n_len (int64 hands with a
      known length), length_sum (int64 plies); with seat and num_players, also
      n_seat (int64[P]) and return_sum_seat (f32[P]): the same per perspective
      seat (a self-play batch that emits every seat sums to the game's total
      payoff over seats, 0 in a zero-sum game, so the per-seat means are its
      information).
  """
  assert reward.shape == done.shape and reward.ndim == 2, (
      reward.shape,
      done.shape,
  )
  reward = reward.astype(F32)
  b, t = done.shape
  idx = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32)[None, :], (b, t))
  last = jnp.where(done, idx, jnp.int32(-1))
  prev = jnp.concatenate(
      [jnp.full((b, 1), -1, jnp.int32), jax.lax.cummax(last, axis=1)[:, :-1]],
      axis=1,
  )
  cum = jnp.cumsum(reward, axis=1)
  base = jnp.where(
      prev >= 0,
      jnp.take_along_axis(cum, jnp.maximum(prev, 0), axis=1),
      F32(0.0),
  )
  ret = jnp.where(done, cum - base, F32(0.0))
  known = done & (prev >= 0)
  out = dict(
      n=jnp.sum(done, dtype=I64),
      return_sum=jnp.sum(ret),
      return_sumsq=jnp.sum(ret * ret),
      n_len=jnp.sum(known, dtype=I64),
      length_sum=jnp.sum(jnp.where(known, idx - prev, 0), dtype=I64),
  )
  if seat is not None:
    assert int(num_players) >= 1, num_players
    seat = jnp.asarray(seat)
    assert seat.shape == done.shape, (seat.shape, done.shape)
    hit = (
        seat[None]
        == jnp.arange(int(num_players), dtype=seat.dtype)[:, None, None]
    )  # [P, B, T]
    out["n_seat"] = jnp.sum(hit & done[None], axis=(1, 2), dtype=I64)
    out["return_sum_seat"] = jnp.sum(
        jnp.where(hit, ret[None], F32(0.0)), axis=(1, 2)
    )
  return out


def every_seat_rows(seat: Any, num_players: int) -> bool:
  """Host: whether a [B, T] batch's rows are every seat of each game.

  Rows b * P + m hold seat m (a self-play batch emitting all seats), where the
  mean over rows is the game's total payoff (0 when zero-sum), not a learning
  signal.

  Args:
    seat: int[B, T] perspective seats (host array).
    num_players: P.

  Returns:
    True for an every-seat batch.
  """
  seat = np.asarray(seat)
  b = seat.shape[0]
  p = int(num_players)
  if p < 2 or b % p:
    return False
  return bool(
      (seat.reshape(b // p, p, -1) == np.arange(p)[None, :, None]).all()
  )


ROW_MEAN_KEYS = (
    "episode_return_mean",
    "reward_per_row",
)  # left out on every_seat_rows batches


def episode_summary(sums: Mapping[str, Any]) -> dict[str, float]:
  """Host: episode return mean / std and length mean of completed hands.

  episode_return_mean / episode_return_std over the completed hands (left out
  without one; the std needs two) and episode_length_mean (left out without a
  hand of known length).

  Args:
    sums: episode_sums() output (host values).

  Returns:
    The metrics dict.
  """
  out: dict[str, float] = {}
  n = int(np.asarray(sums["n"]))
  if n >= 1:
    s = float(np.asarray(sums["return_sum"], np.float64))
    mean = s / n
    out["episode_return_mean"] = mean
    if n >= 2:
      ss = float(np.asarray(sums["return_sumsq"], np.float64))
      out["episode_return_std"] = float(
          np.sqrt(max(ss / n - mean * mean, 0.0) * n / (n - 1))
      )
  n_len = int(np.asarray(sums["n_len"]))
  if n_len >= 1:
    out["episode_length_mean"] = float(np.asarray(sums["length_sum"])) / n_len
  if "n_seat" in sums:
    n_seat = np.asarray(sums["n_seat"])
    r_seat = np.asarray(sums["return_sum_seat"], np.float64)
    for p in range(n_seat.shape[0]):
      if int(n_seat[p]) >= 1:
        out[f"episode_return_mean/seat{p}"] = float(r_seat[p]) / int(n_seat[p])
  return out


def variance_parts(
    pred: jax.Array, target: jax.Array, mask: jax.Array
) -> dict[str, jax.Array]:
  """f32 Var(target) and Var(target - pred) over the masked rows.

  0 rows: 0, 0.

  Args:
    pred: f32 predictions.
    target: f32 targets.
    mask: bool rows.

  Returns:
    {var_target, var_resid}.
  """
  m = mask.astype(F32)
  n = jnp.maximum(jnp.sum(m), F32(1.0))
  target = target.astype(F32)
  resid = target - pred.astype(F32)

  def _var(x):
    """The masked variance of `x`.

    Args:
      x: The values.

    Returns:
      The variance.
    """
    mu = jnp.sum(x * m) / n
    return jnp.sum(jnp.square(x - mu) * m) / n

  return dict(var_target=_var(target), var_resid=_var(resid))


def explained_variance(parts: Mapping[str, Any]) -> dict[str, float]:
  """Host: {"value_explained_var": 1 - Var(resid) / Var(target)}.

  {} when the target has no variance (undefined).

  Args:
    parts: variance_parts() output (host values).

  Returns:
    The metric dict, or {}.
  """
  vt = float(np.asarray(parts["var_target"], np.float64))
  # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
  # pylint: disable-next=unnecessary-negation
  if not vt > 0.0:
    return {}
  return {
      "value_explained_var": (
          1.0 - float(np.asarray(parts["var_resid"], np.float64)) / vt
      )
  }
