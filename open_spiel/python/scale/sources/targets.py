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

"""Value and policy targets written onto TrajectoryBatches.

Imitation and expert-iteration learners regress a value head onto returns and
a policy head onto a teacher's (or a search's) distribution. These builders
write those targets onto a batch once, with a validity label, so a learner
masks its losses to the plies where a target exists.

    batch = targets.with_mc_value_targets(batch, gamma=1.0,
                                          lookahead=next_batch)
    batch = targets.with_q_softmax_targets(batch, temperature=0.5)

Monte Carlo returns use the one recursion, `core.advantages.gae`: with
value 0, bootstrap 0 and lambda 1 its advantages are each valid ply's
discounted return (rewards summed up to the seat's next decision, gamma once
per own decision, cut at the hand's end). A return is a target only when its
hand finished inside the data: labels[TARGET_VALUE_VALID] (bool[B, T]) marks
those plies. `lookahead`, the next segment of the same rows, completes most
hands cut at the segment's end.

target_value is f32[B, T, P] with the target in the perspective seat's column
(steps.seat) and 0 elsewhere.

One writer: every value builder sets labels[TARGET_VALUE_VALID] and refuses
a batch that already carries it (SecondValueWriterError), as the Arena does
when a teacher emits AUX_TARGET_VALUE; the check is on the pytree structure,
so it works under jit. with_search_value_targets replaces the outcome
targets (it computes the outcome itself) and is never chained after them.

Units: value_scale None leaves returns in game points; a number divides them
by it, the units of a value head trained on points / value_scale.

  with_search_value_targets(lam, value_scale)  target_value = (1 - lam) * MC
      outcome / value_scale + lam * aux[AUX_SEARCH_VALUE] (the operator's root
      value from the seat's view, in value-head units), 0 < lam <= 1;
  with_q_softmax_targets(temperature)  target_policy = softmax over the legal
      actions of aux[AUX_SEARCH_Q] / temperature on valid plies, zeros
      elsewhere.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import advantages
from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import operator_vocab

# labels key: bool[B, T], target_value is a target at this ply (core.types,
# which the Arena writes too).
TARGET_VALUE_VALID = types.TARGET_VALUE_VALID
F32 = jnp.float32


class SecondValueWriterError(ValueError):
  """A batch that already has value targets got a second writer."""


def refuse_second_writer(batch: types.TrajectoryBatch, who: str) -> None:
  """Refuses a batch that already has value targets.

  Args:
    batch: the TrajectoryBatch.
    who: the writer's name (for the message).

  Raises:
    SecondValueWriterError: batch already has labels[TARGET_VALUE_VALID]: a
      second writer would silently overwrite them.
  """
  if TARGET_VALUE_VALID in batch.labels:
    raise SecondValueWriterError(
        f"{who}: labels already carry {TARGET_VALUE_VALID!r}: target_value has "
        "a writer (the "
        "Arena's AUX_TARGET_VALUE or another builder); targets are written once"
    )


def _scaled(g, value_scale: Optional[float]):
  """The returns divided by value_scale, if given.

  Args:
    g: The returns.
    value_scale: The scale, or None.

  Returns:
    The scaled returns.

  Raises:
    ValueError: The scale is not finite and positive.
  """
  if value_scale is None:
    return g
  vs = float(value_scale)
  if not (vs > 0.0 and math.isfinite(vs)):
    raise ValueError(f"value_scale must be finite and > 0, got {value_scale}")
  return (jnp.asarray(g, F32) / F32(vs)).astype(F32)


def mc_returns(reward, valid, done, *, gamma: float):
  """Monte Carlo returns of a batch's rows.

  complete[t]: valid and a done at or after t inside the data (the first done at
  or after a decision ply is the end of that decision's hand).

  Args:
    reward: f32[B, T].
    valid: bool[B, T].
    done: bool[B, T].
    gamma: the discount.

  Returns:
    (G f32[B, T], complete bool[B, T]); both zero/False where not valid.

  Raises:
    TypeError: an input has the wrong dtype.
  """
  reward = jnp.asarray(reward)
  valid = jnp.asarray(valid)
  done = jnp.asarray(done)
  if (
      reward.dtype != jnp.float32
      or valid.dtype != jnp.bool_
      or done.dtype != jnp.bool_
  ):
    raise TypeError(
        f"want float32 reward and bool valid/done; got {reward.dtype}, "
        f"{valid.dtype}, {done.dtype}"
    )
  b = reward.shape[0]
  zeros = jnp.zeros_like(reward)
  g = advantages.gae(
      reward,
      zeros,
      valid,
      done,
      jnp.zeros((b,), jnp.float32),
      gamma=gamma,
      lam=1.0,
  ).advantages
  later_done = jnp.flip(
      jnp.cumsum(jnp.flip(done, axis=1).astype(jnp.int32), axis=1) > 0, axis=1
  )
  return g, valid & later_done


def value_targets(g, seat, num_players: int):
  """f32[B, T, P]: g in the seat's column, 0 elsewhere.

  Args:
    g: f32[B, T] returns.
    seat: int[B, T] the perspective seat.
    num_players: P.

  Returns:
    f32[B, T, P].
  """
  oh = jax.nn.one_hot(jnp.asarray(seat), num_players, dtype=jnp.float32)
  return oh * jnp.asarray(g, jnp.float32)[..., None]


def _mc(
    batch: types.TrajectoryBatch,
    gamma: float,
    lookahead: Optional[types.TrajectoryBatch],
):
  """(G f32[B, T], complete bool[B, T]) of batch with its lookahead.

  Args:
    batch: The batch.
    gamma: The discount.
    lookahead: The next segment, or None.

  Returns:
    (G f32[B, T], complete bool[B, T]).

  Raises:
    ValueError: The lookahead covers other rows.
  """
  t = batch.reward.shape[1]
  reward, valid, done = batch.reward, batch.valid, batch.done
  if lookahead is not None:
    if lookahead.reward.shape[0] != reward.shape[0]:
      raise ValueError(
          "lookahead must be the next segment of the same rows: "
          f"{lookahead.reward.shape} vs {reward.shape}"
      )
    reward = jnp.concatenate([reward, lookahead.reward], axis=1)
    valid = jnp.concatenate([valid, lookahead.valid], axis=1)
    done = jnp.concatenate([done, lookahead.done], axis=1)
  g, ok = mc_returns(reward, valid, done, gamma=gamma)
  return g[:, :t], ok[:, :t]


def with_mc_value_targets(
    batch: types.TrajectoryBatch,
    *,
    gamma: float,
    lookahead: Optional[types.TrajectoryBatch] = None,
    value_scale: Optional[float] = None,
) -> types.TrajectoryBatch:
  """Batch with target_value = MC returns and labels[TARGET_VALUE_VALID].

  Args:
    batch: the TrajectoryBatch.
    gamma: the discount.
    lookahead: the next segment of the same rows, completing hands in flight.
    value_scale: divides the returns when given.

  Returns:
    The batch with its value targets.
  """
  refuse_second_writer(batch, "with_mc_value_targets")
  g, ok = _mc(batch, gamma, lookahead)
  num_players = batch.target_value.shape[-1]
  tv = value_targets(
      jnp.where(ok, _scaled(g, value_scale), jnp.float32(0.0)),
      batch.steps.seat,
      num_players,
  )
  return batch._replace(
      target_value=tv, labels={**batch.labels, TARGET_VALUE_VALID: ok}
  )


def with_search_value_targets(
    batch: types.TrajectoryBatch,
    *,
    lam: float,
    value_scale: float,
    gamma: float = 1.0,
    lookahead: Optional[types.TrajectoryBatch] = None,
) -> types.TrajectoryBatch:
  """Batch with the lam-mixed MC outcome / search value targets.

  target_value = (1 - lam) * MC outcome / value_scale + lam *
  aux[AUX_SEARCH_VALUE] where the hand finished inside the data, and
  labels[TARGET_VALUE_VALID] = those plies; f32.

  Args:
    batch: the TrajectoryBatch.
    lam: the search value's weight, in (0, 1].
    value_scale: the operator's value_scale.
    gamma: the discount.
    lookahead: the next segment of the same rows.

  Returns:
    The batch with its value targets.

  Raises:
    ValueError: lam outside (0, 1] (lam = 0 is with_mc_value_targets), no
      value_scale, or a batch that already has value targets.
    KeyError: the batch has no AUX_SEARCH_VALUE.
    TypeError: AUX_SEARCH_VALUE is not f32[B, T].
  """
  refuse_second_writer(batch, "with_search_value_targets")
  lam = float(lam)
  if not 0.0 < lam <= 1.0:
    raise ValueError(
        f"with_search_value_targets: lam must be in (0, 1], got {lam} (lam = "
        "0: "
        "with_mc_value_targets, not this builder)"
    )
  if value_scale is None:
    raise ValueError(
        "with_search_value_targets needs the operator's value_scale (the units "
        "of "
        "AUX_SEARCH_VALUE)"
    )
  if operator_vocab.AUX_SEARCH_VALUE not in batch.aux:
    raise KeyError(
        f"with_search_value_targets: batch aux {sorted(batch.aux)} has no "
        f"{operator_vocab.AUX_SEARCH_VALUE!r}"
    )
  sv = jnp.asarray(batch.aux[operator_vocab.AUX_SEARCH_VALUE])
  if sv.dtype != F32 or sv.shape != batch.reward.shape:
    raise TypeError(
        f"aux {operator_vocab.AUX_SEARCH_VALUE!r}: {sv.dtype}{sv.shape}, want "
        "float32"
        f"{tuple(batch.reward.shape)}"
    )
  g, ok = _mc(batch, gamma, lookahead)
  mix = (F32(1.0 - lam) * _scaled(g, value_scale) + F32(lam) * sv).astype(F32)
  num_players = batch.target_value.shape[-1]
  tv = value_targets(
      jnp.where(ok, mix, F32(0.0)), batch.steps.seat, num_players
  )
  return batch._replace(
      target_value=tv, labels={**batch.labels, TARGET_VALUE_VALID: ok}
  )


def with_q_softmax_targets(
    batch: types.TrajectoryBatch, *, temperature: float
) -> types.TrajectoryBatch:
  """Batch with target_policy = softmax(aux[AUX_SEARCH_Q] / temperature).

  Batch with target_policy = softmax(aux[AUX_SEARCH_Q] / temperature) over
  the legal actions on valid plies, zeros elsewhere (module doc); f32.

  Args:
    batch: the TrajectoryBatch.
    temperature: the softmax temperature, > 0.

  Returns:
    The batch with its policy targets.

  Raises:
    ValueError: temperature is not positive.
    KeyError: the batch has no AUX_SEARCH_Q.
    TypeError: AUX_SEARCH_Q is not f32[B, T, A].
  """
  t = float(temperature)
  if not (t > 0.0 and t != float("inf")):
    raise ValueError(
        f"with_q_softmax_targets: temperature must be finite and > 0, got {t}"
    )
  if operator_vocab.AUX_SEARCH_Q not in batch.aux:
    raise KeyError(
        f"with_q_softmax_targets: batch aux {sorted(batch.aux)} has no "
        f"{operator_vocab.AUX_SEARCH_Q!r}"
    )
  q = jnp.asarray(batch.aux[operator_vocab.AUX_SEARCH_Q])
  legal = jnp.asarray(batch.steps.legal_mask)
  if q.dtype != F32 or q.shape != legal.shape:
    raise TypeError(
        f"aux {operator_vocab.AUX_SEARCH_Q!r}: {q.dtype}{q.shape}, want float32"
        f"{tuple(legal.shape)}"
    )
  z = jnp.where(legal, q / F32(t), F32(-jnp.inf))
  live = jnp.asarray(batch.valid) & jnp.any(legal, axis=-1)
  p = jax.nn.softmax(jnp.where(live[..., None], z, F32(0.0)), axis=-1).astype(
      F32
  )
  return batch._replace(target_policy=jnp.where(live[..., None], p, F32(0.0)))
