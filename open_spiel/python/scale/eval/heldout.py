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

"""Held-out teacher agreement: top-1 and KL of a student on a fixed state set.

Behaviour cloning is measured on states the student never trained on. The
state set is driven by the teacher, generated once from its own seed
(independent of the training seed and of the student) and re-scored at
every evaluation, so successive scores are comparable:

  top1  the student's legal-masked argmax equals the teacher's (the first
        maximal index on both sides);
  kl    KL(teacher || student) over the legal actions;
  ce    the soft-target cross-entropy, kl + the teacher's entropy.

  held = heldout.collect(source, jax.random.key(7), num_states=4096)
  heldout.score(student, params, held)   # {top1, kl, ce, teacher_entropy,
  num_states}

`collect` draws batches from any TrajectorySource whose target_policy is
the teacher's distribution (a DaggerSource at beta 1, a ShardReader over
recorded teacher games) until it holds num_states valid states, and keeps
exactly the first num_states in row-major order in `mask`; batch i uses key
fold_in64(key, i). `score` runs the student's evaluate_seq over the stored
rows (a recurrent student rebuilds its carry from the stored init_carry)
and reduces over `mask` with learners.losses. Metrics are Python floats.
"""

import dataclasses
from typing import Any, Callable
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.learners import losses

F32 = jnp.float32


@dataclasses.dataclass(frozen=True, eq=False)
class HeldoutSet:
  """A fixed held-out state set.

  Attributes:
    batch: [B, T] rows holding the states.
    mask: bool[B, T]: exactly num_states True.
    num_states: the held-out states.
  """

  batch: types.TrajectoryBatch  # [B, T] rows holding the states
  mask: Any  # bool[B, T]: exactly num_states True
  num_states: int


def agreement(logits, legal, target, mask) -> dict[str, jax.Array]:
  """{top1, kl, ce, teacher_entropy} (f32 scalars) over mask rows.

  Args:
    logits: f32[..., A], the student's.
    legal: bool[..., A].
    target: f32[..., A], the teacher's distribution over legal actions.
    mask: bool[...], the rows scored.

  Returns:
    {top1, kl, ce, teacher_entropy}, f32 scalars.
  """
  logits = jnp.asarray(logits, F32)
  target = jnp.asarray(target, F32)
  legal = jnp.asarray(legal, jnp.bool_)
  mask = jnp.asarray(mask, jnp.bool_)
  ref = losses.target_argmax(target, legal)
  kl = losses.kl_from_target(target, logits, legal)
  # H(teacher) over its support (a scripted teacher's -inf logits make
  # t * log pi = 0 * -inf off its support, so ce is kl + H, not a direct sum).
  keep = legal & (target > 0)
  ent = -jnp.sum(
      jnp.where(
          keep, target * jnp.log(jnp.where(keep, target, F32(1.0))), F32(0.0)
      ),
      axis=-1,
  )
  return {
      "top1": losses.top1_agreement(logits, legal, ref, mask),
      "kl": losses.masked_mean(kl, mask),
      "ce": losses.masked_mean(kl + ent, mask),
      "teacher_entropy": losses.masked_mean(ent, mask),
  }


def _concat(batches):
  """Concatenates batches along the rows.

  Args:
    batches: The batches.

  Returns:
    The batch.
  """
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batches)


def collect(
    source: Any, key, num_states: int, max_batches: int = 10_000
) -> HeldoutSet:
  """The first num_states valid states of source.next(fold_in64(key, i)).

  i runs 0, 1, ...

  Args:
    source: a state source with next(key).
    key: the collection key.
    num_states: states to collect.
    max_batches: the most batches drawn before giving up.

  Returns:
    the HeldoutSet.

  Raises:
    ValueError: max_batches batches give fewer than num_states valid states.
  """
  x64.assert_x64("heldout.collect")
  assert int(num_states) >= 1, num_states
  batches, total = [], 0
  for i in range(int(max_batches)):
    b = source.next(x64.fold_in64(key, jnp.int64(i)))
    batches.append(b)
    total += int(jnp.sum(b.valid, dtype=jnp.int64))
    if total >= num_states:
      break
  if total < num_states:
    raise ValueError(
        f"held-out source gave {total} valid states in {max_batches} batches, "
        f"want {num_states}"
    )
  batch = _concat(batches)
  valid = np.asarray(batch.valid)
  # 1-based rank of each valid state
  rank = np.cumsum(valid.reshape(-1)).reshape(valid.shape)
  mask = valid & (rank <= num_states)
  # rows up to the last kept state
  n = int(np.nonzero(mask.any(axis=1))[0][-1]) + 1
  batch = jax.tree.map(lambda x: x[:n], batch)
  mask = jnp.asarray(mask[:n])
  assert int(jnp.sum(mask)) == num_states
  return HeldoutSet(batch=batch, mask=mask, num_states=int(num_states))


# id(policy) -> (weak reference to the policy, its jitted scorer). The scorer
# holds the policy weakly and the entry goes when the policy does, so a long
# run that builds new policies does not keep the old ones alive.
_SCORERS: dict[int, tuple[Any, Callable[..., Any]]] = {}


def _scorer(policy: types.Policy) -> Callable[..., Any]:
  """The jitted held-out agreement scorer of policy, cached per policy.

  The cache holds the policy weakly and drops the entry when it is collected.

  Args:
    policy: the student policy.

  Returns:
    The jitted fn(params, batch, mask) -> agreement dict.
  """
  k = id(policy)
  hit = _SCORERS.get(k)
  if hit is None or hit[0]() is not policy:
    ref = weakref.ref(policy)

    def fn(params, batch, mask):
      """The agreement of the student's last logits with the teacher.

      Args:
        params: the student's params.
        batch: the held-out batch.
        mask: the valid rows.

      Returns:
        the agreement dict.
      """
      _, out = ref().evaluate_seq(params, batch.init_carry, batch.steps, None)
      return agreement(
          out.logits, batch.steps.legal_mask, batch.target_policy, mask
      )

    _SCORERS[k] = (ref, jax.jit(fn))
    weakref.finalize(policy, _SCORERS.pop, k, None)
  return _SCORERS[k][1]


def score(
    policy: types.Policy, params, heldout: HeldoutSet
) -> dict[str, float | int]:
  """{top1, kl, ce, teacher_entropy: float, num_states: int} of the student.

  Args:
    policy: the student policy.
    params: its params.
    heldout: the held-out set.

  Returns:
    {top1, kl, ce, teacher_entropy: float, num_states: int}.
  """
  if policy.spec.stateful:
    assert jax.tree.leaves(
        heldout.batch.init_carry
    ), "a stateful student needs the held-out rows' init_carry"
  m = _scorer(policy)(params, heldout.batch, heldout.mask)
  out = {k: float(v) for k, v in m.items()}
  out["num_states"] = int(heldout.num_states)
  return out
