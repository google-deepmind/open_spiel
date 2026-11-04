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

"""The behaviour-cloning learner: one loss over policy, value and aux targets.

Behaviour cloning, teacher distillation, AlphaZero's network update (BC over
a replay of search targets) and learned belief models (auxiliary heads on
privileged labels) are the same update: a forward pass and
learners.losses.distill_loss over the targets a source wrote into the
TrajectoryBatch.

  learner = BCLearner(student, BCConfig(learning_rate=3e-4))
  state = learner.init(key, env)
  state, metrics = learner.update(state, batch, key)

Sources: sources.dagger.DaggerSource (online, with a teacher),
sources.shards.ShardReader (offline), learners.az.PlyReplay (search targets).

Per batch (TrajectoryBatch [B, T]):
  forward   policy.evaluate_seq(params, batch.init_carry, batch.steps, None):
            observe + evaluate at every ply, so a recurrent student rebuilds
            its carry from the batch's init_carry. A stateful student given
            an empty init_carry () starts from its own init_carry(params, B,
            steps.seat[:, 0], key(0)), the offline case where the stored
            carry is not the student's.
  policy    logits vs target_policy (ce_target, kl_target) or vs an action
            label (ce_action): action_label "taken" = batch.action,
            "target_argmax" = the argmax of the legal target. Masked by
            BCConfig.policy_mask: "valid" = batch.valid;
            "valid&search_decided" = batch.valid & aux[AUX_SEARCH_DECIDED]
            (rows an operator did not decide, for example single-legal
            skips, give no policy gradient); the second raises KeyError when
            the batch has no AUX_SEARCH_DECIDED.
  value     the perspective seat's column of out.value (or aux
            "value_logits" for HL-Gauss) vs the seat's column of
            target_value, masked by valid & labels[TARGET_VALUE_VALID] when
            the source wrote that label.
  aux       each AuxTerm(name): out.aux[name] vs batch.labels[name].

Optimiser: clip_by_global_norm(max_grad_norm), then AdamW (scale_by_adam,
add_decayed_weights, -lr). The step size is BCConfig.schedule(): constant,
or a linear decay learning_rate -> lr_end over total_steps, optionally after
a linear warmup from 0; "cosine" is a cosine decay over total_steps;
"warmup_cosine" is optax.warmup_cosine_decay_schedule (0 -> learning_rate
over warmup_steps, then cosine to lr_end at total_steps). It is evaluated at
the int64 BCState.step and cast to the param dtype.

Reuse: num_epochs passes, each a fresh permutation of the B rows split into
num_minibatches (rows, not plies, so sequences stay whole for recurrent
students); 1 x 1 is one full-batch step with no permutation.

Multi-device: parallel.data_parallel.DataParallel.update runs this update
with rows sharded over the "batch" mesh axis and the state replicated.

Dtypes (x64 on): params as the network declares (f32); the step counter is
int64 (one per optimiser step); optax keeps its own int32 count for Adam's
bias correction only (it saturates, it does not wrap). Metrics are f32:
distill_loss's terms plus grad_norm (pre-clip global norm), means over the
optimiser steps of the update.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.parallel import data_parallel as dp_lib
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.sources import targets

F32 = jnp.float32
ACTION_LABELS = ("taken", "target_argmax")
LR_SCHEDULES = ("linear", "cosine", "warmup_cosine")
POLICY_MASKS = ("valid", "valid&search_decided")


@dataclasses.dataclass(frozen=True)
class BCConfig:
  """The BC learner's settings.

  Attributes:
    loss: the DistillLossConfig.
    action_label: the label the one-hot policy loss reads.
    learning_rate: the (peak) step size.
    lr_end: the final step size; None: constant learning_rate.
    lr_schedule: LR_SCHEDULES: "linear", "cosine" or "warmup_cosine".
    warmup_steps: warmup optimiser steps (warmup_cosine).
    total_steps: optimiser steps the schedule spans (needed with lr_end/warmup).
    weight_decay: AdamW's decoupled weight decay.
    max_grad_norm: the global gradient-norm clip.
    num_epochs: passes over each batch.
    num_minibatches: minibatches per pass.
    policy_mask: POLICY_MASKS (module doc).
  """

  loss: losses.DistillLossConfig = losses.DistillLossConfig()
  action_label: str = "target_argmax"
  learning_rate: float = 3e-4
  lr_end: Optional[float] = None  # None: constant learning_rate
  # LR_SCHEDULES: "linear", "cosine" or "warmup_cosine"
  lr_schedule: str = "linear"
  warmup_steps: int = 0
  # optimiser steps the schedule spans (needed with lr_end/warmup)
  total_steps: int = 0
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0
  num_epochs: int = 1
  num_minibatches: int = 1
  policy_mask: str = "valid"  # POLICY_MASKS (module doc)

  def __post_init__(self):
    assert isinstance(self.loss, losses.DistillLossConfig), type(self.loss)
    if self.policy_mask not in POLICY_MASKS:
      raise ValueError(
          f"policy_mask {self.policy_mask!r} not in {POLICY_MASKS}"
      )
    for t in self.loss.aux_terms:
      if operator_vocab.is_reserved_aux(t.name):
        raise ValueError(
            f"aux term {t.name!r}: the {operator_vocab.RESERVED_AUX_PREFIX!r} "
            "prefix is "
            "reserved for search operator outputs "
            "(policies.operator_vocab.AUX_SPECS)"
        )
    assert self.action_label in ACTION_LABELS, self.action_label
    assert self.learning_rate > 0, self.learning_rate
    assert self.num_epochs >= 1 and self.num_minibatches >= 1
    assert self.max_grad_norm > 0 and self.weight_decay >= 0
    assert self.warmup_steps >= 0
    assert self.lr_schedule in LR_SCHEDULES, self.lr_schedule
    if self.lr_schedule == "cosine":
      assert (
          self.warmup_steps == 0
      ), "the cosine schedule has no warmup (use warmup_cosine)"
    if self.lr_schedule == "warmup_cosine":
      assert (
          self.lr_end is not None and self.total_steps > self.warmup_steps
      ), "warmup_cosine needs lr_end and total_steps > warmup_steps"
    if self.lr_end is not None or self.warmup_steps:
      assert self.total_steps > self.warmup_steps, (
          "a schedule needs total_steps > warmup_steps",
          self.total_steps,
          self.warmup_steps,
      )

  def schedule(self) -> schedules.Schedule:
    """The step size over optimiser steps (module doc).

    Returns:
      The schedules.Schedule of the step size.
    """
    lr = float(self.learning_rate)
    end = lr if self.lr_end is None else float(self.lr_end)
    if self.lr_schedule == "warmup_cosine":
      return schedules.Schedule(
          "cosine",
          lr,
          end,
          decay_steps=float(self.total_steps - self.warmup_steps),
          warmup_steps=float(self.warmup_steps),
      )
    if self.lr_schedule == "cosine":
      if end < lr:
        return schedules.Schedule(
            "cosine", lr, end, decay_steps=float(self.total_steps)
        )
      return schedules.Schedule("constant", lr)
    if self.warmup_steps > 0:
      return schedules.Schedule(
          "linear",
          lr,
          end,
          decay_steps=float(self.total_steps - self.warmup_steps),
          warmup_steps=float(self.warmup_steps),
      )
    if end < lr:
      return schedules.Schedule(
          "linear", lr, end, decay_steps=float(self.total_steps)
      )
    return schedules.Schedule("constant", lr)

  def optimizer(self) -> optax.GradientTransformation:
    """AdamW without its step size (BCLearner applies -schedule(step)).

    Returns:
      The optax GradientTransformation.
    """
    return optax.chain(
        optax.clip_by_global_norm(self.max_grad_norm),
        optax.scale_by_adam(),
        optax.add_decayed_weights(self.weight_decay),
    )


class BCState(NamedTuple):
  """The BC learner's state.

  Attributes:
    params: the network params.
    opt_state: the optimiser state.
    step: i64[] optimiser steps taken.
  """

  params: Any
  opt_state: Any
  step: Any  # i64[] optimiser steps taken


def _seat_column(x, seat):
  """X [B, T, P] -> [B, T] at seat [B, T].

  Args:
    x: [B, T, P] values.
    seat: [B, T] seats.

  Returns:
    [B, T], the seat's column.
  """
  return jnp.take_along_axis(x, seat[..., None].astype(jnp.int32), axis=-1)[
      ..., 0
  ]


class BCLearner:
  """Learner over one Policy (the student).

  The Learner protocol's `cfg` is given at construction; `init(key, env, cfg)`
  accepts None or the same.

  Attributes:
    value_support: the value head's support (losses.check_value_support).
    cfg: the BCConfig.
  """

  def __init__(self, policy: types.Policy, cfg: BCConfig):
    x64.assert_x64("BCLearner")
    assert isinstance(cfg, BCConfig), type(cfg)
    assert policy.spec.framework == "jax", (
        "BCLearner trains a JAX policy; a host-driven student needs its own "
        "learner"
    )
    self.value_support = losses.check_value_support(cfg.loss.value, policy)
    self._policy = policy
    self.cfg = cfg
    self._tx = cfg.optimizer()
    self._lr = cfg.schedule()
    self._update = jax.jit(self._update_impl)

  # --- Learner protocol -----------------------------------------------------
  def init(self, key, env: types.Env) -> BCState:
    """A fresh learner state from freshly initialised params.

    Args:
      key: the PRNG key of the params.
      env: the env; must match the policy's spec.

    Returns:
      The BCState.
    """
    spec = self._policy.spec
    assert (
        env.num_actions == spec.num_actions
        and env.num_players == spec.num_players
    ), (env.num_actions, env.num_players, spec)
    return self.init_from_params(self._policy.init_params(key))

  def init_from_params(self, params) -> BCState:
    """A fresh optimiser over given params (a warm start).

    Args:
      params: the network params.

    Returns:
      The BCState.
    """
    x64.assert_x64("BCLearner.init_from_params")
    return BCState(
        params=params,
        opt_state=self._tx.init(params),
        step=jnp.asarray(0, jnp.int64),
    )

  def policy(self) -> types.Policy:
    """The policy this learner trains.

    Returns:
      The Policy.
    """
    return self._policy

  def params(self, s: BCState):
    """The network params of a learner state.

    Args:
      s: the learner state.

    Returns:
      s.params.
    """
    return s.params

  def update(
      self, s: BCState, batch: types.TrajectoryBatch, key
  ) -> tuple[BCState, Mapping[str, Any]]:
    """One update over a batch.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the PRNG key of the minibatch shuffle.

    Returns:
      (new state, metrics).
    """
    return self._update(s, batch, key)

  def state_dict(self, s: BCState) -> Mapping[str, Any]:
    """{params, opt_state (leaf list), step}.

    opt_state is a leaf list: optax states are NamedTuples, which an orbax round
    trip turns into dicts.

    Args:
      s: the learner state.

    Returns:
      The checkpoint blob.
    """
    return {
        "params": s.params,
        "opt_state": list(jax.tree.leaves(s.opt_state)),
        "step": s.step,
    }

  def load_state(self, blob: Mapping[str, Any]) -> BCState:
    """Inverse of state_dict; raises on any tree, shape or dtype mismatch.

    Args:
      blob: a state_dict() blob.

    Returns:
      The BCState.
    """
    assert set(blob) == {"params", "opt_state", "step"}, sorted(blob)
    step = jnp.asarray(blob["step"])
    assert step.dtype == jnp.int64 and not step.shape, (step.dtype, step.shape)
    ref = self._tx.init(blob["params"])
    ref_leaves, treedef = jax.tree.flatten(ref)
    got = list(blob["opt_state"])
    assert len(got) == len(
        ref_leaves
    ), f"opt_state has {len(got)} leaves, the optimiser {len(ref_leaves)}"
    for g, r in zip(got, ref_leaves):
      assert (jnp.shape(g), jnp.asarray(g).dtype) == (jnp.shape(r), r.dtype), (
          "opt_state leaf does not match the optimiser",
          jnp.shape(g),
          jnp.shape(r),
      )
    return BCState(
        params=blob["params"],
        opt_state=jax.tree.unflatten(treedef, got),
        step=step,
    )

  # --- loss -----------------------------------------------------------------
  def loss(self, params, batch: types.TrajectoryBatch):
    """(total f32, metrics) of one batch (the update's objective).

    Args:
      params: the network params.
      batch: the TrajectoryBatch.

    Returns:
      (total f32, metrics).

    Raises:
      KeyError: a label or aux entry the config needs is missing from the batch.
    """
    cfg = self.cfg.loss
    steps = batch.steps
    carry = batch.init_carry
    if self._policy.spec.stateful and not jax.tree.leaves(carry):
      # A reset-aligned offline batch: the student starts from its own carry.
      seat0 = jnp.asarray(steps.seat[:, 0]).astype(jnp.int32)
      carry = self._policy.init_carry(
          params, seat0.shape[0], seat0, jax.random.key(0)
      )
    _, out = self._policy.evaluate_seq(params, carry, steps, None)
    legal = steps.legal_mask
    seat = steps.seat
    if self.cfg.action_label == "taken":
      action = jnp.maximum(batch.action, 0).astype(jnp.int32)
    else:
      action = losses.target_argmax(batch.target_policy, legal)
    policy_mask = None  # None: distill_loss uses mask = batch.valid
    if self.cfg.policy_mask == "valid&search_decided":
      if operator_vocab.AUX_SEARCH_DECIDED not in batch.aux:
        raise KeyError(
            "policy_mask 'valid&search_decided': the batch has no aux "
            f"{operator_vocab.AUX_SEARCH_DECIDED!r} (aux {sorted(batch.aux)}); "
            "absent never "
            "means decided"
        )
      decided = jnp.asarray(batch.aux[operator_vocab.AUX_SEARCH_DECIDED])
      assert (
          decided.dtype == jnp.bool_ and decided.shape == batch.valid.shape
      ), (decided.dtype, decided.shape)
      policy_mask = batch.valid & decided
    value_mask = batch.valid
    if targets.TARGET_VALUE_VALID in batch.labels:
      value_mask = value_mask & batch.labels[targets.TARGET_VALUE_VALID]
    target_value = _seat_column(batch.target_value, seat)
    value = value_logits = None
    if cfg.value == "mse":
      value = _seat_column(out.value, seat)
    elif cfg.value == "hl_gauss":
      assert (
          "value_logits" in out.aux
      ), "value='hl_gauss' needs a policy with an HL-Gauss head"
      value_logits = out.aux["value_logits"]
    aux_preds, labels = {}, {}
    for t in cfg.aux_terms:
      if t.name not in out.aux:
        raise KeyError(
            f"aux term {t.name!r}: the policy emits aux {sorted(out.aux)}"
        )
      if t.name not in batch.labels:
        raise KeyError(
            f"aux term {t.name!r}: the batch carries labels "
            f"{sorted(batch.labels)}"
        )
      aux_preds[t.name] = out.aux[t.name]
      labels[t.name] = batch.labels[t.name]
    return losses.distill_loss(
        cfg,
        logits=out.logits,
        legal=legal,
        mask=batch.valid,
        action=action,
        target_policy=batch.target_policy,
        value=value,
        value_logits=value_logits,
        target_value=target_value,
        value_mask=value_mask,
        aux_preds=aux_preds,
        labels=labels,
        value_support=self.value_support,
        policy_mask=policy_mask,
    )

  # --- update -----------------------------------------------------------------
  def _step(self, s: BCState, mb: types.TrajectoryBatch):
    """One optimiser step on a minibatch.

    Args:
      s: the learner state.
      mb: the minibatch.

    Returns:
      (the new BCState, the step's metrics).
    """
    (_, metrics), grads = jax.value_and_grad(self.loss, has_aux=True)(
        s.params, mb
    )
    updates, opt_state = self._tx.update(grads, s.opt_state, s.params)
    lr = self._lr.value(s.step)
    updates = jax.tree.map(lambda g: jnp.asarray(-lr, g.dtype) * g, updates)
    params = optax.apply_updates(s.params, updates)
    for new, old in zip(jax.tree.leaves(params), jax.tree.leaves(s.params)):
      assert (
          new.dtype == old.dtype
      ), f"optimiser promoted a param {old.dtype} -> {new.dtype}"
    metrics = dict(metrics, grad_norm=optax.tree.norm(grads).astype(F32))
    return BCState(params, opt_state, s.step + jnp.int64(1)), metrics

  def _update_impl(self, s: BCState, batch: types.TrajectoryBatch, key):
    """The update: num_epochs passes of num_minibatches shuffled steps.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the PRNG key of the epochs' shuffles.

    Returns:
      (the new BCState, metrics: the step's own when a single step, else the
      mean over every minibatch step).
    """
    e, m = self.cfg.num_epochs, self.cfg.num_minibatches
    batch = dp_lib.constrain_rows(
        batch
    )  # data parallel: rows stay on their device
    if e == 1 and m == 1:
      return self._step(s, batch)
    b = batch.valid.shape[0]
    assert b % m == 0, f"batch rows {b} not divisible by num_minibatches {m}"
    size = b // m

    def _epoch(st, k):
      """One epoch over a fresh permutation in minibatches.

      Args:
        st: The learner state.
        k: The epoch's key.

      Returns:
        (state, per-step metrics).
      """
      perm = jax.random.permutation(k, b)
      split = dp_lib.constrain_rows(
          jax.tree.map(
              lambda x: x[perm].reshape((m, size) + x.shape[1:]), batch
          ),
          axis=1,
      )
      return jax.lax.scan(self._step, st, split)

    s, metrics = jax.lax.scan(_epoch, s, jax.random.split(key, e))
    return s, jax.tree.map(lambda x: jnp.mean(x).astype(F32), metrics)
