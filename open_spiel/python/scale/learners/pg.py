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

"""The policy-gradient Learner: PPO, MMD and registered loss variants.

One learner covers the clipped-surrogate family: PPO (Schulman et al. 2017),
magnetic mirror descent (MMD; Sokota et al. 2023, a KL pull toward a
periodically refreshed magnet policy) and any loss registered with
`register_loss`. It trains a NATIVE_JAX Policy, recurrent or not, from
core.types.TrajectoryBatch.

  learner = PGLearner(policy, PGConfig(loss="mmd", lr=2.5e-4, num_updates=1000))
  state = learner.init(key)
  state, metrics = learner.update(state, batch, key)

One update:

  1. prepare: per-seat values from the batch (TrajectoryBatch.value_pred at
     the perspective seat, denormalized by the value EMA when it is on, so
     values are in reward units), the bootstrap value at `bootstrap_step`
     (the carry rebuilt by observe_seq over the segment, then observe +
     evaluate), and advantages.gae: GAE(gae_lambda) advantages and
     TD(gae_lambda) targets, or a registered loss's own targets.
  2. update_epochs x minibatches gradient steps: per epoch the key is split
     and the plies shuffled; per minibatch the value EMA is updated from the
     minibatch's valid rows and its targets normalized, then value_and_grad
     of the loss and one optimizer step.
  3. the magnet is re-taken after the update when (update_step + 1) %
     reference_update_every == 0 (mmd), or every magnet_every updates for a
     registered loss.

Shuffles. "flat" permutes the T x B plies flattened time-major and evaluates
each ply on its own: feed-forward policies only. "env_axis" permutes batch
rows and runs Policy.evaluate_seq over each row's whole segment from
TrajectoryBatch.init_carry, so a recurrent policy's carry is rebuilt through
observe exactly as the Arena built it live.

Decisions only. The flat shuffle trains on the batch's valid plies,
gathered first into a static size (`train_rows`): a self-play batch emits
one row per seat, so about half its plies, and every observe-only terminal
ply, carry no decision. The static size depends only on the batch shape,
the device count, the table layout and the config, and is pinned per batch
shape at the first batch of that shape, so a run compiles one update program
per batch shape. It is a bound the decisions cannot exceed for the layout
(proved in train_rows) unless PGConfig.max_decision_share declares a tighter
one; a batch whose decisions exceed it raises TrainRowsOverflowError and nothing
is silently cut. PGConfig.train_rows_buckets adds up to
MAX_TRAIN_ROWS_BUCKETS smaller static sizes: a batch trains at the smallest
that holds its decisions.

Loss dispatch. Every place the loss matters branches on it explicitly (ppo,
mmd or the registered LossSpec), and an unknown loss raises ValueError. Loss
math lives in learners/losses.py and the registered LossSpecs; advantages in
learners/advantages.py.

Multi-device. parallel.data_parallel.DataParallel.update runs this update
with the batch row-sharded and the state replicated; minibatches are pinned
to the "batch" axis, the shuffle stays global, and reductions are
all-reduces.

Optimizer. clip_by_global_norm, Adam (eps 1e-5) or AdamW (decoupled weight
decay), then the step size PGConfig.lr_schedule() evaluated at the int64
update counter (constant within an update) and cast to the update dtype.

Options, each off by default:
  value_scale      rewards are divided by it before the targets (value
                   targets in points / value_scale).
  value_rows       "module:attr" of f(one InfoState) -> bool: plies where the
                   perspective seat did not act whose value is still trained
                   (the other seat's nodes of a value chain that links
                   through them). They link TD / GAE, their value is the
                   learner network's own value of the seat's InfoState, and
                   they enter the value loss only. Registered losses only
                   (their value mask is separate from the policy mask).
  minibatch_size   one fixed SGD minibatch size in flat plies instead of
                   num_minibatches equal splits: per epoch the trained plies
                   are permuted and dealt into minibatch_size-ply slots. A
                   minibatch holding no trained ply takes no optimizer step.
                   Each step gathers its own plies from the [B, T] batch, so
                   no permuted copy is materialized; padding slots point at
                   trained plies with every mask off, so padding never feeds
                   the network an out-of-domain InfoState.
  sgd_steps        (with minibatch_size) the fewest SGD steps per epoch: the
                   permuted plies are dealt round-robin over max(sgd_steps,
                   ceil(rows / minibatch_size)) steps.
  drop_unbootstrapped  rows whose bootstrap value is non-finite (ply T
                   outside the network's domain) are left out of the update
                   (metric untargeted_rows) instead of raising.
  schedule_step_offset  added to the update counter wherever a schedule or
                   the magnet refresh reads it (a continued run's training
                   step); update_step itself counts this run's updates.
  lr_sched         the step-size Schedule itself; None uses lr, lr_end and
                   anneal_lr.
  schedule_time_scale  the time stretch of a registered loss's magnet
                   temperature.

SGD step gate (LossSpec.step_gate, a registered loss's extension point):
a check run inside the update after each SGD step's loss and gradient and
before its optimizer update. step_gate(loss_config) returns None (no gate
for this config: the update is built exactly as without the hook) or
gate(step_metrics) -> bool scalar, step_metrics being the step's metrics
(the loss's terms, approx_kl, grad_norm and grad_norm/<module>, all
measured at the params the step starts from). False skips that step's
update and every remaining step of the update (later minibatches and
epochs): params, optimizer state and value EMA stay as the last applied step
left them, and the next update starts normally. The skip is a lax.cond on a
bool carried through the SGD scan, so one program serves every update.
With a gate, update() also reports sgd_steps_taken, sgd_steps_planned (the
steps the update would take ungated), <name>_early_stopped (0/1) and
<name>_stop_step (the steps applied before the stop; all of them when
none), name = LossSpec.step_gate_name, and the loss metrics are means over
the steps that computed a loss (the stopping step's included).

Forward chunking. Every learner call of the policy runs on at most
PGConfig.forward_chunk_rows rows at a time: larger calls are split into
equal chunks (actor_critic.chunk_plan), each running the whole policy under
a rematerialized lax.scan, so a gradient keeps one chunk's activations.
Rows are independent, so outputs equal the unchunked ones up to float
reassociation in the parameter gradient's sum over rows.

A minibatch above forward_chunk_rows takes its gradient chunk by chunk. With
ppo / mmd (pg_loss) the gradient is accumulated: each chunk's loss and
gradient are taken alone, every mean divided by the whole minibatch's count
(losses.pg_stats: the normalized advantages, the advantage filter and the
row counts come from the whole minibatch), and the chunks' gradients and
summed metrics add up to the minibatch's, so each chunk's forward runs once.
A registered loss accumulates the same way when its LossSpec declares
chunk_sums (its loss then takes counts=LossCounts, the whole minibatch's
policy and value row counts, as every masked mean's denominator); one that
does not (whose terms need not split over rows) keeps the rematerialized
chunk scan, whose backward runs every chunk's forward again.

Teacher distillation (kickstarting; off by default). PGLearner.set_teacher
gives the learner a frozen teacher: another stateless NATIVE_JAX policy of
the same game, action count and player count (its own spec, params and
forward; never the learner's), with PGConfig.teacher_coef = c(t) > 0. Once
per update, before the SGD epochs, the teacher's forward runs over the rows
the update trains (flat plies, no gradient, at most forward_chunk_rows rows
per call unless set_teacher names its own chunk), and its legal-masked action
probabilities are cached beside the rows (the fixed-minibatch path: per
minibatch, as the magnet's logits). The loss (pg_loss, so loss "ppo" / "mmd"
only) gains c(t) * mean over the trained decision rows of KL(teacher || pi)
(losses.kl_from_target); the magnet is untouched by it, and the teacher's
params are in no optimizer and no PGState. update() adds teacher_kl, the
applied teacher_coef and teacher_top1 (the share of trained rows where the
learner's and the teacher's top legal action agree). With no teacher every
traced program is the one without the option (bit for bit).

x64. Asserted at init and update. Counters (update_step) are int64; params
stay float32 (asserted), every loss input is float32, and the step size is
cast to the param dtype, so nothing promotes to float64.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import itertools
import math
from typing import Any, Mapping, NamedTuple, Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax

from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.core import hash_compat
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import telemetry
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.parallel import data_parallel as dp_lib
from open_spiel.python.scale.policies import registry

PG_LOSSES = ("ppo", "mmd")  # the built-in losses (losses.pg_loss)
SHUFFLES = ("flat", "env_axis")
#: PGConfig.credit values: a seat's own return, or its side's.
CREDITS = ("seat", "side")
#: PGConfig.teacher_coef's default: no teacher distillation.
TEACHER_COEF_OFF = schedules.Schedule("constant", 0.0)
# PGConfig.train_rows_buckets: compiled update programs per shape
MAX_TRAIN_ROWS_BUCKETS = 8
#: The value_rows forward sizes (PGLearner.value_rows_cap): 1024 x 2**i and
#: 1536 x 2**i up to 2**26 rows, ascending.
VALUE_ROWS_CAPS = tuple(
    sorted({m << i for m, i in itertools.product((1024, 1536), range(17))})
)
F32 = jnp.float32
I64 = jnp.int64
tree_map = jax.tree_util.tree_map


@dataclasses.dataclass(frozen=True)
class LossSpec:
  """A PG loss variant beyond ppo / mmd, chosen by PGConfig.loss = name.

  Registered with register_loss. Its config object is PGConfig.loss_config
  (an instance of config_type; None = config_type()).

  Such a loss sees the behaviour policy's full legal-masked log-probs and a
  value mask separate from the policy mask (PGConfig.value_rows may add
  value-only plies); its magnet is uniform unless PGConfig.magnet_every > 0,
  then the network re-taken every that many updates.

  Attributes:
    name: the loss name PGConfig.loss selects.
    config_type: the class of PGConfig.loss_config.
    targets: targets(loss_config, reward, value, node, done, bootstrap)
      -> (advantages f32[B, T], targets f32[B, T]).
    loss: loss(loss_config, logits=, value=, legal=, action=,
      behavior_log_probs=, advantages=, returns=, policy_mask=, value_mask=,
      temperature=, magnet_logits=[, step_coefs=]) -> (total f32, terms
      dict).
    temperature: temperature(loss_config, step int64, floor_step, time_scale)
      -> f32, the magnet temperature of the counter schedules read.
    check: check(pg_config) raises on a PGConfig the loss refuses; None
      accepts every PGConfig.
    chunk_sums: None, or the names of the loss's terms that add up over row
      chunks of a minibatch. Declared, the loss also takes counts=LossCounts
      (None: its own masks' counts) and divides every masked mean by them,
      and a minibatch above forward_chunk_rows takes its gradient accumulated
      over row chunks (module doc "Forward chunking"); every term not named
      must be the same in every chunk (e.g. the temperature). None: the
      rematerialized chunk scan.
    step_coefs: None, or (loss_config, step int64, floor_step, time_scale)
      -> dict of f32 scalars: the loss's OTHER coefficients on the same
      counter (e.g. a guard term's weight), evaluated once per update as the
      temperature is and handed to loss as step_coefs= (the learner passes it
      only when this hook is declared and its value is not None); the same in
      every minibatch and row chunk of the update.
    step_gate: None, or (loss_config) -> None | gate(step_metrics) -> bool
      scalar: the module doc's "SGD step gate" (None from the factory: no
      gate for that config, the update exactly as without one).
    step_gate_name: the prefix of the gate's metrics <name>_early_stopped and
      <name>_stop_step (default "sgd").
    stable_softmax: the loss takes stable_softmax=True (its softmax terms
      through learners.stable_softmax); the learner passes it only when
      PGConfig.pg.stable_softmax is set, and a PGConfig that sets it for a
      loss that does not declare this is refused.
  """

  name: str
  config_type: type[Any]
  targets: Any
  loss: Any
  temperature: Any
  check: Any = None
  chunk_sums: Optional[frozenset[str]] = None
  step_coefs: Any = None
  step_gate: Any = None
  step_gate_name: str = "sgd"
  stable_softmax: bool = False

  def __post_init__(self):
    if not (
        isinstance(self.step_gate_name, str)
        and self.step_gate_name.isidentifier()
    ):
      raise ValueError(
          f"step_gate_name must be an identifier, got {self.step_gate_name!r}"
      )
    if self.chunk_sums is not None:
      assert isinstance(self.chunk_sums, frozenset) and all(
          isinstance(k, str) for k in self.chunk_sums
      ), self.chunk_sums


class LossCounts(NamedTuple):
  """A registered loss's masked-mean denominators over a WHOLE minibatch.

  LossSpec.chunk_sums, given with one chunk of its rows.

  Attributes:
    policy: f32, the policy_mask-True rows.
    value: f32, the value_mask-True rows.
  """

  policy: jax.Array
  value: jax.Array


_EXTENSION_LOSSES: dict[str, LossSpec] = {}
#: Open loss variant -> the module whose import registers it (register_loss),
#: imported when a config names it, so a recorded config needs no import by
#: the caller.
LAZY_LOSSES = {"ppo_magnet": f"{__name__.rpartition('.')[0]}.ppo_magnet"}


def register_loss(spec: LossSpec) -> LossSpec:
  """Adds a PG loss variant (idempotent for the same spec).

  Args:
    spec: the LossSpec to register.

  Returns:
    spec.

  Raises:
    ValueError: spec names a built-in loss, or another spec of that name is
      registered.
  """
  assert isinstance(spec, LossSpec), type(spec)
  if spec.name in PG_LOSSES:
    raise ValueError(f"{spec.name!r} is a built-in loss")
  old = _EXTENSION_LOSSES.get(spec.name)
  if old is not None and old is not spec:
    raise ValueError(f"loss {spec.name!r} is already registered")
  _EXTENSION_LOSSES[spec.name] = spec
  return spec


def losses() -> tuple[str, ...]:
  """Every loss PGConfig.loss may name: ppo, mmd and the registered variants.

  Returns:
    The loss names, built-ins first (the LAZY_LOSSES ones whether or not
    imported yet).
  """
  return PG_LOSSES + tuple(sorted(set(_EXTENSION_LOSSES) | set(LAZY_LOSSES)))


def extension(loss: str) -> Optional[LossSpec]:
  """The registered LossSpec of `loss`, None for ppo / mmd.

  Args:
    loss: a loss name.

  Returns:
    The LossSpec, or None for a built-in loss.

  Raises:
    ValueError: the loss is unknown.
  """
  if loss in PG_LOSSES:
    return None
  if loss not in _EXTENSION_LOSSES and loss in LAZY_LOSSES:
    importlib.import_module(LAZY_LOSSES[loss])
  if loss in _EXTENSION_LOSSES:
    return _EXTENSION_LOSSES[loss]
  raise _unknown_loss(loss)


@dataclasses.dataclass(frozen=True)
class PGConfig:
  """One PG learner config.

  The defaults are not a runnable config on their own: the LR anneals by
  default, so a config sets num_updates (or anneal_lr=False), and the default
  loss "mmd" with kl_coef 0 has no magnet pull, so an MMD config sets kl_coef
  and a PPO config sets loss="ppo". `PGConfig(loss="ppo", anneal_lr=False)`
  is plain PPO at a constant step size.

  Registered-loss options: loss_config, magnet_every, temperature_floor_step,
  schedule_time_scale and value_rows apply only to a loss registered with
  register_loss; "ppo" and "mmd" refuse them at anything but their defaults.

  Attributes:
    loss: "ppo" (clipped surrogate, no magnet), "mmd" (+ kl_coef * KL(pi ||
      magnet), magnet re-taken every reference_update_every updates) or a
      registered variant (LossSpec: register_loss; magnet uniform when
      magnet_every == 0, else re-taken every that many updates).
    lr: the initial step size.
    lr_end: the step size the linear anneal ends at.
    anneal_lr: anneal the step size linearly from lr to lr_end over num_updates.
    num_updates: the run length in updates, and the horizon of the linear LR
      anneal (lr -> lr_end). Required (>= 1) when anneal_lr is on; 0 means
      unset.
    max_grad_norm: the global gradient-norm clip.
    adam_eps: Adam's epsilon.
    use_adamw: AdamW (decoupled weight decay) instead of Adam.
    weight_decay: the AdamW weight decay.
    update_epochs: passes over the batch per update.
    num_minibatches: minibatches per epoch (without minibatch_size).
    shuffle: "flat" or "env_axis" (module doc "Shuffles").
    gamma: the discount.
    gae_lambda: the GAE / TD lambda.
    pg: the ppo/mmd loss (clip, value head, normalisation, advantage filter).
    ent_coef: the entropy coefficient, a schedules.Schedule over outer updates.
    kl_coef: the magnet KL coefficient, a schedules.Schedule over outer updates.
    reference_update_every: mmd re-takes its magnet every that many updates.
    value_ema: per-minibatch normalisation of value targets by a running EMA of
      their moments (scalar head only).
    value_ema_beta: the EMA's decay per minibatch; the default 0.925 averages
      over about 1 / (1 - 0.925), roughly 13 minibatches (a few updates at the
      default 4 minibatches per epoch).
    loss_config: a registered loss's own config (its LossSpec.config_type).
    magnet_every: a registered loss re-takes its magnet every that many updates
      (0: uniform).
    temperature_floor_step: the floor step of a registered loss's
      magnet-temperature schedule.
    max_decision_share: None (default) sizes the decisions-only flat shuffle
      (PGLearner.train_rows) at a bound no batch can exceed; a float in (0, 1]
      declares that a perspective row acts on at most that share of a batch's
      plies (e.g. a two-player game whose seats alternate), for a smaller static
      size. A batch that breaks the declaration raises TrainRowsOverflowError;
      nothing is dropped.
    train_rows_buckets: () (default) trains every batch of a shape at the ONE
      static size (PGLearner.train_rows). Otherwise up to MAX_TRAIN_ROWS_BUCKETS
      strictly increasing shares in (0, 1] of a batch's B x T plies: each,
      rounded up as the static size is, is a smaller size the update may train
      at; a batch trains at the smallest size (bucket) that holds its decisions,
      the static size the top bucket. So the update trains close to the real
      decision count, with at most len(buckets) + 1 compiled update programs per
      batch shape (never a size per batch); decisions over the static size still
      raise TrainRowsOverflowError. The minibatch size and the padding differ by
      bucket, so a run with buckets is a different (equally valid) shuffle than
      one without.
    value_scale: rewards are divided by it before the targets (module doc
      "Options, each off by default").
    value_rows: "module:attr" of the value-only plies of a registered loss
      (module doc "Options, each off by default").
    minibatch_size: one fixed SGD minibatch size in flat plies (module doc
      "Options, each off by default"); 0: num_minibatches equal splits.
    drop_unbootstrapped: leave rows with a non-finite bootstrap value out of the
      update instead of raising (module doc "Options, each off by default").
    schedule_step_offset: added to the update counter wherever a schedule or the
      magnet refresh reads it (module doc "Options, each off by default").
    lr_sched: the step-size Schedule itself; None uses lr, lr_end and anneal_lr.
    schedule_time_scale: the time stretch of a registered loss's
      magnet-temperature schedule.
    forward_chunk_rows: the most rows any learner call of the policy runs on at
      once (module doc "Forward chunking"; execution only, the results are the
      unchunked ones).
    sgd_steps: with minibatch_size, the fewest SGD steps per epoch; the rows are
      dealt evenly over max(sgd_steps, ceil(rows / minibatch_size)) steps
      (module doc "Options, each off by default"). 0: full minibatches.
    magnet_from_behavior: mmd with reference_update_every 1 only. The magnet
      is then the params the update starts from; when the batch was played by
      those same params (synchronous self-play, no replay of older batches),
      the behaviour logits the rollout stored are the magnet's logits up to
      the rollout's own rounding, so the update reads them instead of running
      the magnet's forward. The caller vouches for that condition; nothing in
      the batch can show it.
    credit: "seat" (default) or "side" (CREDITS): whose return a perspective
      seat's trajectory is credited with. "side" makes the seats of a side
      share the side's return (core.sides: the mean over its seats) for the
      advantages and value targets, so a team learner optimises the team, not
      each seat; it differs from "seat" only where partners score
      differently. The source emits it (core.arena Arena credit="side"; the
      league's PG kind builds its Arena from this field) and marks the batch
      (types.SIDE_CREDIT); update refuses a batch whose mark disagrees.
    teacher_coef: the teacher-distillation coefficient c(t), a
      schedules.Schedule over outer updates (module doc "Teacher
      distillation"). The loss gains c(t) * KL(teacher || pi) over the
      trained decision rows. It needs a teacher (PGLearner.set_teacher; the
      league's kinds.PG.teacher); the default constant 0 is off and left out
      of the config hash (core.hash_compat.added).
  """

  loss: str = "mmd"
  lr: float = 2.5e-4
  lr_end: float = 0.0
  anneal_lr: bool = True
  num_updates: int = 0
  max_grad_norm: float = 0.5
  adam_eps: float = 1e-5
  use_adamw: bool = False
  weight_decay: float = 0.0
  update_epochs: int = 4
  num_minibatches: int = 4
  shuffle: str = "flat"
  gamma: float = 0.99
  gae_lambda: float = 0.95
  pg: loss_lib.PGLossConfig = dataclasses.field(
      default_factory=loss_lib.PGLossConfig
  )
  ent_coef: schedules.Schedule = schedules.Schedule("constant", 0.01)
  kl_coef: schedules.Schedule = schedules.Schedule("constant", 0.0)
  reference_update_every: int = 1
  value_ema: bool = False
  value_ema_beta: float = 0.925
  loss_config: Any = None
  magnet_every: int = 0
  temperature_floor_step: int = 0
  max_decision_share: Optional[float] = None
  train_rows_buckets: tuple[float, ...] = ()
  # Module docstring "Options, each off by default".
  value_scale: float = 1.0
  value_rows: str = ""
  minibatch_size: int = 0
  drop_unbootstrapped: bool = False
  schedule_step_offset: int = 0
  lr_sched: Optional[schedules.Schedule] = None
  schedule_time_scale: float = 1.0
  forward_chunk_rows: int = 16384
  sgd_steps: int = 0
  magnet_from_behavior: bool = False
  # N-player credit (CREDITS): "seat" credits each seat's trajectory with its
  # own seat return; "side" with its side's return (the mean over the side's
  # seats, core.sides), so the learner's seats on a team optimise the team's
  # return. The batch's source emits the side reward (core.arena Arena
  # credit="side", which the league's PG kind builds from this field) and
  # marks it with the label types.SIDE_CREDIT; check_batch refuses a batch
  # whose mark disagrees. Left out of the config hash at its default
  # (core.hash_compat.added).
  credit: str = hash_compat.added("seat")
  # Teacher distillation (module doc): off at its default, so left out of the
  # config hash there (core.hash_compat.added).
  teacher_coef: schedules.Schedule = hash_compat.added(TEACHER_COEF_OFF)

  def __post_init__(self):
    if self.credit not in CREDITS:
      raise ValueError(
          f"PGConfig.credit must be one of {CREDITS}, got {self.credit!r}"
      )
    if not isinstance(self.teacher_coef, schedules.Schedule):
      raise ValueError(
          "PGConfig.teacher_coef is a schedules.Schedule, got "
          f"{self.teacher_coef!r}"
      )
    if self.teacher_on() and self.loss not in PG_LOSSES:
      raise ValueError(
          "teacher_coef adds its KL to pg_loss (loss 'ppo' / 'mmd'); "
          f"loss {self.loss!r} is a registered loss"
      )
    ext = extension(self.loss) if self.loss in losses() else None
    assert (
        self.loss in losses()
    ), f"loss must be one of {losses()}, got {self.loss!r}"
    assert (
        self.shuffle in SHUFFLES
    ), f"shuffle must be one of {SHUFFLES}, got {self.shuffle!r}"
    assert isinstance(self.pg, loss_lib.PGLossConfig), type(self.pg)
    assert isinstance(self.ent_coef, schedules.Schedule), type(self.ent_coef)
    assert isinstance(self.kl_coef, schedules.Schedule), type(self.kl_coef)
    if ext is None:
      assert (
          self.loss_config is None
      ), f"loss {self.loss!r} takes no loss_config"
      assert (
          int(self.magnet_every) == 0 and int(self.temperature_floor_step) == 0
      ), (
          "magnet_every / temperature_floor_step are for a registered loss, "
          f"not {self.loss!r}"
      )
      assert (
          float(self.schedule_time_scale) == 1.0
      ), f"schedule_time_scale is for a registered loss, not {self.loss!r}"
    else:
      assert self.loss_config is None or isinstance(
          self.loss_config, ext.config_type
      ), (
          f"loss {self.loss!r}: loss_config must be a "
          f"{ext.config_type.__name__}, got "
          f"{type(self.loss_config).__name__}"
      )
      if self.pg.stable_softmax and not ext.stable_softmax:
        raise ValueError(
            f"loss {self.loss!r} does not declare stable_softmax "
            "(LossSpec.stable_softmax); pg.stable_softmax cannot apply"
        )
    assert int(self.update_epochs) >= 1 and int(self.num_minibatches) >= 1
    assert int(self.num_updates) >= 0, self.num_updates
    assert int(self.reference_update_every) >= 1, self.reference_update_every
    assert int(self.magnet_every) >= 0, self.magnet_every
    for name in (
        "lr",
        "lr_end",
        "max_grad_norm",
        "adam_eps",
        "weight_decay",
        "gamma",
        "gae_lambda",
        "value_ema_beta",
    ):
      v = float(getattr(self, name))
      assert math.isfinite(v) and v >= 0.0, f"{name}={v}"
    assert self.max_grad_norm > 0.0 and self.adam_eps > 0.0
    if self.max_decision_share is not None:
      share = float(self.max_decision_share)
      assert (
          math.isfinite(share) and 0.0 < share <= 1.0
      ), f"max_decision_share={share}"
    buckets = self.train_rows_buckets
    assert isinstance(
        buckets, tuple
    ), f"train_rows_buckets is a tuple, got {type(buckets)}"
    assert len(buckets) <= MAX_TRAIN_ROWS_BUCKETS, (
        f"train_rows_buckets: {len(buckets)} > {MAX_TRAIN_ROWS_BUCKETS} (one "
        "compiled update "
        "program per bucket)"
    )
    for a, b in zip(buckets, buckets[1:]):
      assert float(a) < float(
          b
      ), f"train_rows_buckets must increase strictly: {buckets}"
    for share in buckets:
      assert (
          math.isfinite(float(share)) and 0.0 < float(share) <= 1.0
      ), f"train_rows_buckets share {share} not in (0, 1]"
    if buckets and self.shuffle != "flat":
      raise ValueError(
          "train_rows_buckets size the flat shuffle (env_axis keeps whole "
          "segments)"
      )
    assert (
        self.gamma <= 1.0
        and self.gae_lambda <= 1.0
        and self.value_ema_beta < 1.0
    )
    if not self.use_adamw and self.weight_decay:
      raise ValueError("weight_decay needs use_adamw=True")
    if self.loss == "ppo" and not (
        self.kl_coef.kind == "constant" and self.kl_coef.start == 0.0
    ):
      raise ValueError(
          "loss=ppo has no magnet: kl_coef must be constant 0 (use loss=mmd)"
      )
    if self.value_ema and self.pg.value_head == "hl_gauss":
      raise ValueError(
          "value_ema normalises scalar targets; an hl_gauss value head is "
          "trained on raw targets"
      )
    if ext is not None and ext.check is not None:
      ext.check(self)
    v = float(self.value_scale)
    assert math.isfinite(v) and v > 0.0, f"value_scale={v}"
    t = float(self.schedule_time_scale)
    assert math.isfinite(t) and t > 0.0, f"schedule_time_scale={t}"
    if (
        isinstance(self.forward_chunk_rows, bool)
        or not isinstance(self.forward_chunk_rows, int)
        or self.forward_chunk_rows < 1
    ):
      raise ValueError(
          "forward_chunk_rows must be an int >= 1, got "
          f"{self.forward_chunk_rows!r}"
      )
    assert (
        int(self.minibatch_size) >= 0 and int(self.schedule_step_offset) >= 0
    ), (self.minibatch_size, self.schedule_step_offset)
    if (
        isinstance(self.sgd_steps, bool)
        or not isinstance(self.sgd_steps, int)
        or self.sgd_steps < 0
    ):
      raise ValueError(f"sgd_steps must be an int >= 0, got {self.sgd_steps!r}")
    if self.sgd_steps and not self.minibatch_size:
      raise ValueError(
          "sgd_steps deals rows over fixed-size minibatches: set "
          "minibatch_size > 0"
      )
    if self.lr_sched is not None:
      assert isinstance(self.lr_sched, schedules.Schedule), type(self.lr_sched)
      if self.anneal_lr:
        raise ValueError(
            "lr_sched is the step size: set anneal_lr=False (lr / lr_end "
            "unused)"
        )
    if self.minibatch_size and self.shuffle != "flat":
      raise ValueError(
          "minibatch_size cuts the flat (compacted) plies: shuffle must be "
          "'flat'"
      )
    if not isinstance(self.magnet_from_behavior, bool):
      raise ValueError(
          f"magnet_from_behavior is a bool, got {self.magnet_from_behavior!r}"
      )
    if self.magnet_from_behavior and (
        self.loss != "mmd" or int(self.reference_update_every) != 1
    ):
      raise ValueError(
          "magnet_from_behavior: the magnet must be the update's starting "
          "params "
          f"(loss=mmd, reference_update_every=1), got loss={self.loss!r}, "
          f"reference_update_every={self.reference_update_every}"
      )
    if self.value_rows:
      assert (
          isinstance(self.value_rows, str) and ":" in self.value_rows
      ), f"value_rows is a 'module:attr' string, got {self.value_rows!r}"
      if ext is None:
        raise ValueError(
            "value_rows trains value-only rows: a registered loss (its value "
            "mask is separate from the policy mask); pg_loss has one mask"
        )
      if not self.minibatch_size:
        raise ValueError(
            "value_rows needs minibatch_size > 0 (the compacted path whose "
            "padding never evaluates an untrained ply)"
        )
    if self.anneal_lr and self.lr_sched is None and int(self.num_updates) < 1:
      raise ValueError(
          "anneal_lr anneals lr -> lr_end over num_updates: set num_updates >= "
          "1 "
          "(or anneal_lr=False)"
      )

  def teacher_on(self) -> bool:
    """Whether teacher_coef is anything but the off default (constant 0).

    Returns:
      True when the config distils from a teacher.
    """
    return self.teacher_coef != TEACHER_COEF_OFF

  def lr_schedule(self) -> schedules.Schedule:
    """The step size schedule.

    lr_sched when given, else a linear anneal lr -> lr_end over num_updates, or
    constant lr when anneal_lr is off.

    Returns:
      The schedules.Schedule of the step size.
    """
    if self.lr_sched is not None:
      return self.lr_sched
    if not self.anneal_lr:
      return schedules.Schedule("constant", float(self.lr))
    return schedules.Schedule(
        "linear",
        float(self.lr),
        float(self.lr_end),
        decay_steps=float(self.num_updates),
    )


class PGState(NamedTuple):
  """The PG learner's state.

  Attributes:
    params: f32 network params.
    opt_state: the optimizer state.
    magnet: params-shaped (mmd, a registered loss's moving magnet), else None.
    value_ema: f32[1] leaves; zeros when value_ema is off.
    update_step: i64 scalar: outer updates done.
  """

  params: Any  # f32 network params
  opt_state: Any
  # params-shaped (mmd, a registered loss's moving magnet), else None
  magnet: Any
  value_ema: loss_lib.ValueEMA  # f32[1] leaves; zeros when value_ema is off
  update_step: Any  # i64 scalar: outer updates done


# info entries _summarize turns into metrics (telemetry)
_EPISODES = "_episodes"
_EV = "_explained_variance"


class TrainRowsOverflowError(RuntimeError):
  """A batch holds more decisions than PGLearner.train_rows' static size."""


#: Per-module update metrics (update_norm/<module>, update_ratio/<module>).
_UPDATE_MODULE_PREFIXES = ("update_norm/", "update_ratio/")


def _unknown_loss(loss: str) -> ValueError:
  """The error for an unknown PG loss.

  Args:
    loss: The loss name.

  Returns:
    The ValueError.
  """
  return ValueError(f"unknown PG loss {loss!r}: one of {losses()}")


class Prepared(NamedTuple):
  """Per-ply learning targets, f32[B, T], 0 where the seat did not act.

  Attributes:
    advantages: the advantages.
    targets: value targets in reward units (TD(lambda) returns).
    value_old: the behaviour value in reward units (clipped-value baseline).
  """

  advantages: Any
  targets: Any
  value_old: Any


class RowMasks(NamedTuple):
  """Which plies an update trains (prepare_rows).

  The decisions, plus the value_rows option's value-only rows, minus untargeted
  rows.

  Attributes:
    policy_mask: bool[B, T] decisions trained (valid minus untargeted rows).
    value_mask: bool[B, T] rows the value trains on (policy_mask | value
      rows).
    untargeted: i64: rows left without a target (non-finite bootstrap).
    value_row_overlap: i64: value_rows plies where the seat acted (must be 0).
  """

  policy_mask: Any
  value_mask: Any
  untargeted: Any
  value_row_overlap: Any


def _check_f32_leaf(g):
  """Checks an optimizer update leaf is float32.

  Args:
    g: An update leaf.

  Returns:
    The leaf.
  """
  assert (
      g.dtype == F32
  ), f"an optimizer update leaf is {g.dtype}, params are float32"
  return g


def _check_dtype(name: str, x: Any, want: Any) -> None:
  assert (
      jnp.asarray(x).dtype == want
  ), f"TrajectoryBatch.{name}: want {want}, got {x.dtype}"


class Teacher(NamedTuple):
  """A frozen distillation teacher (PGLearner.set_teacher).

  Attributes:
    policy: the teacher's policy (its own spec and forward).
    params: its params, never trained (not in PGState).
    chunk_rows: the most rows one teacher forward call runs on.
    name: what the teacher is (an artifact id), for logs and checks.
  """

  policy: types.Policy
  params: Any
  chunk_rows: int
  name: str


class PGLearner:
  """The policy-gradient Learner (types.Learner) over one JAX Policy.

  Attributes:
    env_max_len: the Env's max_len, or None (__init__).
    value_support: the value head's support (losses.check_value_support).
    cfg: the PGConfig.
  """

  def __init__(
      self,
      policy: types.Policy,
      cfg: PGConfig,
      env_max_len: Optional[int] = None,
  ):
    """Builds the learner.

    Args:
      policy: the policy to train.
      cfg: the PGConfig.
      env_max_len: the Env's max_len (most decisions in one hand), optional; it
        tightens train_rows for self-play batches (every max_len + 1 plies of a
        row hold a hand's observe-only terminal ply).
    """
    assert isinstance(cfg, PGConfig), type(cfg)
    assert env_max_len is None or int(env_max_len) >= 1, env_max_len
    self.env_max_len = None if env_max_len is None else int(env_max_len)
    spec = policy.spec
    if (
        spec.framework != "jax"
        or spec.capability != types.Capability.NATIVE_JAX
    ):
      raise ValueError(
          f"PGLearner trains NATIVE_JAX policies; got {spec.framework}/"
          f"{spec.capability} (a host-driven policy needs its own Learner)"
      )
    if spec.stateful and cfg.shuffle != "env_axis":
      raise ValueError(
          f"{type(policy).__name__} is stateful (recurrent): shuffle must be "
          "'env_axis' so each row's segment is replayed from its init_carry "
          "(a 'flat' shuffle would evaluate plies without their carry)"
      )
    self.value_support = loss_lib.check_value_support(cfg.pg.value_head, policy)
    self._value_rows_fn = None
    if cfg.value_rows:
      if spec.stateful:
        raise ValueError(
            "value_rows evaluates plies on their own: a stateless policy only"
        )
      one = registry.resolve_callable(cfg.value_rows)
      self._value_rows_fn = jax.vmap(
          jax.vmap(lambda info: jnp.asarray(one(info), jnp.bool_))
      )
      self._value_rows_jit = jax.jit(self._value_rows_fn)
    self._lr = cfg.lr_schedule()
    self._policy = policy
    self.cfg = cfg
    self._teacher: Optional[Teacher] = None  # set_teacher (module doc)
    parts = [
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.scale_by_adam(eps=cfg.adam_eps),
    ]
    if cfg.use_adamw:
      parts.append(optax.add_decayed_weights(cfg.weight_decay))
    self._tx = optax.chain(*parts)
    self._prepare_jit = jax.jit(self._prepare)
    self._update_prepared_jit = jax.jit(self._update_prepared)
    self._update_jit = jax.jit(
        self._update, static_argnames=("capacity", "value_rows_cap")
    )
    # value_rows forward sizes (value_rows_cap) used so far (each a program)
    self._value_rows_caps_used: set[int] = set()
    # (B, T, devices) -> (layout, static size, bucket sizes below it): pinned
    # at the first batch of a shape; the buckets each shape has trained at.
    self._train_rows: dict[
        tuple[int, int, int], tuple[str, Optional[int], tuple[int, ...]]
    ] = {}
    self._buckets_used: dict[tuple[int, int, int], set[int]] = {}
    self._every_seat_by_shape: dict[tuple[int, int], bool] = (
        {}
    )  # telemetry (_every_seat)
    self._plan_jit = jax.jit(self._plan, static_argnames=("capacity",))
    if cfg.minibatch_size:
      minibatches = (
          f"at most {cfg.minibatch_size} plies, at least {cfg.sgd_steps} steps"
      )
    else:
      minibatches = cfg.num_minibatches
    logging.info(
        "PGLearner: loss=%s shuffle=%s epochs=%d minibatches=%s policy=%s "
        "value_rows=%s "
        "value_scale=%s schedule_step_offset=%d forward_chunk_rows=%d "
        "train_rows_buckets=%s",
        cfg.loss,
        cfg.shuffle,
        cfg.update_epochs,
        minibatches,
        spec.arch_name,
        cfg.value_rows or "-",
        cfg.value_scale,
        cfg.schedule_step_offset,
        cfg.forward_chunk_rows,
        list(cfg.train_rows_buckets),
    )

  # -- types.Learner -------------------------------------------------------
  def policy(self) -> types.Policy:
    """The policy this learner trains.

    Returns:
      The Policy.
    """
    return self._policy

  def params(self, s: PGState) -> Any:
    """The network params of a learner state.

    Args:
      s: the learner state.

    Returns:
      s.params.
    """
    return s.params

  @property
  def teacher(self) -> Optional[Teacher]:
    """The distillation teacher (set_teacher), or None."""
    return self._teacher

  def set_teacher(
      self,
      policy: types.Policy,
      params: Any,
      *,
      name: str,
      chunk_rows: Optional[int] = None,
  ) -> None:
    """Gives the learner a frozen teacher (module doc "Teacher distillation").

    The teacher is its own policy: its own spec, params and forward. Its params
    are never trained, never the magnet and not in PGState (a checkpoint does
    not hold them; the caller rebuilds the teacher on resume).

    Args:
      policy: the teacher policy: stateless, NATIVE_JAX, the learner's game,
        action count and player count.
      params: its params (placed as the update's other inputs are; a data
        parallel caller replicates them).
      name: what the teacher is (an artifact id).
      chunk_rows: the most rows one teacher forward runs on; None is
        cfg.forward_chunk_rows (so no learner call of any network exceeds it).

    Raises:
      ValueError: the config has no teacher_coef, the policy does not fit the
        learner, or a teacher is already set.
    """
    cfg = self.cfg
    if not cfg.teacher_on():
      raise ValueError(
          "set_teacher: PGConfig.teacher_coef is the off default (constant 0)"
      )
    if self._teacher is not None:
      raise ValueError(
          f"set_teacher: the teacher is already {self._teacher.name}"
      )
    if cfg.shuffle != "flat":
      raise ValueError(
          "set_teacher: the teacher evaluates flat plies (shuffle 'flat')"
      )
    ts, ls = policy.spec, self._policy.spec
    if ts.framework != "jax" or ts.capability != types.Capability.NATIVE_JAX:
      raise ValueError(
          f"teacher {name}: a NATIVE_JAX policy, got {ts.framework}/"
          f"{ts.capability}"
      )
    if ts.stateful:
      raise ValueError(f"teacher {name}: a stateless policy (flat plies)")
    if (ts.game, ts.num_actions, ts.num_players) != (
        ls.game,
        ls.num_actions,
        ls.num_players,
    ):
      raise ValueError(
          f"teacher {name} plays {ts.game} ({ts.num_actions} actions, "
          f"{ts.num_players} players), the learner {ls.game} "
          f"({ls.num_actions} actions, {ls.num_players} players)"
      )
    chunk_rows = int(
        cfg.forward_chunk_rows if chunk_rows is None else chunk_rows
    )
    if chunk_rows < 1:
      raise ValueError(f"teacher chunk_rows must be >= 1, got {chunk_rows}")
    for leaf in jax.tree_util.tree_leaves(params):
      if jnp.issubdtype(leaf.dtype, jnp.floating):
        assert (
            leaf.dtype == F32
        ), f"teacher params are float32, got {leaf.dtype}"
    self._teacher = Teacher(
        policy=policy, params=params, chunk_rows=chunk_rows, name=str(name)
    )
    logging.info(
        "PGLearner: teacher %s (%s), chunk_rows %d, teacher_coef %s",
        name,
        ts.arch_name,
        chunk_rows,
        cfg.teacher_coef,
    )

  def init(self, key: Any, env: Optional[types.Env] = None) -> PGState:
    """A fresh learner state from freshly initialised params.

    Args:
      key: the PRNG key of the params.
      env: unused (the Learner protocol).

    Returns:
      The PGState.
    """
    x64.assert_x64("PGLearner.init")
    if env is not None:
      spec = self._policy.spec
      assert (env.num_actions, env.num_players) == (
          spec.num_actions,
          spec.num_players,
      ), (env.num_actions, env.num_players, spec)
    return self.init_from_params(self._policy.init_params(key))

  def init_from_params(self, params: Any) -> PGState:
    """A fresh learner state around given params (a warm start).

    Args:
      params: the network params.

    Returns:
      The PGState.
    """
    x64.assert_x64("PGLearner.init_from_params")
    for leaf in jax.tree_util.tree_leaves(params):
      if jnp.issubdtype(leaf.dtype, jnp.floating):
        assert leaf.dtype == F32, f"params are float32, got a {leaf.dtype} leaf"
    return PGState(
        params=params,
        opt_state=self._tx.init(params),
        magnet=params if self._has_magnet else None,
        value_ema=loss_lib.value_ema_init(),
        update_step=jnp.asarray(0, I64),
    )

  def update(
      self, s: PGState, batch: types.TrajectoryBatch, key: Any
  ) -> tuple[PGState, dict[str, float]]:
    """One outer update.

    Metrics: means over every minibatch of every epoch, plus lr (the step size
    this update applied), update_step (after), grad_norm (pre-clip), the value
    EMA moments, batch counts, train_rows (plies trained per epoch) and
    train_rows_headroom (train_rows minus the batch's decisions, >= 0).
    Telemetry (computed in the same jitted update, fetched with the rest):
    grad_norm_max, grad_norm_clipped (the applied norm, min(grad_norm,
    max_grad_norm)), grad_clipped (the fraction of minibatch steps clipped),
    grad_norm/<module>, update_norm and update_ratio (the step's norm over the
    params' before it) and per module update_norm/<module> and
    update_ratio/<module>, approx_kl_max (the largest minibatch approx_kl of
    the update, beside its mean approx_kl), param_norm and param_norm/<module>
    (after the update),
    param_change_norm / _ratio (the whole update), episode_return_mean / _std,
    episode_return_mean/seat<p> and episode_length_mean (hands completed in the
    batch, telemetry.episode_sums; on a batch of every seat of each game the
    mean over rows, and reward_per_row, are the game's total payoff and are left
    out), value_explained_var (behaviour values vs the value targets over the
    decisions), and the policy's game's batch metrics when the game registers
    some (telemetry.register_batch_metrics). Loss terms that are off are left
    out (_not_computed). With an SGD step gate (module doc) the loss metrics
    are means over the steps that computed a loss, plus sgd_steps_taken,
    sgd_steps_planned, <name>_early_stopped and <name>_stop_step. Raises
    FloatingPointError on a non-finite loss term or gradient norm,
    TrainRowsOverflowError when the decisions exceed train_rows.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the PRNG key of the shuffle.

    Returns:
      (new state, metrics dict).

    Raises:
      ValueError: the batch does not fit the config (check_batch).
    """
    x64.assert_x64("PGLearner.update")
    self.check_batch(batch)
    capacity = self.train_rows(batch)
    vr_cap = self.value_rows_cap(batch)
    if self._teacher is None:
      if self.cfg.teacher_on():
        raise ValueError(
            "PGConfig.teacher_coef distils from a teacher: call set_teacher "
            "before update"
        )
      s2, per_mb, info = self._update_jit(
          s, batch, key, capacity=capacity, value_rows_cap=vr_cap
      )
    else:
      s2, per_mb, info = self._update_jit(
          s,
          batch,
          key,
          capacity=capacity,
          teacher_params=self._teacher.params,
          value_rows_cap=vr_cap,
      )
    overlap = int(info.pop("value_row_overlap"))
    if overlap:
      raise ValueError(
          f"PGLearner update {int(s.update_step)}: {overlap} value_rows plies "
          "are "
          "plies where the seat acted (a value row is a non-decision row)"
      )
    untargeted = int(info["untargeted_rows"])
    if untargeted and not self.cfg.drop_unbootstrapped:
      raise ValueError(
          f"PGLearner update {int(s.update_step)}: {untargeted} rows' targets "
          "read a "
          "non-finite bootstrap value (ply T outside the network's domain); "
          "set "
          "drop_unbootstrapped=True to leave them out"
      )
    if self.cfg.minibatch_size and int(info["trained_rows"]) == 0:
      raise ValueError(
          f"PGLearner update {int(s.update_step)}: the batch holds no row to "
          "train on"
      )
    out = self._summarize(s, s2, per_mb, info)
    if self._every_seat(batch):
      # the game's total payoff over seats, not a signal
      for k in telemetry.ROW_MEAN_KEYS:
        out.pop(k, None)
    size = batch.valid.size if capacity is None else capacity
    out["train_rows"] = float(size)
    out["train_rows_headroom"] = float(size - self._n_train(batch))
    if self.cfg.train_rows_buckets:
      out["train_rows_bound"] = float(self.train_rows_bound(batch))
    return s2, out

  def train_rows(self, batch: types.TrajectoryBatch) -> Optional[int]:
    """The plies the flat shuffle trains on.

    The plies the flat shuffle trains on: None = all B x T in time-major order,
    or C < B x T = the batch's valid plies (decisions) gathered first and
    padded with invalid plies to C. The masked loss means over valid plies
    are unchanged; only which empty plies pad a minibatch changes. env_axis
    (recurrent) keeps whole segments: None.

    C is static: computed from the batch shape, the device
    count, the layout and the config at the first batch of each shape and
    pinned for that shape, so the update compiles once per batch shape and
    the valid count never picks the program. A batch whose decisions exceed
    C raises TrainRowsOverflowError (logged first): nothing is dropped.

    The bound. With env_max_len L a hand has at most L decisions followed by
    its observe-only terminal ply, so every L + 1 consecutive plies of an
    Arena row hold a terminal ply and a perspective row has at most
    D = T - floor(T / (L + 1)) decisions (D = T without L). Layouts:
      grouped  every seat of an Arena row is a perspective (TrajectoryBatch
               rows b*P + m: seat m, the same game, so the same to_move at
               every ply, and at most one of them valid per ply: one seat
               acts): at most (B / P) x D decisions.
      seat     any other batch (one perspective seat per Arena row, a
               learner seat vs opponents): at most B x D, tightened to
               ceil(max_decision_share x B x T) when the config declares a
               share (a declaration, not a proof: breaking it raises).
    C is rounded up to num_minibatches x 128 (x lcm(128, devices) / 128 when
    the device count does not divide 128), so for device counts that
    divide 128 C, and with it the minibatches, is the same at 1 and N
    devices; C >= B x T gives None.
    The layout is read from the first batch of a shape (a seat batch is
    never mistaken for grouped unless all its row pairs play identical
    games); a later batch of that shape is held to the pinned C.

    Buckets (PGConfig.train_rows_buckets): each share s gives the size
    ceil(s x B x T) rounded up to the same quantum; the sizes below C are
    pinned with C, and a batch returns the smallest of them that holds its
    decisions (C when none does). The returned size picks the compiled
    update, so a shape has at most len(buckets) + 1 programs; each bucket's
    first use is logged.

    Args:
      batch: the TrajectoryBatch.

    Returns:
      None (every ply) or the static size C the batch trains at.

    Raises:
      TrainRowsOverflowError: the batch's decisions exceed C.
    """
    if self.cfg.shuffle != "flat":
      return None
    b, t = batch.valid.shape
    valid = np.asarray(batch.valid)
    layout, c, sizes, shape_key = self._pinned(batch, valid)
    size = b * t if c is None else c
    n_valid = self._n_train(batch)
    if n_valid > size:
      msg = (
          f"PGLearner.train_rows: {n_valid} decisions in a [{b}, {t}] batch "
          "overflow the "
          f"static size {size} ({layout} layout, "
          f"env_max_len={self.env_max_len}, "
          f"max_decision_share={self.cfg.max_decision_share}) by "
          f"{n_valid - size}; "
          "refusing to drop decisions or recompile (raise max_decision_share "
          "or unset it)"
      )
      logging.error(msg)
      raise TrainRowsOverflowError(msg)
    pick = next((x for x in sizes if x >= n_valid), c)
    used = self._buckets_used.setdefault(shape_key, set())
    if pick not in used:
      used.add(pick)
      logging.info(
          "PGLearner.train_rows: batch [%d, %d]: first update at %s plies (%d "
          "decisions; bucket %d of %d)",
          b,
          t,
          "all" if pick is None else pick,
          n_valid,
          len(used),
          len(sizes) + 1,
      )
    return pick

  def value_rows_cap(self, batch: types.TrajectoryBatch) -> Optional[int]:
    """The static row count of the update's value_rows forward, or None.

    The behaviour values of value-only plies (PGConfig.value_rows) come from
    one learner forward at the update's starting params. Only those plies are
    evaluated: gathered (in time-major order, padded with copies of them) into
    the smallest VALUE_ROWS_CAPS size that holds the batch's count, read on
    the host here, so the program never truncates one. Each size is its own
    compiled update (first use logged); the sizes are about 1.33x apart, so a
    run whose value-row count is stable uses one or two. None: no value_rows,
    or a size of at least B x T (then every ply is evaluated, as before).
    Rows are independent, so the values equal the every-ply forward's up to
    floating-point summation order.

    Args:
      batch: the TrajectoryBatch.

    Returns:
      The size, or None.
    """
    if self._value_rows_fn is None:
      return None
    valid = np.asarray(batch.valid)
    vr = np.asarray(self._value_rows_jit(batch.steps.info)) & ~valid
    k = int(vr.sum(dtype=np.int64))
    n = int(valid.size)
    cap = next((c for c in VALUE_ROWS_CAPS if c >= k), None)
    if cap is None or cap >= n:
      return None
    if cap not in self._value_rows_caps_used:
      self._value_rows_caps_used.add(cap)
      logging.info(
          "PGLearner.value_rows_cap: batch %s: %d value-only plies, forward "
          "size %d of %d plies (size %d of this run)",
          tuple(valid.shape),
          k,
          cap,
          n,
          len(self._value_rows_caps_used),
      )
    return cap

  def train_rows_bound(self, batch: types.TrajectoryBatch) -> int:
    """The static size of `batch`'s shape (the top bucket).

    The most decisions a batch of that shape may hold (B x T when every ply
    trains).

    Args:
      batch: the TrajectoryBatch.

    Returns:
      The size as an int.
    """
    b, t = batch.valid.shape
    if self.cfg.shuffle != "flat":
      return int(b * t)
    _, c, _, _ = self._pinned(batch, np.asarray(batch.valid))
    return int(b * t if c is None else c)

  def _pinned(self, batch, valid: np.ndarray):
    """(layout, static size, bucket sizes, shape key) of batch's shape.

    Pinned at the shape's first batch.

    Args:
      batch: the TrajectoryBatch.
      valid: its host valid mask, bool[B, T].

    Returns:
      (layout, static size, bucket sizes, shape key).
    """
    b, t = valid.shape
    shape_key = (int(b), int(t), int(dp_lib.active_batch_size()))
    pinned = self._train_rows.get(shape_key)
    if pinned is None:
      layout = "grouped" if self._grouped(batch, valid) else "seat"
      c = self._static_rows(b, t, layout)
      pinned = (layout, c, self._bucket_rows(b, t, c))
      self._train_rows[shape_key] = pinned
      logging.info(
          "PGLearner.train_rows: batch [%d, %d] on %d device(s): %s layout, %s "
          "plies trained per epoch (of %d); buckets below it %s",
          b,
          t,
          shape_key[2],
          layout,
          "all" if c is None else c,
          b * t,
          list(pinned[2]),
      )
    return pinned + (shape_key,)

  def _bucket_rows(self, b: int, t: int, c: Optional[int]) -> tuple[int, ...]:
    """The bucket sizes of a [b, t] batch below its static size c.

    c None: every ply. Ascending, each rounded up to the static size's quantum.

    Args:
      b: the batch rows.
      t: the batch plies.
      c: the static size, or None (every ply).

    Returns:
      The bucket sizes, ascending.
    """
    n = b * t
    top = n if c is None else c
    if self.cfg.minibatch_size:
      # The fixed minibatch path (_static_rows): a bucket is a whole number
      # of minibatch slots, at least sgd_steps of them (every step of the
      # plan exists), as the static size is.
      size = int(self.cfg.minibatch_size)
      sizes = {
          max(
              -(-max(math.ceil(float(s) * n), 1) // size),
              int(self.cfg.sgd_steps),
          )
          * size
          for s in self.cfg.train_rows_buckets
      }
      return tuple(sorted(x for x in sizes if x < top))
    q = self.cfg.num_minibatches * math.lcm(128, dp_lib.active_batch_size())
    sizes = {
        -(-math.ceil(float(s) * n) // q) * q
        for s in self.cfg.train_rows_buckets
    }
    return tuple(sorted(x for x in sizes if x < top))

  def _every_seat(self, batch: types.TrajectoryBatch) -> bool:
    """Whether the batch's rows are every seat of each game.

    The pinned grouped layout (flat shuffle), else read once per batch shape (no
    per-update host copy of the seats).

    Args:
      batch: the TrajectoryBatch.

    Returns:
      Whether every seat of each game is a row.
    """
    b, t = batch.valid.shape
    pinned = self._train_rows.get(
        (int(b), int(t), int(dp_lib.active_batch_size()))
    )
    if pinned is not None:
      return pinned[0] == "grouped"
    key = (int(b), int(t))
    if key not in self._every_seat_by_shape:
      self._every_seat_by_shape[key] = telemetry.every_seat_rows(
          batch.steps.seat, self._policy.spec.num_players
      )
    return self._every_seat_by_shape[key]

  def _n_train(self, batch: types.TrajectoryBatch) -> int:
    """Host count of the rows an update can train: decisions plus value_rows.

    Args:
      batch: The batch.

    Returns:
      The trainable row count.
    """
    valid = np.asarray(batch.valid)
    if self._value_rows_fn is None:
      return int(valid.sum(dtype=np.int64))
    vr = np.asarray(self._value_rows_jit(batch.steps.info))
    return int((valid | vr).sum(dtype=np.int64))

  def _grouped(self, batch: types.TrajectoryBatch, valid: np.ndarray) -> bool:
    """Whether every P consecutive rows are the P seats of one Arena row.

    Args:
      batch: The batch.
      valid: bool[B, T], the decision plies.

    Returns:
      True for the grouped layout.
    """
    b, t = valid.shape
    p = self._policy.spec.num_players
    if b % p:
      return False
    seat = np.asarray(batch.steps.seat).reshape(b // p, p, t)
    if not (seat == np.arange(p, dtype=seat.dtype)[None, :, None]).all():
      return False
    to_move = np.asarray(batch.steps.to_move).reshape(b // p, p, t)
    if not (to_move == to_move[:, :1]).all():
      return False
    return int(valid.reshape(b // p, p, t).sum(axis=1).max(initial=0)) <= 1

  def _static_rows(self, b: int, t: int, layout: str) -> Optional[int]:
    """The static size of a [b, t] batch of `layout` (train_rows).

    Args:
      b: The rows.
      t: The plies.
      layout: The row layout.

    Returns:
      The static row count, or None for the whole batch.

    Raises:
      ValueError: minibatch_size is not divisible by the device count.
    """
    n = b * t
    p = self._policy.spec.num_players
    q = self.cfg.num_minibatches * math.lcm(128, dp_lib.active_batch_size())
    per_row = t - (t // (self.env_max_len + 1) if self.env_max_len else 0)
    if self._value_rows_fn is not None:
      # a seat's value row shares its ply with the other seat's decision
      c = b * per_row
    elif layout == "grouped":
      c = (b // p) * per_row
    elif layout == "seat":
      c = b * per_row
    else:
      raise ValueError(f"unknown layout {layout!r}")
    if self.cfg.max_decision_share is not None:
      c = min(c, math.ceil(float(self.cfg.max_decision_share) * n))
    if self.cfg.minibatch_size:
      # Always compacted (padding = masked copies of trained rows), C a
      # multiple of the minibatch size and of the device count; C may exceed
      # B x T (the excess slots are padding).
      size = int(self.cfg.minibatch_size)
      devices = dp_lib.active_batch_size()
      if size % devices:
        raise ValueError(
            f"minibatch_size {size} is not divisible by the {devices} devices"
        )
      c = max(min(c, n), 1)
      # at least sgd_steps minibatch slots, so every step of the plan exists
      return max(-(-c // size), int(self.cfg.sgd_steps)) * size
    c = -(-c // q) * q
    return None if c >= n else c

  def state_dict(self, s: PGState) -> dict[str, Any]:
    """{params, opt_state (leaf list), value_ema, update_step[, magnet]}.

    Args:
      s: the learner state.

    Returns:
      The checkpoint blob. No value is None.
    """
    blob = dict(
        params=s.params,
        opt_state=list(jax.tree_util.tree_leaves(s.opt_state)),
        value_ema=dict(s.value_ema._asdict()),
        update_step=s.update_step,
    )
    if s.magnet is not None:
      blob["magnet"] = s.magnet
    return blob

  def load_state(self, blob: Mapping[str, Any]) -> PGState:
    """Inverse of state_dict; raises on any tree, shape or dtype mismatch.

    Args:
      blob: a state_dict() blob.

    Returns:
      The PGState.

    Raises:
      ValueError: a tree, shape or dtype mismatch.
    """
    want = self.init_from_params(blob["params"])
    opt_def = jax.tree_util.tree_structure(want.opt_state)
    opt_leaves = list(blob["opt_state"])
    if len(opt_leaves) != opt_def.num_leaves:
      raise ValueError(
          f"opt_state has {len(opt_leaves)} leaves, optimizer wants "
          f"{opt_def.num_leaves}"
      )
    for got, ref in zip(opt_leaves, jax.tree_util.tree_leaves(want.opt_state)):
      if (got.shape, got.dtype) != (ref.shape, ref.dtype):
        raise ValueError(
            f"opt_state leaf {got.dtype}{got.shape} != {ref.dtype}{ref.shape}"
        )
    step = jnp.asarray(blob["update_step"])
    if step.dtype != I64 or step.shape:
      raise ValueError(
          f"update_step must be an int64 scalar, got {step.dtype}{step.shape}"
      )
    ema = loss_lib.ValueEMA(
        **{k: jnp.asarray(v) for k, v in blob["value_ema"].items()}
    )
    for leaf in ema:
      if (leaf.shape, leaf.dtype) != ((1,), F32):
        raise ValueError(
            f"value_ema leaf {leaf.dtype}{leaf.shape} != float32(1,)"
        )
    magnet = blob.get("magnet")
    if (magnet is not None) != self._has_magnet:
      raise ValueError(
          f"loss={self.cfg.loss}: magnet "
          f"{'missing' if magnet is None else 'unexpected'}"
      )
    if magnet is not None:
      if jax.tree_util.tree_structure(magnet) != jax.tree_util.tree_structure(
          blob["params"]
      ):
        raise ValueError("magnet tree differs from params")
    return PGState(
        params=blob["params"],
        opt_state=jax.tree_util.tree_unflatten(opt_def, opt_leaves),
        magnet=magnet,
        value_ema=ema,
        update_step=step,
    )

  # -- metrics that are settings ------------------------------------------
  def held_metrics(self) -> tuple[str, ...]:
    """update() metrics that are run settings, not measurements.

    update() metrics that are run settings, not measurements: the applied
    step size and coefficients (a flat schedule's value) and train_rows
    (static, or the bucket an update trained at under train_rows_buckets,
    which then moves and is written as a series). The league runner records them
    in the run's Params and writes them as a series only once they move
    (metrics.MetricsWriter held); update() still returns them.

    Returns:
      The metric names.
    """
    held = ("lr", "ent_coef_effective", "kl_coef_effective", "train_rows")
    if self.cfg.teacher_on():
      held += ("teacher_coef",)
    return held

  def _not_computed(self) -> tuple[str, ...]:
    """Loss metrics whose term is off in this config.

    Loss metrics whose term is off in this config (a constant, not a
    measurement): value clipping off (or an HL-Gauss value head), the
    advantage filter off. update() leaves them out; update_prepared keeps
    every loss metric.

    Returns:
      The metric names.
    """
    pg = self.cfg.pg
    out = []
    if self.cfg.loss in PG_LOSSES:
      if not pg.clip_value or pg.value_head == "hl_gauss":
        out.append("clip_frac_value")
      if not pg.use_advantage_filter:
        out += ["advantage_filter_keep_fraction", "advantage_filter_fallback"]
    return tuple(out)

  # -- public pieces (tests and analysis) ---------------------------------
  def lr_at(self, update_step: Any) -> jax.Array:
    """f64 step size of update `update_step` (int64): cfg.lr_schedule().

    Args:
      update_step: the update counter (int64).

    Returns:
      The f64 step size.
    """
    u = jnp.asarray(update_step)
    assert u.dtype == I64, f"update_step is int64, got {u.dtype}"
    return self._lr.value(self._sched_step(u))

  def temperature_at(self, update_step: Any) -> jax.Array:
    """f32 magnet temperature of a registered loss at `update_step`.

    Args:
      update_step: the update counter (int64).

    Returns:
      The f32 temperature.
    """
    u = jnp.asarray(update_step)
    assert u.dtype == I64, f"update_step is int64, got {u.dtype}"
    ext = extension(self.cfg.loss)
    assert ext is not None, f"loss {self.cfg.loss!r} has no magnet temperature"
    return self._temperature(ext, self._sched_step(u))

  def step_coefs_at(self, update_step: Any) -> Optional[dict[str, Any]]:
    """A registered loss's LossSpec.step_coefs at update `update_step`.

    As the update hands them to its loss.

    Args:
      update_step: the update step (int64).

    Returns:
      The coefficients, or None when the loss declares none.
    """
    u = jnp.asarray(update_step)
    assert u.dtype == I64, f"update_step is int64, got {u.dtype}"
    ext = extension(self.cfg.loss)
    assert ext is not None, f"loss {self.cfg.loss!r} is not a registered loss"
    return self._step_coefs(ext, self._sched_step(u))

  def _step_coefs(self, ext: LossSpec, step: Any) -> Optional[dict[str, Any]]:
    """The registered loss's step coefficients at `step`, or None.

    Args:
      ext: The loss spec.
      step: The schedule step.

    Returns:
      {name: coefficient}, or None when the loss has none.
    """
    if ext.step_coefs is None:
      return None
    return ext.step_coefs(
        self._loss_config(ext),
        step,
        self.cfg.temperature_floor_step,
        self.cfg.schedule_time_scale,
    )

  def _step_gate(self):
    """The SGD step gate of this config (module doc), or None.

    Returns:
      gate(step_metrics) -> bool, or None.
    """
    ext = extension(self.cfg.loss)
    if ext is None or ext.step_gate is None:
      return None
    gate = ext.step_gate(self._loss_config(ext))
    assert gate is None or callable(gate), f"step_gate gave {gate!r}"
    return gate

  def _loss_config(self, ext: LossSpec) -> Any:
    """The loss's config: the configured one, else its default.

    Args:
      ext: The loss spec.

    Returns:
      The config.
    """
    return (
        ext.config_type()
        if self.cfg.loss_config is None
        else self.cfg.loss_config
    )

  def _temperature(self, ext: LossSpec, step: Any) -> jax.Array:
    """The loss's temperature at `step`.

    Args:
      ext: The loss spec.
      step: The schedule step.

    Returns:
      The temperature.
    """
    return ext.temperature(
        self._loss_config(ext),
        step,
        self.cfg.temperature_floor_step,
        self.cfg.schedule_time_scale,
    )

  def _sched_step(self, u: Any) -> jax.Array:
    """The counter schedules read: update_step + schedule_step_offset, int64.

    Args:
      u: The update step.

    Returns:
      The schedule step.
    """
    return u + jnp.int64(self.cfg.schedule_step_offset)

  def check_batch(self, batch: types.TrajectoryBatch) -> None:
    """Host-side dtype / shape / divisibility checks.

    Args:
      batch: the TrajectoryBatch.

    Raises:
      ValueError: the first failed check.
    """
    assert isinstance(batch, types.TrajectoryBatch), type(batch)
    side_marked = types.SIDE_CREDIT in batch.labels
    if side_marked != (self.cfg.credit == "side"):
      raise ValueError(
          f"PGConfig.credit={self.cfg.credit!r}, but the batch's reward is "
          f"the {'side' if side_marked else 'seat'}'s (label "
          f"{types.SIDE_CREDIT!r} {'present' if side_marked else 'absent'}); "
          "the source must emit the credit the learner is configured for "
          "(core.arena Arena credit)"
      )
    b, t = batch.valid.shape
    for name, want in (
        ("reward", F32),
        ("behavior_logp", F32),
        ("behavior_logits", F32),
        ("value_pred", F32),
        ("valid", jnp.bool_),
        ("done", jnp.bool_),
        ("action", jnp.int32),
    ):
      _check_dtype(name, getattr(batch, name), want)
    _check_dtype("steps.legal_mask", batch.steps.legal_mask, jnp.bool_)
    _check_dtype("steps.seat", batch.steps.seat, jnp.int32)
    for name in ("reward", "behavior_logp", "done", "action"):
      assert getattr(batch, name).shape == (b, t), (
          name,
          getattr(batch, name).shape,
      )
    a = self._policy.spec.num_actions
    assert batch.steps.legal_mask.shape == (
        b,
        t,
        a,
    ), batch.steps.legal_mask.shape
    assert batch.value_pred.shape == (
        b,
        t,
        self._policy.spec.num_players,
    ), batch.value_pred.shape
    nm = self.cfg.num_minibatches
    if (
        self.cfg.shuffle == "flat"
        and not self.cfg.minibatch_size
        and (b * t) % nm
    ):
      raise ValueError(
          f"flat shuffle: {b} rows x {t} plies = {b * t} is not divisible by "
          f"num_minibatches={nm}"
      )
    if self.cfg.shuffle == "env_axis" and b % nm:
      raise ValueError(
          f"env_axis shuffle: {b} rows is not divisible by num_minibatches={nm}"
      )

  def prepare(self, s: PGState, batch: types.TrajectoryBatch) -> Prepared:
    """The update's targets (prepare_rows without the RowMasks).

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.

    Returns:
      The Prepared targets.
    """
    return self.prepare_rows(s, batch)[0]

  def prepare_rows(
      self, s: PGState, batch: types.TrajectoryBatch
  ) -> tuple[Prepared, RowMasks]:
    """prepare, with the RowMasks of the plies the update trains.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.

    Returns:
      (Prepared, RowMasks).
    """
    self.check_batch(batch)
    return self._prepare_jit(s, batch)

  def update_prepared(
      self, s: PGState, batch: types.TrajectoryBatch, prep: Prepared, key: Any
  ) -> tuple[PGState, dict[str, jax.Array]]:
    """Step 2-3 on given targets: (state, per-minibatch metrics f32[E, M]).

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      prep: the Prepared targets.
      key: the PRNG key of the shuffle.

    Returns:
      (state, per-minibatch metrics f32[E, M]).

    Raises:
      NotImplementedError: a configuration the prepared path does not support.
    """
    if self.cfg.minibatch_size:
      raise NotImplementedError(
          "minibatch_size compacts to train_rows: use update()"
      )
    if self.cfg.teacher_on():
      raise NotImplementedError("teacher distillation: use update()")
    self.check_batch(batch)
    for x in prep:
      assert x.dtype == F32 and x.shape == batch.valid.shape, (x.dtype, x.shape)
    return self._update_prepared_jit(s, batch, prep, key)

  def sgd_plan(
      self, s: PGState, batch: types.TrajectoryBatch, key: Any
  ) -> list[np.ndarray]:
    """The first epoch's SGD steps of update(s, batch, key).

    The first epoch's SGD steps of update(s, batch, key) (fixed
    minibatch size): per step taken, in order, its trained plies as
    (row, ply) pairs i64[k, 2] in slot order, for inspecting which plies
    each step trains.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the update's PRNG key.

    Returns:
      Per step, i64[k, 2] (row, ply) pairs.

    Raises:
      NotImplementedError: the config has no fixed minibatch size.
    """
    if not self.cfg.minibatch_size:
      raise NotImplementedError(
          "sgd_plan: the fixed minibatch path (minibatch_size > 0) only"
      )
    self.check_batch(batch)
    cap = self.train_rows(batch)
    idx, real = jax.device_get(self._plan_jit(s, batch, key, capacity=cap))
    b = int(batch.valid.shape[0])
    out = []
    for i, r in zip(np.asarray(idx, np.int64), np.asarray(real, bool)):
      if r.any():
        out.append(np.stack([i[r] % b, i[r] // b], axis=1).astype(np.int64))
    return out

  def forward(self, params: Any, carry: Any, steps: types.SeatStep):
    """The learner's forward of the policy over SeatSteps.

    Args:
      params: the network params.
      carry: the rows' init_carry ([Bm, T] steps only).
      steps: SeatSteps [N] (flat; stateless policies; `carry` is ignored and a
        fresh carry is used) or [Bm, T] (evaluate_seq from `carry`).

    Returns:
      (logits f32[..., A], seat value f32[...], value_logits or None).

    Raises:
      ValueError: steps of another rank, or [N] steps for a stateful policy.
    """
    pol = self._policy
    seat = jnp.asarray(steps.seat)
    if seat.ndim == 1:
      if pol.spec.stateful:
        raise ValueError(
            "a stateful policy is evaluated over [B, T] segments only"
        )

      def _run(st):
        """Evaluates rows from their initial carry.

        Args:
          st: The rows' SeatSteps.

        Returns:
          The PolicyOutput.
        """
        s_ = jnp.asarray(st.seat)
        return pol.evaluate(
            params, pol.init_carry(params, s_.shape[0], s_, None), st, None
        )

      out = self._chunked(_run, steps)
    elif seat.ndim == 2:
      out = self._chunked(
          lambda c, st: pol.evaluate_seq(params, c, st, None)[1], carry, steps
      )
    else:
      raise ValueError(f"steps must be [N] or [B, T], got seat {seat.shape}")
    assert out.logits.dtype == F32 and out.value.dtype == F32, (
        out.logits.dtype,
        out.value.dtype,
    )
    value = jnp.take_along_axis(
        out.value, seat[..., None].astype(jnp.int32), axis=-1
    )[..., 0]
    return out.logits, value, out.aux.get("value_logits")

  def _chunked(self, fn, *trees, chunk: Optional[int] = None):
    """fn(*trees) over the trees' shared row axis, in row chunks.

    fn(*trees) over the trees' shared leading row axis, at most
    cfg.forward_chunk_rows rows per call (module doc "Forward chunking"):
    one call when they fit, else equal chunks (actor_critic.chunk_plan; pad
    rows repeat row 0 and are dropped) under a rematerialized lax.scan.
    Chunk k holds rows k, k + num, k + 2 num, ... (strided), so a row-sharded
    batch (data parallel) keeps every chunk spread over all its devices.

    Args:
      fn: the function of the row-batched trees.
      *trees: pytrees sharing a leading row axis.
      chunk: the most rows per call; None is cfg.forward_chunk_rows (the
        teacher's forward passes its own).

    Returns:
      fn(*trees), rows in their input order.
    """
    n = {int(x.shape[0]) for x in jax.tree_util.tree_leaves(trees)}
    if len(n) != 1:
      raise ValueError(
          f"_chunked: the trees disagree on their row count {sorted(n)}"
      )
    n = n.pop()
    chunk = int(self.cfg.forward_chunk_rows if chunk is None else chunk)
    if n <= chunk:
      return fn(*trees)
    num, c = actor_critic.chunk_plan(n, chunk)
    assert c <= chunk and num * c >= n, (n, chunk, num, c)
    pad = num * c - n

    def _split(x):
      """Pads and splits rows into `num` chunks.

      Args:
        x: A leaf, [rows, ...].

      Returns:
        The chunks, leading axis num.
      """
      x = jnp.asarray(x)
      if pad:
        x = jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])
      return jnp.swapaxes(x.reshape((c, num) + x.shape[1:]), 0, 1)

    def _body(carry, xs):
      """Runs `fn` on one chunk.

      Args:
        carry: The scan carry, None (unused).
        xs: One chunk of the arguments.

      Returns:
        (None, fn's output).
      """
      del carry
      return None, fn(*xs)

    _, out = jax.lax.scan(
        jax.checkpoint(_body, prevent_cse=False), None, tree_map(_split, trees)
    )
    return tree_map(
        lambda y: jnp.swapaxes(y, 0, 1).reshape((num * c,) + y.shape[2:])[:n],
        out,
    )

  # -- internals -----------------------------------------------------------
  @property
  def _has_magnet(self) -> bool:
    loss = self.cfg.loss
    if loss == "mmd":
      return True
    if loss == "ppo":
      return False
    extension(loss)  # an unknown loss raises
    return self.cfg.magnet_every > 0

  def _bootstrap_value(self, params, batch) -> jax.Array:
    """f32[B]: the seat's value at bootstrap_step.

    The carry is rebuilt over the segment. evaluate gets a fixed key: PG
    policies are deterministic nets.

    Args:
      params: the network params.
      batch: the TrajectoryBatch.

    Returns:
      The seat's values at bootstrap_step, f32[B].
    """
    pol = self._policy

    def _run(init_carry, steps, bs):  # rows of the batch (_chunked)
      """The bootstrap values of a chunk of rows.

      Args:
        init_carry: The rows' initial carry.
        steps: The rows' SeatSteps.
        bs: The bootstrap SeatSteps.

      Returns:
        The values.
      """
      carry = pol.observe_seq(params, init_carry, steps)
      carry = pol.observe(params, carry, bs)
      return pol.evaluate(params, carry, bs, jax.random.key(0)).value

    bs = batch.bootstrap_step
    value = self._chunked(_run, batch.init_carry, batch.steps, bs)
    seat = jnp.asarray(bs.seat).astype(jnp.int32)
    return jnp.take_along_axis(value, seat[:, None], axis=-1)[:, 0].astype(F32)

  def _value_row_values(
      self, params, batch, vr=None, cap: Optional[int] = None
  ) -> jax.Array:
    """f32[B, T]: the network's own seat value at the value_rows plies.

    cap None: a flat forward over every ply (chunked like every learner
    forward). cap C: only the plies of `vr` (value_rows_cap: C holds them
    all), gathered time-major into C rows padded with copies of them, are
    evaluated and scattered back; every other ply is 0. Read at value_rows
    plies only; a NaN elsewhere is never selected.

    Args:
      params: the network params.
      batch: the TrajectoryBatch.
      vr: bool[B, T], the value-only plies (cap given).
      cap: the static forward size (value_rows_cap), or None.

    Returns:
      The seat values, f32[B, T].
    """
    b, t = batch.valid.shape
    n = b * t
    flat = tree_map(lambda x: x.reshape((n,) + x.shape[2:]), batch.steps)
    if cap is None:
      _, value, _ = self.forward(params, None, flat)
      return jax.lax.stop_gradient(value.reshape(b, t)).astype(F32)
    assert vr is not None and 0 < cap < n, (cap, n)
    tm = self._time_major(vr)  # [T * B]: ply p = (p % B, p // B)
    order = jnp.argsort(jnp.logical_not(tm), stable=True)[:cap]
    k = jnp.sum(tm, dtype=jnp.int32)
    pos = jnp.arange(cap, dtype=jnp.int32)
    real = pos < k
    src = jnp.take(order, jnp.where(real, pos, pos % jnp.maximum(k, 1)))
    rows = (src % b) * t + src // b  # row-major index into flat
    steps = dp_lib.constrain_rows(
        tree_map(lambda x: jnp.take(x, rows, axis=0), flat)
    )
    _, value, _ = self.forward(params, None, steps)
    out = (
        jnp.zeros((n,), F32)
        .at[jnp.where(real, rows, n)]
        .set(value.astype(F32), mode="drop")
    )
    return jax.lax.stop_gradient(out.reshape(b, t))

  def _prepare(
      self, s: PGState, batch, value_rows_cap: Optional[int] = None
  ) -> tuple[Prepared, RowMasks]:
    """Step 1: the per-ply targets and the masks of the plies the update trains.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      value_rows_cap: the value_rows forward size (value_rows_cap), or None
        (every ply).

    Returns:
      (Prepared, RowMasks).
    """
    cfg = self.cfg
    batch = dp_lib.constrain_rows(
        batch
    )  # data parallel: rows stay on their device
    valid = batch.valid
    v = adv_lib.seat_value(batch.value_pred, batch.steps.seat)
    boot = self._bootstrap_value(s.params, batch)
    if cfg.value_ema:  # values in reward units
      v = loss_lib.value_ema_denormalize(v, s.value_ema).astype(F32)
      boot = loss_lib.value_ema_denormalize(boot, s.value_ema).astype(F32)
    overlap = jnp.zeros((), I64)
    node = valid
    if self._value_rows_fn is not None:
      vr = self._value_rows_fn(batch.steps.info)
      overlap = jnp.sum(vr & valid, dtype=I64)
      vr = vr & ~valid
      v = jnp.where(
          valid,
          v,
          jnp.where(
              vr,
              self._value_row_values(s.params, batch, vr, value_rows_cap),
              F32(0.0),
          ),
      )
      node = valid | vr
    # A bootstrap value no row reads may be non-finite (ply T outside the
    # network's domain behind a hand end); one a row reads is "untargeted".
    boot_ok = jnp.isfinite(boot)
    boot = jnp.where(boot_ok, boot, F32(0.0))
    done = batch.done
    done_after = (
        jnp.flip(jnp.cumsum(jnp.flip(done, axis=1), axis=1), axis=1) > 0
    )
    open_ = node & ~done_after & ~boot_ok[:, None]
    untargeted = jnp.sum(open_, dtype=I64)
    if cfg.drop_unbootstrapped:
      node = node & ~open_
    policy_mask = valid & node
    reward = batch.reward
    if cfg.value_scale != 1.0:
      reward = (reward / F32(cfg.value_scale)).astype(F32)
    args = (reward, v, node, done, boot)
    ext = extension(cfg.loss)
    if ext is not None:
      advantages, targets = ext.targets(self._loss_config(ext), *args)
    else:
      g = adv_lib.gae(*args, gamma=cfg.gamma, lam=cfg.gae_lambda)
      advantages, targets = g.advantages, g.returns
    value_old = jnp.where(node, v, F32(0.0))
    return (
        Prepared(advantages=advantages, targets=targets, value_old=value_old),
        RowMasks(
            policy_mask=policy_mask,
            value_mask=node,
            untargeted=untargeted,
            value_row_overlap=overlap,
        ),
    )

  def _rows(
      self, batch, prep: Prepared, masks: Optional[RowMasks] = None
  ) -> dict[str, Any]:
    """The per-ply rows the minibatch loop trains on, leaves [B, T, ...].

    Args:
      batch: the TrajectoryBatch.
      prep: the Prepared targets.
      masks: the RowMasks; None trains every valid ply.

    Returns:
      The rows dict.
    """
    valid = batch.valid if masks is None else masks.policy_mask
    rows = dict(
        steps=batch.steps,
        action=jnp.maximum(batch.action, 0),
        behavior_logp=batch.behavior_logp,
        valid=valid,
        advantages=prep.advantages,
        targets=prep.targets,
        value_old=prep.value_old,
    )
    if extension(self.cfg.loss) is not None:
      rows["behavior_logits"] = batch.behavior_logits
      rows["value_mask"] = valid if masks is None else masks.value_mask
    # the magnet's logits (PGConfig.magnet_from_behavior)
    if self.cfg.magnet_from_behavior:
      rows["magnet_logits"] = batch.behavior_logits
    return rows

  @staticmethod
  def _time_major(x):
    """[B, T, ...] -> [T * B, ...], time-major.

    Args:
      x: A leaf, [B, T, ...].

    Returns:
      The leaf, [T * B, ...].
    """
    return jnp.swapaxes(x, 0, 1).reshape(
        (x.shape[0] * x.shape[1],) + x.shape[2:]
    )

  def _compact(self, rows, capacity: int):
    """Flat [C, ...] plies: every valid ply first, then invalid plies.

    Flat [C, ...] plies: every valid ply (time-major order) first, then
    invalid plies up to C (train_rows checked the valid count fits).

    Args:
      rows: the [B, T] rows.
      capacity: C, the static size.

    Returns:
      The compacted rows, flat [C, ...].
    """
    flat = tree_map(self._time_major, rows)
    keep = jnp.argsort(jnp.logical_not(flat["valid"]), stable=True)[:capacity]
    return dp_lib.constrain_rows(
        tree_map(lambda x: jnp.take(x, keep, axis=0), flat)
    )

  @staticmethod
  def _trained(rows) -> jax.Array:
    """bool: the rows any loss term trains (decisions, plus value rows).

    Args:
      rows: The update rows.

    Returns:
      bool rows.
    """
    return (
        rows["valid"] | rows["value_mask"]
        if "value_mask" in rows
        else rows["valid"]
    )

  def _fixed_plan(self, key, rows, capacity: int):
    """One epoch's fixed-size minibatches as flat ply indices.

    (idx i32[M, size], real bool[M, size]): one epoch's minibatches of
    the fixed size as time-major flat ply indices into the [B, T] rows
    (ply idx // B, row idx % B), M = capacity / size. The trained plies
    (time-major order) fill C = capacity slots, the rest cycled copies of
    them (never an untrained ply, whose InfoState may be outside the
    network's domain); a uniform key per slot orders the trained slots
    first. sgd_steps 0: slot (m, p) takes position m * size + p (full
    minibatches, then the remainder); else steps = max(sgd_steps,
    ceil(k / size)) and slot (m, p) takes position p * steps + m (dealt
    round-robin: floor / ceil(k / steps) plies per step). `real` marks the
    positions below the trained count k (and m < steps).

    Args:
      key: the epoch's PRNG key.
      rows: the [B, T] rows.
      capacity: C, the static size (a multiple of minibatch_size).

    Returns:
      (idx i32[M, size], real bool[M, size]).
    """
    size = int(self.cfg.minibatch_size)
    assert capacity % size == 0, (capacity, size)
    nm = capacity // size
    assert nm >= int(self.cfg.sgd_steps), (nm, self.cfg.sgd_steps)
    train = self._time_major(self._trained(rows))
    order = jnp.argsort(jnp.logical_not(train), stable=True)
    k = jnp.sum(train, dtype=jnp.int32)
    pos = jnp.arange(capacity, dtype=jnp.int32)
    real_c = pos < k
    src = jnp.take(order, jnp.where(real_c, pos, pos % jnp.maximum(k, 1)))
    u = jax.random.uniform(key, (capacity,), F32)
    perm = jnp.argsort(jnp.where(real_c, u, F32(2.0)))
    m = jnp.arange(nm, dtype=jnp.int32)[:, None]
    p = jnp.arange(size, dtype=jnp.int32)[None, :]
    if self.cfg.sgd_steps:
      steps = jnp.maximum(jnp.int32(self.cfg.sgd_steps), (k + size - 1) // size)
      # < steps * size <= capacity (k <= capacity, sgd_steps <= nm)
      q = p * steps + m
      real = (m < steps) & (q < k)
    else:
      q = m * size + p
      real = q < k
    idx = jnp.take(src, jnp.take(perm, q))
    return idx.astype(jnp.int32), real

  def _gather_rows(self, rows, idx, real):
    """One minibatch [size, ...] of the rows at flat indices idx.

    One minibatch [size, ...]: the [B, T] rows at the time-major flat
    indices idx, every mask off where not real.

    Args:
      rows: the [B, T] rows.
      idx: time-major flat ply indices, i32[size].
      real: bool[size], False on padding slots.

    Returns:
      The minibatch rows [size, ...].
    """
    b = rows["valid"].shape[0]
    bi, ti = idx % b, idx // b
    mb = tree_map(lambda x: x[bi, ti], rows)
    mb["valid"] = mb["valid"] & real
    if "value_mask" in mb:
      mb["value_mask"] = mb["value_mask"] & real
    return dp_lib.constrain_rows(mb)

  def _plan(self, s: PGState, batch, key, capacity: int):
    """Update's first-epoch plan (sgd_plan): the same prepare, rows and key.

    Args:
      s: The learner state.
      batch: The batch.
      key: The update key.
      capacity: The plan's row capacity.

    Returns:
      The fixed plan.
    """
    prep, masks = self._prepare(s, batch)
    rows = self._rows(batch, prep, masks)
    _, k = jax.random.split(
        key
    )  # epoch 0's key, as _update_prepared's scan splits it
    return self._fixed_plan(k, rows, capacity)

  def _magnet_logits(self, magnet, rows, init_carry):
    """f32 magnet logits of every row, [C, A] or [B, T, A].

    f32 magnet logits of every row, [C, A] (compacted) or [B, T, A]: the
    magnet is fixed within an update, so its forward runs once per update
    instead of once per minibatch (the same per-ply values with fewer
    forwards). Flat rows evaluate
    each ply alone (stateless); env_axis replays each row's segment from its
    init_carry, as the loss's own forward does.

    Args:
      magnet: the magnet params.
      rows: the rows, flat [C, ...] or [B, T, ...].
      init_carry: the rows' init_carry (env_axis), or None.

    Returns:
      The magnet logits, f32 [C, A] or [B, T, A].
    """
    magnet = jax.lax.stop_gradient(magnet)
    steps = rows["steps"]
    if rows["valid"].ndim == 1 or self.cfg.shuffle == "env_axis":
      logits, _, _ = self.forward(magnet, init_carry, steps)
      return jax.lax.stop_gradient(logits)
    b, t = rows["valid"].shape
    flat = tree_map(lambda x: x.reshape((b * t,) + x.shape[2:]), steps)
    logits, _, _ = self.forward(magnet, None, flat)
    return jax.lax.stop_gradient(logits.reshape((b, t) + logits.shape[1:]))

  def _teacher_probs(self, teacher_params, rows):
    """f32 legal-masked teacher probabilities of every row, [C, A] or [B, T, A].

    The teacher is fixed, so its forward runs once per update over the rows
    (module doc "Teacher distillation"): flat plies (stateless), chunked at the
    Teacher's chunk_rows under no gradient. Illegal actions get 0.

    Args:
      teacher_params: the teacher's params.
      rows: the rows, flat [C, ...] or [B, T, ...].

    Returns:
      The probabilities, f32 [C, A] or [B, T, A].
    """
    t = self._teacher
    assert t is not None, "no teacher"
    pol = t.policy
    params = jax.lax.stop_gradient(teacher_params)
    steps = rows["steps"]
    lead = rows["valid"].shape
    n = int(np.prod(lead))
    flat = tree_map(lambda x: x.reshape((n,) + x.shape[len(lead) :]), steps)

    def _run(st):
      """The teacher's logits of a chunk of plies.

      Args:
        st: The plies' SeatSteps.

      Returns:
        f32 logits.
      """
      s_ = jnp.asarray(st.seat)
      return pol.evaluate(
          params, pol.init_carry(params, s_.shape[0], s_, None), st, None
      ).logits

    logits = self._chunked(_run, flat, chunk=t.chunk_rows)
    assert logits.dtype == F32, logits.dtype
    legal = flat.legal_mask
    probs = jnp.exp(
        loss_lib.masked_log_softmax(
            logits, legal, stable=self.cfg.pg.stable_softmax
        )
    )
    probs = jnp.where(legal, probs, F32(0.0)).astype(F32)
    return jax.lax.stop_gradient(probs.reshape(lead + probs.shape[1:]))

  def _minibatches(self, key, rows, init_carry):
    """Leaves with a leading [num_minibatches] axis.

    rows: [B, T, ...], or flat [C, ...] (compacted, flat shuffle only).

    Args:
      key: the epoch's PRNG key.
      rows: [B, T, ...], or flat [C, ...] (compacted, flat shuffle only).
      init_carry: the rows' init_carry (env_axis), or None.

    Returns:
      (rows, init_carry), leaves with a leading [num_minibatches] axis.
    """
    nm = self.cfg.num_minibatches
    if rows["valid"].ndim == 1:
      c = rows["valid"].shape[0]
      perm = jax.random.permutation(key, c)

      def pick_compacted(x):
        """Shuffles compacted rows into minibatches.

        Args:
          x: A compacted leaf, [c, ...].

        Returns:
          The leaf, [nm, c // nm, ...].
        """
        return jnp.take(x, perm, axis=0).reshape((nm, c // nm) + x.shape[1:])

      return (
          dp_lib.constrain_rows(tree_map(pick_compacted, rows), axis=1),
          None,
      )
    b, t = rows["valid"].shape
    if self.cfg.shuffle == "flat":
      n = b * t
      perm = jax.random.permutation(key, n)

      def _flat(x):  # [B, T, ...] -> time-major [T * B, ...]
        """Shuffles time-major plies into minibatches.

        Args:
          x: A leaf, [B, T, ...].

        Returns:
          The leaf, [nm, n // nm, ...].
        """
        x = jnp.swapaxes(x, 0, 1).reshape((n,) + x.shape[2:])
        return jnp.take(x, perm, axis=0).reshape((nm, n // nm) + x.shape[1:])

      return dp_lib.constrain_rows(tree_map(_flat, rows), axis=1), None
    perm = jax.random.permutation(key, b)

    def pick(x):
      """Shuffles rows into minibatches.

      Args:
        x: A leaf, [B, ...].

      Returns:
        The leaf, [nm, b // nm, ...].
      """
      return jnp.take(x, perm, axis=0).reshape((nm, b // nm) + x.shape[1:])

    return (
        dp_lib.constrain_rows(tree_map(pick, rows), axis=1),
        dp_lib.constrain_rows(tree_map(pick, init_carry), axis=1),
    )

  def _loss(
      self,
      params,
      magnet,
      mb,
      carry,
      targets,
      value_old,
      kl_coef,
      ent_coef,
      temperature,
      is_first,
      stats: Optional[loss_lib.PGStats] = None,
      counts: Optional[dict[str, Any]] = None,
      step_coefs: Optional[dict[str, Any]] = None,
      teacher_coef: Optional[jax.Array] = None,
  ):
    """(total, metrics) of a minibatch, or of one chunk of its rows.

    With `stats` (pg_loss) / `counts` (a registered loss with chunk_sums: the
    minibatch's row counts) of one chunk of its rows, every mean over the
    minibatch's rows (_accumulated_grad).

    Args:
      params: the network params.
      magnet: the magnet params, or None.
      mb: the minibatch rows.
      carry: the rows' init_carry (env_axis), or None.
      targets: the value targets.
      value_old: the behaviour values.
      kl_coef: the magnet KL coefficient.
      ent_coef: the entropy coefficient.
      temperature: a registered loss's magnet temperature.
      is_first: bool, the run's first update.
      stats: the minibatch's losses.PGStats (pg_loss only, one chunk of the
        rows), or None.
      counts: the minibatch's row counts {policy, value, noop} (a registered
        loss with chunk_sums, one chunk of the rows), or None.
      step_coefs: the registered loss's step coefficients (LossSpec
        step_coefs), or None.
      teacher_coef: c(t) of the teacher KL (module doc "Teacher
        distillation"; mb then holds "teacher_probs"), or None (no teacher).

    Returns:
      (total, metrics).
    """
    cfg = self.cfg
    steps = mb["steps"]
    legal = steps.legal_mask
    valid = mb["valid"]
    action = mb["action"]
    logits, value, value_logits = self.forward(params, carry, steps)
    stable = cfg.pg.stable_softmax
    logp = loss_lib.action_log_prob(logits, legal, action, stable=stable)
    log_ratio = logp - mb["behavior_logp"]
    kl_rows = jnp.exp(log_ratio) - 1.0 - log_ratio
    if stats is not None:
      approx_kl = loss_lib.masked_mean(kl_rows, valid, stats.mask_count)
    else:
      approx_kl = loss_lib.masked_mean(
          kl_rows, valid, None if counts is None else counts["policy"]
      )
    magnet_logits = None
    # the magnet's logits, once per update (_magnet_logits, or behaviour)
    if magnet is not None:
      magnet_logits = jax.lax.stop_gradient(mb["magnet_logits"])
    ext = extension(cfg.loss)
    if stats is not None:
      assert ext is None, f"loss {cfg.loss!r}: PGStats are pg_loss's"
    if counts is not None:
      assert ext is not None and ext.chunk_sums is not None, (
          f"loss {cfg.loss!r} declares no chunk_sums: it is not cut into row "
          "chunks"
      )
    if ext is not None:
      kw = {}
      if counts is not None:
        kw["counts"] = LossCounts(
            policy=counts["policy"], value=counts["value"]
        )
      if step_coefs is not None:
        assert (
            ext.step_coefs is not None
        ), f"loss {cfg.loss!r} declares no step_coefs"
        kw["step_coefs"] = step_coefs
      if stable:
        kw["stable_softmax"] = True
      total, m = ext.loss(
          self._loss_config(ext),
          logits=logits,
          value=value,
          legal=legal,
          action=action,
          behavior_log_probs=loss_lib.masked_log_softmax(
              mb["behavior_logits"], legal, stable=stable
          ),
          advantages=mb["advantages"],
          returns=targets,
          policy_mask=valid,
          value_mask=mb["value_mask"],
          temperature=temperature,
          magnet_logits=magnet_logits,
          **kw,
      )
      m = dict(m, loss_total=total)
      # the value loss on decision / value-only plies
      if self._value_rows_fn is not None:
        sq = jax.lax.stop_gradient(loss_lib.value_squared_error(value, targets))
        noop = mb["value_mask"] & ~valid
        m["value_loss_decision"] = loss_lib.masked_mean(
            sq, valid, None if counts is None else counts["policy"]
        )
        m["value_loss_noop"] = loss_lib.masked_mean(
            sq, noop, None if counts is None else counts["noop"]
        )
    else:
      total, m = loss_lib.pg_loss(
          cfg.pg,
          logits=logits,
          value=value,
          legal=legal,
          action=action,
          behavior_logp=mb["behavior_logp"],
          mask=valid,
          advantages=mb["advantages"],
          targets=targets,
          value_old=value_old,
          value_logits=value_logits,
          reference_logits=magnet_logits,
          kl_coef=kl_coef,
          ent_coef=ent_coef,
          is_first_update=is_first,
          value_support=self.value_support,
          stats=stats,
      )
      m = dict(m)
      if cfg.loss == "ppo":
        del m["kl_penalty"], m["kl_coef_effective"]  # no magnet: not computed
    if teacher_coef is not None:  # teacher distillation (module doc)
      assert ext is None, f"loss {cfg.loss!r}: the teacher KL is pg_loss's"
      tp = jax.lax.stop_gradient(mb["teacher_probs"])
      count = None if stats is None else stats.mask_count
      t_kl = loss_lib.masked_mean(
          loss_lib.kl_from_target(tp, logits, legal, stable=stable),
          valid,
          count,
      )
      total = total + teacher_coef * t_kl
      mine = jnp.argmax(
          jnp.where(legal, jax.lax.stop_gradient(logits), -jnp.inf), axis=-1
      )
      agree = (mine == loss_lib.target_argmax(tp, legal)).astype(F32)
      m["loss_total"] = total
      m["teacher_kl"] = loss_lib.f32(jax.lax.stop_gradient(t_kl))
      m["teacher_top1"] = loss_lib.f32(
          loss_lib.masked_mean(agree, valid, count)
      )
      m["teacher_coef"] = loss_lib.f32(teacher_coef)
    m["approx_kl"] = loss_lib.f32(jax.lax.stop_gradient(approx_kl))
    return total, m

  def accumulates(self, rows: int) -> bool:
    """Whether a minibatch's gradient is accumulated over row chunks.

    Module doc "Forward chunking": ppo / mmd, or a registered loss that
    declares chunk_sums, above forward_chunk_rows.

    Args:
      rows: the minibatch's row count.

    Returns:
      True to accumulate.
    """
    ext = extension(self.cfg.loss)
    return (ext is None or ext.chunk_sums is not None) and int(rows) > int(
        self.cfg.forward_chunk_rows
    )

  def _chunk_metrics(self) -> frozenset[Any]:
    """The metrics of _loss that add up over a minibatch's row chunks.

    The others are the minibatch's in every chunk.

    Returns:
      The metric names.
    """
    ext = extension(self.cfg.loss)
    if ext is None:
      out = loss_lib.PG_CHUNK_SUMS | {"approx_kl"}
      if self._teacher is not None:
        out |= {"teacher_kl", "teacher_top1"}
      return out
    assert ext.chunk_sums is not None, self.cfg.loss
    out = ext.chunk_sums | {"loss_total", "approx_kl"}
    if self._value_rows_fn is not None:
      out |= {"value_loss_decision", "value_loss_noop"}
    return out

  def loss_and_grad(
      self,
      params,
      magnet,
      mb,
      carry,
      targets,
      value_old,
      kl_coef,
      ent_coef,
      temperature,
      is_first,
      accumulate: Optional[bool] = None,
      step_coefs: Optional[dict[str, Any]] = None,
      teacher_coef: Optional[jax.Array] = None,
  ):
    """(metrics, grads) of one minibatch's loss.

    Args:
      params: the network params.
      magnet: the magnet params, or None.
      mb: the minibatch rows.
      carry: the rows' init_carry (env_axis), or None.
      targets: the value targets.
      value_old: the behaviour values.
      kl_coef: the magnet KL coefficient.
      ent_coef: the entropy coefficient.
      temperature: a registered loss's magnet temperature.
      is_first: bool, the run's first update.
      accumulate: None: as accumulates() says; False: one value_and_grad over
        the minibatch (its forward the rematerialized chunk scan); True:
        _accumulated_grad.
      step_coefs: a registered loss's LossSpec.step_coefs of this update
        (step_coefs_at), or None.
      teacher_coef: c(t) of the teacher KL (mb holds "teacher_probs"), or
        None.

    Returns:
      (metrics, grads).
    """
    rows = int(mb["valid"].shape[0])
    if accumulate is None:
      accumulate = self.accumulates(rows)
    args = (
        params,
        magnet,
        mb,
        carry,
        targets,
        value_old,
        kl_coef,
        ent_coef,
        temperature,
        is_first,
    )
    kw = {} if teacher_coef is None else {"teacher_coef": teacher_coef}
    if not accumulate:
      (_, m), grads = jax.value_and_grad(
          functools.partial(self._loss, step_coefs=step_coefs, **kw),
          has_aux=True,
      )(*args)
      return m, grads
    return self._accumulated_grad(*args, step_coefs=step_coefs, **kw)

  def _accumulated_grad(
      self,
      params,
      magnet,
      mb,
      carry,
      targets,
      value_old,
      kl_coef,
      ent_coef,
      temperature,
      is_first,
      step_coefs: Optional[dict[str, Any]] = None,
      teacher_coef: Optional[jax.Array] = None,
  ):
    """(metrics, grads) of a minibatch, summed over row chunks.

    (metrics, grads) of a minibatch as the sum over chunks of at most
    forward_chunk_rows rows (strided as _chunked cuts them; pad rows repeat
    row 0 with every mask off) of each chunk's loss under the minibatch's
    pg_stats (pg_loss) or row counts (a registered loss: LossCounts).
    Metrics in _chunk_metrics are summed over the chunks; the others are the
    minibatch's in every chunk.

    Args:
      params: the network params.
      magnet: the magnet params, or None.
      mb: the minibatch rows.
      carry: the rows' init_carry (env_axis), or None.
      targets: the value targets.
      value_old: the behaviour values.
      kl_coef: the magnet KL coefficient.
      ent_coef: the entropy coefficient.
      temperature: a registered loss's magnet temperature.
      is_first: bool, the run's first update.
      step_coefs: the registered loss's step coefficients (LossSpec
        step_coefs), or None.
      teacher_coef: c(t) of the teacher KL (mb holds "teacher_probs"), or
        None.

    Returns:
      (metrics, grads).
    """
    cfg = self.cfg
    ext = extension(cfg.loss)
    assert ext is None or ext.chunk_sums is not None, cfg.loss
    valid = mb["valid"]
    n = int(valid.shape[0])
    stats = counts = None
    if ext is None:
      stats = loss_lib.pg_stats(cfg.pg, valid, mb["advantages"])
    else:  # the whole minibatch's denominators (every mask as _loss reads it)
      vm = mb["value_mask"]

      def cnt(m):
        """The rows of mask `m`.

        Args:
          m: A bool mask.

        Returns:
          f32, its count.
        """
        return jnp.sum(m, dtype=F32)

      counts = dict(policy=cnt(valid), value=cnt(vm), noop=cnt(vm & ~valid))
    num, c = actor_critic.chunk_plan(n, int(cfg.forward_chunk_rows))
    assert c <= cfg.forward_chunk_rows and num * c >= n, (
        n,
        cfg.forward_chunk_rows,
        num,
        c,
    )
    pad = num * c - n
    flat = valid.ndim == 1  # flat plies evaluate with no carry (forward)

    def _split(x):
      """Pads and splits rows into `num` chunks.

      Args:
        x: A leaf, [rows, ...].

      Returns:
        The chunks, leading axis num.
      """
      x = jnp.asarray(x)
      if pad:
        x = jnp.concatenate([x, jnp.broadcast_to(x[:1], (pad,) + x.shape[1:])])
      return jnp.swapaxes(x.reshape((c, num) + x.shape[1:]), 0, 1)

    rows = dict(mb=mb, targets=targets, value_old=value_old)
    if stats is not None:
      rows.update(adv=stats.adv, actor_rows=stats.actor_rows)
    if not flat:
      rows["carry"] = carry
    chunks = tree_map(_split, rows)
    # bool[num, c]: False on the pad rows (split appends them after row n - 1)
    chunks["real"] = jnp.swapaxes(
        (jnp.arange(num * c) < n).reshape(c, num), 0, 1
    )

    def _body(acc, ch):
      """Adds one chunk's gradient to the accumulator.

      Args:
        acc: The gradient so far.
        ch: The chunk.

      Returns:
        (the new sum, the chunk's metrics).
      """
      real = ch["real"].reshape(ch["real"].shape + (1,) * (valid.ndim - 1))
      mb_c = dict(ch["mb"], valid=ch["mb"]["valid"] & real)
      if "value_mask" in mb_c:
        mb_c["value_mask"] = mb_c["value_mask"] & real
      st = None
      if stats is not None:
        st = stats._replace(adv=ch["adv"], actor_rows=ch["actor_rows"] & real)
      carry_c = ch["carry"] if not flat else carry
      kw = {} if teacher_coef is None else {"teacher_coef": teacher_coef}
      (_, m), g = jax.value_and_grad(
          functools.partial(self._loss, **kw), has_aux=True
      )(
          params,
          magnet,
          mb_c,
          carry_c,
          ch["targets"],
          ch["value_old"],
          kl_coef,
          ent_coef,
          temperature,
          is_first,
          st,
          counts,
          step_coefs,
      )
      return tree_map(jnp.add, acc, g), m

    zeros = tree_map(jnp.zeros_like, params)
    grads, per = jax.lax.scan(_body, zeros, chunks)
    summed = self._chunk_metrics()
    m = {k: jnp.sum(v, axis=0) if k in summed else v[0] for k, v in per.items()}
    return m, grads

  def _update_prepared(
      self,
      s: PGState,
      batch,
      prep: Prepared,
      key,
      capacity=None,
      masks: Optional[RowMasks] = None,
      teacher_params=None,
  ):
    """Steps 2-3 on given targets: the epochs of minibatch steps.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      prep: the Prepared targets.
      key: the PRNG key of the shuffle.
      capacity: the rows to compact to (train_rows), or None.
      masks: the RowMasks; None trains every valid ply.
      teacher_params: the teacher's params (set_teacher), else None.

    Returns:
      (the new PGState, per-minibatch metrics f32[E, M]).
    """
    cfg = self.cfg
    if (self._teacher is None) != (teacher_params is None):
      raise ValueError(
          "teacher_params go with set_teacher: teacher "
          f"{'unset' if self._teacher is None else self._teacher.name}, "
          f"params {'missing' if teacher_params is None else 'given'}"
      )
    u = s.update_step
    assert u.dtype == I64, u.dtype
    # The f64 schedule value rounded to the f32 params' dtype once:
    # -f32(lr) == f32(-lr) exactly, so no f64 scalar reaches the minibatch
    # loop.
    us = self._sched_step(
        u
    )  # the counter schedules read (schedule_step_offset)
    neg_lr = -self._lr.f32(us)
    ent = cfg.ent_coef.f32(us)
    kl = cfg.kl_coef.f32(us)
    ext = extension(cfg.loss)
    temperature = (
        F32(0.0) if ext is None else self._temperature(ext, us)
    )  # pg_loss: none
    step_coefs = None if ext is None else self._step_coefs(ext, us)
    teacher_coef = None if teacher_params is None else cfg.teacher_coef.f32(us)
    is_first = u == 0
    rows = self._rows(batch, prep, masks)
    fixed = bool(cfg.minibatch_size)
    if fixed:  # rows stay [B, T]: each step gathers its plies (_fixed_plan)
      assert capacity is not None and cfg.shuffle == "flat", (
          capacity,
          cfg.shuffle,
      )
    elif capacity is not None:
      assert cfg.shuffle == "flat", cfg.shuffle
      rows = self._compact(rows, capacity)
    magnet = s.magnet
    from_behavior = magnet is not None and cfg.magnet_from_behavior
    if magnet is not None and not fixed and not from_behavior:
      rows["magnet_logits"] = self._magnet_logits(
          magnet, rows, batch.init_carry
      )
    if teacher_params is not None and not fixed:  # once per update
      rows["teacher_probs"] = self._teacher_probs(teacher_params, rows)

    # SGD step gate (module doc): the carry gains a bool `stopped`; no gate
    # (None), the carry and every step are exactly as without the hook.
    gate = self._step_gate()
    gated = gate is not None

    def _sgd_step(c, xs):
      """One optimizer step on a minibatch.

      Args:
        c: (params, opt_state, ema, rng), plus the gate's stop flag when
          gated.
        xs: (minibatch, carry).

      Returns:
        (the new carry tuple, the step's metrics).
      """
      params, opt_state, ema, rng = c[:4]
      mb, carry = xs
      targets, value_old = mb["targets"], mb["value_old"]
      if cfg.value_ema:
        ema = loss_lib.value_ema_update(
            ema, targets, cfg.value_ema_beta, mb["valid"]
        )
        targets = loss_lib.value_ema_normalize(targets, ema).astype(F32)
        if cfg.pg.clip_value:
          value_old = loss_lib.value_ema_normalize(value_old, ema).astype(F32)
      # The carried key advances once per minibatch step; the next epoch's
      # shuffle key is split from it, so the key schedule is fixed by the
      # (key, epoch, step) sequence alone.
      rng, _ = jax.random.split(rng)
      m, grads = self.loss_and_grad(
          params,
          magnet,
          mb,
          carry,
          targets,
          value_old,
          kl,
          ent,
          temperature,
          is_first,
          step_coefs=step_coefs,
          teacher_coef=teacher_coef,
      )
      m["grad_norm"] = optax.tree.norm(grads).astype(F32)
      # Telemetry (reads grads / params / the step, changes none of them):
      # clip_by_global_norm rescales when grad_norm >= max_grad_norm, so the
      # applied gradient's norm is min(grad_norm, max_grad_norm).
      max_gn = F32(cfg.max_grad_norm)
      m["grad_norm_clipped"] = jnp.minimum(m["grad_norm"], max_gn)
      m["grad_clipped"] = (m["grad_norm"] >= max_gn).astype(F32)
      for mod, sub in telemetry.module_groups(grads).items():
        m[f"grad_norm/{mod}"] = telemetry.tree_norm(sub)
      param_norm = telemetry.tree_norm(params)
      param_mods = telemetry.module_groups(params)
      upd, opt_state = self._tx.update(grads, opt_state, params)
      upd = tree_map(lambda g: _check_f32_leaf(g) * neg_lr, upd)
      m["update_norm"] = telemetry.tree_norm(upd)
      m["update_ratio"] = m["update_norm"] / jnp.maximum(param_norm, F32(1e-12))
      # Per module: the step's norm and its ratio to the module's params
      # before it (a dead module, or a head outpacing the torso).
      for mod, sub in telemetry.module_groups(upd).items():
        un = telemetry.tree_norm(sub)
        m[f"update_norm/{mod}"] = un
        m[f"update_ratio/{mod}"] = un / jnp.maximum(
            telemetry.tree_norm(param_mods[mod]), F32(1e-12)
        )
      params = optax.apply_updates(params, upd)
      if fixed:
        m["sgd_step"] = F32(1.0)
        m["minibatch_rows"] = jnp.sum(
            self._trained(mb), dtype=jnp.int32
        ).astype(F32)
      if not gated:
        return (params, opt_state, ema, rng), m
      # The gate reads the step's metrics, measured at the params it starts
      # from (before the telemetry of the update itself): False, this step's
      # update is not applied (params, optimizer state and value EMA as the
      # carry held them) and the flag skips every later step of the update
      # (mb_step). Its loss metrics are kept (a measurement at those
      # params); its update norms are the applied 0.
      ok = jnp.asarray(
          gate({
              k: v
              for k, v in m.items()
              if k
              not in (
                  "update_norm",
                  "update_ratio",
                  "sgd_step",
                  "minibatch_rows",
              )
              and not k.startswith(_UPDATE_MODULE_PREFIXES)
          })
      )
      assert (
          not ok.shape and ok.dtype == jnp.bool_
      ), f"step_gate must return a bool scalar, got {ok.dtype}{list(ok.shape)}"
      over = jnp.logical_not(ok)
      params, opt_state, ema = tree_map(
          lambda new, old: jnp.where(over, old, new),
          (params, opt_state, ema),
          tuple(c[:3]),
      )
      for k in list(m):
        if k in ("update_norm", "update_ratio") or k.startswith(
            _UPDATE_MODULE_PREFIXES
        ):
          m[k] = jnp.where(over, F32(0.0), m[k])
      m["sgd_step"] = F32(
          1.0
      )  # a loss was computed (summary means run over these)
      m["gate_stop"] = over.astype(F32)
      return (params, opt_state, ema, rng, over), m

    def _fixed_step(c, plan):
      """One step on a fixed plan's minibatch.

      Args:
        c: (params, opt_state, ema, rng).
        plan: (row indices, real flags).

      Returns:
        (carry, metrics).
      """
      idx, real = plan
      mb = self._gather_rows(rows, idx, real)
      if magnet is not None and not from_behavior:  # fixed within the update
        mb["magnet_logits"] = self._magnet_logits(magnet, mb, None)
      if teacher_params is not None:  # fixed within the update
        mb["teacher_probs"] = self._teacher_probs(teacher_params, mb)
      return _sgd_step(c, (mb, jnp.zeros((), F32)))

    def _mb_step(c, xs):
      """One minibatch step; a padding-only minibatch takes no step.

      Args:
        c: (params, opt_state, ema, rng).
        xs: The minibatch.

      Returns:
        (carry, metrics).
      """
      if not fixed:
        if not gated:
          return _sgd_step(c, xs)
        # stopped: the step is skipped whole (no forward, no update)
        skip_m = tree_map(
            lambda a: jnp.zeros(a.shape, a.dtype),
            jax.eval_shape(_sgd_step, c, xs)[1],
        )
        c, m = jax.lax.cond(c[4], lambda c, _: (c, skip_m), _sgd_step, c, xs)
        return c, dict(m, sgd_planned=F32(1.0))
      # A minibatch with no trained ply (padding only) takes no optimizer
      # step: Adam's moments would otherwise still move the params. Every
      # slot points at a trained ply, so `real` alone says it.
      skip_m = tree_map(
          lambda a: jnp.zeros(a.shape, a.dtype),
          jax.eval_shape(_fixed_step, c, xs)[1],
      )
      if not gated:
        return jax.lax.cond(
            jnp.any(xs[1]), _fixed_step, lambda c, _: (c, skip_m), c, xs
        )
      real = jnp.any(xs[1])
      c, m = jax.lax.cond(
          real & jnp.logical_not(c[4]),
          _fixed_step,
          lambda c, _: (c, skip_m),
          c,
          xs,
      )
      return c, dict(m, sgd_planned=real.astype(F32))

    def _epoch(c, xs):
      """One epoch of minibatch steps.

      Args:
        c: (params, opt_state, ema, rng), plus the gate's stop flag when
          gated.
        xs: The scan's xs, None (unused).

      Returns:
        (carry, per-step metrics).
      """
      del xs
      params, opt_state, ema, rng = c[:4]
      rng, k = jax.random.split(rng)
      c = (params, opt_state, ema, rng) + tuple(c[4:])  # + the gate's stop flag
      if fixed:
        return jax.lax.scan(_mb_step, c, self._fixed_plan(k, rows, capacity))
      mbs, carries = self._minibatches(k, rows, batch.init_carry)
      if carries is None:  # flat rows evaluate without a carry
        carries = jnp.zeros((jax.tree_util.tree_leaves(mbs)[0].shape[0],), F32)
      c, m = jax.lax.scan(_mb_step, c, (mbs, carries))
      return c, m

    init = (s.params, s.opt_state, s.value_ema, key)
    if gated:
      init += (jnp.asarray(False),)
    out_c, per_mb = jax.lax.scan(_epoch, init, None, length=cfg.update_epochs)
    params, opt_state, ema = out_c[:3]
    step_after = u + jnp.int64(1)
    sched_after = self._sched_step(
        step_after
    )  # the refresh reads the schedules' counter
    if cfg.loss == "mmd":
      magnet = loss_lib.refresh_magnet(
          params, magnet, sched_after, cfg.reference_update_every
      )
    elif extension(cfg.loss) is not None:
      if magnet is not None:  # a moving magnet (else uniform: none kept)
        magnet = loss_lib.refresh_magnet(
            params, magnet, sched_after, cfg.magnet_every
        )
    for leaf in jax.tree_util.tree_leaves(params):
      assert leaf.dtype == F32, f"a param leaf became {leaf.dtype}"
    s2 = PGState(
        params=params,
        opt_state=opt_state,
        magnet=magnet,
        value_ema=ema,
        update_step=step_after,
    )
    return s2, per_mb

  def _update(
      self,
      s: PGState,
      batch,
      key,
      capacity=None,
      teacher_params=None,
      value_rows_cap=None,
  ):
    """The full update: prepare, then update_prepared, plus the update's info.

    Args:
      s: the learner state.
      batch: the TrajectoryBatch.
      key: the PRNG key of the shuffle.
      capacity: the rows to compact to (train_rows), or None.
      teacher_params: the teacher's params (set_teacher), else None.
      value_rows_cap: the value_rows forward size (value_rows_cap), or None.

    Returns:
      (the new PGState, per-minibatch metrics, the update's info dict).
    """
    prep, masks = self._prepare(s, batch, value_rows_cap)
    if teacher_params is None:  # the program without the option, unchanged
      s2, per_mb = self._update_prepared(s, batch, prep, key, capacity, masks)
    else:
      s2, per_mb = self._update_prepared(
          s, batch, prep, key, capacity, masks, teacher_params=teacher_params
      )
    valid = batch.valid
    info = dict(
        lr=self._lr.f32(
            self._sched_step(s.update_step)
        ),  # int64 counter -> f64 -> f32
        valid_fraction=jnp.mean(valid.astype(F32)),
        decisions=jnp.sum(valid, dtype=I64),
        hands_done=jnp.sum(batch.done, dtype=I64),
        target_mean=loss_lib.masked_mean(prep.targets, valid),
        reward_per_row=jnp.mean(jnp.sum(batch.reward, axis=1)),
        untargeted_rows=masks.untargeted,
        trained_rows=jnp.sum(masks.value_mask, dtype=I64),
        policy_rows=jnp.sum(masks.policy_mask, dtype=I64),
        value_row_overlap=masks.value_row_overlap,
    )
    if self._value_rows_fn is not None:
      info["value_only_rows"] = jnp.sum(
          masks.value_mask & ~masks.policy_mask, dtype=I64
      )
    if self.cfg.value_ema:
      info["value_ema_mean"] = s2.value_ema.mu_debiased[0]
      info["value_ema_std"] = s2.value_ema.std_debiased[0]
    # Telemetry: the params after the update (global and per module), how far
    # the whole update moved them, completed-hand returns / lengths, and how
    # much of the value targets the behaviour values explain.
    info.update(telemetry.norms(s2.params, "param_norm"))
    delta = telemetry.tree_norm(
        tree_map(lambda a, b: a - b, s2.params, s.params)
    )
    info["param_change_norm"] = delta
    info["param_change_ratio"] = delta / jnp.maximum(
        telemetry.tree_norm(s.params), F32(1e-12)
    )
    info[_EPISODES] = telemetry.episode_sums(
        batch.reward,
        batch.done,
        batch.steps.seat,
        self._policy.spec.num_players,
    )
    game_metrics = telemetry.batch_metrics(self._policy.spec.game)
    if game_metrics is not None:
      extra = dict(game_metrics(batch))
      clash = sorted(set(extra) & set(info))
      if clash:
        raise ValueError(
            f"{self._policy.spec.game} batch metrics {clash} clash with the "
            "learner's own"
        )
      info.update(extra)
    info[_EV] = telemetry.variance_parts(prep.value_old, prep.targets, valid)
    return s2, per_mb, info

  def _summarize(self, s, s2, per_mb, info) -> dict[str, float]:
    """The host metrics of an update.

    Args:
      s: the learner state before the update.
      s2: the learner state after it.
      per_mb: the per-minibatch metrics.
      info: the update's info dict.

    Returns:
      metric name -> float.

    Raises:
      FloatingPointError: a metric is not finite.
    """
    out: dict[str, float] = {}
    per_mb = dict(per_mb)
    gate_stop = per_mb.pop(
        "gate_stop", None
    )  # an SGD step gate only (with sgd_planned)
    planned = per_mb.pop("sgd_planned", None)
    took = None
    # fixed minibatch size, or a step gate: means over the steps that computed a
    # loss
    if "sgd_step" in per_mb:
      took = np.asarray(per_mb.pop("sgd_step"), np.float64).reshape(-1) > 0.5
      rows = per_mb.pop("minibatch_rows", None)
      assert (
          took.any()
      ), "no SGD step taken (update refuses a batch with no trained row)"
      for k, v in per_mb.items():
        out[k] = float(np.mean(np.asarray(v, np.float64).reshape(-1)[took]))
      out["grad_norm_max"] = float(
          np.max(np.asarray(per_mb["grad_norm"], np.float64).reshape(-1)[took])
      )
      if "approx_kl" in per_mb:
        out["approx_kl_max"] = float(
            np.max(
                np.asarray(per_mb["approx_kl"], np.float64).reshape(-1)[took]
            )
        )
      out["sgd_steps"] = float(took.sum())
      if rows is not None:
        rows = np.asarray(rows, np.float64).reshape(-1)
        out["minibatch_rows"] = float(rows.sum())
        out["minibatch_rows_max"] = float(rows[took].max())
        out["minibatch_rows_min"] = float(rows[took].min())
      per_mb = {}
    if (
        gate_stop is not None
    ):  # SGD step gate (module doc): at most one step stops an update
      assert (
          took is not None and planned is not None
      ), "step gate metrics are incomplete"
      stops = np.asarray(gate_stop, np.float64).reshape(-1) > 0.5
      n_stop = int(stops.sum())
      assert n_stop <= 1 and not (stops & ~took).any(), (n_stop, "gate stops")
      taken = int(took.sum()) - n_stop
      n_planned = int(np.sum(np.asarray(planned, np.float64) > 0.5))
      assert n_stop or taken == n_planned, (taken, n_planned)
      name = extension(self.cfg.loss).step_gate_name
      out[f"{name}_early_stopped"] = float(n_stop)
      out[f"{name}_stop_step"] = float(
          taken
      )  # steps applied before the stop; all when none
      out["sgd_steps_taken"] = float(taken)
      out["sgd_steps_planned"] = float(n_planned)
    for k, v in per_mb.items():
      out[k] = float(np.mean(np.asarray(v, np.float64)))
    if per_mb:  # the fixed path took grad_norm_max over its steps taken above
      out["grad_norm_max"] = float(
          np.max(np.asarray(per_mb["grad_norm"], np.float64))
      )
      if "approx_kl" in per_mb:  # the interval's largest minibatch KL
        out["approx_kl_max"] = float(
            np.max(np.asarray(per_mb["approx_kl"], np.float64))
        )
    info = dict(info)
    out.update(telemetry.episode_summary(info.pop(_EPISODES)))
    out.update(telemetry.explained_variance(info.pop(_EV)))
    for k, v in info.items():
      out[k] = float(np.asarray(v))
    for k in self._not_computed():
      out.pop(k, None)
    out["update_step"] = float(int(s2.update_step))
    bad = sorted(k for k, v in out.items() if not math.isfinite(v))
    if bad:
      raise FloatingPointError(
          f"PGLearner update {int(s.update_step)}: non-finite {bad}"
      )
    return out
