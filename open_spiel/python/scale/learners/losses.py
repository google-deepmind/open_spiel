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

"""The loss functions the learners share.

Learners run the forward pass (Policy.evaluate_seq) and hand the outputs
here as arrays; nothing in this module touches a network, so one
implementation serves the policy-gradient learner (ppo and mmd; registered
variants bring their own), behaviour cloning, AlphaZero (BC over search
targets) and learned belief models alike.

  total, terms = losses.pg_loss(cfg, logits=logits, value=value, legal=legal,
                                action=action, behavior_logp=logp, mask=valid,
                                advantages=adv.advantages, targets=adv.returns,
                                ent_coef=0.01)

Contents:
  primitives   masked_mean, masked_log_softmax, action_log_prob, entropy,
               normalize_advantages, ppo_clip_objective, clip_fraction,
               kl_to_reference, kl_to_uniform, value_squared_error,
               clipped_value_squared_error, HL-Gauss (support, target probs,
               logits->scalar, cross-entropy), z_loss, cross_entropy_to_target,
               cross_entropy_to_action, kl_from_target, sigmoid_cross_entropy,
               aux_loss, advantage_filter_mask
  counters     magnet_due / refresh_magnet (the moving magnet), ValueEMA;
               every schedule (coefficients, temperatures, learning rates)
               is the one core.schedules.Schedule
  composed     pg_loss (ppo, mmd; pg_stats cuts a minibatch into row chunks
               whose losses add up to it), distill_loss (BC, AZ, teacher
               distillation)

Conventions:
  * Masks. `legal` bool[..., A] masks actions: illegal logits are REPLACED
    (jnp.where) by NEG before any softmax, so they cannot affect a loss or
    receive gradient. `mask` bool[...] masks rows (TrajectoryBatch.valid):
    every reduction is a masked mean over kept rows (an empty mask gives 0
    with zero gradient).
  * Dtypes: float inputs
    are asserted float32 and every output is float32; Python-float
    coefficients are weak-typed and coefficients given as arrays are cast to
    f32, so nothing promotes to f64. Step counters are int64 (asserted where
    a counter is an input).
  * HL-Gauss support: configured once, on the value head. A loss with an
    HL-Gauss value term takes `value_support` (the policy's
    value_support(), handed in by the learner); the loss configs carry no
    support fields.
  * NEG = -1e9 is finite because a -inf fill gives NaN gradients through
    0 * -inf on rows with no legal action. `decide` (sampling) is separate and
    uses -inf.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.networks import hl_gauss

NEG = -1e9
_F32 = jnp.float32


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def f32(x: Any) -> jax.Array:
  """An f32 array of x (for coefficients/temperatures computed elsewhere).

  Args:
    x: a scalar or array.

  Returns:
    x as a float32 array.
  """
  return jnp.asarray(x, _F32)


def _f(name: str, x: Any) -> jax.Array:
  """Checks `x` is float32.

  Args:
    name: The array's name, for errors.
    x: The array.

  Returns:
    `x` as an array.
  """
  x = jnp.asarray(x)
  assert x.dtype == _F32, f"{name}: want float32, got {x.dtype}"
  return x


def _b(name: str, x: Any) -> jax.Array:
  """Checks `x` is bool.

  Args:
    name: The array's name, for errors.
    x: The array.

  Returns:
    `x` as an array.
  """
  x = jnp.asarray(x)
  assert x.dtype == jnp.bool_, f"{name}: want bool, got {x.dtype}"
  return x


def _i(name: str, x: Any) -> jax.Array:
  """Checks `x` is an integer array.

  Args:
    name: The array's name, for errors.
    x: The array.

  Returns:
    `x` as an array.
  """
  x = jnp.asarray(x)
  assert jnp.issubdtype(
      x.dtype, jnp.integer
  ), f"{name}: want an int dtype, got {x.dtype}"
  return x


def _coef(x: Any) -> Any:
  """A Python float stays weak-typed; an array is cast to f32.

  Args:
    x: A Python number or an array.

  Returns:
    The coefficient.
  """
  if isinstance(x, (int, float)):
    return float(x)
  return f32(x)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
  """Mean of x over mask-True entries; 0 (zero gradient) for an empty mask.

  Args:
    x: f32 values.
    mask: bool, the entries to average (x's shape).

  Returns:
    The f32 mean over the mask, 0 when the mask is empty.
  """
  mask = _b("mask", mask)
  assert x.shape == mask.shape, (x.shape, mask.shape)
  m = mask.astype(x.dtype)
  return jnp.where(mask, x, jnp.zeros_like(x)).sum() / jnp.maximum(m.sum(), 1.0)


def target_argmax(target, legal) -> jax.Array:
  """i32[...]: the teacher's top legal action.

  The first maximal index; illegal entries count -1, below every probability.

  Args:
    target: f32[..., A] the teacher's distribution.
    legal: bool[..., A] legal actions.

  Returns:
    i32[...] the action.
  """
  return jnp.argmax(jnp.where(legal, target, _F32(-1.0)), axis=-1).astype(
      jnp.int32
  )


def top1_agreement(logits, legal, ref_action, mask) -> jax.Array:
  """f32: masked mean of argmax(masked student logits) == ref_action.

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.
    ref_action: i32[...] the reference action.
    mask: bool[...] rows to average.

  Returns:
    The f32 agreement rate over the mask.
  """
  pred = jnp.argmax(masked_logits(logits, legal), axis=-1).astype(jnp.int32)
  return masked_mean(
      (pred == jnp.asarray(ref_action).astype(jnp.int32)).astype(_F32), mask
  )


def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
  """f32 logits with illegal entries replaced by NEG.

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[..., A] logits with NEG at illegal actions.
  """
  logits = _f("logits", logits)
  legal = _b("legal", legal)
  assert logits.shape == legal.shape, (logits.shape, legal.shape)
  return jnp.where(legal, logits, _F32(NEG))


def masked_log_softmax(logits: jax.Array, legal: jax.Array) -> jax.Array:
  """Legal-masked log-softmax (illegal entries ~ NEG, never read).

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[..., A] log-probabilities.
  """
  return jax.nn.log_softmax(masked_logits(logits, legal), axis=-1)


def action_log_prob(
    logits: jax.Array, legal: jax.Array, action: jax.Array
) -> jax.Array:
  """Log pi(action) under the legal-masked softmax, [...].

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.
    action: int[...] the action taken.

  Returns:
    f32[...] log pi(action).
  """
  action = _i("action", action)
  lp = masked_log_softmax(logits, legal)
  return jnp.take_along_axis(lp, action[..., None].astype(jnp.int32), axis=-1)[
      ..., 0
  ]


def entropy(logits: jax.Array, legal: jax.Array) -> jax.Array:
  """Per-row entropy of the legal-masked softmax.

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] entropy per row.
  """
  lp = masked_log_softmax(logits, legal)
  p = jnp.exp(lp)
  return -jnp.sum(jnp.where(legal, p * lp, _F32(0.0)), axis=-1)


def normalize_advantages(
    adv: jax.Array, mask: jax.Array, eps: float = 1e-10
) -> jax.Array:
  """(adv - mean) / (std + eps), mean and std over the mask rows.

  An empty mask uses every row, so the output stays defined.

  Args:
    adv: f32[...] advantages.
    mask: bool[...] rows the moments are taken over.
    eps: added to the std.

  Returns:
    f32[...] the normalised advantages.
  """
  adv = _f("advantages", adv)
  mask = _b("mask", mask)
  m = mask.astype(_F32)
  eff = jnp.where(m.sum() <= 0.0, jnp.ones_like(m), m)
  denom = eff.sum()
  mean = (adv * eff).sum() / denom
  var = (jnp.square(adv - mean) * eff).sum() / denom
  return (adv - mean) / (jnp.sqrt(var) + eps)


def ppo_clip_objective(
    ratio: jax.Array, adv: jax.Array, *, clip_eps: float
) -> jax.Array:
  """Per-row clipped surrogate LOSS: -min(r A, clip(r, 1-eps, 1+eps) A).

  Args:
    ratio: f32[...] importance ratios.
    adv: f32[...] advantages.
    clip_eps: the clip range.

  Returns:
    f32[...] the per-row surrogate loss.
  """
  # f32 bounds: a python-float bound would be traced as an f64 scalar;
  # f32(1 - eps) rounds the same double, so the clipped values are the same.
  lo, hi = jnp.float32(1.0 - clip_eps), jnp.float32(1.0 + clip_eps)
  return -jnp.minimum(ratio * adv, jnp.clip(ratio, lo, hi) * adv)


def clip_fraction(ratio: jax.Array, *, clip_eps: float) -> jax.Array:
  """Per-row 1.0 where |ratio - 1| > clip_eps.

  Args:
    ratio: f32[...] importance ratios.
    clip_eps: the clip range.

  Returns:
    f32[...] 1.0 where clipped, else 0.0.
  """
  return (jnp.abs(ratio - 1.0) > clip_eps).astype(_F32)


def kl_to_reference(
    logits: jax.Array, reference_logits: jax.Array, legal: jax.Array
) -> jax.Array:
  """Per-row reverse KL(pi || ref) over legal actions.

  Both distributions are renormalised over the legal set.

  Args:
    logits: f32[..., A] policy logits.
    reference_logits: f32[..., A] the reference policy's logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] KL per row.
  """
  ref = jax.lax.stop_gradient(
      masked_logits(_f("reference_logits", reference_logits), legal)
  )
  lp = masked_log_softmax(logits, legal)
  lr = jax.nn.log_softmax(ref, axis=-1)
  p = jnp.exp(lp)
  return jnp.sum(jnp.where(legal, p * (lp - lr), _F32(0.0)), axis=-1)


def kl_to_uniform(logits: jax.Array, legal: jax.Array) -> jax.Array:
  """Per-row KL(pi || uniform over legal) = log(#legal) - H(pi).

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] KL per row.
  """
  n = jnp.maximum(legal.sum(-1), 1).astype(_F32)
  return jnp.log(n) - entropy(logits, legal)


def value_squared_error(value: jax.Array, target: jax.Array) -> jax.Array:
  """Per-row (value - target)^2 (no 0.5: each composition states its own).

  Args:
    value: f32[...] predictions.
    target: f32[...] targets.

  Returns:
    f32[...] squared error per row.
  """
  return jnp.square(_f("value", value) - _f("target", target))


def clipped_value_squared_error(
    value: jax.Array,
    value_old: jax.Array,
    target: jax.Array,
    *,
    clip_eps: float,
) -> jax.Array:
  """Per-row PPO clipped value error.

  max((v-t)^2, (v_old + clip(v-v_old)-t)^2).

  Args:
    value: f32[...] predictions.
    value_old: f32[...] the behaviour values.
    target: f32[...] targets.
    clip_eps: the clip range.

  Returns:
    f32[...] the clipped squared error per row.
  """
  value = _f("value", value)
  value_old = _f("value_old", value_old)
  target = _f("target", target)
  clipped = value_old + jnp.clip(value - value_old, -clip_eps, clip_eps)
  return jnp.maximum(jnp.square(value - target), jnp.square(clipped - target))


# --- HL-Gauss (Farebrother et al. 2024): the ONE implementation is
# networks.hl_gauss (the value head decodes with it); the loss uses the
# same Support type and functions, adding only the f32 input checks and the
# stop-gradient on the target.

HLGaussSupport = hl_gauss.Support
hl_gauss_support = hl_gauss.make_support
hl_gauss_logits_to_scalar = hl_gauss.logits_to_scalar


def check_value_support(
    loss_value: str, policy: Any
) -> Optional[HLGaussSupport]:
  """The support an HL-Gauss value loss trains on.

  The policy's value head's (policy.value_support()), so loss and network can
  never disagree.

  Args:
    loss_value: the loss's value kind; any kind but "hl_gauss" returns None.
    policy: the policy being trained.

  Returns:
    The HLGaussSupport, or None for a non-HL-Gauss loss.

  Raises:
    ValueError: the loss is HL-Gauss and the policy declares no support (a
      scalar head).
  """
  if loss_value != "hl_gauss":
    return None
  get = getattr(policy, "value_support", None)
  if get is None:
    raise ValueError(
        "an hl_gauss value loss needs a policy that declares its value "
        "support "
        f"(value_support()); {type(policy).__name__} does not"
    )
  head = get()
  if head is None:
    raise ValueError(
        f"an hl_gauss value loss over {type(policy).__name__} whose value head "
        "is "
        "scalar"
    )
  assert isinstance(head, HLGaussSupport), type(head)
  return head


def hl_gauss_target_probs(
    target: jax.Array, support: HLGaussSupport
) -> jax.Array:
  """Scalar targets -> HL-Gauss soft labels [..., num_bins].

  Targets are clamped to the support and renormalised by the in-support Gaussian
  mass.

  Args:
    target: f32[...] scalar targets.
    support: the HLGaussSupport.

  Returns:
    f32[..., num_bins] soft labels.
  """
  return hl_gauss.target_to_probs(_f("target", target), support)


def hl_gauss_cross_entropy(
    logits: jax.Array, target: jax.Array, support: HLGaussSupport
) -> jax.Array:
  """Per-row CE between softmax(logits) and the HL-Gauss target.

  Args:
    logits: f32[..., num_bins] value logits.
    target: f32[...] scalar targets.
    support: the HLGaussSupport.

  Returns:
    f32[...] CE per row.
  """
  logits = _f("value_logits", logits)
  assert logits.shape[-1] == support.num_bins, (logits.shape, support.num_bins)
  return hl_gauss.cross_entropy(
      logits, jax.lax.stop_gradient(_f("target", target)), support
  )


def z_loss(logits: jax.Array, legal: jax.Array) -> jax.Array:
  """Per-row logsumexp(legal logits)^2 (keeps the log-normaliser near 0).

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] the squared log-normaliser per row.
  """
  return jnp.square(jax.nn.logsumexp(masked_logits(logits, legal), axis=-1))


def cross_entropy_to_target(
    logits: jax.Array, target_policy: jax.Array, legal: jax.Array
) -> jax.Array:
  """Per-row soft-target CE -sum_a t(a) log pi(a) over legal actions.

  Args:
    logits: f32[..., A] policy logits.
    target_policy: f32[..., A] the target distribution.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] CE per row.
  """
  t = jax.lax.stop_gradient(_f("target_policy", target_policy))
  lp = masked_log_softmax(logits, legal)
  return -jnp.sum(jnp.where(legal, t * lp, _F32(0.0)), axis=-1)


def cross_entropy_to_action(
    logits: jax.Array, legal: jax.Array, action: jax.Array
) -> jax.Array:
  """Per-row -log pi(action) (one-hot BC).

  Args:
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.
    action: int[...] the target action.

  Returns:
    f32[...] CE per row.
  """
  return -action_log_prob(logits, legal, action)


def kl_from_target(
    target_policy: jax.Array, logits: jax.Array, legal: jax.Array
) -> jax.Array:
  """Per-row forward KL(target || pi) over legal actions (distillation).

  Args:
    target_policy: f32[..., A] the target distribution.
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.

  Returns:
    f32[...] KL per row.
  """
  t = jax.lax.stop_gradient(_f("target_policy", target_policy))
  lp = masked_log_softmax(logits, legal)
  keep = legal & (t > 0)
  log_t = jnp.log(jnp.where(keep, t, _F32(1.0)))
  return jnp.sum(jnp.where(keep, t * (log_t - lp), _F32(0.0)), axis=-1)


def sigmoid_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
  """Per-row mean over the last axis of binary CE (optax form).

  Args:
    logits: f32[..., K] logits.
    labels: f32[..., K] labels in {0, 1}.

  Returns:
    f32[...] mean binary CE per row.
  """
  logits = _f("aux_logits", logits)
  labels = jax.lax.stop_gradient(_f("labels", labels))
  per = -labels * jax.nn.log_sigmoid(logits) - (
      1.0 - labels
  ) * jax.nn.log_sigmoid(-logits)
  return jnp.mean(per, axis=-1)


AUX_KINDS = ("sigmoid_ce", "softmax_ce", "mse")


def aux_loss(kind: str, pred: jax.Array, label: jax.Array) -> jax.Array:
  """Per-row aux-head loss on a privileged label (TrajectoryBatch.labels).

  sigmoid_ce: pred logits [..., K], label {0,1} [..., K], mean over K.
  softmax_ce: pred logits [..., K], label int class [...].
  mse:        pred [..., K], label [..., K], mean over K.

  Args:
    kind: one of AUX_KINDS (above).
    pred: the aux head's output.
    label: the privileged label.

  Returns:
    f32[...] the loss per row.

  Raises:
    ValueError: kind is not in AUX_KINDS.
  """
  if kind == "sigmoid_ce":
    return sigmoid_cross_entropy(pred, jnp.asarray(label).astype(_F32))
  if kind == "softmax_ce":
    label = _i("label", label)
    lp = jax.nn.log_softmax(_f("aux_logits", pred), axis=-1)
    return -jnp.take_along_axis(
        lp, label[..., None].astype(jnp.int32), axis=-1
    )[..., 0]
  if kind == "mse":
    return jnp.mean(
        jnp.square(_f("aux_pred", pred) - jnp.asarray(label).astype(_F32)),
        axis=-1,
    )
  raise ValueError(f"unknown aux loss kind {kind!r}; want one of {AUX_KINDS}")


def _masked_quantile(values: jax.Array, mask: jax.Array, q: float) -> jax.Array:
  """The q-quantile of the masked entries, by linear interpolation.

  Args:
    values: The values.
    mask: Which values count.
    q: The quantile in [0, 1].

  Returns:
    The quantile, or inf when no entry counts.
  """
  # Linear interpolation between order statistics of the mask-True entries.
  flat = jnp.where(mask, values, jnp.inf).reshape(-1)
  s = jnp.sort(flat)
  n = mask.sum().astype(values.dtype)
  pos = jnp.clip(q * (n - 1.0), 0.0, flat.size - 1)
  lo = jnp.floor(pos).astype(jnp.int32)
  hi = jnp.ceil(pos).astype(jnp.int32)
  frac = pos - lo.astype(values.dtype)
  qv = s[lo] * (1.0 - frac) + s[hi] * frac
  return jnp.where(n > 0.0, qv, jnp.inf)


def advantage_filter_mask(
    adv: jax.Array, mask: jax.Array, *, quantile: float, floor: float
) -> jax.Array:
  """Bool rows with |A| >= masked quantile(|A|, q) and |A| >= floor.

  Within mask (a quantile filter on advantages, mask form).

  Args:
    adv: f32[...] advantages.
    mask: bool[...] candidate rows.
    quantile: q in [0, 1].
    floor: the minimum |A|.

  Returns:
    bool[...] the kept rows.
  """
  a = jnp.abs(_f("advantages", adv))
  mask = _b("mask", mask)
  thr = jnp.where(
      mask.all(), jnp.quantile(a, quantile), _masked_quantile(a, mask, quantile)
  )
  return (a >= thr) & (a >= floor) & mask


# ---------------------------------------------------------------------------
# Counters: the moving magnet, value EMA
# ---------------------------------------------------------------------------


_step = schedules.counter  # pylint: disable=invalid-name


def magnet_due(step_after: Any, every: int) -> jax.Array:
  """bool: the magnet is re-taken after this update.

  (update_step + 1) % every == 0 with step_after = update_step + 1 (int64).

  Args:
    step_after: the update counter after this update (int64).
    every: the refresh period in updates.

  Returns:
    A bool scalar.
  """
  assert every >= 1, every
  return (_step(step_after) % jnp.int64(every)) == 0


def refresh_magnet(
    params: Any, magnet: Any, step_after: Any, every: int
) -> Any:
  """The moving magnet: params where magnet_due(step_after, every) else magnet.

  Args:
    params: the current params.
    magnet: the current magnet (params-shaped).
    step_after: the update counter after this update (int64).
    every: the refresh period in updates.

  Returns:
    The new magnet.
  """
  due = magnet_due(step_after, every)
  return jax.tree.map(lambda p, m: jnp.where(due, p, m), params, magnet)


class ValueEMA(NamedTuple):
  """Running EMA of value-target moments, f32[1] leaves.

  Attributes:
    mu: the EMA of the mean.
    std: the EMA of the std.
    debias: the bias correction factor.
    mu_debiased: the debiased mean.
    std_debiased: the debiased std.
  """

  mu: jax.Array
  std: jax.Array
  debias: jax.Array
  mu_debiased: jax.Array
  std_debiased: jax.Array


def value_ema_init() -> ValueEMA:
  """The fresh ValueEMA (all zeros).

  Returns:
    A ValueEMA of f32[1] zeros.
  """
  z = jnp.zeros((1,), _F32)
  return ValueEMA(z, z, z, z, z)


def value_ema_update(
    ema: ValueEMA, targets: jax.Array, beta: float, mask: jax.Array
) -> ValueEMA:
  """One minibatch update from the mask rows; an empty mask keeps the state.

  Args:
    ema: the current ValueEMA.
    targets: f32[...] value targets.
    beta: the EMA decay.
    mask: bool[...] rows to read.

  Returns:
    The updated ValueEMA.
  """
  targets = _f("targets", targets)
  mask = _b("mask", mask)
  mean = jnp.where(mask.all(), targets.mean(), masked_mean(targets, mask))
  var = masked_mean(jnp.square(targets - mean), mask)
  std = jnp.where(mask.all(), jnp.std(targets), jnp.sqrt(var))
  mu = beta * ema.mu + (1 - beta) * mean
  sd = beta * ema.std + (1 - beta) * std
  db = beta * ema.debias + (1 - beta)
  new = ValueEMA(mu, sd, db, mu / db, sd / db)
  return jax.tree.map(
      lambda n, o: jnp.where(jnp.any(mask), n, o).astype(_F32), new, ema
  )


# The smallest scale value_ema_normalize divides by: targets of (nearly)
# constant value (std 0) would otherwise be scaled, with their gradients, by
# about 1e10.
VALUE_EMA_MIN_STD = 1e-3


def _ema_small(ema: ValueEMA) -> jax.Array:
  """Whether the value EMA's std is below its floor.

  Args:
    ema: The value EMA.

  Returns:
    bool[].
  """
  return ema.std_debiased + 1e-10 < VALUE_EMA_MIN_STD


def value_ema_normalize(x: jax.Array, ema: ValueEMA) -> jax.Array:
  """(x - mean) / std of the EMA, with std floored at VALUE_EMA_MIN_STD.

  Args:
    x: f32 values.
    ema: the ValueEMA.

  Returns:
    The normalised values.
  """
  scale = jnp.where(
      _ema_small(ema),
      jnp.asarray(VALUE_EMA_MIN_STD, _F32),
      ema.std_debiased + 1e-10,
  )
  return (x - ema.mu_debiased) / scale


def value_ema_denormalize(x: jax.Array, ema: ValueEMA) -> jax.Array:
  """X * std + mean of the EMA, unfloored.

  The fresh EMA (all zeros, before its first update) maps every value to 0. It
  inverts value_ema_normalize wherever the EMA's std is at least
  VALUE_EMA_MIN_STD.

  Args:
    x: normalised values.
    ema: the ValueEMA.

  Returns:
    The values in target units.
  """
  return x * ema.std_debiased + ema.mu_debiased


# ---------------------------------------------------------------------------
# Composed losses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PGLossConfig:
  """The policy-gradient loss: PPO's clipped surrogate, plus MMD's magnet KL.

  total = L_clip(normalized A) + vf_coef * L_value - ent_coef * H
          + kl_coef * KL(pi || magnet)          [mmd: reference_logits given]
  L_value: scalar head 0.5 * (clipped if clip_value) squared error, or the
  HL-Gauss cross-entropy (value_head="hl_gauss"; Imani and White 2018,
  Farebrother et al. 2024).

  Attributes:
    clip_eps: the surrogate's (and the clipped value's) clip range.
    vf_coef: the value loss's weight.
    clip_value: clip the scalar value around the behaviour value.
    normalize_advantages: normalize the advantages over the minibatch's rows.
    value_head: "scalar" or "hl_gauss" (the support is the value head's,
      value_support).
    kl_init_to_uniform: on the run's first update, the magnet KL pulls toward
      uniform instead of the magnet.
    use_advantage_filter: train the surrogate and the KL only on rows with
      |A| at or above the masked quantile and the floor (all rows when none
      is kept).
    advantage_filter_quantile: the filter's quantile q of |A|.
    advantage_filter_floor: the filter's minimum |A|.
  """

  clip_eps: float = 0.2
  vf_coef: float = 0.5
  clip_value: bool = False
  normalize_advantages: bool = True
  value_head: str = "scalar"
  kl_init_to_uniform: bool = False
  use_advantage_filter: bool = False
  advantage_filter_quantile: float = 0.75
  advantage_filter_floor: float = 0.01

  def __post_init__(self):
    assert self.value_head in ("scalar", "hl_gauss"), self.value_head
    assert self.clip_eps > 0.0, self.clip_eps
    if self.value_head == "hl_gauss":
      assert not self.clip_value, "clip_value applies to the scalar head only"


def _sum_over(x: jax.Array, mask: jax.Array, count: jax.Array) -> jax.Array:
  """Sum of x over mask-True entries / max(count, 1).

  masked_mean when count is mask's own count, a share of a larger set's mean
  when count is that set's (pg_stats).

  Args:
    x: the values.
    mask: bool, x's shape: the entries summed.
    count: the divisor (floored at 1).

  Returns:
    The scalar sum / max(count, 1).
  """
  mask = _b("mask", mask)
  assert x.shape == mask.shape, (x.shape, mask.shape)
  return jnp.where(mask, x, jnp.zeros_like(x)).sum() / jnp.maximum(
      count.astype(x.dtype), 1.0
  )


def _value_term(
    cfg_head: str,
    support,
    *,
    value,
    value_logits,
    value_old,
    targets,
    mask,
    count,
    clip_value: bool,
    clip_eps: float,
):
  """The value loss and the fraction of clipped values.

  Args:
    cfg_head: the value head, "scalar" or "hl_gauss".
    support: the hl_gauss head's support.
    value: the predicted values.
    value_logits: the hl_gauss head's logits.
    value_old: the behavior values (clip_value only).
    targets: the value targets.
    mask: bool, value's shape: the entries summed.
    count: the divisor (floored at 1).
    clip_value: whether to clip the scalar value error.
    clip_eps: the clip range.

  Returns:
    (the value loss, the clipped fraction).
  """
  if cfg_head == "hl_gauss":
    assert value_logits is not None, "value_head=hl_gauss needs value_logits"
    assert (
        support is not None
    ), "value_head=hl_gauss needs value_support (the head's support)"
    return (
        _sum_over(
            hl_gauss_cross_entropy(value_logits, targets, support), mask, count
        ),
        _F32(0.0),
    )
  if clip_value:
    assert value_old is not None, "clip_value needs value_old"
    err = clipped_value_squared_error(
        value, value_old, targets, clip_eps=clip_eps
    )
    frac = _sum_over(
        (jnp.abs(value - value_old) > clip_eps).astype(_F32), mask, count
    )
    return 0.5 * _sum_over(err, mask, count), frac
  return 0.5 * _sum_over(
      value_squared_error(value, targets), mask, count
  ), _F32(0.0)


class PGStats(NamedTuple):
  """The terms of pg_loss that read no network output, over a minibatch.

  The terms of pg_loss that read no network output, over a whole
  minibatch (pg_stats). Given to pg_loss for a chunk of the minibatch's rows
  (adv and actor_rows cut to the chunk, the rest as they are), every mean
  divides by the minibatch's count, so the chunks' losses, their gradients
  and the metrics in PG_CHUNK_SUMS add up to the minibatch's; the other
  metrics are the minibatch's in every chunk.

  Attributes:
    adv: f32[rows]: the advantages the surrogate trains on (normalized per
      cfg).
    actor_rows: bool[rows]: the rows the surrogate and the KL train on.
    mask_count: f32: mask-True rows.
    actor_count: f32: actor_rows-True rows.
    adv_mean: f32: the raw advantages' mean over the mask.
    adv_std: f32: the raw advantages' std over the mask.
    keep_fraction: f32: the share of mask rows in actor_rows.
    fallback: f32: 1 when the advantage filter kept no row (every mask row
      trains), else 0.
  """

  adv: jax.Array
  actor_rows: jax.Array
  mask_count: jax.Array
  actor_count: jax.Array
  adv_mean: jax.Array
  adv_std: jax.Array
  keep_fraction: jax.Array
  fallback: jax.Array


#: pg_loss metrics that add up over the chunks of a minibatch (PGStats).
PG_CHUNK_SUMS = frozenset({
    "loss_total",
    "loss_policy",
    "loss_value",
    "entropy",
    "kl_penalty",
    "clip_frac_policy",
    "clip_frac_value",
})


def pg_stats(
    cfg: PGLossConfig, mask: jax.Array, advantages: jax.Array
) -> PGStats:
  """PGStats of a minibatch's rows.

  Args:
    cfg: the PGLossConfig.
    mask: bool[...] the rows that train (TrajectoryBatch.valid).
    advantages: f32[...] advantages, any leading shape.

  Returns:
    The PGStats.
  """
  mask = _b("mask", mask)
  advantages = _f("advantages", advantages)
  assert mask.shape == advantages.shape, (mask.shape, advantages.shape)
  adv = (
      normalize_advantages(advantages, mask)
      if cfg.normalize_advantages
      else advantages
  )
  actor_rows = mask
  fallback = _F32(0.0)
  if cfg.use_advantage_filter:
    keep = advantage_filter_mask(
        advantages,
        mask,
        quantile=cfg.advantage_filter_quantile,
        floor=cfg.advantage_filter_floor,
    )
    none_kept = ~jnp.any(keep)
    actor_rows = jnp.where(
        none_kept, mask, keep
    )  # explicit fallback, reported below
    fallback = none_kept.astype(_F32)
  adv_mean = masked_mean(advantages, mask)
  return PGStats(
      adv=adv,
      actor_rows=actor_rows,
      mask_count=mask.astype(_F32).sum(),
      actor_count=actor_rows.astype(_F32).sum(),
      adv_mean=adv_mean,
      adv_std=jnp.sqrt(masked_mean(jnp.square(advantages - adv_mean), mask)),
      keep_fraction=masked_mean(actor_rows.astype(_F32), mask),
      fallback=fallback,
  )


def pg_loss(
    cfg: PGLossConfig,
    *,
    logits: jax.Array,
    value: jax.Array,
    legal: jax.Array,
    action: jax.Array,
    behavior_logp: jax.Array,
    mask: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    value_old: Optional[jax.Array] = None,
    value_logits: Optional[jax.Array] = None,
    reference_logits: Optional[jax.Array] = None,
    kl_coef: Any = 0.0,
    ent_coef: Any = 0.0,
    is_first_update: Any = False,
    value_support: Optional[HLGaussSupport] = None,
    stats: Optional[PGStats] = None,
):
  """PPO (reference_logits=None) or MMD (magnet KL) loss over rows.

  Rows are any leading shape ([N] or [B, T]); logits [..., A].

  Args:
    cfg: the PGLossConfig.
    logits: f32[..., A] policy logits.
    value: f32[...] value predictions.
    legal: bool[..., A] legal actions.
    action: int[...] the actions taken.
    behavior_logp: f32[...] log pi_behaviour(action)
      (TrajectoryBatch.behavior_logp).
    mask: bool[...] the rows that train (TrajectoryBatch.valid).
    advantages: f32[...] from advantages.gae.
    targets: f32[...] value targets from advantages.gae.
    value_old: f32[...] behaviour values (clipped value loss).
    value_logits: f32[..., num_bins] value logits (value_head="hl_gauss").
    reference_logits: the magnet's logits on the same rows (stop-gradient);
      is_first_update with cfg.kl_init_to_uniform swaps them for uniform.
    kl_coef: the magnet KL coefficient.
    ent_coef: the entropy coefficient.
    is_first_update: bool, the run's first update.
    value_support: the value head's HL-Gauss support (value_head="hl_gauss").
    stats: the minibatch's PGStats when these rows are one chunk of it (None:
      these rows are the minibatch, pg_stats of them).

  Returns:
    (total f32, metrics dict of f32 scalars).
  """
  mask = _b("mask", mask)
  logits = _f("logits", logits)
  legal = _b("legal", legal)
  behavior_logp = _f("behavior_logp", behavior_logp)
  advantages = _f("advantages", advantages)
  assert (
      logits.shape[:-1] == mask.shape == advantages.shape == behavior_logp.shape
  ), (logits.shape, mask.shape, advantages.shape, behavior_logp.shape)
  kl_coef = _coef(kl_coef)
  ent_coef = _coef(ent_coef)

  if stats is None:
    stats = pg_stats(cfg, mask, advantages)
  adv, actor_rows = stats.adv, _b("actor_rows", stats.actor_rows)
  assert adv.shape == actor_rows.shape == mask.shape, (
      adv.shape,
      actor_rows.shape,
      mask.shape,
  )
  n_mask, n_actor = stats.mask_count, stats.actor_count

  logp = action_log_prob(logits, legal, action)
  ratio = jnp.exp(logp - behavior_logp)
  value_loss, clip_frac_value = _value_term(
      cfg.value_head,
      value_support,
      value=value,
      value_logits=value_logits,
      value_old=value_old,
      targets=targets,
      mask=mask,
      count=n_mask,
      clip_value=cfg.clip_value,
      clip_eps=cfg.clip_eps,
  )
  ent = _sum_over(entropy(logits, legal), mask, n_mask)

  loss_policy = _sum_over(
      ppo_clip_objective(ratio, adv, clip_eps=cfg.clip_eps), actor_rows, n_actor
  )
  clip_frac_policy = _sum_over(
      clip_fraction(ratio, clip_eps=cfg.clip_eps), actor_rows, n_actor
  )

  if reference_logits is None:
    kl = _F32(jnp.nan)
    kl_term = _F32(0.0)
  else:
    ref = _f("reference_logits", reference_logits)
    if cfg.kl_init_to_uniform:
      ref = jnp.where(jnp.asarray(is_first_update), jnp.zeros_like(ref), ref)
    kl = _sum_over(kl_to_reference(logits, ref, legal), actor_rows, n_actor)
    kl_term = kl_coef * kl

  total = loss_policy + cfg.vf_coef * value_loss - ent_coef * ent + kl_term
  metrics = {
      "loss_total": total,
      "loss_policy": loss_policy,
      "loss_value": value_loss,
      "entropy": ent,
      "kl_penalty": kl,
      "clip_frac_policy": clip_frac_policy,
      "clip_frac_value": clip_frac_value,
      "advantage_mean": stats.adv_mean,
      "advantage_std": stats.adv_std,
      "kl_coef_effective": f32(kl_coef),
      "ent_coef_effective": f32(ent_coef),
      "advantage_filter_keep_fraction": stats.keep_fraction,
      "advantage_filter_fallback": stats.fallback,
  }
  metrics = {k: f32(v) for k, v in metrics.items()}
  assert total.dtype == _F32, total.dtype
  return total, metrics


class AuxTerm(NamedTuple):
  """An aux head on a privileged label: aux_preds[name] vs labels[name].

  Attributes:
    name: the label (and aux head) name.
    kind: one of AUX_KINDS.
    coef: the term's weight in the total.
  """

  name: str
  kind: str
  coef: float


@dataclasses.dataclass(frozen=True)
class DistillLossConfig:
  """BC / AZ / teacher distillation over target_policy, target_value, labels.

  total = L_policy + value_coef * L_value + z_loss_coef * z
          + sum aux coef * L_aux

  Attributes:
    policy: "ce_action" (one-hot on `action`), "ce_target" (soft CE to
      target_policy; AZ visits, BC teacher distribution), "kl_target"
      (KL(target || pi); teacher distillation). Same gradient for
      ce_target/kl_target.
    value: "none", "mse" (no 0.5: AZ's l2_loss is value_coef=0.5), "hl_gauss"
      (on the value head's support, passed to distill_loss as value_support).
    value_coef: the value loss's weight.
    z_loss_coef: the weight of the logits' z-loss (0: off).
    aux_terms: the AuxTerms on privileged labels.
  """

  policy: str = "ce_target"
  value: str = "mse"
  value_coef: float = 1.0
  z_loss_coef: float = 0.0
  aux_terms: Sequence[AuxTerm] = ()

  def __post_init__(self):
    assert self.policy in ("ce_action", "ce_target", "kl_target"), self.policy
    assert self.value in ("none", "mse", "hl_gauss"), self.value
    for t in self.aux_terms:
      assert t.kind in AUX_KINDS, t


def distill_loss(
    cfg: DistillLossConfig,
    *,
    logits: jax.Array,
    legal: jax.Array,
    mask: jax.Array,
    action: Optional[jax.Array] = None,
    target_policy: Optional[jax.Array] = None,
    value: Optional[jax.Array] = None,
    value_logits: Optional[jax.Array] = None,
    target_value: Optional[jax.Array] = None,
    value_mask: Optional[jax.Array] = None,
    aux_preds: Optional[Mapping[str, jax.Array]] = None,
    labels: Optional[Mapping[str, jax.Array]] = None,
    value_support: Optional[HLGaussSupport] = None,
    policy_mask: Optional[jax.Array] = None,
):
  """BC/AZ/distillation loss over rows.

  policy_accuracy is the argmax agreement with `action` if given, else with
  argmax(target_policy).

  Args:
    cfg: the DistillLossConfig.
    logits: f32[..., A] policy logits.
    legal: bool[..., A] legal actions.
    mask: bool[...] policy / z / aux rows.
    action: int[...] the reference action (one-hot BC, accuracy).
    target_policy: f32[..., A] target distribution.
    value: f32[...] value predictions.
    value_logits: f32[..., num_bins] value logits (value="hl_gauss").
    target_value: f32[...] value targets.
    value_mask: bool[...] value rows (default = mask; e.g. BC drops states whose
      game did not finish).
    aux_preds: {name: aux head output}.
    labels: {name: privileged label}.
    value_support: the value head's HL-Gauss support.
    policy_mask: policy / z / accuracy rows when given (default = mask;
      BCConfig.policy_mask "valid&search_decided" narrows it to the rows a
      search decided, the aux heads keep `mask`).

  Returns:
    (total f32, metrics dict of f32 scalars).
  """
  logits = _f("logits", logits)
  legal = _b("legal", legal)
  mask = _b("mask", mask)
  assert logits.shape[:-1] == mask.shape, (logits.shape, mask.shape)
  value_mask = mask if value_mask is None else _b("value_mask", value_mask)
  aux_mask = mask
  if policy_mask is not None:
    mask = _b("policy_mask", policy_mask)
    assert mask.shape == aux_mask.shape, (mask.shape, aux_mask.shape)

  if cfg.policy == "ce_action":
    assert action is not None, "policy=ce_action needs action"
    per = cross_entropy_to_action(logits, legal, action)
  elif cfg.policy == "ce_target":
    assert target_policy is not None, "policy=ce_target needs target_policy"
    per = cross_entropy_to_target(logits, target_policy, legal)
  else:
    assert target_policy is not None, "policy=kl_target needs target_policy"
    per = kl_from_target(target_policy, logits, legal)
  loss_policy = masked_mean(per, mask)
  total = loss_policy
  metrics = {"loss_policy": loss_policy}

  if cfg.value == "mse":
    assert value is not None and target_value is not None
    lv = masked_mean(value_squared_error(value, target_value), value_mask)
  elif cfg.value == "hl_gauss":
    assert value_logits is not None and target_value is not None
    assert (
        value_support is not None
    ), "value=hl_gauss needs value_support (the head's support)"
    lv = masked_mean(
        hl_gauss_cross_entropy(value_logits, target_value, value_support),
        value_mask,
    )
  else:
    lv = _F32(0.0)
  total = total + cfg.value_coef * lv
  metrics["loss_value"] = lv

  if cfg.z_loss_coef:
    z = masked_mean(z_loss(logits, legal), mask)
    total = total + cfg.z_loss_coef * z
    metrics["z_loss"] = z

  aux_preds = aux_preds or {}
  labels = labels or {}
  for t in cfg.aux_terms:
    assert (
        t.name in aux_preds
    ), f"aux term {t.name!r}: missing aux_preds[{t.name!r}]"
    assert t.name in labels, f"aux term {t.name!r}: missing labels[{t.name!r}]"
    la = masked_mean(
        aux_loss(t.kind, aux_preds[t.name], labels[t.name]), aux_mask
    )
    total = total + t.coef * la
    metrics[f"aux/{t.name}"] = la

  ref_a = (
      _i("action", action)
      if action is not None
      else target_argmax(target_policy, legal)
  )
  metrics["policy_accuracy"] = top1_agreement(logits, legal, ref_a, mask)
  metrics["loss_total"] = total
  metrics = {k: f32(v) for k, v in metrics.items()}
  assert total.dtype == _F32, total.dtype
  return total, metrics
