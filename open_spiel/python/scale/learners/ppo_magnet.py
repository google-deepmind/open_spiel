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

"""PPO with a behaviour KL and an annealed magnet KL (loss "ppo_magnet").

A registered PG loss (learners.pg register_loss): importing this module
registers it, and PGConfig(loss="ppo_magnet", loss_config=PPOMagnetConfig())
selects it. Per minibatch of rows:

  total = policy_coef * PPO-clip surrogate on RAW advantages (no
          normalisation)
        + kl_coef * KL(pi || behaviour)
        + temperature(n) * KL(pi || magnet)
        + vf_coef * MSE(value, returns)
        - entropy_coef * entropy            (0 by default)

The magnet is the uniform policy over the legal actions, or with
PGConfig.magnet_every > 0 the network re-taken every that many updates
(magnetic mirror descent, Sokota et al. 2023). Advantages are GAE(gae_lambda)
and the value targets TD(td_lambda) returns, both at gamma (two
core.advantages.gae calls). The temperature is the inverse-power schedule
clip(temperature_coef / (n + 1)**temperature_decay, floor, ceil) on the
update counter n (PGConfig's schedule_step_offset, temperature_floor_step and
schedule_time_scale apply). With rescale_step S > 0 the schedule reads
n = t * rescale_time_scale from update S on (the clock of a run continued at
S under a stretched time scale, in one run; the counter is not restarted).

Field credit (field_players P > 0): the advantages and value targets are
computed from each seat's reward relative to the field, its own reward minus
the mean of the other P - 1 seats' rewards, (P r_i - sum_j r_j) / (P - 1),
ply by ply. It needs the batch layout of self-play over every seat
(core.arena self_play_table(all_seats=True)): each P consecutive batch rows
are the P seats of one table, in seat order, so a ply's P rewards are the
P rows' rewards at that ply. In a free-for-all game this makes self-play
zero-sum (in hearts a moon then costs every other seat, where 26 - points
is general-sum).

The defaults are the recipe of configs.recipes.ppo_magnet_config: clip 0.2,
behaviour KL 0.1, no entropy bonus, value coefficient 1, gamma 1, GAE(0.5)
advantages, TD(0.8) returns, temperature 0.05 / (n + 1)**0.3 in [0.001, 0.1].
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners import stable_softmax as ss

LOSS = "ppo_magnet"
_F32 = jnp.float32


def _f(name: str, x: Any) -> jax.Array:
  """The input as an array, asserted float32.

  Args:
    name: the input's name (for the message).
    x: the input.

  Returns:
    The array.
  """
  x = jnp.asarray(x)
  assert x.dtype == _F32, f"{name}: want float32, got {x.dtype}"
  return x


def _b(name: str, x: Any) -> jax.Array:
  """The input as an array, asserted bool.

  Args:
    name: the input's name (for the message).
    x: the input.

  Returns:
    The array.
  """
  x = jnp.asarray(x)
  assert x.dtype == jnp.bool_, f"{name}: want bool, got {x.dtype}"
  return x


def _coef(x: Any) -> Any:
  """A Python number stays weak-typed; an array is cast to f32.

  Args:
    x: the coefficient.

  Returns:
    The coefficient.
  """
  if isinstance(x, (int, float)):
    return float(x)
  return loss_lib.f32(x)


@dataclasses.dataclass(frozen=True)
class PPOMagnetConfig:
  """The loss's coefficients and target lambdas (module doc).

  Attributes:
    clip_range: the PPO ratio clip.
    vf_coef: the value-loss coefficient.
    policy_coef: the surrogate's coefficient.
    kl_coef: the KL(pi || behaviour) coefficient.
    entropy_coef: the entropy bonus (0: none).
    gamma: the discount.
    td_lambda: the lambda of the value targets.
    gae_lambda: the lambda of the advantages.
    temperature_coef: the magnet temperature at n = 0 (before the clip).
    temperature_decay: its power.
    temperature_ceil: its upper clip.
    temperature_floor: its lower clip.
    rescale_step: 0 (off), or the update from which the temperature's clock
      runs at rescale_time_scale (core.schedules.Schedule rescale_step).
    rescale_time_scale: that time scale.
    field_players: 0 (off), or P: field credit over groups of P batch rows
      (module doc).
  """

  clip_range: float = 0.2
  vf_coef: float = 1.0
  policy_coef: float = 1.0
  kl_coef: float = 0.1
  entropy_coef: float = 0.0
  gamma: float = 1.0
  td_lambda: float = 0.8
  gae_lambda: float = 0.5
  temperature_coef: float = 0.05
  temperature_decay: float = 0.3
  temperature_ceil: float = 0.1
  temperature_floor: float = 0.001
  rescale_step: int = 0
  rescale_time_scale: float = 1.0
  field_players: int = 0

  def __post_init__(self):
    for name in (
        "clip_range",
        "vf_coef",
        "policy_coef",
        "kl_coef",
        "entropy_coef",
        "gamma",
        "td_lambda",
        "gae_lambda",
        "temperature_coef",
        "temperature_decay",
        "temperature_ceil",
        "temperature_floor",
    ):
      v = getattr(self, name)
      if (
          isinstance(v, bool)
          or not isinstance(v, (int, float))
          or not math.isfinite(v)
      ):
        raise ValueError(
            f"PPOMagnetConfig.{name} must be a finite number, got {v!r}"
        )
    if not 0.0 < self.temperature_floor <= self.temperature_ceil:
      raise ValueError(
          "PPOMagnetConfig needs 0 < temperature_floor <= temperature_ceil,"
          f" got {self.temperature_floor}, {self.temperature_ceil}"
      )
    r, k = self.rescale_step, self.rescale_time_scale
    if isinstance(r, bool) or not isinstance(r, int) or r < 0:
      raise ValueError(
          f"PPOMagnetConfig.rescale_step must be an int >= 0, got {r!r}"
      )
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, float))
        or not math.isfinite(k)
        or not k > 0
        or (r == 0 and float(k) != 1.0)
    ):
      raise ValueError(
          "PPOMagnetConfig.rescale_time_scale must be > 0 (1.0 with "
          f"rescale_step 0), got {k!r}"
      )
    f = self.field_players
    if isinstance(f, bool) or not isinstance(f, int) or f < 0 or f == 1:
      raise ValueError(
          f"PPOMagnetConfig.field_players must be 0 or an int >= 2, got {f!r}"
      )

  def temperature_schedule(
      self, floor_step: int = 0, time_scale: float = 1.0, rescaled: bool = False
  ) -> schedules.Schedule:
    """The magnet temperature's inverse-power Schedule.

    Args:
      floor_step: PGConfig.temperature_floor_step.
      time_scale: PGConfig.schedule_time_scale.
      rescaled: the schedule from rescale_step on (time scale
        rescale_time_scale).

    Returns:
      The Schedule.
    """
    if rescaled:
      if floor_step or time_scale != 1.0:
        raise ValueError(
            "rescale_step takes no temperature_floor_step and no "
            "schedule_time_scale"
        )
      time_scale = float(self.rescale_time_scale)
    return schedules.Schedule(
        "inverse_power",
        self.temperature_coef,
        power=self.temperature_decay,
        floor=self.temperature_floor,
        ceil=self.temperature_ceil,
        floor_step=floor_step,
        time_scale=time_scale,
    )

  def temperature(
      self, step: Any, floor_step: int = 0, time_scale: float = 1.0
  ) -> jax.Array:
    """The f32 temperature at the int64 counter `step`.

    Args:
      step: the counter.
      floor_step: PGConfig.temperature_floor_step.
      time_scale: PGConfig.schedule_time_scale.

    Returns:
      f32[].
    """
    base = self.temperature_schedule(floor_step, time_scale).f32(step)
    if not self.rescale_step:
      return base
    after = self.temperature_schedule(
        floor_step, time_scale, rescaled=True
    ).f32(step)
    return jnp.where(schedules.counter(step) < self.rescale_step, base, after)


def ppo_magnet_loss(
    cfg: PPOMagnetConfig,
    *,
    logits: jax.Array,
    value: jax.Array,
    legal: jax.Array,
    action: jax.Array,
    behavior_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    policy_mask: jax.Array,
    value_mask: jax.Array,
    temperature: Any,
    magnet_logits: Optional[jax.Array] = None,
    counts: Optional[pg.LossCounts] = None,
    stable_softmax: bool = False,
):
  """The loss over rows [N] (module doc).

  Args:
    cfg: the PPOMagnetConfig.
    logits: f32[N, A], the policy's raw logits.
    value: f32[N], the value head.
    legal: bool[N, A].
    action: int[N], the actions taken.
    behavior_log_probs: f32[N, A], the behaviour policy's legal-masked
      log-probs (illegal entries are never read).
    advantages: f32[N].
    returns: f32[N], the value targets.
    policy_mask: bool[N], the decision rows (policy, KL, magnet, entropy).
    value_mask: bool[N], the value-loss rows.
    temperature: the magnet temperature of this update.
    magnet_logits: None (the uniform magnet) or f32[N, A].
    counts: None, or the whole minibatch's row counts these rows are a chunk
      of (pg.LossSpec.chunk_sums): every masked mean divides by them.
    stable_softmax: every softmax term through learners.stable_softmax.

  Returns:
    (total f32[], terms dict of f32[]).
  """
  logits = _f("logits", logits)
  legal = _b("legal", legal)
  pm = _b("policy_mask", policy_mask)
  vm = _b("value_mask", value_mask)
  adv = _f("advantages", advantages)
  beh = loss_lib.masked_logits(
      _f("behavior_log_probs", behavior_log_probs), legal
  )
  assert logits.shape[:-1] == pm.shape == vm.shape == adv.shape, (
      logits.shape,
      pm.shape,
  )
  temperature = _coef(temperature)
  action = jnp.asarray(action)
  assert jnp.issubdtype(action.dtype, jnp.integer), action.dtype
  action = action.astype(jnp.int32)
  pc = vc = None
  if counts is not None:
    assert isinstance(counts, pg.LossCounts), type(counts)
    pc, vc = _f("counts.policy", counts.policy), _f(
        "counts.value", counts.value
    )

  def pmean(x):
    """The policy-masked mean of x.

    Args:
      x: per-row values.

    Returns:
      Their mean over the policy rows.
    """
    return loss_lib.masked_mean(x, pm, pc)

  lp = loss_lib.masked_log_softmax(logits, legal, stable=stable_softmax)
  p = jnp.where(legal, jnp.exp(lp), _F32(0.0))
  logp_a = jnp.take_along_axis(lp, action[..., None], axis=-1)[..., 0]
  old_a = jnp.take_along_axis(beh, action[..., None], axis=-1)[..., 0]
  ratio = jnp.exp(logp_a - old_a)
  surrogate = jnp.minimum(
      adv * ratio, adv * jnp.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
  )
  zero = _F32(0.0)
  if stable_softmax:
    kl_beh = ss.reverse_kl(
        loss_lib.masked_logits(lp, legal), jax.lax.stop_gradient(beh)
    )
  else:
    kl_beh = jnp.sum(jnp.where(legal, p * (lp - beh), zero), axis=-1)
  ent = -jnp.sum(jnp.where(legal, p * lp, zero), axis=-1)
  uniform_kl = pmean(
      loss_lib.kl_to_uniform(logits, legal, stable=stable_softmax)
  )
  if magnet_logits is None:
    magnet_kl = uniform_kl
  elif stable_softmax:
    mz = jax.lax.stop_gradient(
        loss_lib.masked_logits(_f("magnet_logits", magnet_logits), legal)
    )
    magnet_kl = pmean(ss.reverse_kl(loss_lib.masked_logits(logits, legal), mz))
  else:
    mlp = jax.lax.stop_gradient(
        loss_lib.masked_log_softmax(_f("magnet_logits", magnet_logits), legal)
    )
    xe = -jnp.sum(jnp.where(legal, p * mlp, zero), axis=-1)
    magnet_kl = pmean(xe - ent)
  terms = {
      "policy_loss": -pmean(surrogate),
      "kl_loss": pmean(kl_beh),
      "magnet_kl": magnet_kl,
      "entropy": pmean(ent),
      "clip_fraction": pmean(
          loss_lib.clip_fraction(ratio, clip_eps=cfg.clip_range)
      ),
      "uniform_kl": uniform_kl,
      "value_loss": loss_lib.masked_mean(
          loss_lib.value_squared_error(value, returns), vm, vc
      ),
  }
  total = (
      cfg.policy_coef * terms["policy_loss"]
      + temperature * terms["magnet_kl"]
      + cfg.vf_coef * terms["value_loss"]
      + cfg.kl_coef * terms["kl_loss"]
  )
  if cfg.entropy_coef:
    total = total - cfg.entropy_coef * terms["entropy"]
  terms["temperature"] = loss_lib.f32(temperature)
  terms = {k: loss_lib.f32(v) for k, v in terms.items()}
  assert total.dtype == _F32, total.dtype
  return total, terms


def field_rewards(reward: jax.Array, num_players: int) -> jax.Array:
  """Rewards relative to the field over groups of P batch rows (module doc).

  Args:
    reward: f32[B, T], B a multiple of P; rows b P .. b P + P - 1 are the P
      seats of one table.
    num_players: P.

  Returns:
    f32[B, T]: (P r_i - sum_j r_j) / (P - 1) within each group, per ply.
  """
  reward = _f("reward", reward)
  b, t = reward.shape
  p = int(num_players)
  if b % p:
    raise ValueError(
        f"field credit: {b} batch rows are not groups of {p} seats"
    )
  r = reward.reshape(b // p, p, t)
  out = (_F32(p) * r - jnp.sum(r, axis=1, keepdims=True)) / _F32(p - 1)
  return out.reshape(b, t).astype(_F32)


def _targets(cfg: PPOMagnetConfig, reward, value, node, done, boot):
  """(GAE(gae_lambda) advantages, TD(td_lambda) returns).

  Args:
    cfg: the PPOMagnetConfig.
    reward: f32[B, T].
    value: f32[B, T].
    node: bool[B, T], the plies of the chain.
    done: bool[B, T].
    boot: f32[B], the bootstrap values.

  Returns:
    (advantages, returns).
  """
  if cfg.field_players:
    reward = field_rewards(reward, cfg.field_players)
  args = (reward, value, node, done, boot)
  advantages = adv_lib.gae(
      *args, gamma=cfg.gamma, lam=cfg.gae_lambda
  ).advantages
  targets = adv_lib.gae(*args, gamma=cfg.gamma, lam=cfg.td_lambda).returns
  return advantages, targets


def _temperature(cfg: PPOMagnetConfig, step, floor_step, time_scale):
  """The magnet temperature of the update counter `step`.

  Args:
    cfg: the PPOMagnetConfig.
    step: the int64 counter.
    floor_step: PGConfig.temperature_floor_step.
    time_scale: PGConfig.schedule_time_scale.

  Returns:
    f32[].
  """
  return cfg.temperature(step, floor_step=floor_step, time_scale=time_scale)


def _check(c: pg.PGConfig) -> None:
  """Refuses a PGConfig the loss cannot train.

  Args:
    c: the PGConfig.

  Raises:
    ValueError: value_ema on, or a value head that is not scalar.
  """
  if c.value_ema:
    raise ValueError(
        "ppo_magnet trains the value on raw TD(lambda) returns: "
        "value_ema must be off"
    )
  if c.pg.value_head != "scalar":
    raise ValueError(
        "ppo_magnet uses an MSE value loss: scalar value head only"
    )


#: The terms that add up over row chunks of a minibatch (every one a masked
#: mean under counts); "temperature" is the update's in every chunk.
CHUNK_SUMS = frozenset({
    "policy_loss",
    "kl_loss",
    "magnet_kl",
    "entropy",
    "clip_fraction",
    "uniform_kl",
    "value_loss",
})

SPEC = pg.register_loss(
    pg.LossSpec(
        name=LOSS,
        config_type=PPOMagnetConfig,
        targets=_targets,
        loss=ppo_magnet_loss,
        temperature=_temperature,
        check=_check,
        chunk_sums=CHUNK_SUMS,
        stable_softmax=True,
    )
)
