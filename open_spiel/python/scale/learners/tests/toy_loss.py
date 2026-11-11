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

"""A toy registered PG loss for the loss-dispatch contract tests.

A toy registered PG loss (learners.pg.register_loss) for the loss-dispatch
contract tests: the extension path (its own targets, loss, magnet
temperature, behaviour log-probs and value mask) with a loss simple enough to
state exactly.

  targets      GAE(lam) advantages and TD(lam) returns at gamma
  loss         -mean(adv * log pi(a)) over policy rows
               + temperature * KL(pi || magnet or uniform) over policy rows
               + vf_coef * MSE(value, returns) over value rows
  temperature  coef / (step + 1) (f32)
  check        refuses value_ema (raw returns)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import advantages as adv_lib
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg

NAME = "toy_pg"
F32 = jnp.float32


@dataclasses.dataclass(frozen=True)
class ToyLossConfig:
  """The toy loss's config.

  Attributes:
    gamma: the discount.
    lam: the GAE / TD lambda.
    vf_coef: the value loss weight.
    temperature_coef: the magnet temperature's scale.
  """

  gamma: float = 1.0
  lam: float = 0.9
  vf_coef: float = 0.5
  temperature_coef: float = 0.1


def targets(cfg: ToyLossConfig, reward, value, node, done, boot):
  """GAE(lam) advantages and TD(lam) returns.

  Args:
    cfg: the ToyLossConfig.
    reward: f32[B, T].
    value: f32[B, T].
    node: bool[B, T] valid plies.
    done: bool[B, T].
    boot: f32[B] bootstrap values.

  Returns:
    (advantages, returns).
  """
  g = adv_lib.gae(reward, value, node, done, boot, gamma=cfg.gamma, lam=cfg.lam)
  return g.advantages, g.returns


def loss(
    cfg: ToyLossConfig,
    *,
    logits,
    value,
    legal,
    action,
    behavior_log_probs,
    advantages,
    returns,
    policy_mask,
    value_mask,
    temperature,
    magnet_logits=None,
):
  """The toy loss (module doc).

  Args:
    cfg: the ToyLossConfig.
    logits: f32[..., A].
    value: f32[...].
    legal: bool[..., A].
    action: int[...].
    behavior_log_probs: unused.
    advantages: f32[...].
    returns: f32[...].
    policy_mask: bool[...].
    value_mask: bool[...].
    temperature: the magnet temperature.
    magnet_logits: f32[..., A], or None (uniform).

  Returns:
    (total f32, terms dict).
  """
  del behavior_log_probs
  lp = loss_lib.masked_log_softmax(logits, legal)
  logp_a = jnp.take_along_axis(
      lp, action[..., None].astype(jnp.int32), axis=-1
  )[..., 0]
  p = jnp.where(legal, jnp.exp(lp), F32(0.0))
  if magnet_logits is None:
    kl = loss_lib.kl_to_uniform(logits, legal)
  else:
    mlp = jax.lax.stop_gradient(
        loss_lib.masked_log_softmax(magnet_logits, legal)
    )
    kl = jnp.sum(jnp.where(legal, p * (lp - mlp), F32(0.0)), axis=-1)
  terms = {
      "policy_loss": -loss_lib.masked_mean(advantages * logp_a, policy_mask),
      "magnet_kl": loss_lib.masked_mean(kl, policy_mask),
      "value_loss": loss_lib.masked_mean(
          loss_lib.value_squared_error(value, returns), value_mask
      ),
  }
  total = (
      terms["policy_loss"]
      + temperature * terms["magnet_kl"]
      + cfg.vf_coef * terms["value_loss"]
  )
  terms["temperature"] = loss_lib.f32(temperature)
  return loss_lib.f32(total), {k: loss_lib.f32(v) for k, v in terms.items()}


def temperature_schedule(cfg: ToyLossConfig, step, floor_step, time_scale):
  """The temperature coef / (step + 1) (f32).

  Args:
    cfg: the ToyLossConfig.
    step: the int64 counter.
    floor_step: unused.
    time_scale: multiplies the counter.

  Returns:
    The f32 temperature.
  """
  del floor_step
  n = jnp.asarray(step, jnp.float64) * float(time_scale)
  return (cfg.temperature_coef / (n + 1.0)).astype(F32)


def check(c) -> None:
  """Refuses value_ema (the loss trains on raw returns).

  Args:
    c: the PGConfig.

  Raises:
    ValueError: c.value_ema is set.
  """
  if c.value_ema:
    raise ValueError(
        f"{NAME} trains the value on raw returns: value_ema must be off"
    )


SPEC = pg.register_loss(
    pg.LossSpec(
        name=NAME,
        config_type=ToyLossConfig,
        targets=targets,
        loss=loss,
        temperature=temperature_schedule,
        check=check,
    )
)
