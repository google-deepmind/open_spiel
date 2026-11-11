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

"""LossSpec.step_gate.

The PG learner's per-SGD-step gate (module doc "SGD step gate"), on a gated
variant of the toy registered loss whose gate passes a step while its approx_kl
is at most a configured bound.

An oversized step (a large step size on leduc self-play) drives approx_kl
over the bound at the first step of the second epoch: that step and every
later one are skipped, and the state is the state of the same update run for
one epoch: integer leaves exactly, float leaves up to f32 rounding (rel 1e-5),
as the two are different programs and XLA:CPU's rounding of them is not
portable across CPU models. A gate that always passes gives the ungated update
(up to f32 rounding of the other program); no gate adds nothing to the carry or
the metrics.
"""

import dataclasses
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx
from open_spiel.python.scale.learners.tests import toy_loss

NAME = "toy_pg_gated"
GATE_METRICS = (
    "sgd_early_stopped",
    "sgd_stop_step",
    "sgd_steps_taken",
    "sgd_steps_planned",
)


@dataclasses.dataclass(frozen=True)
class GatedToyConfig(toy_loss.ToyLossConfig):
  """The toy loss's config with an SGD step gate on its KL."""

  max_kl: Optional[float] = None  # None: no gate


def _gate(cfg: GatedToyConfig):
  """The toy loss's step gate: stop once approx_kl passes max_kl.

  Args:
    cfg: The loss config.

  Returns:
    gate(step_metrics) -> bool, or None without max_kl.
  """
  if cfg.max_kl is None:
    return None
  bound = jnp.float32(cfg.max_kl)
  return lambda m: m["approx_kl"] <= bound


SPEC = pg.register_loss(
    dataclasses.replace(
        toy_loss.SPEC, name=NAME, config_type=GatedToyConfig, step_gate=_gate
    )
)

#: A gate returning the f32 metric itself, not a bool: refused.
BAD = pg.register_loss(
    dataclasses.replace(
        SPEC,
        name=NAME + "_bad",
        step_gate=lambda cfg: (lambda m: m["approx_kl"]),
    )
)


def _cfg(max_kl=None, **kw):
  """A PGConfig of the gated toy loss.

  Args:
    max_kl: The gate's bound, or None (no gate).
    **kw: PGConfig fields over the defaults.

  Returns:
    The PGConfig.
  """
  base = dict(
      loss=NAME,
      lr=0.05,
      anneal_lr=False,
      update_epochs=2,
      num_minibatches=4,
      shuffle="flat",
      max_grad_norm=1e9,
      ent_coef=schedules.constant(0.0),
      kl_coef=schedules.constant(0.0),
      loss_config=GatedToyConfig(max_kl=max_kl),
  )
  base.update(kw)
  return pg.PGConfig(**base)


def _setup(b=8, t=16, **kw):
  """A leduc learner, its state and a self-play batch.

  Args:
    b: Rows.
    t: Steps.
    **kw: PGConfig fields.

  Returns:
    (policy, learner, state, batch).
  """
  pol = fx.make_policy("leduc_poker")
  learner = pg.PGLearner(pol, _cfg(**kw))
  s = learner.init(jax.random.key(0), fx.make_env("leduc_poker"))
  batch = fx.self_play_batch(
      pol, s.params, game="leduc_poker", b=b, t=t, seed=1
  )
  return pol, learner, s, batch


def _with(learner, max_kl=None, **kw):
  """`learner` with another max_kl and config fields.

  Args:
    learner: The learner.
    max_kl: The gate's bound, or None.
    **kw: PGConfig fields.

  Returns:
    The PGLearner.
  """
  cfg = dataclasses.replace(
      learner.cfg, loss_config=GatedToyConfig(max_kl=max_kl), **kw
  )
  return pg.PGLearner(learner.policy(), cfg)


def _raw(learner, s, batch, key):
  """(state, per-step metrics [E, M], info) of update(), before the summary.

  Args:
    learner: The learner.
    s: Its state.
    batch: The batch.
    key: The PRNG key.

  Returns:
    (state, metrics, info).
  """
  return learner._update_jit(s, batch, key, capacity=learner.train_rows(batch))  # pylint: disable=protected-access


def _same_state(a, b) -> bool:
  """Whether two learner states are equal, leaf for leaf.

  Integer leaves exactly, float leaves up to f32 rounding (fx.leaves_close):
  the two states come from different programs, and XLA:CPU's rounding of a
  program is not portable across CPU models, so bit for bit across programs
  holds on some hosts only.

  Args:
    a: A PGState.
    b: A PGState.

  Returns:
    True when equal.
  """
  return (
      fx.leaves_close(a.params, b.params)
      and fx.leaves_close(a.opt_state, b.opt_state)
      and fx.leaves_close(a.value_ema, b.value_ema)
      and int(a.update_step) == int(b.update_step)
  )


def test_spec_validation():
  """Spec validation."""
  with pytest.raises(ValueError, match="step_gate_name"):
    dataclasses.replace(SPEC, step_gate_name="not a name")
  assert toy_loss.SPEC.step_gate is None and SPEC.step_gate_name == "sgd"


@pytest.mark.parametrize(
    "fixed", [False, True], ids=["num_minibatches", "minibatch_size"]
)
def test_gate_stops_the_update_and_skips_the_remaining_steps(fixed):
  """Gate stops the update and skips the remaining steps."""
  extra = dict(minibatch_size=32, sgd_steps=4) if fixed else {}
  _, off, s, batch = _setup(**extra)
  key = jax.random.key(3)
  _, per, _ = _raw(off, s, batch, key)
  # [2 epochs, M]: KL at each step's start
  kl = np.asarray(per["approx_kl"], np.float64)
  real = (
      (np.asarray(per["sgd_step"]) > 0.5) if fixed else np.ones(kl.shape, bool)
  )
  m = int(real[0].sum())
  assert m >= 2 and kl.shape[0] == 2, (kl.shape, m)
  first = kl[0][real[0]]
  second = kl[1][real[1]][0]
  # the fixture's premise: the epoch's steps drift the policy past every KL
  # measured inside the epoch (the oversized update)
  assert second > 1.05 * first.max() and second > 1e-4, (first, second)
  bound = float((first.max() + second) / 2)

  on = _with(off, max_kl=bound)
  one_epoch = _with(off, update_epochs=1)
  s_on, per_on, _ = _raw(on, s, batch, key)
  s_one, _, _ = _raw(one_epoch, s, batch, key)
  # stopped at the first step of epoch 2: the state is the one-epoch update's
  assert _same_state(s_on, s_one)
  stops = np.asarray(per_on["gate_stop"]) > 0.5
  assert stops.sum() == 1 and stops[1][np.argmax(real[1])], stops
  took = np.asarray(per_on["sgd_step"]) > 0.5
  # the stopping step computed its loss only
  assert took[0].sum() == m and took[1].sum() == 1
  np.testing.assert_array_equal(
      np.asarray(per_on["approx_kl"])[0], np.asarray(per["approx_kl"])[0]
  )
  assert float(np.asarray(per_on["update_norm"])[1][np.argmax(real[1])]) == 0.0

  _, out = on.update(s, batch, key)
  assert out["sgd_early_stopped"] == 1.0
  assert out["sgd_steps_taken"] == out["sgd_stop_step"] == float(m)
  assert out["sgd_steps_planned"] == float(real.sum())
  assert out["approx_kl"] == pytest.approx(
      float(np.mean(np.r_[first, second])), rel=1e-5
  )

  # the next update proceeds normally: the flag starts clear (a fresh batch
  # from the new params, so its first steps are under the bound)
  batch2 = fx.self_play_batch(
      on.policy(), s_on.params, game="leduc_poker", b=8, t=16, seed=2
  )
  _, out2 = on.update(s_on, batch2, jax.random.key(4))
  assert out2["sgd_steps_taken"] >= 1.0


@pytest.mark.parametrize(
    "fixed", [False, True], ids=["num_minibatches", "minibatch_size"]
)
def test_gate_that_always_passes_is_the_ungated_update(fixed):
  """Gate that always passes is the ungated update."""
  extra = dict(minibatch_size=32, sgd_steps=4) if fixed else {}
  _, off, s, batch = _setup(lr=1e-3, **extra)
  on = _with(off, max_kl=1e9)
  key = jax.random.key(5)
  s_off, m_off = off.update(s, batch, key)
  s_on, m_on = on.update(s, batch, key)
  # The gated program is another program (a cond per step, a select of the
  # state): XLA may round a fused op differently, so equal up to f32 rounding
  # (bit for bit is the no-gate path, whose program is unchanged).
  assert fx.max_abs_diff(s_off.params, s_on.params) < 1e-8
  assert fx.max_abs_diff(s_off.opt_state, s_on.opt_state) < 1e-8
  assert fx.leaves_equal(s_off.value_ema, s_on.value_ema)
  assert m_on["sgd_early_stopped"] == 0.0
  assert (
      m_on["sgd_steps_taken"]
      == m_on["sgd_stop_step"]
      == m_on["sgd_steps_planned"]
  )
  for k in GATE_METRICS:
    assert k not in m_off, k
  for k, v in m_off.items():  # every ungated metric, up to that rounding
    assert m_on[k] == pytest.approx(v, rel=1e-5, abs=1e-9), (k, v, m_on[k])


def test_no_gate_adds_nothing_to_the_carry_or_the_metrics():
  """No gate adds nothing to the carry or the metrics."""
  _, off, s, batch = _setup(lr=1e-3)
  _, per, _ = _raw(off, s, batch, jax.random.key(6))
  assert not {"gate_stop", "sgd_planned", "sgd_step"} & set(per)
  _, m = off.update(s, batch, jax.random.key(6))
  assert not set(GATE_METRICS) & set(m)


def test_stop_at_the_first_step_keeps_the_params():
  """A batch already past the bound (a behaviour policy far from the params).

  No step is applied; the metrics are the measurement at the start.
  """
  _, off, s, batch = _setup()
  far = batch._replace(behavior_logp=batch.behavior_logp - jnp.float32(1.0))
  on = _with(off, max_kl=0.05)
  s2, out = on.update(s, far, jax.random.key(7))
  assert fx.leaves_equal(s2.params, s.params) and fx.leaves_equal(
      s2.opt_state, s.opt_state
  )
  assert int(s2.update_step) == 1
  assert out["sgd_early_stopped"] == 1.0 and out["sgd_steps_taken"] == 0.0
  assert out["sgd_stop_step"] == 0.0 and out["approx_kl"] > 0.05


def test_gate_must_return_a_bool_scalar():
  """Gate must return a bool scalar."""
  pol, off, s, batch = _setup()
  learner = pg.PGLearner(pol, dataclasses.replace(off.cfg, loss=BAD.name))
  with pytest.raises(AssertionError, match="bool scalar"):
    learner.update(s, batch, jax.random.key(8))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
