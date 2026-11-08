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

"""core.schedules: known values of every schedule kind.

core.schedules: known values of every schedule kind, computed in float64
from an int64 counter (no int32 wrap, no f32 rounding of the counter) and
returned as f64 (`value`) or f32 (`f32`).
"""

import dataclasses
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import hash_compat
from open_spiel.python.scale.core import schedules

I64 = jnp.int64


def _v(s, t):
  """The schedule's value at t.

  Args:
    s: The schedule.
    t: The step.

  Returns:
    The value.
  """
  return float(s.value(jnp.asarray(t, I64)))


def test_constant_and_dtypes():
  """Constant and dtypes."""
  s = schedules.Schedule("constant", 0.3)
  assert _v(s, 0) == 0.3 and _v(s, 10**12) == 0.3
  assert s.value(jnp.asarray(5, I64)).dtype == jnp.float64
  assert s.f32(jnp.asarray(5, I64)).dtype == jnp.float32
  assert s.value(7).dtype == jnp.float64  # a python int is taken as int64
  with pytest.raises(AssertionError, match="int64"):
    s.value(jnp.asarray(5, jnp.int32))  # counters are int64


def test_linear_known_answers_past_int32():
  """Linear known answers past int32."""
  s = schedules.Schedule("linear", 1e-3, 0.0, decay_steps=4)
  np.testing.assert_allclose(
      [_v(s, t) for t in range(6)],
      [1e-3, 7.5e-4, 5e-4, 2.5e-4, 0.0, 0.0],
      atol=1e-15,
  )
  big = schedules.Schedule("linear", 1.0, 0.0, decay_steps=2**32)
  u = 2**31 + 5
  np.testing.assert_allclose(_v(big, u), 1.0 - u / 2**32, rtol=1e-12)
  assert _v(
      schedules.Schedule("linear", 1.0, 0.25, decay_steps=10.0), 2**40
  ) == pytest.approx(0.25)


def test_cosine_is_optax_cosine_decay():
  """Cosine is optax cosine decay."""
  lr, end, total = 1e-3, 1e-4, 100
  s = schedules.Schedule("cosine", lr, end, decay_steps=total)
  for t in (0, 25, 50, 75, 100, 150, 2**33):
    want = end + (lr - end) * 0.5 * (
        1 + math.cos(math.pi * min(t, total) / total)
    )
    np.testing.assert_allclose(_v(s, t), want, rtol=1e-12)


def test_inverse_power_floor_ceil_and_scaling():
  # The clipped inverse-power schedule: clip(coef / (n + 1)**decay, floor, ceil)
  """Inverse power floor ceil and scaling."""
  s = schedules.Schedule(
      "inverse_power", 0.05, power=0.3, floor=0.001, ceil=0.1
  )
  np.testing.assert_allclose(_v(s, 0), 0.05, rtol=1e-15)
  np.testing.assert_allclose(_v(s, 999), 0.05 / 1000**0.3, rtol=1e-15)
  assert _v(s, 2**33) == 0.001  # past 2**31: no wrap, floored
  # floor_step: the step scale that reaches the floor at floor_step
  h = schedules.Schedule(
      "inverse_power", 0.05, power=0.3, floor=0.001, ceil=0.1, floor_step=3000
  )
  np.testing.assert_allclose(_v(h, 3000), 0.001, rtol=1e-12)
  np.testing.assert_allclose(_v(h, 0), 0.05, rtol=1e-15)
  ts = schedules.Schedule(
      "inverse_power", 0.05, power=0.3, floor=0.001, ceil=0.1, time_scale=14.0
  )
  np.testing.assert_allclose(_v(ts, 100), 0.05 / 1401**0.3, rtol=1e-12)
  with pytest.raises(ValueError, match="exclusive"):
    schedules.Schedule(
        "inverse_power",
        0.05,
        power=0.3,
        floor=0.001,
        floor_step=10,
        time_scale=2.0,
    )
  # unclipped, and floored at `floor`
  np.testing.assert_allclose(
      _v(schedules.Schedule("inverse_power", 1.0, power=0.5), 3), 0.5
  )
  assert (
      0.0
      < _v(schedules.Schedule("inverse_power", 1.0, power=0.5), 2**40)
      < 1e-4
  )


def test_power_law_known_answer():
  """Power law known answer."""
  s = schedules.Schedule("power_law", 0.2, 0.0, power=2.0, decay_steps=100)
  np.testing.assert_allclose(_v(s, 50), 0.2 * 0.25, rtol=1e-12)
  assert _v(s, 100) == 0.0 and _v(s, 10**9) == 0.0


def test_warmup_forms():
  # a constant warmup: start * min((t + 1) / W, 1)  (warmup_offset=1)
  """Warmup forms."""
  w = schedules.Schedule("constant", 0.2, warmup_steps=10, warmup_offset=1)
  np.testing.assert_allclose(
      [_v(w, t) for t in (0, 4, 9, 50)], [0.02, 0.1, 0.2, 0.2]
  )
  # linear 0 -> lr over W, then linear lr -> end over decay_steps (optax join)
  b = schedules.Schedule("linear", 1.0, 0.0, decay_steps=10, warmup_steps=5)
  np.testing.assert_allclose(
      [_v(b, t) for t in (0, 1, 5, 10, 15, 99)],
      [0.0, 0.2, 1.0, 0.5, 0.0, 0.0],
      atol=1e-15,
  )


def test_validation():
  """Validation."""
  with pytest.raises(ValueError):
    schedules.Schedule("step", 1.0)
  with pytest.raises(ValueError):
    schedules.Schedule("linear", 1.0, 0.0)  # needs decay_steps
  with pytest.raises(ValueError):
    schedules.Schedule("cosine", 1.0, 0.0, decay_steps=0)
  with pytest.raises(ValueError):
    schedules.Schedule("constant", 1.0, warmup_steps=-1)


def test_is_constant():
  """Is constant."""
  assert schedules.Schedule("constant", 0.1).is_constant()
  assert not schedules.Schedule("constant", 0.1, warmup_steps=3).is_constant()
  assert not schedules.Schedule("linear", 1.0, 0.5, decay_steps=3).is_constant()
  assert schedules.Schedule("linear", 1.0, 1.0, decay_steps=3).is_constant()


def test_jittable_with_traced_counter():
  """Jittable with traced counter."""
  s = schedules.Schedule("cosine", 1.0, 0.1, decay_steps=10)
  f = jax.jit(s.f32)
  np.testing.assert_allclose(
      float(f(jnp.asarray(5, I64))), float(s.f32(jnp.asarray(5, I64)))
  )


def test_anneal_freezes_the_value_at_start_then_decays_linearly_and_holds():
  """The step-size anneal (module doc) against hand values.

  The plain value before A, then the value AT A times a factor linear 1 -> f
  over [A, E], held at f after.
  """
  base = schedules.Schedule(
      "inverse_power", 0.5, power=1.1, floor=5e-6, ceil=1e-4
  )
  s = dataclasses.replace(
      base, anneal_start=3000, anneal_end=3250, anneal_final_frac=0.1
  )
  v0 = 0.5 / 3001.0**1.1
  for t in (0, 2500, 2999):
    assert _v(s, t) == pytest.approx(_v(base, t), rel=1e-12)
  assert _v(s, 3000) == pytest.approx(v0, rel=1e-12)
  assert _v(s, 3125) == pytest.approx(v0 * 0.55, rel=1e-12)
  assert _v(s, 3250) == pytest.approx(v0 * 0.1, rel=1e-12)
  assert _v(s, 9000) == pytest.approx(v0 * 0.1, rel=1e-12)
  assert not s.is_constant()
  f = jax.jit(s.f32)
  np.testing.assert_allclose(
      float(f(jnp.asarray(3100, I64))), float(s.f32(jnp.asarray(3100, I64)))
  )


def test_anneal_fields_validate_and_stay_out_of_the_hash_at_default():
  """All 0 is off; on needs A < E and f in (0, 1); the fields are added.

  They are hash_compat.added: an off schedule hashes as before they existed.
  """
  assert set(hash_compat.added_fields(schedules.Schedule)) == {
      "anneal_start",
      "anneal_end",
      "anneal_final_frac",
  }
  for kw in (
      {"anneal_start": 5},
      {"anneal_final_frac": 0.1},
      {"anneal_start": 5, "anneal_end": 5, "anneal_final_frac": 0.1},
      {"anneal_start": 0, "anneal_end": 5, "anneal_final_frac": 1.0},
      {"anneal_start": 0, "anneal_end": 5, "anneal_final_frac": 0.0},
      {"anneal_start": 1.5, "anneal_end": 5, "anneal_final_frac": 0.1},
  ):
    with pytest.raises(ValueError):
      schedules.Schedule("constant", 1.0, **kw)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
