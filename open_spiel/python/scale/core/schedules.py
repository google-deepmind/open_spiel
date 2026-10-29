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

"""Scalar schedules over int64 training counters.

Every value that moves with a counter is a `Schedule`: learning rates, the
entropy and KL coefficients of policy-gradient losses, a magnet
temperature, the DAgger beta. One implementation keeps their arithmetic
identical everywhere: values are computed in float64 from the int64
counter, so a long run neither wraps an int32 step nor rounds the counter
to f32.

  lr = Schedule("linear", 3e-4, 0.0, decay_steps=1e6)
  lr.value(step)    # f64[]
  lr.f32(step)      # f32[], cast once, for coefficients that feed f32 losses

A Python int step is taken as int64; any other counter must be int64
(asserted). Jittable: the counter may be traced.

kind (t = the step as f64, D = decay_steps):
  "constant"       start
  "linear"         start + (end - start) * clip(t / D, 0, 1)
  "cosine"         end + (start - end) * (1 + cos(pi * min(t, D) / D)) / 2
                   (optax.cosine_decay_schedule(start, D, alpha=end/start))
  "inverse_power"  start / (n + 1)**power with n = t * scale, clipped to
                   [floor, ceil] where given (None: unclipped on that side).
                   scale = time_scale, or (floor_step > 0) the scale that
                   reaches `floor` at t = floor_step (exclusive with
                   time_scale != 1).
  "power_law"      end + (start - end) * (1 - clip(t / D, 0, 1))**power

warmup_steps W > 0 prepends a linear ramp from 0: with s = t +
warmup_offset, value = start * s / W while s < W, else the kind's value at
t - W. warmup_offset 0 is optax.join_schedules of a 0 -> start ramp;
warmup_offset 1 with "constant" gives start * min((t + 1) / W, 1).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp

KINDS = ("constant", "linear", "cosine", "inverse_power", "power_law")
F64 = jnp.float64
I64 = jnp.int64


def counter(step: Any) -> jax.Array:
  """Step as an int64 array.

  A Python int is converted; anything else must already be int64.

  Args:
    step: the counter.

  Returns:
    The int64 array.
  """
  if isinstance(step, int) and not isinstance(step, bool):
    return jnp.asarray(step, I64)
  s = jnp.asarray(step)
  assert s.dtype == I64, f"step counters are int64, got {s.dtype}"
  return s


@dataclasses.dataclass(frozen=True)
class Schedule:
  """A scalar schedule over an int64 step counter (module doc).

  Attributes:
    kind: one of KINDS.
    start: the value at step 0 (after warmup).
    end: the final value.
    power: the polynomial / power-law exponent.
    decay_steps: the steps from start to end.
    warmup_steps: steps of the linear warmup.
    warmup_offset: the counter offset of the warmup.
    floor: a lower clamp, optional.
    ceil: an upper clamp, optional.
    time_scale: multiplies the counter.
    floor_step: the step a floor applies from.
  """

  kind: str = "constant"
  start: float = 0.0
  end: float = 0.0
  power: float = 1.0
  decay_steps: Optional[float] = None
  warmup_steps: float = 0.0
  warmup_offset: int = 0
  floor: Optional[float] = None
  ceil: Optional[float] = None
  time_scale: float = 1.0
  floor_step: int = 0

  def __post_init__(self):
    if self.kind not in KINDS:
      raise ValueError(f"schedule kind {self.kind!r} not in {KINDS}")
    for name in ("start", "end", "power", "warmup_steps", "time_scale"):
      v = float(getattr(self, name))
      if not math.isfinite(v):
        raise ValueError(f"schedule {name} must be finite, got {v}")
    if self.kind in ("linear", "cosine", "power_law"):
      # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
      # pylint: disable-next=unnecessary-negation
      if self.decay_steps is None or not float(self.decay_steps) > 0:
        raise ValueError(
            f"schedule kind={self.kind} needs decay_steps > 0, got "
            f"{self.decay_steps}"
        )
    if float(self.warmup_steps) < 0.0:
      raise ValueError(
          f"schedule warmup_steps must be >= 0, got {self.warmup_steps}"
      )
    if int(self.warmup_offset) not in (0, 1):
      raise ValueError(
          f"schedule warmup_offset must be 0 or 1, got {self.warmup_offset}"
      )
    # NaN-safe: `not x > 0` also refuses NaN, which `x <= 0` would accept.
    # pylint: disable-next=unnecessary-negation
    if int(self.floor_step) < 0 or not float(self.time_scale) > 0.0:
      raise ValueError(
          "schedule floor_step must be >= 0 and time_scale > 0, got "
          f"{self.floor_step}, {self.time_scale}"
      )
    if self.kind != "inverse_power" and not (
        self.floor is None
        and self.ceil is None
        and not self.floor_step
        and self.time_scale == 1.0
    ):
      raise ValueError(
          f"floor/ceil/floor_step/time_scale are inverse_power's ({self})"
      )
    if self.floor_step and self.time_scale != 1.0:
      raise ValueError(
          f"floor_step {self.floor_step} and time_scale {self.time_scale} are "
          "exclusive"
      )
    if self.floor_step and not (
        self.floor is not None and self.start > self.floor > 0
    ):
      raise ValueError(
          f"floor_step needs start > floor > 0, got start {self.start}, floor "
          f"{self.floor}"
      )
    if self.kind == "cosine" and self.end > self.start:
      raise ValueError(
          f"a cosine schedule decays start -> end, got start {self.start}, end "
          f"{self.end}"
      )

  def is_constant(self) -> bool:
    """True when value(t) is `start` for every t.

    A resume may then extend the run's length.

    Returns:
      Whether the schedule is constant.
    """
    if float(self.warmup_steps) > 0.0:
      return False
    if self.kind == "constant":
      return True
    if self.kind in ("linear", "cosine", "power_law"):
      return self.start == self.end
    return self.start == 0.0

  def _base(self, t: jax.Array) -> jax.Array:
    """The value at step `t`, ignoring warmup.

    Args:
      t: the f64 step.

    Returns:
      The f64 value.
    """
    start, end = jnp.asarray(self.start, F64), jnp.asarray(self.end, F64)
    if self.kind == "constant":
      return start
    if self.kind == "inverse_power":
      if self.floor_step:
        scale = (
            (self.start / self.floor) ** (1.0 / self.power) - 1.0
        ) / self.floor_step
      else:
        scale = float(self.time_scale)
      n = t if scale == 1.0 else t * jnp.asarray(scale, F64)
      x = start / (n + 1.0) ** jnp.asarray(self.power, F64)
      if self.floor is not None:
        x = jnp.maximum(x, jnp.asarray(self.floor, F64))
      if self.ceil is not None:
        x = jnp.minimum(x, jnp.asarray(self.ceil, F64))
      return x
    d = jnp.asarray(float(self.decay_steps), F64)
    frac = jnp.clip(t / d, 0.0, 1.0)
    if self.kind == "linear":
      return start + (end - start) * frac
    if self.kind == "cosine":
      return end + (start - end) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
    return end + (start - end) * (1.0 - frac) ** jnp.asarray(self.power, F64)

  def value(self, step: Any) -> jax.Array:
    """f64 value at int64 counter `step`.

    Args:
      step: the int64 counter.

    Returns:
      The f64 value.
    """
    t = counter(step).astype(F64)
    w = float(self.warmup_steps)
    if w <= 0.0:
      return self._base(t)
    s = t + float(self.warmup_offset)
    ramp = jnp.asarray(self.start, F64) * s / w
    return jnp.where(s < w, ramp, self._base(jnp.maximum(t - w, 0.0)))

  def f32(self, step: Any) -> jax.Array:
    """f32 value at int64 counter `step`.

    Args:
      step: the int64 counter.

    Returns:
      The f32 value.
    """
    return self.value(step).astype(jnp.float32)


def constant(v: float) -> Schedule:
  """The constant schedule of v.

  Args:
    v: the value.

  Returns:
    The Schedule.
  """
  return Schedule("constant", float(v))
