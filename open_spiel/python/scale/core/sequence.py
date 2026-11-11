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

"""The default scans behind Policy.observe_seq and evaluate_seq (JAX policies).

Running a policy over a [B, T] SeatStep sequence has one implementation:
learners, serving replay and testing.check_policy go through these scans, or
through a fused override that check_policy proves equal to them.

  carry = sequence.scan_observe(policy, params, carry, steps)
  carry, outputs = sequence.scan_evaluate(policy, params, carry, steps, keys)
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from open_spiel.python.scale.core import types

tree_map = jax.tree_util.tree_map


def _time_major(x):
  """Swaps the batch and time axes.

  Args:
    x: A leaf, [B, T, ...].

  Returns:
    The leaf, time-major.
  """
  return jnp.swapaxes(x, 0, 1)


def scan_observe(
    policy: types.Policy, params: Any, carry: Any, steps: types.SeatStep
):
  """Final carry after observing steps [B, T] in order.

  Args:
    policy: the policy.
    params: one member's params.
    carry: the carry before steps.
    steps: SeatSteps [B, T, ...].

  Returns:
    the carry after the last step.
  """
  steps_tm = tree_map(_time_major, steps)

  def body(c, st):
    """One ply of the scan.

    Args:
      c: the carry.
      st: the ply's SeatStep.

    Returns:
      (carry after observing st, None).
    """
    return policy.observe(params, c, st), None

  carry, _ = jax.lax.scan(body, carry, steps_tm)
  return carry


def scan_evaluate(
    policy: types.Policy,
    params: Any,
    carry: Any,
    steps: types.SeatStep,
    keys: Optional[Any],
):
  """Observe then evaluate per ply.

  keys: [T] keys (ply t's evaluate key, the one key evaluate receives for the
  batch) or None. The key is shared by the batch's B rows, while the Arena gives
  every row its own key, so for B > 1 a stochastic policy's outputs here do not
  reproduce the Arena's sampling (a one-row batch does).

  Args:
    policy: the policy.
    params: one member's params.
    carry: the carry before steps.
    steps: SeatSteps [B, T, ...].
    keys: the [T] per-ply evaluate keys, or None.

  Returns:
    (final carry, PolicyOutput [B, T, ...]).
  """
  steps_tm = tree_map(_time_major, steps)
  keys_tm = keys

  def body(c, xs):
    """One ply of the scan.

    Args:
      c: the carry.
      xs: (the ply's SeatStep, its key or None).

    Returns:
      (carry after observing, the ply's PolicyOutput).
    """
    st, k = xs
    c = policy.observe(params, c, st)
    return c, policy.evaluate(params, c, st, k)

  carry, outs = jax.lax.scan(body, carry, (steps_tm, keys_tm))
  return carry, tree_map(_time_major, outs)
