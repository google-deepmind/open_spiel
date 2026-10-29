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

"""Tests for RoutedPolicy under the Arena's per-row vmap.

A JAX part without evaluate_rows is evaluated only on the vmapped rows routed to
it, gathered into core.arena.on_rows' chunks (R/32, R/4, R/2, R), instead of on
every row with a select. Known answers by counting the rows the part computes (a
host callback); the outputs equal the ungated per-row program's bit for bit.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies.tests import named_fns

A = 3
ROUTE = named_fns.ROUTED_ROWS
ROWS = []


class Counting(types.Policy):
  """NATIVE_JAX, stateless: logits = x * w per action; counts computed rows.

  Attributes:
    scale: the logits' scale w.
    count: the host-side counter of computed rows.
    spec: the PolicySpec.
  """

  def __init__(self, scale, count):
    self.scale, self.count = float(scale), count
    self.spec = types.PolicySpec(
        game="x",
        obs_format="x",
        num_actions=A,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="testing.vmap_counting",
        arch_kwargs={"scale": scale, "count": count},
    )

  def init_params(self, key=None):
    """The weights 1..A times `scale`.

    Args:
      key: unused.

    Returns:
      {"w": f32[A]}.
    """
    del key  # Unused.
    return {"w": jnp.arange(1.0, A + 1.0, dtype=jnp.float32) * self.scale}

  def encode(self, info):
    """The "x" features.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      info["x"].
    """
    return info["x"]

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key  # Unused.
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step  # Unused.
    return carry

  def evaluate(self, params, carry, step, key):
    """Logits x[:, :1] * w; optionally records the rows.

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    del carry, key  # Unused.
    x = step.info["x"]
    if self.count:
      jax.debug.callback(
          lambda r: ROWS.append(int(np.asarray(r).size)), x[:, 0]
      )
    lg = x[:, :1] * params["w"][None, :]
    return types.PolicyOutput(
        logits=lg.astype(jnp.float32),
        value=jnp.zeros((x.shape[0], 2), jnp.float32),
        aux={},
    )


def _steps(sign):
  """Toy SeatSteps whose feature 0 has the given signs.

  Args:
    sign: f32[b], the signs.

  Returns:
    The SeatStep.
  """
  sign = np.asarray(sign, np.float32)
  b = sign.shape[0]
  x = (np.abs(np.random.default_rng(0).normal(size=(b, 4))) + 0.1).astype(
      np.float32
  )
  x[:, 0] *= sign
  return types.SeatStep(
      info={"x": jnp.asarray(x)},
      legal_mask=jnp.ones((b, A), jnp.bool_),
      seat=jnp.zeros((b,), jnp.int32),
      to_move=jnp.zeros((b,), jnp.int32),
      new_hand=jnp.zeros((b,), jnp.bool_),
      new_match=jnp.zeros((b,), jnp.bool_),
      last_action=jnp.full((b,), -1, jnp.int32),
      last_actor=jnp.full((b,), -1, jnp.int32),
  )


def _per_row(pol, params, st):
  """The Arena's call style: vmap of batch-1 evaluate calls.

  Args:
    pol: The policy.
    params: Its params.
    st: The SeatSteps.

  Returns:
    (logits, value).
  """
  carry = {"base": (), "override": ()}

  def one(s):
    """The outputs of one vmapped row.

    Args:
      s: one row's SeatStep.

    Returns:
      (logits, value) of the row.
    """
    s1 = jax.tree_util.tree_map(lambda x: x[None], s)
    out = pol.evaluate(params, carry, s1, None)
    return jax.tree_util.tree_map(lambda x: x[0], (out.logits, out.value))

  return jax.jit(jax.vmap(one))(st)


def _policy():
  """A routed policy over two counting parts.

  Returns:
    (policy, params).
  """
  pol = routed.RoutedPolicy(Counting(1.0, 0), Counting(2.0, 1), ROUTE)
  return pol, pol.make_params(
      pol.base.init_params(), pol.override.init_params()
  )


def test_override_rows_counted_under_vmap():
  """Override rows counted under vmap."""
  pol, params = _policy()
  # R = 8: on_rows chunks 1, 2, 4, 8.
  # no routed row: never evaluated
  for sign, want in (
      ([-1] * 8, 0),
      ([1, -1, -1, -1, -1, -1, -1, -1], 1),  # one row: chunk 1
      ([1, -1, 1, -1, -1, 1, -1, -1], 4),  # three rows: chunk 4
      ([1] * 8, 8),
  ):  # every row
    ROWS.clear()
    jax.block_until_ready(_per_row(pol, params, _steps(sign)))
    assert sum(ROWS) == want, (sign, ROWS)


def test_outputs_equal_the_batched_and_ungated_programs():
  """Outputs equal the batched and ungated programs."""
  pol, params = _policy()
  sign = [1, -1, 1, -1, -1, 1, -1, -1]
  st = _steps(sign)
  lg, v = _per_row(pol, params, st)
  use = np.asarray(sign) > 0
  base = np.asarray(st.info["x"][:, :1] * params["base"]["w"][None, :])
  over = np.asarray(st.info["x"][:, :1] * params["override"]["w"][None, :])
  np.testing.assert_array_equal(np.asarray(lg)[use], over[use])
  np.testing.assert_array_equal(np.asarray(lg)[~use], base[~use])
  # the batched (no vmap) call: lax.cond gating, the same values
  out = pol.evaluate(params, {"base": (), "override": ()}, st, None)
  np.testing.assert_array_equal(np.asarray(out.logits), np.asarray(lg))
  np.testing.assert_array_equal(np.asarray(out.value), np.asarray(v))


class Mat(Counting):
  """A matrix policy whose params are big enough to see in a jaxpr.

  logits = (x @ W)[:, :A] with a [D, D] weight: big enough that a per-row
  copy of the params shows as an [R, D, D] value in the jaxpr.
  """

  D = 16

  def init_params(self, key=None):
    """The identity weight times `scale`.

    Args:
      key: unused.

    Returns:
      {"w": f32[D, D]}.
    """
    return {"w": jnp.eye(self.D, dtype=jnp.float32) * self.scale}

  def evaluate(self, params, carry, step, key):
    """Logits (x @ W)[:, :A].

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    x = jnp.pad(step.info["x"], ((0, 0), (0, self.D - 4)))
    lg = (x @ params["w"])[:, :A]
    return types.PolicyOutput(
        logits=lg.astype(jnp.float32),
        value=jnp.zeros((x.shape[0], 2), jnp.float32),
        aux={},
    )


def _subjaxprs(v):
  if hasattr(v, "eqns"):
    yield v
  elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
    yield v.jaxpr
  elif isinstance(v, (tuple, list)):
    for x in v:
      yield from _subjaxprs(x)


def _avals(jaxpr, out):
  """Every equation output's shape in `jaxpr` and its sub-jaxprs.

  Args:
    jaxpr: The jaxpr.
    out: The list to extend.

  Returns:
    `out`.
  """
  for eqn in jaxpr.eqns:
    out.extend(
        tuple(v.aval.shape) for v in eqn.outvars if hasattr(v.aval, "shape")
    )
    for p in eqn.params.values():
      for sub in _subjaxprs(p):
        _avals(sub, out)
  return out


def test_base_params_never_batched_under_vmap():
  """Base params never batched under vmap.

  Under the Arena's per-row vmap the base's params stay unbatched: no
  per-row weight copy exists. A vmapped lax.cond batches every operand (the
  closed-over params too), so each row would carry its own copy of the
  weights. Known answer: in the Arena's call style at R = 8 no value of shape
  [8, D, D] exists, for the base and for the gated override, and the base's
  rows equal the batched program's; the control (the vmapped cond itself)
  has one.
  """
  base, over = Mat(1.0, 0), Mat(2.0, 0)
  pol = routed.RoutedPolicy(base, over, ROUTE)
  params = pol.make_params(base.init_params(), over.init_params())
  sign = [1, -1, 1, -1, -1, 1, -1, -1]
  st = _steps(sign)
  carry = {"base": (), "override": ()}

  def one(s):
    """The routed policy's logits on one vmapped row.

    Args:
      s: one row's SeatStep.

    Returns:
      The row's logits.
    """
    s1 = jax.tree_util.tree_map(lambda x: x[None], s)
    return pol.evaluate(params, carry, s1, None).logits[0]

  per_row = jax.vmap(one)
  shapes = _avals(jax.make_jaxpr(per_row)(st).jaxpr, [])
  assert (8, Mat.D, Mat.D) not in shapes, sorted(
      set(s for s in shapes if len(s) == 3)
  )
  want = np.asarray(pol.evaluate(params, carry, st, None).logits)
  np.testing.assert_array_equal(np.asarray(jax.jit(per_row)(st)), want)

  def cond_one(s):  # the control: the part under a plain vmapped lax.cond
    """The base under a plain vmapped lax.cond (the control).

    Args:
      s: one row's SeatStep.

    Returns:
      The row's logits.
    """
    s1 = jax.tree_util.tree_map(lambda x: x[None], s)
    rows = named_fns.routed_rows(s1.info) <= 0
    # Reaches private _cond_evaluate of the code under test.
    return routed._cond_evaluate(  # pylint: disable=protected-access
        base, params["base"], (), s1, None, rows
    ).logits[0]

  assert (8, Mat.D, Mat.D) in _avals(
      jax.make_jaxpr(jax.vmap(cond_one))(st).jaxpr, []
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
