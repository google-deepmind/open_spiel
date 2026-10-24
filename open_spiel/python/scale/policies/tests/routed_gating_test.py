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

"""Tests for RoutedPolicy branch gating, known answers by counting.

* a host (CALLBACK_BATCHED) base with evaluate_rows receives exactly the
rows the route leaves to it: forwarded rows == count(~route) for a
batched call, a jitted call and a per-row vmap (how the Arena calls a
slot policy); when every row is routed it forwards none;
* a part without evaluate_rows is not evaluated on a batch with no row for
it (lax.cond), counted by a host callback;
* outputs are the per-row selection of the two parts (base where not
routed, override where routed);
* nested routes gate through evaluate_rows.
"""

import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies.tests import named_fns

A = 3
ROUTE = named_fns.FIRST_FEATURE_POSITIVE
CALLS = []


class ObsInfo(NamedTuple):
  """An InfoState that is a plain feature array.

  Attributes:
    obs: the features [B, ...].
  """

  obs: jax.Array


class CountingPolicy(types.Policy):
  """A NATIVE_JAX policy that records each evaluation's batch size.

  Logits 1 on action 1; each evaluation records its batch size through a host
  callback (CALLS).

  Attributes:
    spec: the PolicySpec.
  """

  def __init__(self):
    self.spec = types.PolicySpec(
        game="x",
        obs_format="x",
        num_actions=A,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="testing.counting",
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key  # Unused.
    return {}

  def encode(self, info):
    """The obs features.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      info.obs.
    """
    return info.obs

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
    """Logits 1 on action 1; records the batch size.

    Args:
      params: the params.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput.
    """
    del params, carry, key  # Unused.
    b = step.legal_mask.shape[0]
    jax.debug.callback(lambda n: CALLS.append(int(n)), jnp.int32(b))
    lg = jnp.zeros((b, A), jnp.float32).at[:, 1].set(1.0)
    return types.PolicyOutput(
        logits=lg, value=jnp.zeros((b, 2), jnp.float32), aux={}
    )


class RowRecorder(types.Policy):
  """A host policy reached by jax.pure_callback.

  logits = the first A obs features; `forwarded_rows` counts the rows its
  host function computed. evaluate_rows receives the row mask, so rows
  outside it are never computed.

  Attributes:
    spec: the PolicySpec.
    forwarded_rows: the rows the host function computed.
  """

  def __init__(self):
    self.spec = types.PolicySpec(
        game="x",
        obs_format="x",
        num_actions=A,
        num_players=2,
        stateful=False,
        capability=types.Capability.CALLBACK_BATCHED,
        framework="jax",
        arch_name="testing.row_recorder",
        arch_kwargs={},
    )
    self.forwarded_rows = 0

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key  # Unused.
    return {}

  def encode(self, info):
    """The obs features.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      info.obs.
    """
    return info.obs

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

  def _host(self, x, active):
    """Counts the forwarded rows and returns their first A features.

    Args:
      x: The features.
      active: The forwarded rows.

    Returns:
      f32 logits.
    """
    x = np.asarray(x, np.float32)
    act = np.asarray(active, np.bool_)
    self.forwarded_rows += int(act.sum())
    return np.where(act[..., None], x[..., :A], np.float32(0.0)).astype(
        np.float32
    )

  def evaluate_rows(self, params, carry, step, key, rows):
    """The first A obs features as logits, computed on `rows` only.

    Args:
      params: unused.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.
      rows: bool[B], the rows to evaluate.

    Returns:
      The PolicyOutput (zeros outside `rows`).
    """
    del params, carry, key  # Unused.
    x = step.info.obs
    b = x.shape[0]
    shape = jax.ShapeDtypeStruct((b, A), jnp.float32)
    lg = jax.pure_callback(
        self._host,
        shape,
        x,
        jnp.asarray(rows, jnp.bool_),
        vmap_method="broadcast_all",
    )
    return types.PolicyOutput(
        logits=lg, value=jnp.zeros((b, 2), jnp.float32), aux={}
    )

  def evaluate(self, params, carry, step, key):
    """evaluate_rows on every row.

    Args:
      params: the params.
      carry: the carry.
      step: the SeatStep, leaves [B, ...].
      key: a PRNG key.

    Returns:
      The PolicyOutput.
    """
    return self.evaluate_rows(
        params, carry, step, key, jnp.ones(step.legal_mask.shape[:1], jnp.bool_)
    )


def _steps(sign, b=8, seed=0):
  """sign: f32[b] multiplier of feature 0 (> 0: routed to the override).

  Args:
    sign: The sign of each row.
    b: The rows.
    seed: The seed.

  Returns:
    The SeatStep.
  """
  rng = np.random.default_rng(seed)
  x = rng.normal(size=(b, 5)).astype(np.float32)
  x[:, 0] = np.abs(x[:, 0]) * np.asarray(sign, np.float32) + 0.1 * np.sign(sign)
  legal = rng.random((b, A)) < 0.6
  legal[:, 0] = True
  return types.SeatStep(
      info=ObsInfo(obs=jnp.asarray(x)),
      legal_mask=jnp.asarray(legal),
      seat=jnp.ones((b,), jnp.int32),
      to_move=jnp.ones((b,), jnp.int32),
      new_hand=jnp.ones((b,), jnp.bool_),
      new_match=jnp.ones((b,), jnp.bool_),
      last_action=jnp.full((b,), types.NO_ACTION, jnp.int32),
      last_actor=jnp.full((b,), types.NO_ACTOR, jnp.int32),
  )


def _policy():
  """A routed policy over a row recorder and a counting override.

  Returns:
    (policy, recorder, params).
  """
  net = RowRecorder()
  pol = routed.RoutedPolicy(net, CountingPolicy(), ROUTE)
  return pol, net, pol.make_params({}, {})


def _eval(pol, params, st):
  """Evaluates the routed policy with empty carries.

  Args:
    pol: The routed policy.
    params: Its params.
    st: The SeatSteps.

  Returns:
    The PolicyOutput.
  """
  return pol.evaluate(params, {"base": (), "override": ()}, st, None)


def test_host_base_receives_only_its_rows():
  """Host base receives only its rows."""
  pol, net, params = _policy()
  # 4 routed, 4 base
  sign = np.asarray([1, -1, -1, 1, -1, 1, 1, -1], np.float32)
  st = _steps(sign)
  use = np.asarray(named_fns.first_feature_positive(st.info))
  np.testing.assert_array_equal(use, sign > 0)
  out = _eval(pol, params, st)
  assert net.forwarded_rows == int((~use).sum()) == 4
  lg = np.asarray(out.logits)
  np.testing.assert_array_equal(lg[~use], np.asarray(st.info.obs)[~use, :A])
  np.testing.assert_array_equal(lg[use], np.tile([0.0, 1.0, 0.0], (4, 1)))
  net.forwarded_rows = 0
  jax.block_until_ready(
      jax.jit(lambda p, s: _eval(pol, p, s).logits)(params, st)
  )
  assert net.forwarded_rows == 4
  net.forwarded_rows = 0

  def one(s):
    """The logits of one vmapped row.

    Args:
      s: one row's SeatStep.

    Returns:
      The row's logits.
    """
    s1 = jax.tree_util.tree_map(lambda x: x[None], s)
    return jax.tree_util.tree_map(lambda x: x[0], _eval(pol, params, s1).logits)

  per_row = jax.block_until_ready(jax.jit(jax.vmap(one))(st))
  assert net.forwarded_rows == 4
  np.testing.assert_array_equal(np.asarray(per_row), lg)


def test_all_rows_routed_forwards_no_host_row():
  """All rows routed forwards no host row."""
  pol, net, params = _policy()
  st = _steps(np.ones(8, np.float32))
  out = jax.jit(lambda p, s: _eval(pol, p, s).logits)(params, st)
  np.testing.assert_array_equal(
      np.asarray(out), np.tile([0.0, 1.0, 0.0], (8, 1))
  )
  assert net.forwarded_rows == 0


def test_override_not_evaluated_on_a_batch_without_its_rows():
  """Override not evaluated on a batch without its rows."""
  pol, net, params = _policy()
  CALLS.clear()
  st = _steps(-np.ones(8, np.float32))  # no routed row
  jax.block_until_ready(_eval(pol, params, st).logits)
  assert not CALLS and net.forwarded_rows == 8
  st = _steps(np.asarray([1, -1, -1, -1, -1, -1, -1, -1], np.float32))
  jax.block_until_ready(_eval(pol, params, st).logits)
  assert CALLS == [8] and net.forwarded_rows == 8 + 7


def test_nested_routes_gate_through_evaluate_rows():
  """Nested routes gate through evaluate rows."""
  net = RowRecorder()
  inner = routed.RoutedPolicy(net, CountingPolicy(), ROUTE)
  outer = routed.RoutedPolicy(inner, CountingPolicy(), named_fns.NEVER_OBS)
  params = outer.make_params(inner.make_params({}, {}), {})
  st = _steps(np.asarray([1, -1, 1, -1, -1, -1, 1, 1], np.float32))
  carry = {"base": {"base": (), "override": ()}, "override": ()}
  jax.block_until_ready(outer.evaluate(params, carry, st, None).logits)
  assert net.forwarded_rows == 4


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
