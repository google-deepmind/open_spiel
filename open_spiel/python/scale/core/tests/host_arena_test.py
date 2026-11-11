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

"""HostArena.

with only in-graph slots it reproduces the Arena exactly; a HOST_DRIVEN numpy
slot (NumpyHistoryPolicy) reproduces the same maths' NATIVE_JAX slot
(HistoryPolicy) in the Arena under the same keys; a host slot sees only the
seats it owns, one member per call; the Arena refuses a HOST_DRIVEN slot.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import host_arena
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core.tests import numpy_policies
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

tree_map = jax.tree_util.tree_map


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The kuhn env, shared by the module.

  Returns:
    a KuhnEnv.
  """
  return kuhn.KuhnEnv()


def _stack(params, k=1):
  """Stacks `params` k times on a new leading axis.

  Args:
    params: The params.
    k: The copies.

  Returns:
    The stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


def _assert_batches_equal(a, b):
  la, ta = jax.tree_util.tree_flatten(a)
  lb, tb = jax.tree_util.tree_flatten(b)
  assert ta == tb, (ta, tb)
  for x, y in zip(la, lb):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_in_graph_slots_match_arena_exactly(env):
  """In-graph slots in HostArena match the Arena exactly."""
  hist = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  slots = [
      arena_lib.Slot(hist, 1, types.SAMPLE),
      arena_lib.Slot(
          uniform.UniformPolicy("kuhn_poker", 2, 2), 1, types.SAMPLE
      ),
  ]
  params = [
      _stack(hist.make_params(w=jnp.array([[0.3, -0.2], [0.1, 0.4]]))),
      _stack({}),
  ]
  table = arena_lib.rotated_table(8, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(0), 8, 2)
  _, ba, la = arena_lib.Arena(env, slots).collect(params, table, dk, pk, 12)
  _, bh, lh = host_arena.HostArena(env, slots).collect(
      params, table, dk, pk, 12
  )
  _assert_batches_equal(ba, bh)
  _assert_batches_equal(la, lh)


def test_numpy_host_driven_slot_matches_jax_slot(env):
  """A numpy HOST_DRIVEN slot plays like the same JAX slot."""
  w = np.array([[0.3, -0.2], [0.1, 0.4]], np.float32)
  b_ = np.array([0.0, 0.25], np.float32)
  jax_pol = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  host_pol = numpy_policies.NumpyHistoryPolicy("kuhn_poker", 2, 2)
  unif = uniform.UniformPolicy("kuhn_poker", 2, 2)
  table = arena_lib.rotated_table(8, [0, 1], [0, 0])
  dk, pk = arena_lib.duplicate_keys(jax.random.key(1), 8, 2)
  ref = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(jax_pol, 1, types.SAMPLE),
          arena_lib.Slot(unif, 1, types.SAMPLE),
      ],
  )
  _, ba, la = ref.collect(
      [_stack(jax_pol.make_params(w=w, b=b_)), _stack({})], table, dk, pk, 16
  )
  host = host_arena.HostArena(
      env,
      [
          arena_lib.Slot(host_pol, 1, types.SAMPLE),
          arena_lib.Slot(unif, 1, types.SAMPLE),
      ],
  )
  _, bh, lh = host.collect(
      [[host_pol.make_params(w=w, b=b_)], _stack({})], table, dk, pk, 16
  )
  np.testing.assert_array_equal(np.asarray(lh.action), np.asarray(la.action))
  np.testing.assert_allclose(
      np.asarray(bh.behavior_logits), np.asarray(ba.behavior_logits), atol=1e-6
  )
  np.testing.assert_allclose(
      np.asarray(bh.behavior_logp), np.asarray(ba.behavior_logp), atol=1e-6
  )
  np.testing.assert_array_equal(np.asarray(bh.reward), np.asarray(ba.reward))


def test_host_slot_sees_only_its_seats_grouped_by_member(env):
  """A HOST_DRIVEN slot with K=2 receives only rows it owns.

  One member per call, never a seat of another slot.

  Args:
    env: the kuhn env fixture.
  """
  seen = []

  class Recording(numpy_policies.NumpyHistoryPolicy):
    """A numpy HistoryPolicy that records its calls."""

    def evaluate(self, params, carry, step, key):
      """Records the call, then evaluates as NumpyHistoryPolicy.

      Args:
        params: the policy params.
        carry: the carry.
        step: the SeatStep.
        key: unused.

      Returns:
        NumpyHistoryPolicy's PolicyOutput.
      """
      seen.append((
          int(params["tag"]),
          np.asarray(step.seat).copy(),
          np.asarray(key).shape,
      ))
      return super().evaluate(params, carry, step, key)

  pol = Recording("kuhn_poker", 2, 2)
  unif = uniform.UniformPolicy("kuhn_poker", 2, 2)
  ar = host_arena.HostArena(
      env,
      [
          arena_lib.Slot(unif, 1, types.SAMPLE),
          arena_lib.Slot(pol, 2, types.SAMPLE),
      ],
  )
  b = 8
  member = (np.arange(b) % 2).astype(np.int32)
  table = arena_lib.SlotTable(
      seat_slot=np.tile(np.array([[0, 1]], np.int32), (b, 1)),
      seat_member=np.stack([np.zeros(b, np.int32), member], axis=1),
      perspective_seat=np.zeros(b, np.int32),
  )
  params_m = []
  for m in range(2):
    p = pol.make_params()
    p["tag"] = np.float32(m)
    params_m.append(p)
  dk, pk = arena_lib.independent_keys(jax.random.key(2), b, 2)
  _, _, log = ar.collect([_stack({}), params_m], table, dk, pk, 10)
  assert seen
  for tag, seats, kshape in seen:
    assert (seats == 1).all()
    assert kshape == (seats.size, 2), "host keys are raw uint32[n, 2]"
    assert tag in (0, 1)
  slot1 = np.asarray(log.slot) == 1
  np.testing.assert_array_equal(
      np.asarray(log.member)[slot1],
      np.broadcast_to(member[:, None], slot1.shape)[slot1],
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
