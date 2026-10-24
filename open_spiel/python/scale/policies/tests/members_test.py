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

"""Tests for policies.members.

JAX members stack as values; host-weights members are
copied into the owner (new handles, nothing on the owner overwritten), freeze
copies, release frees storage for reuse, RoutedPolicy recurses part by part,
and another architecture is refused.
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import members
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.policies.tests import named_fns

A = 3
ROUTE = named_fns.NEVER


class HostWeights:
  """A host-weights policy: params {"member": i32} index weights it holds.

  Attributes:
    spec: the PolicySpec.
    store: the weights held, by member index.
    free: the released member indices.
  """

  def __init__(self, tag: str = "x"):
    self.spec = types.PolicySpec(
        game="x",
        obs_format="x",
        num_actions=A,
        num_players=2,
        stateful=False,
        capability=types.Capability.CALLBACK_BATCHED,
        framework="jax",
        arch_name="testing.host_weights",
        arch_kwargs={"tag": tag},
    )
    self.store = [np.zeros(A)]
    self.free = set()

  def weights(self, params):
    """The weights member `params` names.

    Args:
      params: {"member": i32}.

    Returns:
      The stored weights.
    """
    return self.store[int(np.asarray(params["member"]))]

  def export_member(self, params):
    """A copy of the member's weights.

    Args:
      params: {"member": i32}.

    Returns:
      The weights, copied.
    """
    return self.weights(params).copy()

  def import_member(self, blob):
    """Stores weights in a free slot (or a new one).

    Args:
      blob: the weights.

    Returns:
      {"member": i32} naming the slot.
    """
    if self.free:
      idx = min(self.free)
      self.free.discard(idx)
      self.store[idx] = np.array(blob)
    else:
      idx = len(self.store)
      self.store.append(np.array(blob))
    return {"member": jnp.int32(idx)}

  def release_member(self, params):
    """Frees the slot `params` names.

    Args:
      params: {"member": i32}.
    """
    self.free.add(int(np.asarray(params["member"])))


def loaded(value):
  """A HostWeights whose member 0 holds `value` everywhere.

  Args:
    value: the weight value.

  Returns:
    (policy, params naming member 0).
  """
  pol = HostWeights()
  pol.store[0] = np.full(A, float(value))
  return pol, {"member": jnp.int32(0)}


def test_jax_members_are_stacked_values():
  """Jax members are stacked values."""
  tab = tabular.TabularPolicy("x", 2, A, 2, named_fns.TAB_INDEX)
  a = tab.params_from_logits(jnp.ones((2, A)))
  b = tab.params_from_logits(2 * jnp.ones((2, A)))
  other = tabular.TabularPolicy("x", 2, A, 2, named_fns.TAB_INDEX)
  got = members.stack(tab, [(tab, a), (other, b)])
  np.testing.assert_array_equal(
      np.asarray(got["logits"]), np.stack([a["logits"], b["logits"]])
  )
  assert got["logits"].dtype == jnp.float32


def test_host_weights_members_move_into_the_owner():
  """Host weights members move into the owner."""
  (p0, a0), (p1, a1) = loaded(1.0), loaded(2.0)
  stacked = members.stack(p0, [(p0, a0), (p1, a1)])
  np.testing.assert_array_equal(np.asarray(stacked["member"]), [0, 1])
  np.testing.assert_array_equal(p0.store[1], np.full(A, 2.0))
  # never overwritten
  np.testing.assert_array_equal(p0.store[0], np.full(A, 1.0))
  assert len(p1.store) == 1  # the source is untouched


def test_same_object_keeps_its_handle():
  """Same object keeps its handle."""
  p0, a0 = loaded(1.0)
  assert members.adopt(p0, p0, a0) is a0


def test_freeze_copies_and_release_frees_for_reuse():
  """Freeze copies and release frees for reuse."""
  p0, a0 = loaded(1.0)
  frozen = members.freeze(p0, a0)
  assert int(frozen["member"]) == 1
  p0.store[0][:] = 5.0  # the live member trains on
  np.testing.assert_array_equal(p0.weights(frozen), np.full(A, 1.0))
  members.release(p0, frozen)
  again = members.freeze(p0, a0)
  assert int(again["member"]) == 1  # the released storage is reused


def test_routed_members_recurse():
  """Routed members recurse."""
  (p0, a0), (p1, a1) = loaded(1.0), loaded(2.0)
  r0 = routed.RoutedPolicy(p0, uniform.UniformPolicy("x", A, 2), ROUTE)
  r1 = routed.RoutedPolicy(p1, uniform.UniformPolicy("x", A, 2), ROUTE)
  stacked = members.stack(
      r0, [(r0, r0.make_params(a0, {})), (r1, r1.make_params(a1, {}))]
  )
  np.testing.assert_array_equal(np.asarray(stacked["base"]["member"]), [0, 1])
  assert not stacked["override"]


def test_refuses_another_architecture_and_an_empty_slot():
  """Refuses another architecture and an empty slot."""
  p0, a0 = loaded(1.0)
  other = HostWeights(tag="y")
  with pytest.raises(ValueError, match="adopt"):
    members.adopt(p0, other, a0)
  with pytest.raises(ValueError, match="at least one"):
    members.stack(p0, [])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
