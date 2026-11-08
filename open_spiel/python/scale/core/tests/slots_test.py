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

"""core.slots.

Slot, SlotTable, TemperatureDrop and IN_GRAPH live in a module that imports no
array framework, and their validation holds. This file imports no jax.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import slots
from open_spiel.python.scale.core import types
from open_spiel.python.scale.launch import subprocess_env


class _Stub:

  def __init__(self, capability):
    self.spec = types.PolicySpec(
        "g", "o", 2, 2, False, capability, "numpy", "stub"
    )


def test_slots_imports_no_framework():
  """core.slots imports no array framework."""
  probe = (
      f"import sys\nimport {slots.__name__}\n"
      "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
      "('jax', 'jaxlib', 'flax', 'optax', 'orbax'))\nassert not bad, bad\n"
  )
  r = subprocess_env.run_python(["-c", probe], timeout=120)
  assert r.returncode == 0, r.stderr


def test_slot_in_graph_by_capability():
  """Slot.in_graph follows the policy's capability."""
  assert slots.IN_GRAPH == frozenset(
      {types.Capability.NATIVE_JAX, types.Capability.CALLBACK_BATCHED}
  )
  for cap in types.Capability:
    s = slots.Slot(_Stub(cap), 3, types.ARGMAX)
    assert s.num_members == 3 and s.rule == types.ARGMAX
    assert s.in_graph == (cap in slots.IN_GRAPH)
  with pytest.raises(AssertionError):
    slots.Slot(_Stub(types.Capability.NATIVE_JAX), 0)
  with pytest.raises(AssertionError):
    slots.Slot(_Stub(types.Capability.NATIVE_JAX), 1, "argmax")


def test_temperature_drop_validation():
  """TemperatureDrop validates its fields."""
  td = slots.TemperatureDrop(ply=3, temperature=0.5)
  s = slots.Slot(_Stub(types.Capability.NATIVE_JAX), 1, types.SAMPLE, td)
  assert s.temperature_drop is td
  for bad in ((0, 1.0), (2, 0.0), (2, float("inf")), (2, -1.0)):
    with pytest.raises(AssertionError):
      slots.TemperatureDrop(*bad)


def test_slot_table_fields():
  """SlotTable's fields."""
  t = slots.SlotTable(
      np.zeros((4, 2), np.int32),
      np.ones((4, 2), np.int32),
      np.arange(4, dtype=np.int32) % 2,
  )
  assert t._fields == ("seat_slot", "seat_member", "perspective_seat")
  assert t.seat_member.sum() == 8


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
