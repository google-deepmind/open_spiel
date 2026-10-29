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

"""The Arena's slot records: Slot, SlotTable, TemperatureDrop and IN_GRAPH.

They import no array framework, so the population-training control plane
(sampler tables, artifact-store slots, an evaluator's duplicate tables) can
build them without jax. core.arena re-exports them, and its module doc
describes how a rollout uses them.

  slot = Slot(policy, num_members=4, rule=types.SAMPLE,
              temperature_drop=TemperatureDrop(ply=8, temperature=1e-3))
"""

import dataclasses
import math
from typing import Any, NamedTuple, Optional

from open_spiel.python.scale.core import types

IN_GRAPH = frozenset(
    {types.Capability.NATIVE_JAX, types.Capability.CALLBACK_BATCHED}
)


@dataclasses.dataclass(frozen=True)
class TemperatureDrop:
  """The slot acts at `temperature` from the hand's 0-based ply `ply` on.

  Before it, at its rule's temperature (core.arena, Temperature drops).

  Attributes:
    ply: the hand's 0-based ply from which the slot acts at `temperature`.
    temperature: the temperature from `ply` on.
  """

  ply: int
  temperature: float

  def __post_init__(self):
    assert int(self.ply) >= 1, self.ply
    t = float(self.temperature)
    assert t > 0.0 and math.isfinite(t), self.temperature


@dataclasses.dataclass(frozen=True, eq=False)
class Slot:
  """One architecture in an Arena.

  a Policy, its decision rule, its number of stacked members K, and optionally
  a per-ply temperature drop.

  Attributes:
    policy: the Policy.
    num_members: K, the number of stacked members.
    rule: its decision rule.
    temperature_drop: an optional per-ply temperature drop.
  """

  policy: types.Policy
  num_members: int = 1
  rule: types.DecisionRule = types.SAMPLE
  temperature_drop: Optional[TemperatureDrop] = None

  def __post_init__(self):
    assert self.num_members >= 1, self.num_members
    assert isinstance(self.rule, types.DecisionRule), self.rule
    assert isinstance(self.policy.spec.capability, types.Capability)
    assert self.temperature_drop is None or isinstance(
        self.temperature_drop, TemperatureDrop
    ), self.temperature_drop

  @property
  def in_graph(self) -> bool:
    """Whether the slot runs in the jitted Arena (not HOST_DRIVEN)."""
    return self.policy.spec.capability in IN_GRAPH


class SlotTable(NamedTuple):
  """Which slot and member plays each seat of each row.

  It is traced, so a new table does not recompile the rollout.

  Attributes:
    seat_slot: i32[B, P] slot index per seat.
    seat_member: i32[B, P] member index (into that slot's [K] stack).
    perspective_seat: i32[B] or i32[B, M]: the seat(s) TrajectoryBatch rows are
      emitted for. With [B, M] the batch has B*M rows, row b*M + m for seat
      perspective_seat[b, m] (e.g. every seat of a self-play row, M = P). Every
      perspective seat must sit in the Arena's perspective slot.
  """

  seat_slot: Any
  seat_member: Any
  perspective_seat: Any
